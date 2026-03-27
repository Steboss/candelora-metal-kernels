#[cfg(feature = "metal")]
use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp3, DType, Layout, Result, Shape, Storage, Tensor, Var};
use half::{bf16, f16};
#[cfg(feature = "metal")]
use std::collections::HashMap;
#[cfg(feature = "metal")]
use std::sync::{OnceLock, RwLock};

#[derive(Clone, Copy, Debug)]
pub struct AdamWStepParams {
    pub beta1: f64,
    pub beta2: f64,
    pub scale_m: f64,
    pub scale_v: f64,
    pub eps: f64,
    pub lr: f64,
    pub lr_lambda: f64,
}

#[derive(Clone, Debug)]
struct AdamWPackedUpdateOp {
    grad: Tensor,
    params: AdamWStepParams,
}

#[cfg(feature = "metal")]
#[derive(Debug)]
struct AdamWMetalDeviceCache {
    library: candle_metal_kernels::metal::Library,
    pipelines: HashMap<&'static str, candle_metal_kernels::metal::ComputePipeline>,
}

#[cfg(feature = "metal")]
type AdamWMetalDeviceId = candle_core::metal_backend::DeviceId;

#[cfg(feature = "metal")]
static ADAMW_METAL_CACHE: OnceLock<RwLock<HashMap<AdamWMetalDeviceId, AdamWMetalDeviceCache>>> =
    OnceLock::new();

impl AdamWPackedUpdateOp {
    fn cpu_grad_storage<'a>(
        &'a self,
    ) -> Result<(std::sync::RwLockReadGuard<'a, Storage>, &'a Layout)> {
        let (storage, layout) = self.grad.storage_and_layout();
        Ok((storage, layout))
    }
}

#[cfg(feature = "metal")]
fn get_or_create_adamw_pipeline(
    device_id: AdamWMetalDeviceId,
    device: &candle_metal_kernels::metal::Device,
    kernel_name: &'static str,
) -> Result<candle_metal_kernels::metal::ComputePipeline> {
    let cache = ADAMW_METAL_CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    let mut cache = cache
        .write()
        .map_err(|_| candle_core::Error::msg("adamw metal cache lock poisoned"))?;
    if let Some(dev_cache) = cache.get_mut(&device_id) {
        if let Some(pipeline) = dev_cache.pipelines.get(kernel_name) {
            return Ok(pipeline.clone());
        }
        let func = dev_cache
            .library
            .get_function(kernel_name, None)
            .map_err(|e| {
                candle_core::Error::msg(format!("failed loading adamw metal function: {e}"))
            })?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| {
                candle_core::Error::msg(format!("failed creating adamw metal pipeline: {e}"))
            })?;
        dev_cache.pipelines.insert(kernel_name, pipeline.clone());
        return Ok(pipeline);
    }

    let options = objc2_metal::MTLCompileOptions::new();
    let library = device
        .new_library_with_source(ADAMW_PACK_METAL, Some(&options))
        .map_err(|e| {
            candle_core::Error::msg(format!("failed compiling adamw metal source: {e}"))
        })?;
    let func = library.get_function(kernel_name, None).map_err(|e| {
        candle_core::Error::msg(format!("failed loading adamw metal function: {e}"))
    })?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| {
            candle_core::Error::msg(format!("failed creating adamw metal pipeline: {e}"))
        })?;

    let mut pipelines = HashMap::new();
    pipelines.insert(kernel_name, pipeline.clone());
    cache.insert(device_id, AdamWMetalDeviceCache { library, pipelines });
    Ok(pipeline)
}

fn contiguous_slice<'a, T>(
    values: &'a [T],
    layout: &Layout,
    name: &'static str,
) -> Result<&'a [T]> {
    match layout.contiguous_offsets() {
        Some((start, end)) => Ok(&values[start..end]),
        None => candle_core::bail!("{name} must be contiguous for fused AdamW op"),
    }
}

fn cpu_pack_f32(
    theta: &[f32],
    m: &[f32],
    v: &[f32],
    g: &[f32],
    params: AdamWStepParams,
) -> Vec<f32> {
    let n = theta.len();
    let mut out = vec![0f32; 3 * n];
    let beta1 = params.beta1 as f32;
    let beta2 = params.beta2 as f32;
    let scale_m = params.scale_m as f32;
    let scale_v = params.scale_v as f32;
    let eps = params.eps as f32;
    let lr = params.lr as f32;
    let one_minus_lr_lambda = (1f64 - params.lr_lambda) as f32;
    let one_minus_beta1 = 1f32 - beta1;
    let one_minus_beta2 = 1f32 - beta2;
    for i in 0..n {
        let gi = g[i];
        let next_m = beta1 * m[i] + one_minus_beta1 * gi;
        let next_v = beta2 * v[i] + one_minus_beta2 * gi * gi;
        let m_hat = next_m * scale_m;
        let v_hat = next_v * scale_v;
        let next_theta = one_minus_lr_lambda * theta[i] - lr * (m_hat / (v_hat.sqrt() + eps));
        out[i] = next_m;
        out[i + n] = next_v;
        out[i + 2 * n] = next_theta;
    }
    out
}

fn cpu_pack_f16(
    theta: &[f16],
    m: &[f16],
    v: &[f16],
    g: &[f16],
    params: AdamWStepParams,
) -> Vec<f16> {
    let n = theta.len();
    let mut out = vec![f16::from_f32(0.0); 3 * n];
    let beta1 = params.beta1 as f32;
    let beta2 = params.beta2 as f32;
    let scale_m = params.scale_m as f32;
    let scale_v = params.scale_v as f32;
    let eps = params.eps as f32;
    let lr = params.lr as f32;
    let one_minus_lr_lambda = (1f64 - params.lr_lambda) as f32;
    let one_minus_beta1 = 1f32 - beta1;
    let one_minus_beta2 = 1f32 - beta2;
    for i in 0..n {
        let gi = g[i].to_f32();
        let next_m = beta1 * m[i].to_f32() + one_minus_beta1 * gi;
        let next_v = beta2 * v[i].to_f32() + one_minus_beta2 * gi * gi;
        let m_hat = next_m * scale_m;
        let v_hat = next_v * scale_v;
        let next_theta =
            one_minus_lr_lambda * theta[i].to_f32() - lr * (m_hat / (v_hat.sqrt() + eps));
        out[i] = f16::from_f32(next_m);
        out[i + n] = f16::from_f32(next_v);
        out[i + 2 * n] = f16::from_f32(next_theta);
    }
    out
}

fn cpu_pack_bf16(
    theta: &[bf16],
    m: &[bf16],
    v: &[bf16],
    g: &[bf16],
    params: AdamWStepParams,
) -> Vec<bf16> {
    let n = theta.len();
    let mut out = vec![bf16::from_f32(0.0); 3 * n];
    let beta1 = params.beta1 as f32;
    let beta2 = params.beta2 as f32;
    let scale_m = params.scale_m as f32;
    let scale_v = params.scale_v as f32;
    let eps = params.eps as f32;
    let lr = params.lr as f32;
    let one_minus_lr_lambda = (1f64 - params.lr_lambda) as f32;
    let one_minus_beta1 = 1f32 - beta1;
    let one_minus_beta2 = 1f32 - beta2;
    for i in 0..n {
        let gi = g[i].to_f32();
        let next_m = beta1 * m[i].to_f32() + one_minus_beta1 * gi;
        let next_v = beta2 * v[i].to_f32() + one_minus_beta2 * gi * gi;
        let m_hat = next_m * scale_m;
        let v_hat = next_v * scale_v;
        let next_theta =
            one_minus_lr_lambda * theta[i].to_f32() - lr * (m_hat / (v_hat.sqrt() + eps));
        out[i] = bf16::from_f32(next_m);
        out[i + n] = bf16::from_f32(next_v);
        out[i + 2 * n] = bf16::from_f32(next_theta);
    }
    out
}

impl CustomOp3 for AdamWPackedUpdateOp {
    fn name(&self) -> &'static str {
        "candelora-adamw-packed-update"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (g_storage, g_layout) = self.cpu_grad_storage()?;
        let elem_count = l1.shape().elem_count();
        if elem_count != l2.shape().elem_count()
            || elem_count != l3.shape().elem_count()
            || elem_count != g_layout.shape().elem_count()
        {
            candle_core::bail!("inconsistent tensor sizes in fused AdamW op");
        }
        let shape = Shape::from((3, elem_count));
        match (s1, s2, s3, &*g_storage) {
            (
                CpuStorage::F32(theta),
                CpuStorage::F32(m),
                CpuStorage::F32(v),
                Storage::Cpu(CpuStorage::F32(g)),
            ) => {
                let theta = contiguous_slice(theta, l1, "theta")?;
                let m = contiguous_slice(m, l2, "first_moment")?;
                let v = contiguous_slice(v, l3, "second_moment")?;
                let g = contiguous_slice(g, g_layout, "grad")?;
                let out = cpu_pack_f32(theta, m, v, g, self.params);
                Ok((CpuStorage::F32(out), shape))
            }
            (
                CpuStorage::F16(theta),
                CpuStorage::F16(m),
                CpuStorage::F16(v),
                Storage::Cpu(CpuStorage::F16(g)),
            ) => {
                let theta = contiguous_slice(theta, l1, "theta")?;
                let m = contiguous_slice(m, l2, "first_moment")?;
                let v = contiguous_slice(v, l3, "second_moment")?;
                let g = contiguous_slice(g, g_layout, "grad")?;
                let out = cpu_pack_f16(theta, m, v, g, self.params);
                Ok((CpuStorage::F16(out), shape))
            }
            (
                CpuStorage::BF16(theta),
                CpuStorage::BF16(m),
                CpuStorage::BF16(v),
                Storage::Cpu(CpuStorage::BF16(g)),
            ) => {
                let theta = contiguous_slice(theta, l1, "theta")?;
                let m = contiguous_slice(m, l2, "first_moment")?;
                let v = contiguous_slice(v, l3, "second_moment")?;
                let g = contiguous_slice(g, g_layout, "grad")?;
                let out = cpu_pack_bf16(theta, m, v, g, self.params);
                Ok((CpuStorage::BF16(out), shape))
            }
            _ => candle_core::bail!("unsupported dtype/device combination for fused AdamW op"),
        }
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
        s2: &candle_core::MetalStorage,
        l2: &Layout,
        s3: &candle_core::MetalStorage,
        l3: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_core::Storage;
        use candle_metal_kernels::BufferOffset;
        use objc2_metal::MTLSize;

        let (g_storage, g_layout) = self.grad.storage_and_layout();
        let g_storage = match &*g_storage {
            Storage::Metal(storage) => storage,
            _ => candle_core::bail!("grad tensor must be on metal for metal fused AdamW op"),
        };

        if !l1.is_contiguous()
            || !l2.is_contiguous()
            || !l3.is_contiguous()
            || !g_layout.is_contiguous()
        {
            candle_core::bail!("fused AdamW op expects contiguous layouts");
        }

        let elem_count = l1.shape().elem_count();
        if elem_count != l2.shape().elem_count()
            || elem_count != l3.shape().elem_count()
            || elem_count != g_layout.shape().elem_count()
        {
            candle_core::bail!("inconsistent tensor sizes in fused AdamW op");
        }
        if elem_count == 0 {
            let out = s1.device().new_buffer(0, s1.dtype(), "adamw-packed-out")?;
            return Ok((
                candle_core::MetalStorage::new(out, s1.device().clone(), 0, s1.dtype()),
                Shape::from((3, 0usize)),
            ));
        }

        let dtype = s1.dtype();
        if dtype != s2.dtype() || dtype != s3.dtype() || dtype != g_storage.dtype() {
            candle_core::bail!("dtype mismatch in fused AdamW op");
        }

        let kernel_name = match dtype {
            DType::F32 => "adamw_pack_f32",
            DType::F16 => "adamw_pack_f16",
            DType::BF16 => "adamw_pack_bf16",
            _ => candle_core::bail!("unsupported dtype for fused AdamW op: {:?}", dtype),
        };

        if s1.device().id() != s2.device().id()
            || s1.device().id() != s3.device().id()
            || s1.device().id() != g_storage.device().id()
        {
            candle_core::bail!("all tensors must be on the same metal device");
        }

        let metal = s1.device().metal_device();
        let pipeline = get_or_create_adamw_pipeline(s1.device().id(), metal, kernel_name)?;

        let output = s1
            .device()
            .new_buffer(3 * elem_count, dtype, "adamw-packed-out")?;
        let encoder = s1.device().command_encoder()?;
        encoder.set_label("candelora_adamw_pack");
        encoder.set_compute_pipeline_state(&pipeline);

        let theta = BufferOffset {
            buffer: s1.buffer(),
            offset_in_bytes: l1.start_offset() * dtype.size_in_bytes(),
        };
        let first_moment = BufferOffset {
            buffer: s2.buffer(),
            offset_in_bytes: l2.start_offset() * dtype.size_in_bytes(),
        };
        let second_moment = BufferOffset {
            buffer: s3.buffer(),
            offset_in_bytes: l3.start_offset() * dtype.size_in_bytes(),
        };
        let grad = BufferOffset {
            buffer: g_storage.buffer(),
            offset_in_bytes: g_layout.start_offset() * dtype.size_in_bytes(),
        };
        let out = BufferOffset {
            buffer: &output,
            offset_in_bytes: 0,
        };

        let beta1 = self.params.beta1 as f32;
        let beta2 = self.params.beta2 as f32;
        let scale_m = self.params.scale_m as f32;
        let scale_v = self.params.scale_v as f32;
        let eps = self.params.eps as f32;
        let lr = self.params.lr as f32;
        let lr_lambda = self.params.lr_lambda as f32;
        let encoder_ref = &encoder;

        candle_metal_kernels::set_params!(
            encoder_ref,
            (
                elem_count,
                beta1,
                beta2,
                scale_m,
                scale_v,
                eps,
                lr,
                lr_lambda,
                &theta,
                &first_moment,
                &second_moment,
                &grad,
                &out
            )
        );

        let threads = pipeline
            .max_total_threads_per_threadgroup()
            .min(elem_count.max(1));
        let groups = elem_count.div_ceil(threads);
        let tg_count = MTLSize {
            width: groups,
            height: 1,
            depth: 1,
        };
        let tg_size = MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(tg_count, tg_size);

        Ok((
            candle_core::MetalStorage::new(output, s1.device().clone(), 3 * elem_count, dtype),
            Shape::from((3, elem_count)),
        ))
    }
}

#[cfg(feature = "metal")]
const ADAMW_PACK_METAL: &str = include_str!("metal_src/adamw_pack.metal");

pub fn adamw_packed_update(
    theta: &Tensor,
    first_moment: &Tensor,
    second_moment: &Tensor,
    grad: &Tensor,
    params: AdamWStepParams,
) -> Result<Tensor> {
    theta.apply_op3_no_bwd(
        first_moment,
        second_moment,
        &AdamWPackedUpdateOp {
            grad: grad.clone(),
            params,
        },
    )
}

pub fn adamw_step_in_place(
    theta: &Var,
    first_moment: &Var,
    second_moment: &Var,
    grad: &Tensor,
    params: AdamWStepParams,
) -> Result<bool> {
    if std::env::var("CANDLORA_DISABLE_FUSED_ADAMW_METAL")
        .map(|v| {
            matches!(
                v.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "y" | "on"
            )
        })
        .unwrap_or(false)
    {
        return Ok(false);
    }

    let theta_t = theta.as_tensor();
    let m_t = first_moment.as_tensor();
    let v_t = second_moment.as_tensor();

    if theta_t.elem_count() != m_t.elem_count()
        || theta_t.elem_count() != v_t.elem_count()
        || theta_t.elem_count() != grad.elem_count()
    {
        return Ok(false);
    }
    if theta_t.dtype() != m_t.dtype()
        || theta_t.dtype() != v_t.dtype()
        || theta_t.dtype() != grad.dtype()
    {
        return Ok(false);
    }
    match theta_t.dtype() {
        DType::F32 | DType::F16 | DType::BF16 => {}
        _ => return Ok(false),
    }
    if !theta_t.device().same_device(m_t.device())
        || !theta_t.device().same_device(v_t.device())
        || !theta_t.device().same_device(grad.device())
    {
        return Ok(false);
    }

    let packed = match adamw_packed_update(theta_t, m_t, v_t, grad, params) {
        Ok(t) => t,
        Err(_) => return Ok(false),
    };
    let n = theta_t.elem_count();
    let flat = packed.flatten_all()?;
    let next_m = flat.narrow(0, 0, n)?.reshape(theta_t.dims())?;
    let next_v = flat.narrow(0, n, n)?.reshape(theta_t.dims())?;
    let next_theta = flat.narrow(0, 2 * n, n)?.reshape(theta_t.dims())?;
    first_moment.set(&next_m)?;
    second_moment.set(&next_v)?;
    theta.set(&next_theta)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_all_close(actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - e).abs();
            let tol = atol + rtol * e.abs();
            assert!(
                diff <= tol,
                "mismatch at index {idx}: actual={a} expected={e} diff={diff} tol={tol}"
            );
        }
    }

    fn tolerances(dtype: DType) -> (f32, f32) {
        match dtype {
            DType::F32 => (1e-6, 1e-5),
            DType::F16 => (3e-3, 3e-3),
            DType::BF16 => (3e-2, 3e-2),
            _ => (1e-6, 1e-5),
        }
    }

    fn reference_adamw(
        theta: &[f32],
        m: &[f32],
        v: &[f32],
        g: &[f32],
        p: AdamWStepParams,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut next_m = vec![0f32; theta.len()];
        let mut next_v = vec![0f32; theta.len()];
        let mut next_theta = vec![0f32; theta.len()];
        let beta1 = p.beta1 as f32;
        let beta2 = p.beta2 as f32;
        let one_minus_beta1 = 1f32 - beta1;
        let one_minus_beta2 = 1f32 - beta2;
        let scale_m = p.scale_m as f32;
        let scale_v = p.scale_v as f32;
        let eps = p.eps as f32;
        let lr = p.lr as f32;
        let one_minus_lr_lambda = (1f64 - p.lr_lambda) as f32;
        for i in 0..theta.len() {
            let nm = beta1 * m[i] + one_minus_beta1 * g[i];
            let nv = beta2 * v[i] + one_minus_beta2 * g[i] * g[i];
            let m_hat = nm * scale_m;
            let v_hat = nv * scale_v;
            let nt = one_minus_lr_lambda * theta[i] - lr * (m_hat / (v_hat.sqrt() + eps));
            next_m[i] = nm;
            next_v[i] = nv;
            next_theta[i] = nt;
        }
        (next_m, next_v, next_theta)
    }

    fn tensor_to_f32_vec(t: &Tensor) -> Result<Vec<f32>> {
        t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()
    }

    fn params_for_step(step_idx: usize) -> AdamWStepParams {
        let beta1 = 0.9f64;
        let beta2 = 0.999f64;
        let step = (step_idx + 1) as i32;
        AdamWStepParams {
            beta1: 0.9,
            beta2: 0.999,
            scale_m: 1.0 / (1.0 - beta1.powi(step)),
            scale_v: 1.0 / (1.0 - beta2.powi(step)),
            eps: 1e-8,
            lr: 1e-3,
            lr_lambda: 0.01,
        }
    }

    fn make_theta(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 * 0.137).sin() * 0.7) - 0.2)
            .collect()
    }

    fn make_first_moment(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 * 0.097).cos() * 0.04) - 0.01)
            .collect()
    }

    fn make_second_moment(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| 0.001 + ((i as f32 * 0.071).sin().abs() * 0.009))
            .collect()
    }

    fn make_grad(len: usize, step: usize) -> Vec<f32> {
        let s = step as f32 + 1.0;
        (0..len)
            .map(|i| (((i as f32 + 1.0) * 0.113 + s * 0.037).sin()) * 0.25)
            .collect()
    }

    fn unfused_step_in_place(
        theta: &Var,
        first_moment: &Var,
        second_moment: &Var,
        grad: &Tensor,
        params: AdamWStepParams,
    ) -> Result<()> {
        let next_m = ((first_moment.as_tensor() * params.beta1)? + (grad * (1.0 - params.beta1))?)?;
        let next_v =
            ((second_moment.as_tensor() * params.beta2)? + (grad.sqr()? * (1.0 - params.beta2))?)?;
        let m_hat = (&next_m * params.scale_m)?;
        let v_hat = (&next_v * params.scale_v)?;
        let next_theta = (theta.as_tensor() * (1f64 - params.lr_lambda))?;
        let adjusted_grad = (m_hat / (v_hat.sqrt()? + params.eps)?)?;
        let next_theta = (next_theta - (adjusted_grad * params.lr)?)?;
        first_moment.set(&next_m)?;
        second_moment.set(&next_v)?;
        theta.set(&next_theta)?;
        Ok(())
    }

    fn run_packed_update_case(
        device: &candle_core::Device,
        dtype: DType,
        shape: (usize, usize),
        atol: f32,
        rtol: f32,
    ) -> Result<()> {
        let len = shape.0 * shape.1;
        let theta = Tensor::from_vec(make_theta(len), shape, device)?.to_dtype(dtype)?;
        let first_moment =
            Tensor::from_vec(make_first_moment(len), shape, device)?.to_dtype(dtype)?;
        let second_moment =
            Tensor::from_vec(make_second_moment(len), shape, device)?.to_dtype(dtype)?;
        let grad = Tensor::from_vec(make_grad(len, 0), shape, device)?.to_dtype(dtype)?;
        let params = params_for_step(0);
        let theta_ref = tensor_to_f32_vec(&theta)?;
        let m_ref = tensor_to_f32_vec(&first_moment)?;
        let v_ref = tensor_to_f32_vec(&second_moment)?;
        let g_ref = tensor_to_f32_vec(&grad)?;
        let (m_expected, v_expected, theta_expected) =
            reference_adamw(&theta_ref, &m_ref, &v_ref, &g_ref, params);

        let packed = adamw_packed_update(&theta, &first_moment, &second_moment, &grad, params)?;
        let packed = tensor_to_f32_vec(&packed)?;
        let n = theta_ref.len();
        assert_eq!(packed.len(), 3 * n);
        assert_all_close(&packed[0..n], &m_expected, atol, rtol);
        assert_all_close(&packed[n..2 * n], &v_expected, atol, rtol);
        assert_all_close(&packed[2 * n..3 * n], &theta_expected, atol, rtol);
        Ok(())
    }

    fn run_fused_vs_unfused_sequence_case(
        device: &candle_core::Device,
        dtype: DType,
        shape: (usize, usize),
        steps: usize,
        atol: f32,
        rtol: f32,
    ) -> Result<()> {
        let len = shape.0 * shape.1;
        let theta_init = Tensor::from_vec(make_theta(len), shape, device)?.to_dtype(dtype)?;
        let m_init = Tensor::from_vec(make_first_moment(len), shape, device)?.to_dtype(dtype)?;
        let v_init = Tensor::from_vec(make_second_moment(len), shape, device)?.to_dtype(dtype)?;

        let fused_theta = Var::from_tensor(&theta_init)?;
        let fused_m = Var::from_tensor(&m_init)?;
        let fused_v = Var::from_tensor(&v_init)?;
        let unfused_theta = Var::from_tensor(&theta_init)?;
        let unfused_m = Var::from_tensor(&m_init)?;
        let unfused_v = Var::from_tensor(&v_init)?;

        for step in 0..steps {
            let params = params_for_step(step);
            let grad = Tensor::from_vec(make_grad(len, step), shape, device)?.to_dtype(dtype)?;
            let fused_used = adamw_step_in_place(&fused_theta, &fused_m, &fused_v, &grad, params)?;
            assert!(
                fused_used,
                "expected fused AdamW path to run for dtype={dtype:?} shape={shape:?}"
            );
            unfused_step_in_place(&unfused_theta, &unfused_m, &unfused_v, &grad, params)?;

            let fused_m_vals = tensor_to_f32_vec(fused_m.as_tensor())?;
            let fused_v_vals = tensor_to_f32_vec(fused_v.as_tensor())?;
            let fused_theta_vals = tensor_to_f32_vec(fused_theta.as_tensor())?;
            let unfused_m_vals = tensor_to_f32_vec(unfused_m.as_tensor())?;
            let unfused_v_vals = tensor_to_f32_vec(unfused_v.as_tensor())?;
            let unfused_theta_vals = tensor_to_f32_vec(unfused_theta.as_tensor())?;

            assert_all_close(&fused_m_vals, &unfused_m_vals, atol, rtol);
            assert_all_close(&fused_v_vals, &unfused_v_vals, atol, rtol);
            assert_all_close(&fused_theta_vals, &unfused_theta_vals, atol, rtol);
        }
        Ok(())
    }

    #[test]
    fn packed_update_matches_reference_cpu_all_dtypes_and_shapes() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let dtypes = [DType::F32, DType::F16, DType::BF16];
        let shapes = [(2usize, 3usize), (4, 4), (1, 17)];
        for dtype in dtypes {
            let (atol, rtol) = tolerances(dtype);
            for shape in shapes {
                run_packed_update_case(&device, dtype, shape, atol, rtol)?;
            }
        }
        Ok(())
    }

    #[test]
    fn fused_vs_unfused_across_steps_cpu_all_dtypes_and_shapes() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let dtypes = [DType::F32, DType::F16, DType::BF16];
        let shapes = [(2usize, 3usize), (4, 4), (1, 17)];
        for dtype in dtypes {
            let (atol, rtol) = tolerances(dtype);
            for shape in shapes {
                run_fused_vs_unfused_sequence_case(&device, dtype, shape, 5, atol, rtol)?;
            }
        }
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn fused_vs_unfused_metal_f32_f16() -> Result<()> {
        let device = match candle_core::Device::metal_if_available(0) {
            Ok(device) => device,
            Err(_) => return Ok(()),
        };
        for dtype in [DType::F32, DType::F16] {
            let (atol, rtol) = tolerances(dtype);
            run_fused_vs_unfused_sequence_case(&device, dtype, (8, 8), 5, atol, rtol)?;
        }
        Ok(())
    }
}
