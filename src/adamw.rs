use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp3, DType, Layout, Result, Shape, Storage, Tensor, Var};
use half::{bf16, f16};

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

impl AdamWPackedUpdateOp {
    fn cpu_grad_storage<'a>(&'a self) -> Result<(std::sync::RwLockReadGuard<'a, Storage>, &'a Layout)> {
        let (storage, layout) = self.grad.storage_and_layout();
        Ok((storage, layout))
    }
}

fn contiguous_slice<'a, T>(values: &'a [T], layout: &Layout, name: &'static str) -> Result<&'a [T]> {
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
        let next_theta = one_minus_lr_lambda * theta[i].to_f32() - lr * (m_hat / (v_hat.sqrt() + eps));
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
        let next_theta = one_minus_lr_lambda * theta[i].to_f32() - lr * (m_hat / (v_hat.sqrt() + eps));
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
            (CpuStorage::F32(theta), CpuStorage::F32(m), CpuStorage::F32(v), Storage::Cpu(CpuStorage::F32(g))) => {
                let theta = contiguous_slice(theta, l1, "theta")?;
                let m = contiguous_slice(m, l2, "first_moment")?;
                let v = contiguous_slice(v, l3, "second_moment")?;
                let g = contiguous_slice(g, g_layout, "grad")?;
                let out = cpu_pack_f32(theta, m, v, g, self.params);
                Ok((CpuStorage::F32(out), shape))
            }
            (CpuStorage::F16(theta), CpuStorage::F16(m), CpuStorage::F16(v), Storage::Cpu(CpuStorage::F16(g))) => {
                let theta = contiguous_slice(theta, l1, "theta")?;
                let m = contiguous_slice(m, l2, "first_moment")?;
                let v = contiguous_slice(v, l3, "second_moment")?;
                let g = contiguous_slice(g, g_layout, "grad")?;
                let out = cpu_pack_f16(theta, m, v, g, self.params);
                Ok((CpuStorage::F16(out), shape))
            }
            (CpuStorage::BF16(theta), CpuStorage::BF16(m), CpuStorage::BF16(v), Storage::Cpu(CpuStorage::BF16(g))) => {
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
        use objc2_metal::{MTLCompileOptions, MTLSize};

        let (g_storage, g_layout) = self.grad.storage_and_layout();
        let g_storage = match &*g_storage {
            Storage::Metal(storage) => storage,
            _ => candle_core::bail!("grad tensor must be on metal for metal fused AdamW op"),
        };

        if !l1.is_contiguous() || !l2.is_contiguous() || !l3.is_contiguous() || !g_layout.is_contiguous() {
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
        let options = MTLCompileOptions::new();
        let lib = metal
            .new_library_with_source(ADAMW_PACK_METAL, Some(&options))
            .map_err(candle_core::MetalError::from)?;
        let func = lib
            .get_function(kernel_name, None)
            .map_err(candle_core::MetalError::from)?;
        let pipeline = metal
            .new_compute_pipeline_state_with_function(&func)
            .map_err(candle_core::MetalError::from)?;

        let output = s1.device().new_buffer(3 * elem_count, dtype, "adamw-packed-out")?;
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
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "y" | "on"))
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
    if theta_t.dtype() != m_t.dtype() || theta_t.dtype() != v_t.dtype() || theta_t.dtype() != grad.dtype() {
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
