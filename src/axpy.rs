#[cfg(feature = "metal")]
use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp3, Layout, Result, Shape, Tensor};
#[cfg(feature = "metal")]
use candle_core::DType;
use half::{bf16, f16};
#[cfg(feature = "metal")]
use std::collections::HashMap;
#[cfg(feature = "metal")]
use std::sync::{OnceLock, RwLock};

#[derive(Clone, Debug)]
struct AxpyOp {
    alpha: f64,
}

#[cfg(feature = "metal")]
#[derive(Debug)]
struct AxpyMetalDeviceCache {
    library: candle_metal_kernels::metal::Library,
    pipelines: HashMap<&'static str, candle_metal_kernels::metal::ComputePipeline>,
}

#[cfg(feature = "metal")]
type AxpyMetalDeviceId = candle_core::metal_backend::DeviceId;

#[cfg(feature = "metal")]
static AXPY_METAL_CACHE: OnceLock<RwLock<HashMap<AxpyMetalDeviceId, AxpyMetalDeviceCache>>> =
    OnceLock::new();

#[cfg(feature = "metal")]
fn get_or_create_axpy_pipeline(
    device_id: AxpyMetalDeviceId,
    device: &candle_metal_kernels::metal::Device,
    kernel_name: &'static str,
) -> Result<candle_metal_kernels::metal::ComputePipeline> {
    let cache = AXPY_METAL_CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    let mut cache = cache
        .write()
        .map_err(|_| candle_core::Error::msg("axpy metal cache lock poisoned"))?;
    if let Some(dev_cache) = cache.get_mut(&device_id) {
        if let Some(pipeline) = dev_cache.pipelines.get(kernel_name) {
            return Ok(pipeline.clone());
        }
        let func = dev_cache
            .library
            .get_function(kernel_name, None)
            .map_err(|e| {
                candle_core::Error::msg(format!("failed loading axpy metal function: {e}"))
            })?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| {
                candle_core::Error::msg(format!("failed creating axpy metal pipeline: {e}"))
            })?;
        dev_cache.pipelines.insert(kernel_name, pipeline.clone());
        return Ok(pipeline);
    }

    let options = objc2_metal::MTLCompileOptions::new();
    let library = device
        .new_library_with_source(AXPY_METAL, Some(&options))
        .map_err(|e| candle_core::Error::msg(format!("failed compiling axpy metal source: {e}")))?;
    let func = library
        .get_function(kernel_name, None)
        .map_err(|e| candle_core::Error::msg(format!("failed loading axpy metal function: {e}")))?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| candle_core::Error::msg(format!("failed creating axpy metal pipeline: {e}")))?;

    let mut pipelines = HashMap::new();
    pipelines.insert(kernel_name, pipeline.clone());
    cache.insert(device_id, AxpyMetalDeviceCache { library, pipelines });
    Ok(pipeline)
}

fn contiguous_slice<'a, T>(values: &'a [T], layout: &Layout, name: &'static str) -> Result<&'a [T]> {
    match layout.contiguous_offsets() {
        Some((start, end)) => Ok(&values[start..end]),
        None => candle_core::bail!("{name} must be contiguous for fused axpy op"),
    }
}

fn cpu_axpy_f32(y: &[f32], x: &[f32], alpha: f64) -> Vec<f32> {
    let mut out = vec![0f32; y.len()];
    let alpha = alpha as f32;
    for i in 0..y.len() {
        out[i] = y[i] + alpha * x[i];
    }
    out
}

fn cpu_axpy_f16(y: &[f16], x: &[f16], alpha: f64) -> Vec<f16> {
    let mut out = vec![f16::from_f32(0.0); y.len()];
    let alpha = alpha as f32;
    for i in 0..y.len() {
        out[i] = f16::from_f32(y[i].to_f32() + alpha * x[i].to_f32());
    }
    out
}

fn cpu_axpy_bf16(y: &[bf16], x: &[bf16], alpha: f64) -> Vec<bf16> {
    let mut out = vec![bf16::from_f32(0.0); y.len()];
    let alpha = alpha as f32;
    for i in 0..y.len() {
        out[i] = bf16::from_f32(y[i].to_f32() + alpha * x[i].to_f32());
    }
    out
}

impl CustomOp3 for AxpyOp {
    fn name(&self) -> &'static str {
        "candelora-axpy-fused"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let elem_count = l1.shape().elem_count();
        if elem_count != l2.shape().elem_count() {
            candle_core::bail!("inconsistent tensor sizes in fused axpy op");
        }
        let shape = l1.shape().clone();
        match (s1, s2) {
            (CpuStorage::F32(y), CpuStorage::F32(x)) => {
                let y = contiguous_slice(y, l1, "y")?;
                let x = contiguous_slice(x, l2, "x")?;
                Ok((CpuStorage::F32(cpu_axpy_f32(y, x, self.alpha)), shape))
            }
            (CpuStorage::F16(y), CpuStorage::F16(x)) => {
                let y = contiguous_slice(y, l1, "y")?;
                let x = contiguous_slice(x, l2, "x")?;
                Ok((CpuStorage::F16(cpu_axpy_f16(y, x, self.alpha)), shape))
            }
            (CpuStorage::BF16(y), CpuStorage::BF16(x)) => {
                let y = contiguous_slice(y, l1, "y")?;
                let x = contiguous_slice(x, l2, "x")?;
                Ok((CpuStorage::BF16(cpu_axpy_bf16(y, x, self.alpha)), shape))
            }
            _ => candle_core::bail!("unsupported dtype combination for fused axpy op"),
        }
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
        s2: &candle_core::MetalStorage,
        l2: &Layout,
        _s3: &candle_core::MetalStorage,
        _l3: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_metal_kernels::BufferOffset;
        use objc2_metal::MTLSize;

        if !l1.is_contiguous() || !l2.is_contiguous() {
            candle_core::bail!("fused axpy op expects contiguous layouts");
        }
        let elem_count = l1.shape().elem_count();
        if elem_count != l2.shape().elem_count() {
            candle_core::bail!("inconsistent tensor sizes in fused axpy op");
        }
        if elem_count == 0 {
            let out = s1.device().new_buffer(0, s1.dtype(), "axpy-out")?;
            return Ok((
                candle_core::MetalStorage::new(out, s1.device().clone(), 0, s1.dtype()),
                l1.shape().clone(),
            ));
        }
        let dtype = s1.dtype();
        if dtype != s2.dtype() {
            candle_core::bail!("dtype mismatch in fused axpy op");
        }
        if s1.device().id() != s2.device().id() {
            candle_core::bail!("all tensors must be on the same metal device for fused axpy op");
        }
        let kernel_name = match dtype {
            DType::F32 => "axpy_f32",
            DType::F16 => "axpy_f16",
            DType::BF16 => "axpy_bf16",
            _ => candle_core::bail!("unsupported dtype for fused axpy op: {:?}", dtype),
        };

        let metal = s1.device().metal_device();
        let pipeline = get_or_create_axpy_pipeline(s1.device().id(), metal, kernel_name)?;

        let output = s1.device().new_buffer(elem_count, dtype, "axpy-out")?;
        let encoder = s1.device().command_encoder()?;
        encoder.set_label("candelora_axpy");
        encoder.set_compute_pipeline_state(&pipeline);

        let y = BufferOffset {
            buffer: s1.buffer(),
            offset_in_bytes: l1.start_offset() * dtype.size_in_bytes(),
        };
        let x = BufferOffset {
            buffer: s2.buffer(),
            offset_in_bytes: l2.start_offset() * dtype.size_in_bytes(),
        };
        let out = BufferOffset {
            buffer: &output,
            offset_in_bytes: 0,
        };

        let alpha = self.alpha as f32;
        let encoder_ref = &encoder;
        candle_metal_kernels::set_params!(encoder_ref, (elem_count, alpha, &y, &x, &out));

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
            candle_core::MetalStorage::new(output, s1.device().clone(), elem_count, dtype),
            l1.shape().clone(),
        ))
    }
}

#[cfg(feature = "metal")]
const AXPY_METAL: &str = include_str!("metal_src/axpy.metal");

pub fn axpy(y: &Tensor, x: &Tensor, alpha: f64) -> Result<Tensor> {
    if y.elem_count() != x.elem_count() {
        candle_core::bail!(
            "inconsistent tensor sizes in fused axpy op: y={}, x={}",
            y.elem_count(),
            x.elem_count()
        );
    }
    if y.dtype() != x.dtype() {
        candle_core::bail!(
            "dtype mismatch in fused axpy op: y={:?}, x={:?}",
            y.dtype(),
            x.dtype()
        );
    }
    if !y.device().same_device(x.device()) {
        candle_core::bail!(
            "device mismatch in fused axpy op: y={:?}, x={:?}",
            y.device(),
            x.device()
        );
    }
    y.apply_op3_no_bwd(x, x, &AxpyOp { alpha })
}

#[cfg(test)]
mod tests {
    use super::axpy;
    use candle_core::{DType, Device, Result, Tensor};

    fn assert_close(got: &[f32], expected: &[f32], atol: f32, rtol: f32) {
        assert_eq!(got.len(), expected.len());
        for (idx, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            let diff = (g - e).abs();
            let tol = atol + rtol * e.abs().max(g.abs());
            assert!(
                diff <= tol,
                "mismatch at {idx}: got={g} expected={e} diff={diff} tol={tol}"
            );
        }
    }

    fn run_case(device: &Device, dtype: DType) -> Result<()> {
        let y = Tensor::new(&[1.0f32, -2.0, 3.5, 0.0], device)?.to_dtype(dtype)?;
        let x = Tensor::new(&[4.0f32, 1.5, -2.0, 8.0], device)?.to_dtype(dtype)?;
        let out = axpy(&y, &x, 0.25)?;
        let out = out.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let expected = [2.0f32, -1.625, 3.0, 2.0];
        assert_close(&out, &expected, 1e-3, 1e-3);
        Ok(())
    }

    #[test]
    fn axpy_cpu_all_dtypes() -> Result<()> {
        let device = Device::Cpu;
        run_case(&device, DType::F32)?;
        run_case(&device, DType::F16)?;
        run_case(&device, DType::BF16)?;
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn axpy_metal_f32_f16() -> Result<()> {
        let device = match Device::metal_if_available(0) {
            Ok(d) if !d.is_cpu() => d,
            _ => return Ok(()),
        };
        run_case(&device, DType::F32)?;
        run_case(&device, DType::F16)?;
        Ok(())
    }
}
