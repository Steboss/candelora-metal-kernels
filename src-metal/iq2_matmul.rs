#[cfg(feature = "metal")]
use candle_core::backend::BackendStorage;
#[cfg(feature = "metal")]
use candle_core::DType;
use candle_core::{Result, Tensor};
#[cfg(feature = "metal")]
use std::collections::HashMap;
#[cfg(feature = "metal")]
use std::sync::{OnceLock, RwLock};

#[cfg(feature = "metal")]
const IQ2_QK: usize = 256;
#[cfg(feature = "metal")]
const IQ2_XXS_BYTES_PER_BLOCK: usize = 64;
#[cfg(feature = "metal")]
const IQ2_XXS_SCALES_PER_BLOCK: usize = 2;
#[cfg(feature = "metal")]
const IQ2_XS_BYTES_PER_BLOCK: usize = 64;
#[cfg(feature = "metal")]
const IQ2_XS_SCALES_PER_BLOCK: usize = 10;
#[cfg(feature = "metal")]
const IQ2_S_BYTES_PER_BLOCK: usize = 72;
#[cfg(feature = "metal")]
const IQ2_S_SCALES_PER_BLOCK: usize = 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Iq2MatmulVariant {
    Iq2Xxs,
    Iq2Xs,
    Iq2S,
}

#[cfg(feature = "metal")]
impl Iq2MatmulVariant {
    fn bytes_per_block(self) -> usize {
        match self {
            Self::Iq2Xxs => IQ2_XXS_BYTES_PER_BLOCK,
            Self::Iq2Xs => IQ2_XS_BYTES_PER_BLOCK,
            Self::Iq2S => IQ2_S_BYTES_PER_BLOCK,
        }
    }

    fn scales_per_block(self) -> usize {
        match self {
            Self::Iq2Xxs => IQ2_XXS_SCALES_PER_BLOCK,
            Self::Iq2Xs => IQ2_XS_SCALES_PER_BLOCK,
            Self::Iq2S => IQ2_S_SCALES_PER_BLOCK,
        }
    }

    fn kernel_name(self, dtype: DType) -> Result<&'static str> {
        Ok(match (self, dtype) {
            (Self::Iq2Xxs, DType::F32) => "iq2_xxs_matmul_f32",
            (Self::Iq2Xxs, DType::F16) => "iq2_xxs_matmul_f16",
            (Self::Iq2Xxs, DType::BF16) => "iq2_xxs_matmul_bf16",
            (Self::Iq2Xs, DType::F32) => "iq2_xs_matmul_f32",
            (Self::Iq2Xs, DType::F16) => "iq2_xs_matmul_f16",
            (Self::Iq2Xs, DType::BF16) => "iq2_xs_matmul_bf16",
            (Self::Iq2S, DType::F32) => "iq2_s_matmul_f32",
            (Self::Iq2S, DType::F16) => "iq2_s_matmul_f16",
            (Self::Iq2S, DType::BF16) => "iq2_s_matmul_bf16",
            (_, dt) => candle_core::bail!("iq2_matmul unsupported x dtype: {:?}", dt),
        })
    }
}

#[cfg(feature = "metal")]
#[derive(Debug)]
struct Iq2MatmulMetalDeviceCache {
    library: candle_metal_kernels::metal::Library,
    pipelines: HashMap<&'static str, candle_metal_kernels::metal::ComputePipeline>,
}

#[cfg(feature = "metal")]
type Iq2MatmulMetalDeviceId = candle_core::metal_backend::DeviceId;

#[cfg(feature = "metal")]
static IQ2_MATMUL_METAL_CACHE: OnceLock<RwLock<HashMap<Iq2MatmulMetalDeviceId, Iq2MatmulMetalDeviceCache>>> =
    OnceLock::new();

#[cfg(feature = "metal")]
fn get_or_create_iq2_matmul_pipeline(
    device_id: Iq2MatmulMetalDeviceId,
    device: &candle_metal_kernels::metal::Device,
    kernel_name: &'static str,
) -> Result<candle_metal_kernels::metal::ComputePipeline> {
    let cache = IQ2_MATMUL_METAL_CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    let mut cache = cache
        .write()
        .map_err(|_| candle_core::Error::msg("iq2-matmul metal cache lock poisoned"))?;
    if let Some(dev_cache) = cache.get_mut(&device_id) {
        if let Some(pipeline) = dev_cache.pipelines.get(kernel_name) {
            return Ok(pipeline.clone());
        }
        let func = dev_cache
            .library
            .get_function(kernel_name, None)
            .map_err(|e| {
                candle_core::Error::msg(format!("failed loading iq2-matmul metal function: {e}"))
            })?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| {
                candle_core::Error::msg(format!("failed creating iq2-matmul metal pipeline: {e}"))
            })?;
        dev_cache.pipelines.insert(kernel_name, pipeline.clone());
        return Ok(pipeline);
    }

    let options = objc2_metal::MTLCompileOptions::new();
    let library = device
        .new_library_with_source(IQ2_MATMUL_METAL, Some(&options))
        .map_err(|e| {
            candle_core::Error::msg(format!("failed compiling iq2-matmul metal source: {e}"))
        })?;
    let func = library
        .get_function(kernel_name, None)
        .map_err(|e| candle_core::Error::msg(format!("failed loading iq2-matmul metal function: {e}")))?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| candle_core::Error::msg(format!("failed creating iq2-matmul metal pipeline: {e}")))?;

    let mut pipelines = HashMap::new();
    pipelines.insert(kernel_name, pipeline.clone());
    cache.insert(device_id, Iq2MatmulMetalDeviceCache { library, pipelines });
    Ok(pipeline)
}

#[cfg(feature = "metal")]
fn parse_x_shape(x: &Tensor, in_dim: usize) -> Result<(Tensor, usize, bool)> {
    match x.dims() {
        [d] => {
            if *d != in_dim {
                candle_core::bail!(
                    "iq2_matmul input mismatch: x dim={} expected in_dim={}",
                    d,
                    in_dim
                );
            }
            Ok((x.reshape((1, in_dim))?, 1, true))
        }
        [m, d] => {
            if *d != in_dim {
                candle_core::bail!(
                    "iq2_matmul input mismatch: x last dim={} expected in_dim={}",
                    d,
                    in_dim
                );
            }
            Ok((x.clone(), *m, false))
        }
        dims => candle_core::bail!(
            "iq2_matmul expects x rank 1 or 2, got shape {:?}",
            dims
        ),
    }
}

#[cfg(feature = "metal")]
fn iq2_matmul_metal(
    x: &Tensor,
    weight_bytes: &Tensor,
    weight_scales: &Tensor,
    out_dim: usize,
    in_dim: usize,
    variant: Iq2MatmulVariant,
) -> Result<Tensor> {
    use candle_metal_kernels::BufferOffset;
    use objc2_metal::MTLSize;

    if out_dim == 0 || in_dim == 0 {
        candle_core::bail!("iq2_matmul expects non-zero dimensions")
    }
    if !x.device().is_metal() {
        candle_core::bail!("iq2_matmul requires a Metal device");
    }
    if !x.device().same_device(weight_bytes.device()) || !x.device().same_device(weight_scales.device()) {
        candle_core::bail!(
            "iq2_matmul device mismatch: x={:?}, bytes={:?}, scales={:?}",
            x.device(),
            weight_bytes.device(),
            weight_scales.device()
        );
    }
    if weight_bytes.dtype() != DType::U8 || weight_scales.dtype() != DType::U8 {
        candle_core::bail!(
            "iq2_matmul expects U8 weight bytes/scales, got bytes={:?}, scales={:?}",
            weight_bytes.dtype(),
            weight_scales.dtype()
        );
    }

    let (x2, m, squeeze_out) = parse_x_shape(x, in_dim)?;
    let x2 = x2.contiguous()?;

    let blocks_per_row = in_dim.div_ceil(IQ2_QK);
    let expected_bytes = out_dim
        .checked_mul(blocks_per_row)
        .and_then(|v| v.checked_mul(variant.bytes_per_block()))
        .ok_or_else(|| candle_core::Error::msg("iq2_matmul bytes size overflow"))?;
    let expected_scales = out_dim
        .checked_mul(blocks_per_row)
        .and_then(|v| v.checked_mul(variant.scales_per_block()))
        .ok_or_else(|| candle_core::Error::msg("iq2_matmul scales size overflow"))?;
    if weight_bytes.elem_count() < expected_bytes {
        candle_core::bail!(
            "iq2_matmul weight_bytes too small: got {} expected at least {}",
            weight_bytes.elem_count(),
            expected_bytes
        );
    }
    if weight_scales.elem_count() < expected_scales {
        candle_core::bail!(
            "iq2_matmul weight_scales too small: got {} expected at least {}",
            weight_scales.elem_count(),
            expected_scales
        );
    }

    let out = Tensor::zeros((m, out_dim), DType::F32, x.device())?;
    let total = m
        .checked_mul(out_dim)
        .ok_or_else(|| candle_core::Error::msg("iq2_matmul output size overflow"))?;
    if total == 0 {
        return if squeeze_out { out.squeeze(0) } else { Ok(out) };
    }

    {
        let (x_storage, x_layout) = x2.storage_and_layout();
        let (bytes_storage, bytes_layout) = weight_bytes.storage_and_layout();
        let (scales_storage, scales_layout) = weight_scales.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();

        let (x_metal, bytes_metal, scales_metal, out_metal) =
            match (&*x_storage, &*bytes_storage, &*scales_storage, &*out_storage) {
                (
                    candle_core::Storage::Metal(xm),
                    candle_core::Storage::Metal(wb),
                    candle_core::Storage::Metal(ws),
                    candle_core::Storage::Metal(outm),
                ) => (xm, wb, ws, outm),
                _ => candle_core::bail!("iq2_matmul expected Metal storage"),
            };
        if !x_layout.is_contiguous()
            || !bytes_layout.is_contiguous()
            || !scales_layout.is_contiguous()
            || !out_layout.is_contiguous()
        {
            candle_core::bail!("iq2_matmul expects contiguous layouts");
        }

        let kernel_name = variant.kernel_name(x_metal.dtype())?;

        let m_u32 = u32::try_from(m).map_err(|_| candle_core::Error::msg("m too large"))?;
        let out_u32 = u32::try_from(out_dim).map_err(|_| candle_core::Error::msg("out_dim too large"))?;
        let in_u32 = u32::try_from(in_dim).map_err(|_| candle_core::Error::msg("in_dim too large"))?;
        let blocks_u32 = u32::try_from(blocks_per_row)
            .map_err(|_| candle_core::Error::msg("blocks_per_row too large"))?;

        let metal = x_metal.device().metal_device();
        let pipeline = get_or_create_iq2_matmul_pipeline(x_metal.device().id(), metal, kernel_name)?;
        let encoder = x_metal.device().command_encoder()?;
        encoder.set_label("candelora_iq2_matmul");
        encoder.set_compute_pipeline_state(&pipeline);

        let x_bo = BufferOffset {
            buffer: x_metal.buffer(),
            offset_in_bytes: x_layout.start_offset() * x_metal.dtype().size_in_bytes(),
        };
        let bytes_bo = BufferOffset {
            buffer: bytes_metal.buffer(),
            offset_in_bytes: bytes_layout.start_offset() * DType::U8.size_in_bytes(),
        };
        let scales_bo = BufferOffset {
            buffer: scales_metal.buffer(),
            offset_in_bytes: scales_layout.start_offset() * DType::U8.size_in_bytes(),
        };
        let out_bo = BufferOffset {
            buffer: out_metal.buffer(),
            offset_in_bytes: out_layout.start_offset() * DType::F32.size_in_bytes(),
        };

        let encoder_ref = &encoder;
        candle_metal_kernels::set_params!(
            encoder_ref,
            (m_u32, out_u32, in_u32, blocks_u32, &x_bo, &bytes_bo, &scales_bo, &out_bo)
        );

        let threads = pipeline.max_total_threads_per_threadgroup().min(total.max(1));
        let groups = total.div_ceil(threads);
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
    }

    if squeeze_out {
        out.squeeze(0)
    } else {
        Ok(out)
    }
}

#[cfg(not(feature = "metal"))]
fn iq2_matmul_metal(
    _x: &Tensor,
    _weight_bytes: &Tensor,
    _weight_scales: &Tensor,
    _out_dim: usize,
    _in_dim: usize,
    _variant: Iq2MatmulVariant,
) -> Result<Tensor> {
    candle_core::bail!("iq2_matmul requires `candelora-metal-kernels` built with `--features metal`")
}

/// Generic entry point for IQ2 matmul variants.
pub fn iq2_matmul(
    x: &Tensor,
    weight_bytes: &Tensor,
    weight_scales: &Tensor,
    out_dim: usize,
    in_dim: usize,
    variant: Iq2MatmulVariant,
) -> Result<Tensor> {
    iq2_matmul_metal(x, weight_bytes, weight_scales, out_dim, in_dim, variant)
}

pub fn iq2_xxs_matmul(
    x: &Tensor,
    weight_bytes: &Tensor,
    weight_scales: &Tensor,
    out_dim: usize,
    in_dim: usize,
) -> Result<Tensor> {
    iq2_matmul(
        x,
        weight_bytes,
        weight_scales,
        out_dim,
        in_dim,
        Iq2MatmulVariant::Iq2Xxs,
    )
}

pub fn iq2_xs_matmul(
    x: &Tensor,
    weight_bytes: &Tensor,
    weight_scales: &Tensor,
    out_dim: usize,
    in_dim: usize,
) -> Result<Tensor> {
    iq2_matmul(
        x,
        weight_bytes,
        weight_scales,
        out_dim,
        in_dim,
        Iq2MatmulVariant::Iq2Xs,
    )
}

pub fn iq2_s_matmul(
    x: &Tensor,
    weight_bytes: &Tensor,
    weight_scales: &Tensor,
    out_dim: usize,
    in_dim: usize,
) -> Result<Tensor> {
    iq2_matmul(
        x,
        weight_bytes,
        weight_scales,
        out_dim,
        in_dim,
        Iq2MatmulVariant::Iq2S,
    )
}

#[cfg(feature = "metal")]
const IQ2_MATMUL_METAL: &str = concat!(
    include_str!("metal_src/iq2_tables.metal"),
    "\n",
    include_str!("metal_src/iq2_matmul.metal"),
);
