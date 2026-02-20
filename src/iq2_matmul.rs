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

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use half::f16;
    use std::sync::OnceLock;

    struct Iq2RefTables {
        kmask_iq2xs: Vec<u8>,
        ksigns_iq2xs: Vec<u8>,
        iq2xxs_grid: Vec<u64>,
        iq2xs_grid: Vec<u64>,
        iq2s_grid: Vec<u64>,
    }

    fn parse_numeric_u64(token: &str) -> Option<u64> {
        let tok = token.trim();
        if tok.is_empty() {
            return None;
        }
        if let Some(hex) = tok.strip_prefix("0x") {
            return u64::from_str_radix(hex, 16).ok();
        }
        tok.parse::<u64>().ok()
    }

    fn extract_table_body<'a>(source: &'a str, name: &str) -> &'a str {
        let needle = format!(", {},", name);
        let begin = source
            .lines()
            .scan(0usize, |offset, line| {
                let start = *offset;
                *offset += line.len() + 1;
                Some((start, line))
            })
            .find(|(_, line)| line.contains("GGML_TABLE_BEGIN(") && line.contains(&needle))
            .map(|(start, _)| start)
            .unwrap_or_else(|| panic!("table `{}` not found in iq2_tables.metal", name));

        let after_begin = source[begin..]
            .find('\n')
            .map(|idx| begin + idx + 1)
            .unwrap_or_else(|| panic!("table `{}` begin line malformed", name));
        let end = source[after_begin..]
            .find("GGML_TABLE_END()")
            .map(|idx| after_begin + idx)
            .unwrap_or_else(|| panic!("table `{}` end marker not found", name));
        &source[after_begin..end]
    }

    fn parse_table_u8(source: &str, name: &str) -> Vec<u8> {
        extract_table_body(source, name)
            .split(|c: char| c == ',' || c.is_ascii_whitespace())
            .filter_map(parse_numeric_u64)
            .map(|v| u8::try_from(v).expect("u8 table value out of range"))
            .collect()
    }

    fn parse_table_u64(source: &str, name: &str) -> Vec<u64> {
        extract_table_body(source, name)
            .split(|c: char| c == ',' || c.is_ascii_whitespace())
            .filter_map(parse_numeric_u64)
            .collect()
    }

    fn ref_tables() -> &'static Iq2RefTables {
        static TABLES: OnceLock<Iq2RefTables> = OnceLock::new();
        TABLES.get_or_init(|| {
            let source = include_str!("metal_src/iq2_tables.metal");
            Iq2RefTables {
                kmask_iq2xs: parse_table_u8(source, "kmask_iq2xs"),
                ksigns_iq2xs: parse_table_u8(source, "ksigns_iq2xs"),
                iq2xxs_grid: parse_table_u64(source, "iq2xxs_grid"),
                iq2xs_grid: parse_table_u64(source, "iq2xs_grid"),
                iq2s_grid: parse_table_u64(source, "iq2s_grid"),
            }
        })
    }

    fn read_u16_le(data: &[u8], off: usize) -> u16 {
        u16::from(data[off]) | (u16::from(data[off + 1]) << 8)
    }

    fn lcg_fill(len: usize, seed: u32) -> Vec<u8> {
        let mut state = seed;
        let mut out = vec![0u8; len];
        for v in &mut out {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *v = (state >> 24) as u8;
        }
        out
    }

    fn deterministic_x(m: usize, in_dim: usize) -> Vec<f32> {
        (0..(m * in_dim))
            .map(|i| (((i * 17) % 41) as f32 - 20.0) / 13.0)
            .collect()
    }

    fn force_f16_one_per_block(scales: &mut [u8], scales_per_block: usize) {
        for block in scales.chunks_exact_mut(scales_per_block) {
            // 1.0f16 in little-endian.
            block[0] = 0x00;
            block[1] = 0x3c;
        }
    }

    fn accum_word_grid(
        grid_pack: u64,
        signs: u8,
        dl: f32,
        base_col: usize,
        in_dim: usize,
        x: &[f32],
        x_base: usize,
        kmask: &[u8],
        acc: &mut f32,
    ) {
        for (j, &mask) in kmask.iter().enumerate().take(8) {
            let col = base_col + j;
            if col >= in_dim {
                break;
            }
            let gv = ((grid_pack >> (8 * j)) & 0xff) as u8;
            let sign = if (signs & mask) != 0 { -1.0 } else { 1.0 };
            *acc += x[x_base + col] * (dl * gv as f32 * sign);
        }
    }

    fn iq2_matmul_reference(
        x: &[f32],
        m: usize,
        out_dim: usize,
        in_dim: usize,
        weight_bytes: &[u8],
        weight_scales: &[u8],
        variant: Iq2MatmulVariant,
    ) -> Vec<f32> {
        let tables = ref_tables();
        let mut out = vec![0f32; m * out_dim];
        let blocks_per_row = in_dim.div_ceil(IQ2_QK);

        for row in 0..m {
            let x_base = row * in_dim;
            for out_idx in 0..out_dim {
                let mut acc = 0f32;
                for blk in 0..blocks_per_row {
                    let block_idx = out_idx * blocks_per_row + blk;
                    match variant {
                        Iq2MatmulVariant::Iq2Xxs => {
                            let byte_base = block_idx * IQ2_XXS_BYTES_PER_BLOCK;
                            let scale_base = block_idx * IQ2_XXS_SCALES_PER_BLOCK;
                            let d = f16::from_bits(read_u16_le(weight_scales, scale_base)).to_f32();
                            for ib32 in 0..8usize {
                                let q2_base = byte_base + ib32 * 8;
                                let q20 = read_u16_le(weight_bytes, q2_base);
                                let q21 = read_u16_le(weight_bytes, q2_base + 2);
                                let q22 = read_u16_le(weight_bytes, q2_base + 4);
                                let q23 = read_u16_le(weight_bytes, q2_base + 6);
                                let aux_g = u32::from(q20) | (u32::from(q21) << 16);
                                let aux_s = u32::from(q22) | (u32::from(q23) << 16);
                                let dl = d * (0.5 + ((aux_s >> 28) & 0xF) as f32) * 0.25;
                                for g in 0..4usize {
                                    let grid_idx = ((aux_g >> (8 * g)) & 0xFF) as usize;
                                    let signs_idx = ((aux_s >> (7 * g)) & 0x7F) as usize;
                                    let base_col = blk * IQ2_QK + ib32 * 32 + g * 8;
                                    accum_word_grid(
                                        tables.iq2xxs_grid[grid_idx],
                                        tables.ksigns_iq2xs[signs_idx],
                                        dl,
                                        base_col,
                                        in_dim,
                                        x,
                                        x_base,
                                        &tables.kmask_iq2xs,
                                        &mut acc,
                                    );
                                }
                            }
                        }
                        Iq2MatmulVariant::Iq2Xs => {
                            let byte_base = block_idx * IQ2_XS_BYTES_PER_BLOCK;
                            let scale_base = block_idx * IQ2_XS_SCALES_PER_BLOCK;
                            let d = f16::from_bits(read_u16_le(weight_scales, scale_base)).to_f32();
                            for ib32 in 0..8usize {
                                let q2_base = byte_base + ib32 * 8;
                                let q20 = read_u16_le(weight_bytes, q2_base);
                                let q21 = read_u16_le(weight_bytes, q2_base + 2);
                                let q22 = read_u16_le(weight_bytes, q2_base + 4);
                                let q23 = read_u16_le(weight_bytes, q2_base + 6);
                                let scale_byte = weight_scales[scale_base + 2 + ib32];
                                let dl0 = d * (0.5 + (scale_byte & 0x0F) as f32) * 0.25;
                                let dl1 = d * (0.5 + ((scale_byte >> 4) & 0x0F) as f32) * 0.25;
                                let base_col = blk * IQ2_QK + ib32 * 32;
                                for (w, dl, off) in
                                    [(q20, dl0, 0usize), (q21, dl0, 8), (q22, dl1, 16), (q23, dl1, 24)]
                                {
                                    let grid_idx = (w & 0x1FF) as usize;
                                    let signs_idx = ((w >> 9) & 0x7F) as usize;
                                    accum_word_grid(
                                        tables.iq2xs_grid[grid_idx],
                                        tables.ksigns_iq2xs[signs_idx],
                                        dl,
                                        base_col + off,
                                        in_dim,
                                        x,
                                        x_base,
                                        &tables.kmask_iq2xs,
                                        &mut acc,
                                    );
                                }
                            }
                        }
                        Iq2MatmulVariant::Iq2S => {
                            let byte_base = block_idx * IQ2_S_BYTES_PER_BLOCK;
                            let scale_base = block_idx * IQ2_S_SCALES_PER_BLOCK;
                            let d = f16::from_bits(read_u16_le(weight_scales, scale_base)).to_f32();
                            let qs = &weight_bytes[byte_base..byte_base + 64];
                            let qh = &weight_bytes[byte_base + 64..byte_base + 72];
                            for ib32 in 0..8usize {
                                let qh_byte = qh[ib32];
                                let scale_byte = weight_scales[scale_base + 2 + ib32];
                                for il in 0..2usize {
                                    let qs_off = 4 * ib32 + 2 * il;
                                    let qs0 = qs[qs_off];
                                    let qs1 = qs[qs_off + 1];
                                    let sign0 = qs[32 + qs_off];
                                    let sign1 = qs[32 + qs_off + 1];
                                    let qh_nib = qh_byte >> (4 * il);
                                    let dl = d * (0.5 + ((scale_byte >> (4 * il)) & 0x0F) as f32) * 0.25;
                                    let grid_idx0 = usize::from(qs0) | (((usize::from(qh_nib)) << 8) & 0x300);
                                    let grid_idx1 = usize::from(qs1) | (((usize::from(qh_nib)) << 6) & 0x300);
                                    let base_col = blk * IQ2_QK + ib32 * 32 + il * 16;
                                    accum_word_grid(
                                        tables.iq2s_grid[grid_idx0],
                                        sign0,
                                        dl,
                                        base_col,
                                        in_dim,
                                        x,
                                        x_base,
                                        &tables.kmask_iq2xs,
                                        &mut acc,
                                    );
                                    accum_word_grid(
                                        tables.iq2s_grid[grid_idx1],
                                        sign1,
                                        dl,
                                        base_col + 8,
                                        in_dim,
                                        x,
                                        x_base,
                                        &tables.kmask_iq2xs,
                                        &mut acc,
                                    );
                                }
                            }
                        }
                    }
                }
                out[row * out_dim + out_idx] = acc;
            }
        }
        out
    }

    fn parity_case(variant: Iq2MatmulVariant) -> Result<()> {
        let device = match Device::metal_if_available(0) {
            Ok(d) if d.is_metal() => d,
            _ => return Ok(()),
        };

        let m = 2usize;
        let out_dim = 7usize;
        let in_dim = 320usize;
        let blocks_per_row = in_dim.div_ceil(IQ2_QK);
        let num_blocks = out_dim * blocks_per_row;

        let mut weight_bytes = lcg_fill(num_blocks * variant.bytes_per_block(), 0xA11CE5ED);
        let mut weight_scales = lcg_fill(num_blocks * variant.scales_per_block(), 0xC0FFEE12);
        force_f16_one_per_block(&mut weight_scales, variant.scales_per_block());

        // Keep the tail deterministic for better reproducibility across runs.
        if let Some(last) = weight_bytes.last_mut() {
            *last = 0x5A;
        }

        let x_host = deterministic_x(m, in_dim);
        let y_ref = iq2_matmul_reference(
            &x_host,
            m,
            out_dim,
            in_dim,
            &weight_bytes,
            &weight_scales,
            variant,
        );

        let x = Tensor::from_slice(&x_host, (m, in_dim), &device)?;
        let bytes = Tensor::from_slice(&weight_bytes, weight_bytes.len(), &device)?;
        let scales = Tensor::from_slice(&weight_scales, weight_scales.len(), &device)?;
        let y = iq2_matmul(&x, &bytes, &scales, out_dim, in_dim, variant)?;
        device.synchronize()?;
        let y_host = y
            .to_device(&Device::Cpu)?
            .reshape((m * out_dim,))?
            .to_vec1::<f32>()?;

        let mut max_abs = 0f32;
        for (idx, (a, b)) in y_host.iter().zip(y_ref.iter()).enumerate() {
            let diff = (a - b).abs();
            if diff > max_abs {
                max_abs = diff;
            }
            if diff > 2e-3 {
                panic!(
                    "IQ2 parity mismatch variant={:?} idx={} metal={} ref={} diff={}",
                    variant, idx, a, b, diff
                );
            }
        }
        assert!(
            max_abs <= 2e-3,
            "IQ2 parity max_abs too high for {:?}: {}",
            variant,
            max_abs
        );
        Ok(())
    }

    #[test]
    fn cpu_vs_metal_parity_iq2_xxs() -> Result<()> {
        parity_case(Iq2MatmulVariant::Iq2Xxs)
    }

    #[test]
    fn cpu_vs_metal_parity_iq2_xs() -> Result<()> {
        parity_case(Iq2MatmulVariant::Iq2Xs)
    }

    #[test]
    fn cpu_vs_metal_parity_iq2_s() -> Result<()> {
        parity_case(Iq2MatmulVariant::Iq2S)
    }
}

#[cfg(feature = "metal")]
const IQ2_MATMUL_METAL: &str = concat!(
    include_str!("metal_src/iq2_tables.metal"),
    "\n",
    include_str!("metal_src/iq2_matmul.metal"),
);
