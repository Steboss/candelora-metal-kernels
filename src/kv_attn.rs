#[cfg(feature = "metal")]
use candle_core::backend::BackendStorage;
use candle_core::quantized::{QMatMul, QTensor};
use candle_core::{
    CpuStorage, CustomOp1, CustomOp2, CustomOp3, DType, Layout, Module, Result, Shape, Tensor,
};
use half::{bf16, f16};
#[cfg(feature = "metal")]
use std::collections::HashMap;
use std::f32::consts::FRAC_1_SQRT_2;
#[cfg(feature = "metal")]
use std::sync::{OnceLock, RwLock};

#[derive(Clone, Debug)]
struct QkScoresOp {
    scale: f32,
}

#[derive(Clone, Debug)]
struct AttnWeightedSumOp;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowwiseQuantKind {
    Int8,
    Int4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TurboQuantKind {
    Turbo3,
    Turbo4,
}

impl TurboQuantKind {
    fn signed_max(self) -> i32 {
        match self {
            Self::Turbo3 => 3,
            Self::Turbo4 => 7,
        }
    }
}

#[derive(Clone, Debug)]
struct RowwiseQkScoresOp {
    kv_heads: usize,
    repeat_factor: usize,
    tokens: usize,
    head_dim: usize,
    scale: f32,
    kind: RowwiseQuantKind,
}

#[derive(Clone, Debug)]
struct RowwiseAttnWeightedSumOp {
    kv_heads: usize,
    repeat_factor: usize,
    tokens: usize,
    head_dim: usize,
    kind: RowwiseQuantKind,
}

#[derive(Clone, Debug)]
struct RowwisePackScalesOp {
    head_dim: usize,
    kind: RowwiseQuantKind,
}

#[derive(Clone, Debug)]
struct RowwisePackDataOp {
    head_dim: usize,
    kind: RowwiseQuantKind,
}

#[cfg(feature = "metal")]
#[derive(Debug)]
struct KvAttnMetalDeviceCache {
    library: candle_metal_kernels::metal::Library,
    pipelines: HashMap<&'static str, candle_metal_kernels::metal::ComputePipeline>,
}

#[cfg(feature = "metal")]
type KvAttnMetalDeviceId = candle_core::metal_backend::DeviceId;

#[cfg(feature = "metal")]
static KV_ATTN_METAL_CACHE: OnceLock<RwLock<HashMap<KvAttnMetalDeviceId, KvAttnMetalDeviceCache>>> =
    OnceLock::new();

#[cfg(feature = "metal")]
fn get_or_create_kv_attn_pipeline(
    device_id: KvAttnMetalDeviceId,
    device: &candle_metal_kernels::metal::Device,
    kernel_name: &'static str,
) -> Result<candle_metal_kernels::metal::ComputePipeline> {
    let cache = KV_ATTN_METAL_CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    let mut cache = cache
        .write()
        .map_err(|_| candle_core::Error::msg("kv-attn metal cache lock poisoned"))?;
    if let Some(dev_cache) = cache.get_mut(&device_id) {
        if let Some(pipeline) = dev_cache.pipelines.get(kernel_name) {
            return Ok(pipeline.clone());
        }
        let func = dev_cache
            .library
            .get_function(kernel_name, None)
            .map_err(|e| {
                candle_core::Error::msg(format!("failed loading kv-attn metal function: {e}"))
            })?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| {
                candle_core::Error::msg(format!("failed creating kv-attn metal pipeline: {e}"))
            })?;
        dev_cache.pipelines.insert(kernel_name, pipeline.clone());
        return Ok(pipeline);
    }

    let options = objc2_metal::MTLCompileOptions::new();
    let library = device
        .new_library_with_source(KV_ATTN_METAL, Some(&options))
        .map_err(|e| {
            candle_core::Error::msg(format!("failed compiling kv-attn metal source: {e}"))
        })?;
    let func = library.get_function(kernel_name, None).map_err(|e| {
        candle_core::Error::msg(format!("failed loading kv-attn metal function: {e}"))
    })?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| {
            candle_core::Error::msg(format!("failed creating kv-attn metal pipeline: {e}"))
        })?;

    let mut pipelines = HashMap::new();
    pipelines.insert(kernel_name, pipeline.clone());
    cache.insert(device_id, KvAttnMetalDeviceCache { library, pipelines });
    Ok(pipeline)
}

fn contiguous_slice<'a, T>(
    values: &'a [T],
    layout: &Layout,
    name: &'static str,
) -> Result<&'a [T]> {
    match layout.contiguous_offsets() {
        Some((start, end)) => Ok(&values[start..end]),
        None => candle_core::bail!("{name} must be contiguous for fused kv-attn op"),
    }
}

fn parse_qk_shapes(l1: &Layout, l2: &Layout) -> Result<(usize, usize, usize, usize)> {
    let qd = l1.shape().dims();
    let kd = l2.shape().dims();
    if qd.len() != 3 {
        candle_core::bail!("q must be rank-3 [b,h,d], got {:?}", qd)
    }
    if kd.len() != 4 {
        candle_core::bail!("k must be rank-4 [b,h,t,d], got {:?}", kd)
    }
    let (b, h, d) = (qd[0], qd[1], qd[2]);
    let (bk, hk, t, dk) = (kd[0], kd[1], kd[2], kd[3]);
    if b != bk || h != hk || d != dk {
        candle_core::bail!("q/k shape mismatch: q=[{b},{h},{d}], k=[{bk},{hk},{t},{dk}]")
    }
    Ok((b, h, t, d))
}

fn parse_attn_v_shapes(l1: &Layout, l2: &Layout) -> Result<(usize, usize, usize, usize)> {
    let ad = l1.shape().dims();
    let vd = l2.shape().dims();
    if ad.len() != 3 {
        candle_core::bail!("attn probs must be rank-3 [b,h,t], got {:?}", ad)
    }
    if vd.len() != 4 {
        candle_core::bail!("v must be rank-4 [b,h,t,d], got {:?}", vd)
    }
    let (b, h, t) = (ad[0], ad[1], ad[2]);
    let (bv, hv, tv, d) = (vd[0], vd[1], vd[2], vd[3]);
    if b != bv || h != hv || t != tv {
        candle_core::bail!("attn/v shape mismatch: attn=[{b},{h},{t}], v=[{bv},{hv},{tv},{d}]")
    }
    Ok((b, h, t, d))
}

fn cpu_qk_scores_f32(
    q: &[f32],
    k: &[f32],
    b: usize,
    h: usize,
    t: usize,
    d: usize,
    scale: f32,
) -> Vec<f32> {
    let bh = b * h;
    let mut out = vec![0f32; bh * t];
    for bh_idx in 0..bh {
        let q_base = bh_idx * d;
        for tok in 0..t {
            let k_base = (bh_idx * t + tok) * d;
            let mut acc = 0f32;
            for i in 0..d {
                acc += q[q_base + i] * k[k_base + i];
            }
            out[bh_idx * t + tok] = acc * scale;
        }
    }
    out
}

fn cpu_qk_scores_f16(
    q: &[f16],
    k: &[f16],
    b: usize,
    h: usize,
    t: usize,
    d: usize,
    scale: f32,
) -> Vec<f32> {
    let bh = b * h;
    let mut out = vec![0f32; bh * t];
    for bh_idx in 0..bh {
        let q_base = bh_idx * d;
        for tok in 0..t {
            let k_base = (bh_idx * t + tok) * d;
            let mut acc = 0f32;
            for i in 0..d {
                acc += q[q_base + i].to_f32() * k[k_base + i].to_f32();
            }
            out[bh_idx * t + tok] = acc * scale;
        }
    }
    out
}

fn cpu_qk_scores_bf16(
    q: &[bf16],
    k: &[bf16],
    b: usize,
    h: usize,
    t: usize,
    d: usize,
    scale: f32,
) -> Vec<f32> {
    let bh = b * h;
    let mut out = vec![0f32; bh * t];
    for bh_idx in 0..bh {
        let q_base = bh_idx * d;
        for tok in 0..t {
            let k_base = (bh_idx * t + tok) * d;
            let mut acc = 0f32;
            for i in 0..d {
                acc += q[q_base + i].to_f32() * k[k_base + i].to_f32();
            }
            out[bh_idx * t + tok] = acc * scale;
        }
    }
    out
}

fn cpu_attn_weighted_sum_f32(
    attn: &[f32],
    v: &[f32],
    b: usize,
    h: usize,
    t: usize,
    d: usize,
) -> Vec<f32> {
    let bh = b * h;
    let mut out = vec![0f32; bh * d];
    for bh_idx in 0..bh {
        let attn_base = bh_idx * t;
        for dim in 0..d {
            let mut acc = 0f32;
            for tok in 0..t {
                let v_idx = (bh_idx * t + tok) * d + dim;
                acc += attn[attn_base + tok] * v[v_idx];
            }
            out[bh_idx * d + dim] = acc;
        }
    }
    out
}

fn cpu_attn_weighted_sum_f16(
    attn: &[f32],
    v: &[f16],
    b: usize,
    h: usize,
    t: usize,
    d: usize,
) -> Vec<f32> {
    let bh = b * h;
    let mut out = vec![0f32; bh * d];
    for bh_idx in 0..bh {
        let attn_base = bh_idx * t;
        for dim in 0..d {
            let mut acc = 0f32;
            for tok in 0..t {
                let v_idx = (bh_idx * t + tok) * d + dim;
                acc += attn[attn_base + tok] * v[v_idx].to_f32();
            }
            out[bh_idx * d + dim] = acc;
        }
    }
    out
}

fn cpu_attn_weighted_sum_bf16(
    attn: &[f32],
    v: &[bf16],
    b: usize,
    h: usize,
    t: usize,
    d: usize,
) -> Vec<f32> {
    let bh = b * h;
    let mut out = vec![0f32; bh * d];
    for bh_idx in 0..bh {
        let attn_base = bh_idx * t;
        for dim in 0..d {
            let mut acc = 0f32;
            for tok in 0..t {
                let v_idx = (bh_idx * t + tok) * d + dim;
                acc += attn[attn_base + tok] * v[v_idx].to_f32();
            }
            out[bh_idx * d + dim] = acc;
        }
    }
    out
}

fn expected_packed_cols(head_dim: usize, kind: RowwiseQuantKind) -> usize {
    match kind {
        RowwiseQuantKind::Int8 => head_dim,
        RowwiseQuantKind::Int4 => head_dim.div_ceil(2),
    }
}

fn dequant_row_elem(
    data: &[u8],
    row: usize,
    col: usize,
    head_dim: usize,
    kind: RowwiseQuantKind,
) -> f32 {
    match kind {
        RowwiseQuantKind::Int8 => {
            let idx = row * head_dim + col;
            let q = i32::from(data[idx]) - 128;
            q as f32
        }
        RowwiseQuantKind::Int4 => {
            let packed_cols = head_dim.div_ceil(2);
            let idx = row * packed_cols + (col / 2);
            let byte = data[idx];
            let nib = if (col & 1) == 0 {
                byte & 0x0f
            } else {
                (byte >> 4) & 0x0f
            };
            let q = if nib >= 8 {
                i32::from(nib) - 16
            } else {
                i32::from(nib)
            };
            q as f32
        }
    }
}

fn parse_rowwise_pack_input(l: &Layout, head_dim: usize, name: &'static str) -> Result<usize> {
    let dims = l.shape().dims();
    if dims.is_empty() {
        candle_core::bail!("{name} must have rank >= 1 for rowwise packing")
    }
    let last = dims[dims.len() - 1];
    if last != head_dim {
        candle_core::bail!(
            "{name} last dimension mismatch for rowwise packing: got {} expected {}",
            last,
            head_dim
        )
    }
    Ok(l.shape().elem_count() / head_dim)
}

fn cpu_pack_scales_rowwise_f32(
    x: &[f32],
    rows: usize,
    head_dim: usize,
    kind: RowwiseQuantKind,
) -> Vec<f32> {
    let denom = match kind {
        RowwiseQuantKind::Int8 => 127.0f32,
        RowwiseQuantKind::Int4 => 7.0f32,
    };
    let mut scales = vec![1f32; rows];
    for row in 0..rows {
        let base = row * head_dim;
        let mut max_abs = 0f32;
        for i in 0..head_dim {
            max_abs = max_abs.max(x[base + i].abs());
        }
        let s = if max_abs == 0.0 {
            1e-8
        } else {
            (max_abs / denom).max(1e-8)
        };
        scales[row] = s;
    }
    scales
}

fn cpu_pack_scales_rowwise_f16(
    x: &[f16],
    rows: usize,
    head_dim: usize,
    kind: RowwiseQuantKind,
) -> Vec<f32> {
    let xf = x.iter().map(|v| v.to_f32()).collect::<Vec<_>>();
    cpu_pack_scales_rowwise_f32(&xf, rows, head_dim, kind)
}

fn cpu_pack_scales_rowwise_bf16(
    x: &[bf16],
    rows: usize,
    head_dim: usize,
    kind: RowwiseQuantKind,
) -> Vec<f32> {
    let xf = x.iter().map(|v| v.to_f32()).collect::<Vec<_>>();
    cpu_pack_scales_rowwise_f32(&xf, rows, head_dim, kind)
}

fn cpu_pack_data_rowwise_f32(
    x: &[f32],
    scales: &[f32],
    rows: usize,
    head_dim: usize,
    kind: RowwiseQuantKind,
) -> Vec<u8> {
    match kind {
        RowwiseQuantKind::Int8 => {
            let mut out = vec![0u8; rows * head_dim];
            for row in 0..rows {
                let base = row * head_dim;
                let s = scales[row].max(1e-8);
                for i in 0..head_dim {
                    let q = (x[base + i] / s).round().clamp(-127.0, 127.0) as i32;
                    out[base + i] = (q + 128) as u8;
                }
            }
            out
        }
        RowwiseQuantKind::Int4 => {
            let packed_cols = head_dim.div_ceil(2);
            let mut out = vec![0u8; rows * packed_cols];
            for row in 0..rows {
                let x_base = row * head_dim;
                let out_base = row * packed_cols;
                let s = scales[row].max(1e-8);
                for col in 0..packed_cols {
                    let i0 = col * 2;
                    let i1 = i0 + 1;
                    let q0 = (x[x_base + i0] / s).round().clamp(-8.0, 7.0) as i32;
                    let mut byte = (q0 as u8) & 0x0f;
                    if i1 < head_dim {
                        let q1 = (x[x_base + i1] / s).round().clamp(-8.0, 7.0) as i32;
                        byte |= ((q1 as u8) & 0x0f) << 4;
                    }
                    out[out_base + col] = byte;
                }
            }
            out
        }
    }
}

fn cpu_pack_data_rowwise_f16(
    x: &[f16],
    scales: &[f32],
    rows: usize,
    head_dim: usize,
    kind: RowwiseQuantKind,
) -> Vec<u8> {
    let xf = x.iter().map(|v| v.to_f32()).collect::<Vec<_>>();
    cpu_pack_data_rowwise_f32(&xf, scales, rows, head_dim, kind)
}

fn cpu_pack_data_rowwise_bf16(
    x: &[bf16],
    scales: &[f32],
    rows: usize,
    head_dim: usize,
    kind: RowwiseQuantKind,
) -> Vec<u8> {
    let xf = x.iter().map(|v| v.to_f32()).collect::<Vec<_>>();
    cpu_pack_data_rowwise_f32(&xf, scales, rows, head_dim, kind)
}

fn cpu_qk_scores_rowwise_f32(
    q: &[f32],
    packed: &[u8],
    scales: &[f32],
    b: usize,
    full_heads: usize,
    kv_heads: usize,
    repeat_factor: usize,
    tokens: usize,
    head_dim: usize,
    kind: RowwiseQuantKind,
    scale: f32,
) -> Vec<f32> {
    let bh = b * full_heads;
    let mut out = vec![0f32; bh * tokens];
    for bh_idx in 0..bh {
        let b_idx = bh_idx / full_heads;
        let h_idx = bh_idx % full_heads;
        let kv_h = h_idx / repeat_factor;
        let q_base = bh_idx * head_dim;
        for tok in 0..tokens {
            let row = (b_idx * kv_heads + kv_h) * tokens + tok;
            let row_scale = scales[row];
            let mut acc = 0f32;
            for i in 0..head_dim {
                let kv = dequant_row_elem(packed, row, i, head_dim, kind) * row_scale;
                acc += q[q_base + i] * kv;
            }
            out[bh_idx * tokens + tok] = acc * scale;
        }
    }
    out
}

fn cpu_qk_scores_rowwise_f16(
    q: &[f16],
    packed: &[u8],
    scales: &[f32],
    b: usize,
    full_heads: usize,
    kv_heads: usize,
    repeat_factor: usize,
    tokens: usize,
    head_dim: usize,
    kind: RowwiseQuantKind,
    scale: f32,
) -> Vec<f32> {
    let qf = q.iter().map(|x| x.to_f32()).collect::<Vec<_>>();
    cpu_qk_scores_rowwise_f32(
        &qf,
        packed,
        scales,
        b,
        full_heads,
        kv_heads,
        repeat_factor,
        tokens,
        head_dim,
        kind,
        scale,
    )
}

fn cpu_qk_scores_rowwise_bf16(
    q: &[bf16],
    packed: &[u8],
    scales: &[f32],
    b: usize,
    full_heads: usize,
    kv_heads: usize,
    repeat_factor: usize,
    tokens: usize,
    head_dim: usize,
    kind: RowwiseQuantKind,
    scale: f32,
) -> Vec<f32> {
    let qf = q.iter().map(|x| x.to_f32()).collect::<Vec<_>>();
    cpu_qk_scores_rowwise_f32(
        &qf,
        packed,
        scales,
        b,
        full_heads,
        kv_heads,
        repeat_factor,
        tokens,
        head_dim,
        kind,
        scale,
    )
}

fn cpu_attn_weighted_sum_rowwise(
    attn: &[f32],
    packed: &[u8],
    scales: &[f32],
    b: usize,
    full_heads: usize,
    kv_heads: usize,
    repeat_factor: usize,
    tokens: usize,
    head_dim: usize,
    kind: RowwiseQuantKind,
) -> Vec<f32> {
    let bh = b * full_heads;
    let mut out = vec![0f32; bh * head_dim];
    for bh_idx in 0..bh {
        let b_idx = bh_idx / full_heads;
        let h_idx = bh_idx % full_heads;
        let kv_h = h_idx / repeat_factor;
        let attn_base = bh_idx * tokens;
        for dim in 0..head_dim {
            let mut acc = 0f32;
            for tok in 0..tokens {
                let row = (b_idx * kv_heads + kv_h) * tokens + tok;
                let vv = dequant_row_elem(packed, row, dim, head_dim, kind) * scales[row];
                acc += attn[attn_base + tok] * vv;
            }
            out[bh_idx * head_dim + dim] = acc;
        }
    }
    out
}

impl CustomOp1 for RowwisePackScalesOp {
    fn name(&self) -> &'static str {
        "candelora-kv-attn-rowwise-pack-scales"
    }

    fn cpu_fwd(&self, s1: &CpuStorage, l1: &Layout) -> Result<(CpuStorage, Shape)> {
        let rows = parse_rowwise_pack_input(l1, self.head_dim, "x")?;
        let out = match s1 {
            CpuStorage::F32(x) => cpu_pack_scales_rowwise_f32(
                contiguous_slice(x, l1, "x")?,
                rows,
                self.head_dim,
                self.kind,
            ),
            CpuStorage::F16(x) => cpu_pack_scales_rowwise_f16(
                contiguous_slice(x, l1, "x")?,
                rows,
                self.head_dim,
                self.kind,
            ),
            CpuStorage::BF16(x) => cpu_pack_scales_rowwise_bf16(
                contiguous_slice(x, l1, "x")?,
                rows,
                self.head_dim,
                self.kind,
            ),
            _ => candle_core::bail!("rowwise pack scales only supports F32/F16/BF16 inputs"),
        };
        Ok((CpuStorage::F32(out), Shape::from(rows)))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_metal_kernels::BufferOffset;
        use objc2_metal::MTLSize;

        if !l1.is_contiguous() {
            candle_core::bail!("rowwise pack scales expects contiguous input")
        }
        let rows = parse_rowwise_pack_input(l1, self.head_dim, "x")?;
        if rows == 0 {
            let out = s1
                .device()
                .new_buffer(0, DType::F32, "kv-attn-pack-scales-out")?;
            return Ok((
                candle_core::MetalStorage::new(out, s1.device().clone(), 0, DType::F32),
                Shape::from(0usize),
            ));
        }

        let kernel_name = match (self.kind, s1.dtype()) {
            (RowwiseQuantKind::Int8, DType::F32) => "pack_scales_rowwise_q8_f32",
            (RowwiseQuantKind::Int8, DType::F16) => "pack_scales_rowwise_q8_f16",
            (RowwiseQuantKind::Int8, DType::BF16) => "pack_scales_rowwise_q8_bf16",
            (RowwiseQuantKind::Int4, DType::F32) => "pack_scales_rowwise_q8_f32",
            (RowwiseQuantKind::Int4, DType::F16) => "pack_scales_rowwise_q8_f16",
            (RowwiseQuantKind::Int4, DType::BF16) => "pack_scales_rowwise_q8_bf16",
            (_, dt) => candle_core::bail!("unsupported dtype for rowwise pack scales op: {:?}", dt),
        };
        let denom = match self.kind {
            RowwiseQuantKind::Int8 => 127.0f32,
            RowwiseQuantKind::Int4 => 7.0f32,
        };

        let rows_u32 =
            u32::try_from(rows).map_err(|_| candle_core::Error::msg("rows too large"))?;
        let d_u32 = u32::try_from(self.head_dim)
            .map_err(|_| candle_core::Error::msg("head_dim too large"))?;
        let metal = s1.device().metal_device();
        let pipeline = get_or_create_kv_attn_pipeline(s1.device().id(), metal, kernel_name)?;
        let output = s1
            .device()
            .new_buffer(rows, DType::F32, "kv-attn-pack-scales-out")?;
        let encoder = s1.device().command_encoder()?;
        encoder.set_label("candelora_kv_attn_rowwise_pack_scales");
        encoder.set_compute_pipeline_state(&pipeline);

        let x = BufferOffset {
            buffer: s1.buffer(),
            offset_in_bytes: l1.start_offset() * s1.dtype().size_in_bytes(),
        };
        let out = BufferOffset {
            buffer: &output,
            offset_in_bytes: 0,
        };
        let encoder_ref = &encoder;
        candle_metal_kernels::set_params!(encoder_ref, (rows_u32, d_u32, denom, &x, &out));

        let threads = pipeline
            .max_total_threads_per_threadgroup()
            .min(rows.max(1));
        let groups = rows.div_ceil(threads);
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
            candle_core::MetalStorage::new(output, s1.device().clone(), rows, DType::F32),
            Shape::from(rows),
        ))
    }
}

impl CustomOp2 for RowwisePackDataOp {
    fn name(&self) -> &'static str {
        "candelora-kv-attn-rowwise-pack-data"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let rows = parse_rowwise_pack_input(l1, self.head_dim, "x")?;
        let scale_dims = l2.shape().dims();
        if scale_dims.len() != 1 || scale_dims[0] != rows {
            candle_core::bail!(
                "rowwise pack data expects scales shape [{}], got {:?}",
                rows,
                scale_dims
            )
        }
        let scales = match s2 {
            CpuStorage::F32(v) => contiguous_slice(v, l2, "scales")?,
            _ => candle_core::bail!("rowwise pack data expects F32 scales"),
        };
        let out = match s1 {
            CpuStorage::F32(x) => cpu_pack_data_rowwise_f32(
                contiguous_slice(x, l1, "x")?,
                scales,
                rows,
                self.head_dim,
                self.kind,
            ),
            CpuStorage::F16(x) => cpu_pack_data_rowwise_f16(
                contiguous_slice(x, l1, "x")?,
                scales,
                rows,
                self.head_dim,
                self.kind,
            ),
            CpuStorage::BF16(x) => cpu_pack_data_rowwise_bf16(
                contiguous_slice(x, l1, "x")?,
                scales,
                rows,
                self.head_dim,
                self.kind,
            ),
            _ => candle_core::bail!("rowwise pack data only supports F32/F16/BF16 inputs"),
        };
        let packed_cols = expected_packed_cols(self.head_dim, self.kind);
        Ok((CpuStorage::U8(out), Shape::from((rows, packed_cols))))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
        s2: &candle_core::MetalStorage,
        l2: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_metal_kernels::BufferOffset;
        use objc2_metal::MTLSize;

        if !l1.is_contiguous() || !l2.is_contiguous() {
            candle_core::bail!("rowwise pack data expects contiguous inputs")
        }
        if s1.device().id() != s2.device().id() {
            candle_core::bail!("rowwise pack data expects x/scales on same device")
        }
        if s2.dtype() != DType::F32 {
            candle_core::bail!("rowwise pack data expects F32 scales, got {:?}", s2.dtype())
        }
        let rows = parse_rowwise_pack_input(l1, self.head_dim, "x")?;
        let scale_dims = l2.shape().dims();
        if scale_dims.len() != 1 || scale_dims[0] != rows {
            candle_core::bail!(
                "rowwise pack data expects scales shape [{}], got {:?}",
                rows,
                scale_dims
            )
        }
        if rows == 0 {
            let out = s1
                .device()
                .new_buffer(0, DType::U8, "kv-attn-pack-data-out")?;
            return Ok((
                candle_core::MetalStorage::new(out, s1.device().clone(), 0, DType::U8),
                Shape::from((0usize, expected_packed_cols(self.head_dim, self.kind))),
            ));
        }

        let kernel_name = match (self.kind, s1.dtype()) {
            (RowwiseQuantKind::Int8, DType::F32) => "pack_data_rowwise_q8_f32",
            (RowwiseQuantKind::Int8, DType::F16) => "pack_data_rowwise_q8_f16",
            (RowwiseQuantKind::Int8, DType::BF16) => "pack_data_rowwise_q8_bf16",
            (RowwiseQuantKind::Int4, DType::F32) => "pack_data_rowwise_q4_f32",
            (RowwiseQuantKind::Int4, DType::F16) => "pack_data_rowwise_q4_f16",
            (RowwiseQuantKind::Int4, DType::BF16) => "pack_data_rowwise_q4_bf16",
            (_, dt) => candle_core::bail!("unsupported dtype for rowwise pack data op: {:?}", dt),
        };
        let packed_cols = expected_packed_cols(self.head_dim, self.kind);
        let out_elems = rows * packed_cols;
        let rows_u32 =
            u32::try_from(rows).map_err(|_| candle_core::Error::msg("rows too large"))?;
        let d_u32 = u32::try_from(self.head_dim)
            .map_err(|_| candle_core::Error::msg("head_dim too large"))?;
        let metal = s1.device().metal_device();
        let pipeline = get_or_create_kv_attn_pipeline(s1.device().id(), metal, kernel_name)?;
        let output = s1
            .device()
            .new_buffer(out_elems, DType::U8, "kv-attn-pack-data-out")?;
        let encoder = s1.device().command_encoder()?;
        encoder.set_label("candelora_kv_attn_rowwise_pack_data");
        encoder.set_compute_pipeline_state(&pipeline);

        let x = BufferOffset {
            buffer: s1.buffer(),
            offset_in_bytes: l1.start_offset() * s1.dtype().size_in_bytes(),
        };
        let scales = BufferOffset {
            buffer: s2.buffer(),
            offset_in_bytes: l2.start_offset() * DType::F32.size_in_bytes(),
        };
        let out = BufferOffset {
            buffer: &output,
            offset_in_bytes: 0,
        };
        let encoder_ref = &encoder;
        candle_metal_kernels::set_params!(encoder_ref, (rows_u32, d_u32, &x, &scales, &out));

        let threads = pipeline
            .max_total_threads_per_threadgroup()
            .min(rows.max(1));
        let groups = rows.div_ceil(threads);
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
            candle_core::MetalStorage::new(output, s1.device().clone(), out_elems, DType::U8),
            Shape::from((rows, packed_cols)),
        ))
    }
}

impl CustomOp2 for QkScoresOp {
    fn name(&self) -> &'static str {
        "candelora-kv-attn-qk-scores"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (b, h, t, d) = parse_qk_shapes(l1, l2)?;
        let out_shape = Shape::from((b, h, t));
        match (s1, s2) {
            (CpuStorage::F32(q), CpuStorage::F32(k)) => {
                let q = contiguous_slice(q, l1, "q")?;
                let k = contiguous_slice(k, l2, "k")?;
                Ok((
                    CpuStorage::F32(cpu_qk_scores_f32(q, k, b, h, t, d, self.scale)),
                    out_shape,
                ))
            }
            (CpuStorage::F16(q), CpuStorage::F16(k)) => {
                let q = contiguous_slice(q, l1, "q")?;
                let k = contiguous_slice(k, l2, "k")?;
                Ok((
                    CpuStorage::F32(cpu_qk_scores_f16(q, k, b, h, t, d, self.scale)),
                    out_shape,
                ))
            }
            (CpuStorage::BF16(q), CpuStorage::BF16(k)) => {
                let q = contiguous_slice(q, l1, "q")?;
                let k = contiguous_slice(k, l2, "k")?;
                Ok((
                    CpuStorage::F32(cpu_qk_scores_bf16(q, k, b, h, t, d, self.scale)),
                    out_shape,
                ))
            }
            _ => candle_core::bail!("unsupported dtype combination for fused kv-attn qk op"),
        }
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
        s2: &candle_core::MetalStorage,
        l2: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_metal_kernels::BufferOffset;
        use objc2_metal::MTLSize;

        if !l1.is_contiguous() || !l2.is_contiguous() {
            candle_core::bail!("fused kv-attn qk op expects contiguous layouts");
        }
        if s1.device().id() != s2.device().id() {
            candle_core::bail!("q and k must be on the same metal device for fused kv-attn qk op");
        }
        let (b, h, t, d) = parse_qk_shapes(l1, l2)?;
        let bh = b * h;
        let out_elems = bh * t;
        let out_shape = Shape::from((b, h, t));
        if out_elems == 0 {
            let out = s1.device().new_buffer(0, DType::F32, "kv-attn-qk-out")?;
            return Ok((
                candle_core::MetalStorage::new(out, s1.device().clone(), 0, DType::F32),
                out_shape,
            ));
        }

        let dtype = s1.dtype();
        if dtype != s2.dtype() {
            candle_core::bail!(
                "dtype mismatch in fused kv-attn qk op: q={:?}, k={:?}",
                dtype,
                s2.dtype()
            );
        }
        let kernel_name = match dtype {
            DType::F32 => "qk_scores_f32",
            DType::F16 => "qk_scores_f16",
            DType::BF16 => "qk_scores_bf16",
            _ => candle_core::bail!("unsupported dtype for fused kv-attn qk op: {:?}", dtype),
        };

        let bh_u32 = u32::try_from(bh).map_err(|_| candle_core::Error::msg("bh too large"))?;
        let t_u32 = u32::try_from(t).map_err(|_| candle_core::Error::msg("t too large"))?;
        let d_u32 = u32::try_from(d).map_err(|_| candle_core::Error::msg("d too large"))?;

        let metal = s1.device().metal_device();
        let pipeline = get_or_create_kv_attn_pipeline(s1.device().id(), metal, kernel_name)?;
        let output = s1
            .device()
            .new_buffer(out_elems, DType::F32, "kv-attn-qk-out")?;
        let encoder = s1.device().command_encoder()?;
        encoder.set_label("candelora_kv_attn_qk");
        encoder.set_compute_pipeline_state(&pipeline);

        let q = BufferOffset {
            buffer: s1.buffer(),
            offset_in_bytes: l1.start_offset() * dtype.size_in_bytes(),
        };
        let k = BufferOffset {
            buffer: s2.buffer(),
            offset_in_bytes: l2.start_offset() * dtype.size_in_bytes(),
        };
        let out = BufferOffset {
            buffer: &output,
            offset_in_bytes: 0,
        };

        let encoder_ref = &encoder;
        candle_metal_kernels::set_params!(
            encoder_ref,
            (bh_u32, t_u32, d_u32, self.scale, &q, &k, &out)
        );

        let threads = pipeline
            .max_total_threads_per_threadgroup()
            .min(out_elems.max(1));
        let groups = out_elems.div_ceil(threads);
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
            candle_core::MetalStorage::new(output, s1.device().clone(), out_elems, DType::F32),
            out_shape,
        ))
    }
}

impl CustomOp2 for AttnWeightedSumOp {
    fn name(&self) -> &'static str {
        "candelora-kv-attn-weighted-sum"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (b, h, t, d) = parse_attn_v_shapes(l1, l2)?;
        let out_shape = Shape::from((b, h, d));
        match (s1, s2) {
            (CpuStorage::F32(attn), CpuStorage::F32(v)) => {
                let attn = contiguous_slice(attn, l1, "attn_probs")?;
                let v = contiguous_slice(v, l2, "v")?;
                Ok((
                    CpuStorage::F32(cpu_attn_weighted_sum_f32(attn, v, b, h, t, d)),
                    out_shape,
                ))
            }
            (CpuStorage::F32(attn), CpuStorage::F16(v)) => {
                let attn = contiguous_slice(attn, l1, "attn_probs")?;
                let v = contiguous_slice(v, l2, "v")?;
                Ok((
                    CpuStorage::F32(cpu_attn_weighted_sum_f16(attn, v, b, h, t, d)),
                    out_shape,
                ))
            }
            (CpuStorage::F32(attn), CpuStorage::BF16(v)) => {
                let attn = contiguous_slice(attn, l1, "attn_probs")?;
                let v = contiguous_slice(v, l2, "v")?;
                Ok((
                    CpuStorage::F32(cpu_attn_weighted_sum_bf16(attn, v, b, h, t, d)),
                    out_shape,
                ))
            }
            _ => candle_core::bail!(
                "unsupported dtype combination for fused kv-attn weighted-sum op"
            ),
        }
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
        s2: &candle_core::MetalStorage,
        l2: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_metal_kernels::BufferOffset;
        use objc2_metal::MTLSize;

        if !l1.is_contiguous() || !l2.is_contiguous() {
            candle_core::bail!("fused kv-attn weighted-sum op expects contiguous layouts");
        }
        if s1.device().id() != s2.device().id() {
            candle_core::bail!(
                "attn probs and v must be on the same metal device for fused kv-attn weighted-sum op"
            );
        }
        if s1.dtype() != DType::F32 {
            candle_core::bail!(
                "fused kv-attn weighted-sum op expects F32 attn probs, got {:?}",
                s1.dtype()
            );
        }
        let (b, h, t, d) = parse_attn_v_shapes(l1, l2)?;
        let bh = b * h;
        let out_elems = bh * d;
        let out_shape = Shape::from((b, h, d));
        if out_elems == 0 {
            let out = s1
                .device()
                .new_buffer(0, DType::F32, "kv-attn-weighted-sum-out")?;
            return Ok((
                candle_core::MetalStorage::new(out, s1.device().clone(), 0, DType::F32),
                out_shape,
            ));
        }

        let v_dtype = s2.dtype();
        let kernel_name = match v_dtype {
            DType::F32 => "attn_weighted_sum_v_f32",
            DType::F16 => "attn_weighted_sum_v_f16",
            DType::BF16 => "attn_weighted_sum_v_bf16",
            _ => candle_core::bail!(
                "unsupported v dtype for fused kv-attn weighted-sum op: {:?}",
                v_dtype
            ),
        };

        let bh_u32 = u32::try_from(bh).map_err(|_| candle_core::Error::msg("bh too large"))?;
        let t_u32 = u32::try_from(t).map_err(|_| candle_core::Error::msg("t too large"))?;
        let d_u32 = u32::try_from(d).map_err(|_| candle_core::Error::msg("d too large"))?;

        let metal = s1.device().metal_device();
        let pipeline = get_or_create_kv_attn_pipeline(s1.device().id(), metal, kernel_name)?;
        let output = s1
            .device()
            .new_buffer(out_elems, DType::F32, "kv-attn-weighted-sum-out")?;
        let encoder = s1.device().command_encoder()?;
        encoder.set_label("candelora_kv_attn_weighted_sum");
        encoder.set_compute_pipeline_state(&pipeline);

        let attn = BufferOffset {
            buffer: s1.buffer(),
            offset_in_bytes: l1.start_offset() * DType::F32.size_in_bytes(),
        };
        let v = BufferOffset {
            buffer: s2.buffer(),
            offset_in_bytes: l2.start_offset() * v_dtype.size_in_bytes(),
        };
        let out = BufferOffset {
            buffer: &output,
            offset_in_bytes: 0,
        };

        let encoder_ref = &encoder;
        candle_metal_kernels::set_params!(encoder_ref, (bh_u32, t_u32, d_u32, &attn, &v, &out));

        let threads = pipeline
            .max_total_threads_per_threadgroup()
            .min(out_elems.max(1));
        let groups = out_elems.div_ceil(threads);
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
            candle_core::MetalStorage::new(output, s1.device().clone(), out_elems, DType::F32),
            out_shape,
        ))
    }
}

impl CustomOp3 for RowwiseQkScoresOp {
    fn name(&self) -> &'static str {
        "candelora-kv-attn-rowwise-qk"
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
        let qd = l1.shape().dims();
        if qd.len() != 3 {
            candle_core::bail!("rowwise qk expects q rank-3 [b,h,d], got {:?}", qd)
        }
        let (b, full_heads, d) = (qd[0], qd[1], qd[2]);
        if d != self.head_dim {
            candle_core::bail!("rowwise qk head_dim mismatch: q={} op={}", d, self.head_dim)
        }
        if full_heads % self.repeat_factor != 0 {
            candle_core::bail!(
                "rowwise qk invalid heads config: full_heads={} repeat_factor={}",
                full_heads,
                self.repeat_factor
            )
        }
        if full_heads / self.repeat_factor != self.kv_heads {
            candle_core::bail!(
                "rowwise qk kv head mismatch: derived={} expected={}",
                full_heads / self.repeat_factor,
                self.kv_heads
            )
        }
        let data_dims = l2.shape().dims();
        if data_dims.len() != 2 {
            candle_core::bail!(
                "rowwise qk expects packed data rank-2 [rows, packed_cols], got {:?}",
                data_dims
            )
        }
        let expected_rows = b * self.kv_heads * self.tokens;
        let expected_cols = expected_packed_cols(self.head_dim, self.kind);
        if data_dims[0] != expected_rows || data_dims[1] != expected_cols {
            candle_core::bail!(
                "rowwise qk packed data shape mismatch: got {:?}, expected [{}, {}]",
                data_dims,
                expected_rows,
                expected_cols
            )
        }
        let scale_dims = l3.shape().dims();
        if scale_dims.len() != 1 || scale_dims[0] != expected_rows {
            candle_core::bail!(
                "rowwise qk scales shape mismatch: got {:?}, expected [{}]",
                scale_dims,
                expected_rows
            )
        }

        let out_shape = Shape::from((b, full_heads, self.tokens));
        let packed = match s2 {
            CpuStorage::U8(v) => contiguous_slice(v, l2, "packed_k")?,
            _ => candle_core::bail!("rowwise qk expects U8 packed data"),
        };
        let scales_f16_to_f32;
        let scales = match s3 {
            CpuStorage::F32(v) => contiguous_slice(v, l3, "k_scales")?,
            CpuStorage::F16(v) => {
                scales_f16_to_f32 = contiguous_slice(v, l3, "k_scales")?
                    .iter()
                    .map(|x| x.to_f32())
                    .collect::<Vec<_>>();
                &scales_f16_to_f32
            }
            _ => candle_core::bail!("rowwise qk expects F32 or F16 scales"),
        };
        let out = match s1 {
            CpuStorage::F32(q) => cpu_qk_scores_rowwise_f32(
                contiguous_slice(q, l1, "q")?,
                packed,
                scales,
                b,
                full_heads,
                self.kv_heads,
                self.repeat_factor,
                self.tokens,
                self.head_dim,
                self.kind,
                self.scale,
            ),
            CpuStorage::F16(q) => cpu_qk_scores_rowwise_f16(
                contiguous_slice(q, l1, "q")?,
                packed,
                scales,
                b,
                full_heads,
                self.kv_heads,
                self.repeat_factor,
                self.tokens,
                self.head_dim,
                self.kind,
                self.scale,
            ),
            CpuStorage::BF16(q) => cpu_qk_scores_rowwise_bf16(
                contiguous_slice(q, l1, "q")?,
                packed,
                scales,
                b,
                full_heads,
                self.kv_heads,
                self.repeat_factor,
                self.tokens,
                self.head_dim,
                self.kind,
                self.scale,
            ),
            _ => candle_core::bail!("unsupported q dtype for rowwise qk op"),
        };
        Ok((CpuStorage::F32(out), out_shape))
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
        use candle_metal_kernels::BufferOffset;
        use objc2_metal::MTLSize;

        if !l1.is_contiguous() || !l2.is_contiguous() || !l3.is_contiguous() {
            candle_core::bail!("rowwise qk op expects contiguous layouts");
        }
        if s1.device().id() != s2.device().id() || s1.device().id() != s3.device().id() {
            candle_core::bail!("rowwise qk op expects all tensors on the same device");
        }

        let qd = l1.shape().dims();
        if qd.len() != 3 {
            candle_core::bail!("rowwise qk expects q rank-3 [b,h,d], got {:?}", qd)
        }
        let (b, full_heads, d) = (qd[0], qd[1], qd[2]);
        if d != self.head_dim {
            candle_core::bail!("rowwise qk head_dim mismatch: q={} op={}", d, self.head_dim)
        }
        if full_heads % self.repeat_factor != 0 || full_heads / self.repeat_factor != self.kv_heads
        {
            candle_core::bail!(
                "rowwise qk invalid head config: full_heads={} kv_heads={} repeat_factor={}",
                full_heads,
                self.kv_heads,
                self.repeat_factor
            )
        }

        let data_dims = l2.shape().dims();
        let expected_rows = b * self.kv_heads * self.tokens;
        let expected_cols = expected_packed_cols(self.head_dim, self.kind);
        if data_dims.len() != 2 || data_dims[0] != expected_rows || data_dims[1] != expected_cols {
            candle_core::bail!(
                "rowwise qk packed data shape mismatch: got {:?}, expected [{}, {}]",
                data_dims,
                expected_rows,
                expected_cols
            )
        }
        let scale_dims = l3.shape().dims();
        if scale_dims.len() != 1 || scale_dims[0] != expected_rows {
            candle_core::bail!(
                "rowwise qk scales shape mismatch: got {:?}, expected [{}]",
                scale_dims,
                expected_rows
            )
        }

        if s2.dtype() != DType::U8 {
            candle_core::bail!("rowwise qk expects U8 packed data, got {:?}", s2.dtype())
        }
        if !matches!(s3.dtype(), DType::F32 | DType::F16) {
            candle_core::bail!("rowwise qk expects F32 or F16 scales, got {:?}", s3.dtype())
        }

        let kernel_name = match (self.kind, s1.dtype(), s3.dtype()) {
            (RowwiseQuantKind::Int8, DType::F32, DType::F32) => "qk_scores_rowwise_q8_f32",
            (RowwiseQuantKind::Int8, DType::F32, DType::F16) => "qk_scores_rowwise_q8_f32_sf16",
            (RowwiseQuantKind::Int8, DType::F16, DType::F32) => "qk_scores_rowwise_q8_f16",
            (RowwiseQuantKind::Int8, DType::F16, DType::F16) => "qk_scores_rowwise_q8_f16_sf16",
            (RowwiseQuantKind::Int8, DType::BF16, DType::F32) => "qk_scores_rowwise_q8_bf16",
            (RowwiseQuantKind::Int8, DType::BF16, DType::F16) => "qk_scores_rowwise_q8_bf16_sf16",
            (RowwiseQuantKind::Int4, DType::F32, DType::F32) => "qk_scores_rowwise_q4_f32",
            (RowwiseQuantKind::Int4, DType::F32, DType::F16) => "qk_scores_rowwise_q4_f32_sf16",
            (RowwiseQuantKind::Int4, DType::F16, DType::F32) => "qk_scores_rowwise_q4_f16",
            (RowwiseQuantKind::Int4, DType::F16, DType::F16) => "qk_scores_rowwise_q4_f16_sf16",
            (RowwiseQuantKind::Int4, DType::BF16, DType::F32) => "qk_scores_rowwise_q4_bf16",
            (RowwiseQuantKind::Int4, DType::BF16, DType::F16) => "qk_scores_rowwise_q4_bf16_sf16",
            (_, q_dt, scale_dt) => candle_core::bail!(
                "unsupported dtypes for rowwise qk op: q={:?} scales={:?}",
                q_dt,
                scale_dt
            ),
        };

        let bh_full = b * full_heads;
        let out_elems = bh_full * self.tokens;
        let out_shape = Shape::from((b, full_heads, self.tokens));
        if out_elems == 0 {
            let out = s1
                .device()
                .new_buffer(0, DType::F32, "kv-attn-rowwise-qk-out")?;
            return Ok((
                candle_core::MetalStorage::new(out, s1.device().clone(), 0, DType::F32),
                out_shape,
            ));
        }

        let bh_u32 =
            u32::try_from(bh_full).map_err(|_| candle_core::Error::msg("bh_full too large"))?;
        let full_heads_u32 = u32::try_from(full_heads)
            .map_err(|_| candle_core::Error::msg("full_heads too large"))?;
        let kv_heads_u32 = u32::try_from(self.kv_heads)
            .map_err(|_| candle_core::Error::msg("kv_heads too large"))?;
        let repeat_u32 = u32::try_from(self.repeat_factor)
            .map_err(|_| candle_core::Error::msg("repeat_factor too large"))?;
        let t_u32 =
            u32::try_from(self.tokens).map_err(|_| candle_core::Error::msg("tokens too large"))?;
        let d_u32 = u32::try_from(self.head_dim)
            .map_err(|_| candle_core::Error::msg("head_dim too large"))?;

        let metal = s1.device().metal_device();
        let pipeline = get_or_create_kv_attn_pipeline(s1.device().id(), metal, kernel_name)?;
        let output = s1
            .device()
            .new_buffer(out_elems, DType::F32, "kv-attn-rowwise-qk-out")?;
        let encoder = s1.device().command_encoder()?;
        encoder.set_label("candelora_kv_attn_rowwise_qk");
        encoder.set_compute_pipeline_state(&pipeline);

        let q = BufferOffset {
            buffer: s1.buffer(),
            offset_in_bytes: l1.start_offset() * s1.dtype().size_in_bytes(),
        };
        let kq = BufferOffset {
            buffer: s2.buffer(),
            offset_in_bytes: l2.start_offset() * DType::U8.size_in_bytes(),
        };
        let k_scales = BufferOffset {
            buffer: s3.buffer(),
            offset_in_bytes: l3.start_offset() * DType::F32.size_in_bytes(),
        };
        let out = BufferOffset {
            buffer: &output,
            offset_in_bytes: 0,
        };
        let encoder_ref = &encoder;
        candle_metal_kernels::set_params!(
            encoder_ref,
            (
                bh_u32,
                full_heads_u32,
                kv_heads_u32,
                repeat_u32,
                t_u32,
                d_u32,
                self.scale,
                &q,
                &kq,
                &k_scales,
                &out
            )
        );

        let threads = pipeline
            .max_total_threads_per_threadgroup()
            .min(out_elems.max(1));
        let groups = out_elems.div_ceil(threads);
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
            candle_core::MetalStorage::new(output, s1.device().clone(), out_elems, DType::F32),
            out_shape,
        ))
    }
}

impl CustomOp3 for RowwiseAttnWeightedSumOp {
    fn name(&self) -> &'static str {
        "candelora-kv-attn-rowwise-weighted-sum"
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
        let ad = l1.shape().dims();
        if ad.len() != 3 {
            candle_core::bail!(
                "rowwise weighted-sum expects attn rank-3 [b,h,t], got {:?}",
                ad
            )
        }
        let (b, full_heads, t) = (ad[0], ad[1], ad[2]);
        if t != self.tokens {
            candle_core::bail!(
                "rowwise weighted-sum token mismatch: attn={} op={}",
                t,
                self.tokens
            )
        }
        if full_heads % self.repeat_factor != 0 || full_heads / self.repeat_factor != self.kv_heads
        {
            candle_core::bail!(
                "rowwise weighted-sum invalid head config: full_heads={} kv_heads={} repeat_factor={}",
                full_heads,
                self.kv_heads,
                self.repeat_factor
            )
        }
        let data_dims = l2.shape().dims();
        let expected_rows = b * self.kv_heads * self.tokens;
        let expected_cols = expected_packed_cols(self.head_dim, self.kind);
        if data_dims.len() != 2 || data_dims[0] != expected_rows || data_dims[1] != expected_cols {
            candle_core::bail!(
                "rowwise weighted-sum packed data shape mismatch: got {:?}, expected [{}, {}]",
                data_dims,
                expected_rows,
                expected_cols
            )
        }
        let scale_dims = l3.shape().dims();
        if scale_dims.len() != 1 || scale_dims[0] != expected_rows {
            candle_core::bail!(
                "rowwise weighted-sum scales shape mismatch: got {:?}, expected [{}]",
                scale_dims,
                expected_rows
            )
        }

        let attn = match s1 {
            CpuStorage::F32(v) => contiguous_slice(v, l1, "attn_probs")?,
            _ => candle_core::bail!("rowwise weighted-sum expects F32 attn probs"),
        };
        let packed = match s2 {
            CpuStorage::U8(v) => contiguous_slice(v, l2, "packed_v")?,
            _ => candle_core::bail!("rowwise weighted-sum expects U8 packed data"),
        };
        let scales_f16_to_f32;
        let scales = match s3 {
            CpuStorage::F32(v) => contiguous_slice(v, l3, "v_scales")?,
            CpuStorage::F16(v) => {
                scales_f16_to_f32 = contiguous_slice(v, l3, "v_scales")?
                    .iter()
                    .map(|x| x.to_f32())
                    .collect::<Vec<_>>();
                &scales_f16_to_f32
            }
            _ => candle_core::bail!("rowwise weighted-sum expects F32 or F16 scales"),
        };
        let out = cpu_attn_weighted_sum_rowwise(
            attn,
            packed,
            scales,
            b,
            full_heads,
            self.kv_heads,
            self.repeat_factor,
            self.tokens,
            self.head_dim,
            self.kind,
        );
        Ok((
            CpuStorage::F32(out),
            Shape::from((b, full_heads, self.head_dim)),
        ))
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
        use candle_metal_kernels::BufferOffset;
        use objc2_metal::MTLSize;

        if !l1.is_contiguous() || !l2.is_contiguous() || !l3.is_contiguous() {
            candle_core::bail!("rowwise weighted-sum op expects contiguous layouts");
        }
        if s1.device().id() != s2.device().id() || s1.device().id() != s3.device().id() {
            candle_core::bail!("rowwise weighted-sum op expects all tensors on the same device");
        }
        if s1.dtype() != DType::F32 {
            candle_core::bail!(
                "rowwise weighted-sum expects F32 attn probs, got {:?}",
                s1.dtype()
            )
        }
        if s2.dtype() != DType::U8 {
            candle_core::bail!(
                "rowwise weighted-sum expects U8 packed data, got {:?}",
                s2.dtype()
            )
        }
        if !matches!(s3.dtype(), DType::F32 | DType::F16) {
            candle_core::bail!(
                "rowwise weighted-sum expects F32 or F16 scales, got {:?}",
                s3.dtype()
            )
        }

        let ad = l1.shape().dims();
        if ad.len() != 3 {
            candle_core::bail!(
                "rowwise weighted-sum expects attn rank-3 [b,h,t], got {:?}",
                ad
            )
        }
        let (b, full_heads, t) = (ad[0], ad[1], ad[2]);
        if t != self.tokens {
            candle_core::bail!(
                "rowwise weighted-sum token mismatch: attn={} op={}",
                t,
                self.tokens
            )
        }
        if full_heads % self.repeat_factor != 0 || full_heads / self.repeat_factor != self.kv_heads
        {
            candle_core::bail!(
                "rowwise weighted-sum invalid head config: full_heads={} kv_heads={} repeat_factor={}",
                full_heads,
                self.kv_heads,
                self.repeat_factor
            )
        }
        let data_dims = l2.shape().dims();
        let expected_rows = b * self.kv_heads * self.tokens;
        let expected_cols = expected_packed_cols(self.head_dim, self.kind);
        if data_dims.len() != 2 || data_dims[0] != expected_rows || data_dims[1] != expected_cols {
            candle_core::bail!(
                "rowwise weighted-sum packed data shape mismatch: got {:?}, expected [{}, {}]",
                data_dims,
                expected_rows,
                expected_cols
            )
        }
        let scale_dims = l3.shape().dims();
        if scale_dims.len() != 1 || scale_dims[0] != expected_rows {
            candle_core::bail!(
                "rowwise weighted-sum scales shape mismatch: got {:?}, expected [{}]",
                scale_dims,
                expected_rows
            )
        }

        let kernel_name = match (self.kind, s3.dtype()) {
            (RowwiseQuantKind::Int8, DType::F32) => "attn_weighted_sum_rowwise_q8",
            (RowwiseQuantKind::Int8, DType::F16) => "attn_weighted_sum_rowwise_q8_sf16",
            (RowwiseQuantKind::Int4, DType::F32) => "attn_weighted_sum_rowwise_q4",
            (RowwiseQuantKind::Int4, DType::F16) => "attn_weighted_sum_rowwise_q4_sf16",
            (_, dt) => candle_core::bail!(
                "unsupported scale dtype for rowwise weighted-sum op: {:?}",
                dt
            ),
        };

        let bh_full = b * full_heads;
        let out_elems = bh_full * self.head_dim;
        let out_shape = Shape::from((b, full_heads, self.head_dim));
        if out_elems == 0 {
            let out = s1
                .device()
                .new_buffer(0, DType::F32, "kv-attn-rowwise-weighted-sum-out")?;
            return Ok((
                candle_core::MetalStorage::new(out, s1.device().clone(), 0, DType::F32),
                out_shape,
            ));
        }

        let bh_u32 =
            u32::try_from(bh_full).map_err(|_| candle_core::Error::msg("bh_full too large"))?;
        let full_heads_u32 = u32::try_from(full_heads)
            .map_err(|_| candle_core::Error::msg("full_heads too large"))?;
        let kv_heads_u32 = u32::try_from(self.kv_heads)
            .map_err(|_| candle_core::Error::msg("kv_heads too large"))?;
        let repeat_u32 = u32::try_from(self.repeat_factor)
            .map_err(|_| candle_core::Error::msg("repeat_factor too large"))?;
        let t_u32 =
            u32::try_from(self.tokens).map_err(|_| candle_core::Error::msg("tokens too large"))?;
        let d_u32 = u32::try_from(self.head_dim)
            .map_err(|_| candle_core::Error::msg("head_dim too large"))?;

        let metal = s1.device().metal_device();
        let pipeline = get_or_create_kv_attn_pipeline(s1.device().id(), metal, kernel_name)?;
        let output =
            s1.device()
                .new_buffer(out_elems, DType::F32, "kv-attn-rowwise-weighted-sum-out")?;
        let encoder = s1.device().command_encoder()?;
        encoder.set_label("candelora_kv_attn_rowwise_weighted_sum");
        encoder.set_compute_pipeline_state(&pipeline);

        let attn = BufferOffset {
            buffer: s1.buffer(),
            offset_in_bytes: l1.start_offset() * DType::F32.size_in_bytes(),
        };
        let vq = BufferOffset {
            buffer: s2.buffer(),
            offset_in_bytes: l2.start_offset() * DType::U8.size_in_bytes(),
        };
        let v_scales = BufferOffset {
            buffer: s3.buffer(),
            offset_in_bytes: l3.start_offset() * DType::F32.size_in_bytes(),
        };
        let out = BufferOffset {
            buffer: &output,
            offset_in_bytes: 0,
        };
        let encoder_ref = &encoder;
        candle_metal_kernels::set_params!(
            encoder_ref,
            (
                bh_u32,
                full_heads_u32,
                kv_heads_u32,
                repeat_u32,
                t_u32,
                d_u32,
                &attn,
                &vq,
                &v_scales,
                &out
            )
        );

        let threads = pipeline
            .max_total_threads_per_threadgroup()
            .min(out_elems.max(1));
        let groups = out_elems.div_ceil(threads);
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
            candle_core::MetalStorage::new(output, s1.device().clone(), out_elems, DType::F32),
            out_shape,
        ))
    }
}

#[cfg(feature = "metal")]
const KV_ATTN_METAL: &str = include_str!("metal_src/kv_attn.metal");

/// Computes per-head scaled q·k^T scores for decode-shaped tensors.
///
/// Expected shapes:
/// - `q`: `[batch, heads, dim]`
/// - `k`: `[batch, heads, tokens, dim]`
///
/// Returns:
/// - `scores`: `[batch, heads, tokens]` in `F32`.
pub fn qk_scores(q: &Tensor, k: &Tensor, scale: f64) -> Result<Tensor> {
    if !q.device().same_device(k.device()) {
        candle_core::bail!(
            "device mismatch in fused kv-attn qk op: q={:?}, k={:?}",
            q.device(),
            k.device()
        );
    }
    if q.dtype() != k.dtype() {
        candle_core::bail!(
            "dtype mismatch in fused kv-attn qk op: q={:?}, k={:?}",
            q.dtype(),
            k.dtype()
        );
    }
    if q.rank() != 3 || k.rank() != 4 {
        candle_core::bail!(
            "fused kv-attn qk op expects q rank-3 and k rank-4, got q={:?}, k={:?}",
            q.dims(),
            k.dims()
        );
    }
    q.apply_op2_no_bwd(
        k,
        &QkScoresOp {
            scale: scale as f32,
        },
    )
}

/// Computes weighted sum over values using attention probabilities.
///
/// Expected shapes:
/// - `attn_probs`: `[batch, heads, tokens]` in `F32`
/// - `v`: `[batch, heads, tokens, dim]` in `F32/F16/BF16`
///
/// Returns:
/// - `[batch, heads, dim]` in `F32`.
pub fn attn_weighted_sum(attn_probs: &Tensor, v: &Tensor) -> Result<Tensor> {
    if !attn_probs.device().same_device(v.device()) {
        candle_core::bail!(
            "device mismatch in fused kv-attn weighted-sum op: attn={:?}, v={:?}",
            attn_probs.device(),
            v.device()
        );
    }
    if attn_probs.dtype() != DType::F32 {
        candle_core::bail!(
            "fused kv-attn weighted-sum expects F32 attn probs, got {:?}",
            attn_probs.dtype()
        );
    }
    if attn_probs.rank() != 3 || v.rank() != 4 {
        candle_core::bail!(
            "fused kv-attn weighted-sum expects attn rank-3 and v rank-4, got attn={:?}, v={:?}",
            attn_probs.dims(),
            v.dims()
        );
    }
    attn_probs.apply_op2_no_bwd(v, &AttnWeightedSumOp)
}

/// Computes q·k^T where `k` is quantized and shaped as `[tokens, dim]`.
///
/// Expected shapes:
/// - `q`: `[rows, dim]` in `F32/F16/BF16`
/// - `k`: quantized tensor with shape `[tokens, dim]`
///
/// Returns:
/// - `[rows, tokens]` in the same dtype behavior as candle quantized matmul.
pub fn qk_scores_quantized_k(q: &Tensor, k: std::sync::Arc<QTensor>, scale: f64) -> Result<Tensor> {
    if q.rank() != 2 {
        candle_core::bail!(
            "qk_scores_quantized_k expects q rank-2 [rows, dim], got {:?}",
            q.dims()
        );
    }
    let k_shape = k.shape().dims();
    if k_shape.len() != 2 {
        candle_core::bail!(
            "qk_scores_quantized_k expects quantized k rank-2 [tokens, dim], got {:?}",
            k_shape
        );
    }
    if !q.device().same_device(&k.device()) {
        candle_core::bail!(
            "device mismatch in qk_scores_quantized_k: q={:?}, k={:?}",
            q.device(),
            k.device()
        );
    }
    let q = if q.dtype() != DType::F32 {
        q.to_dtype(DType::F32)?
    } else {
        q.clone()
    };
    let mm = QMatMul::from_arc(k)?;
    let out = mm.forward(&q)?;
    if (scale - 1.0).abs() < f64::EPSILON {
        Ok(out)
    } else {
        out.affine(scale, 0.0)
    }
}

/// Computes attention weighted sum where `v_t` is quantized transposed values.
///
/// Expected shapes:
/// - `attn_probs`: `[rows, tokens]` in `F32/F16/BF16`
/// - `v_t`: quantized tensor with shape `[dim, tokens]`
///
/// Returns:
/// - `[rows, dim]`
pub fn attn_weighted_sum_quantized_vt(
    attn_probs: &Tensor,
    v_t: std::sync::Arc<QTensor>,
) -> Result<Tensor> {
    if attn_probs.rank() != 2 {
        candle_core::bail!(
            "attn_weighted_sum_quantized_vt expects attn_probs rank-2 [rows, tokens], got {:?}",
            attn_probs.dims()
        );
    }
    let v_shape = v_t.shape().dims();
    if v_shape.len() != 2 {
        candle_core::bail!(
            "attn_weighted_sum_quantized_vt expects quantized v_t rank-2 [dim, tokens], got {:?}",
            v_shape
        );
    }
    if !attn_probs.device().same_device(&v_t.device()) {
        candle_core::bail!(
            "device mismatch in attn_weighted_sum_quantized_vt: attn={:?}, v_t={:?}",
            attn_probs.device(),
            v_t.device()
        );
    }
    let attn_probs = if attn_probs.dtype() != DType::F32 {
        attn_probs.to_dtype(DType::F32)?
    } else {
        attn_probs.clone()
    };
    let mm = QMatMul::from_arc(v_t)?;
    mm.forward(&attn_probs)
}

#[cfg(feature = "metal")]
fn try_pack_rowwise_quantized_metal(
    x: &Tensor,
    head_dim: usize,
    kind: RowwiseQuantKind,
) -> Result<Option<(Tensor, Tensor)>> {
    use candle_metal_kernels::BufferOffset;
    use objc2_metal::MTLSize;

    if !x.device().is_metal() {
        return Ok(None);
    }
    let x = x.contiguous()?;
    let rows = x.elem_count() / head_dim;
    let packed_cols = expected_packed_cols(head_dim, kind);
    let packed = Tensor::zeros((rows, packed_cols), DType::U8, x.device())?;
    let scales = Tensor::zeros(rows, DType::F32, x.device())?;
    if rows == 0 {
        return Ok(Some((packed, scales)));
    }

    {
        let (x_storage, x_layout) = x.storage_and_layout();
        let (packed_storage, packed_layout) = packed.storage_and_layout();
        let (scales_storage, scales_layout) = scales.storage_and_layout();

        let (x_metal, packed_metal, scales_metal) =
            match (&*x_storage, &*packed_storage, &*scales_storage) {
                (
                    candle_core::Storage::Metal(xm),
                    candle_core::Storage::Metal(pm),
                    candle_core::Storage::Metal(sm),
                ) => (xm, pm, sm),
                _ => return Ok(None),
            };
        if !x_layout.is_contiguous()
            || !packed_layout.is_contiguous()
            || !scales_layout.is_contiguous()
        {
            candle_core::bail!("fused rowwise pack expects contiguous layouts")
        }

        let kernel_name = match (kind, x_metal.dtype()) {
            (RowwiseQuantKind::Int8, DType::F32) => "pack_rowwise_fused_q8_f32",
            (RowwiseQuantKind::Int8, DType::F16) => "pack_rowwise_fused_q8_f16",
            (RowwiseQuantKind::Int8, DType::BF16) => "pack_rowwise_fused_q8_bf16",
            (RowwiseQuantKind::Int4, DType::F32) => "pack_rowwise_fused_q4_f32",
            (RowwiseQuantKind::Int4, DType::F16) => "pack_rowwise_fused_q4_f16",
            (RowwiseQuantKind::Int4, DType::BF16) => "pack_rowwise_fused_q4_bf16",
            (_, dt) => {
                candle_core::bail!("unsupported dtype for fused rowwise pack op: {:?}", dt)
            }
        };
        let rows_u32 =
            u32::try_from(rows).map_err(|_| candle_core::Error::msg("rows too large"))?;
        let d_u32 =
            u32::try_from(head_dim).map_err(|_| candle_core::Error::msg("head_dim too large"))?;
        let metal = x_metal.device().metal_device();
        let pipeline = get_or_create_kv_attn_pipeline(x_metal.device().id(), metal, kernel_name)?;
        let encoder = x_metal.device().command_encoder()?;
        encoder.set_label("candelora_kv_attn_rowwise_pack_fused");
        encoder.set_compute_pipeline_state(&pipeline);

        let x_bo = BufferOffset {
            buffer: x_metal.buffer(),
            offset_in_bytes: x_layout.start_offset() * x_metal.dtype().size_in_bytes(),
        };
        let packed_bo = BufferOffset {
            buffer: packed_metal.buffer(),
            offset_in_bytes: packed_layout.start_offset() * DType::U8.size_in_bytes(),
        };
        let scales_bo = BufferOffset {
            buffer: scales_metal.buffer(),
            offset_in_bytes: scales_layout.start_offset() * DType::F32.size_in_bytes(),
        };
        let encoder_ref = &encoder;
        candle_metal_kernels::set_params!(
            encoder_ref,
            (rows_u32, d_u32, &x_bo, &packed_bo, &scales_bo)
        );

        let threads = pipeline
            .max_total_threads_per_threadgroup()
            .min(rows.max(1));
        let groups = rows.div_ceil(threads);
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

    Ok(Some((packed, scales)))
}

#[cfg(not(feature = "metal"))]
fn try_pack_rowwise_quantized_metal(
    _x: &Tensor,
    _head_dim: usize,
    _kind: RowwiseQuantKind,
) -> Result<Option<(Tensor, Tensor)>> {
    Ok(None)
}

/// Packs a dense rowwise tensor into quantized bytes + per-row scales.
///
/// Shapes:
/// - `x`: `[..., head_dim]` in `F32/F16/BF16`.
///
/// Returns:
/// - `packed`: `[rows, packed_cols]` in `U8`
/// - `scales`: `[rows]` in `F32`
/// where `rows = prod(x.shape()) / head_dim`.
pub fn pack_rowwise_quantized(
    x: &Tensor,
    head_dim: usize,
    kind: RowwiseQuantKind,
) -> Result<(Tensor, Tensor)> {
    if head_dim == 0 {
        candle_core::bail!("pack_rowwise_quantized requires head_dim > 0")
    }
    if x.rank() < 1 {
        candle_core::bail!(
            "pack_rowwise_quantized expects rank >= 1 tensor, got {:?}",
            x.dims()
        )
    }
    let x = x.contiguous()?;
    let disable_fused = std::env::var("CANDELORA_DISABLE_FUSED_PACK")
        .ok()
        .map(|v| {
            let lv = v.trim().to_ascii_lowercase();
            lv == "1" || lv == "true" || lv == "yes" || lv == "on"
        })
        .unwrap_or(false);
    if !disable_fused {
        if let Some((packed, scales)) = try_pack_rowwise_quantized_metal(&x, head_dim, kind)? {
            return Ok((packed, scales));
        }
    }
    let scales = x.apply_op1_no_bwd(&RowwisePackScalesOp { head_dim, kind })?;
    let packed = x.apply_op2_no_bwd(&scales, &RowwisePackDataOp { head_dim, kind })?;
    Ok((packed, scales))
}

/// Computes q·k^T for row-wise packed quantized K without materializing dense K.
///
/// Shapes:
/// - `q`: `[batch, full_heads, head_dim]`
/// - `packed_k`: `[rows, packed_cols]` where `rows = batch * kv_heads * tokens`
/// - `k_scales`: `[rows]`
///
/// Returns:
/// - `[batch, full_heads, tokens]` in `F32`.
pub fn qk_scores_quantized_rowwise(
    q: &Tensor,
    packed_k: &Tensor,
    k_scales: &Tensor,
    kv_heads: usize,
    repeat_factor: usize,
    tokens: usize,
    head_dim: usize,
    kind: RowwiseQuantKind,
    scale: f64,
) -> Result<Tensor> {
    if repeat_factor == 0 {
        candle_core::bail!("rowwise qk requires repeat_factor > 0")
    }
    let qd = q.dims();
    if qd.len() != 3 {
        candle_core::bail!("rowwise qk expects q rank-3 [b,h,d], got {:?}", qd)
    }
    if qd[2] != head_dim {
        candle_core::bail!("rowwise qk head_dim mismatch: q={} arg={}", qd[2], head_dim)
    }
    if !q.device().same_device(packed_k.device()) || !q.device().same_device(k_scales.device()) {
        candle_core::bail!(
            "rowwise qk device mismatch: q={:?}, packed_k={:?}, scales={:?}",
            q.device(),
            packed_k.device(),
            k_scales.device()
        )
    }
    q.apply_op3_no_bwd(
        packed_k,
        k_scales,
        &RowwiseQkScoresOp {
            kv_heads,
            repeat_factor,
            tokens,
            head_dim,
            scale: scale as f32,
            kind,
        },
    )
}

/// Computes attention weighted sum using row-wise packed quantized V.
///
/// Shapes:
/// - `attn_probs`: `[batch, full_heads, tokens]` (F32)
/// - `packed_v`: `[rows, packed_cols]` where `rows = batch * kv_heads * tokens`
/// - `v_scales`: `[rows]`
///
/// Returns:
/// - `[batch, full_heads, head_dim]` in `F32`.
pub fn attn_weighted_sum_quantized_rowwise(
    attn_probs: &Tensor,
    packed_v: &Tensor,
    v_scales: &Tensor,
    kv_heads: usize,
    repeat_factor: usize,
    tokens: usize,
    head_dim: usize,
    kind: RowwiseQuantKind,
) -> Result<Tensor> {
    if repeat_factor == 0 {
        candle_core::bail!("rowwise weighted-sum requires repeat_factor > 0")
    }
    if attn_probs.dtype() != DType::F32 {
        candle_core::bail!(
            "rowwise weighted-sum expects F32 attn_probs, got {:?}",
            attn_probs.dtype()
        )
    }
    if !attn_probs.device().same_device(packed_v.device())
        || !attn_probs.device().same_device(v_scales.device())
    {
        candle_core::bail!(
            "rowwise weighted-sum device mismatch: attn={:?}, packed_v={:?}, scales={:?}",
            attn_probs.device(),
            packed_v.device(),
            v_scales.device()
        )
    }
    attn_probs.apply_op3_no_bwd(
        packed_v,
        v_scales,
        &RowwiseAttnWeightedSumOp {
            kv_heads,
            repeat_factor,
            tokens,
            head_dim,
            kind,
        },
    )
}

fn cpu_qk_scores_turboquant_impl<T: Copy, F: Fn(T) -> f32 + Copy>(
    q: &[T],
    q_to_f32: F,
    codes: &[u8],
    scales: &[f32],
    pair_signs: &[f32],
    residual_signs: Option<&[u8]>,
    residual_scales: Option<&[f32]>,
    batch: usize,
    full_heads: usize,
    kv_heads: usize,
    repeat_factor: usize,
    tokens: usize,
    head_dim: usize,
    subvector_dim: usize,
    scale_block_dim: usize,
    signed_max: i32,
    scale: f32,
) -> Vec<f32> {
    let num_subvectors = head_dim / subvector_dim;
    let num_scale_blocks = head_dim / scale_block_dim;
    let mut out = vec![0f32; batch * full_heads * tokens];
    let mut q_rot = vec![0f32; head_dim];
    for b in 0..batch {
        for full_head in 0..full_heads {
            let kv_head = full_head / repeat_factor;
            let q_base = (b * full_heads + full_head) * head_dim;
            for (pair_idx, sign) in pair_signs.iter().copied().enumerate() {
                let i = pair_idx * 2;
                let qa = q_to_f32(q[q_base + i]);
                let qb = q_to_f32(q[q_base + i + 1]);
                q_rot[i] = (qa + sign * qb) * FRAC_1_SQRT_2;
                q_rot[i + 1] = (-sign * qa + qb) * FRAC_1_SQRT_2;
            }
            for tok in 0..tokens {
                let row = ((b * kv_heads + kv_head) * tokens) + tok;
                let mut acc = 0f32;
                for sub in 0..num_subvectors {
                    let block = (sub * subvector_dim) / scale_block_dim;
                    let scale_idx = row * num_scale_blocks + block;
                    let sub_scale = scales[scale_idx];
                    let residual_scale = residual_scales
                        .as_ref()
                        .map(|vals| vals[scale_idx])
                        .unwrap_or(0.0);
                    let start = sub * subvector_dim;
                    let end = start + subvector_dim;
                    for idx in start..end {
                        let qv = (codes[row * head_dim + idx] as i32) - signed_max;
                        acc += q_rot[idx] * ((qv as f32) * sub_scale);
                        if let Some(bits) = residual_signs {
                            let sign = if bits[row * head_dim + idx] == 0 {
                                -1.0
                            } else {
                                1.0
                            };
                            acc += q_rot[idx] * sign * residual_scale;
                        }
                    }
                }
                out[(b * full_heads + full_head) * tokens + tok] = acc * scale;
            }
        }
    }
    out
}

fn unpack_packed_u8_values(packed: &[u8], bits_per_value: usize, value_count: usize) -> Vec<u8> {
    debug_assert!(bits_per_value > 0 && bits_per_value <= 8);
    (0..value_count)
        .map(|idx| {
            let start_bit = idx * bits_per_value;
            let byte_idx = start_bit / 8;
            let bit_offset = start_bit % 8;
            let mut word = u16::from(*packed.get(byte_idx).unwrap_or(&0));
            if bit_offset + bits_per_value > 8 {
                word |= u16::from(*packed.get(byte_idx + 1).unwrap_or(&0)) << 8;
            }
            let mask = (1u16 << bits_per_value) - 1;
            ((word >> bit_offset) & mask) as u8
        })
        .collect()
}

fn cpu_qk_scores_turboquant(
    q: &Tensor,
    codes: &Tensor,
    scales: &Tensor,
    pair_signs: &Tensor,
    residual_signs: Option<&Tensor>,
    residual_scales: Option<&Tensor>,
    kv_heads: usize,
    repeat_factor: usize,
    tokens: usize,
    head_dim: usize,
    subvector_dim: usize,
    kind: TurboQuantKind,
    scale: f64,
) -> Result<Tensor> {
    let q = q
        .to_dtype(DType::F32)?
        .to_device(&candle_core::Device::Cpu)?;
    let codes = codes.to_device(&candle_core::Device::Cpu)?;
    let scales = scales
        .to_dtype(DType::F32)?
        .to_device(&candle_core::Device::Cpu)?;
    let pair_signs = pair_signs.to_device(&candle_core::Device::Cpu)?;
    let residual_signs = match residual_signs {
        Some(t) => Some(t.to_device(&candle_core::Device::Cpu)?),
        None => None,
    };
    let residual_scales = match residual_scales {
        Some(t) => Some(
            t.to_dtype(DType::F32)?
                .to_device(&candle_core::Device::Cpu)?,
        ),
        None => None,
    };

    let (batch, full_heads, _) = q.dims3()?;
    let q = q.flatten_all()?.to_vec1::<f32>()?;
    let codes = codes.flatten_all()?.to_vec1::<u8>()?;
    let scales = scales.flatten_all()?.to_vec1::<f32>()?;
    let pair_signs = pair_signs.flatten_all()?.to_vec1::<f32>()?;
    let residual_signs = match residual_signs {
        Some(t) => Some(t.flatten_all()?.to_vec1::<u8>()?),
        None => None,
    };
    let residual_scales = match residual_scales {
        Some(t) => Some(t.flatten_all()?.to_vec1::<f32>()?),
        None => None,
    };
    let out = cpu_qk_scores_turboquant_impl(
        &q,
        |v| v,
        &codes,
        &scales,
        &pair_signs,
        residual_signs.as_deref(),
        residual_scales.as_deref(),
        batch,
        full_heads,
        kv_heads,
        repeat_factor,
        tokens,
        head_dim,
        subvector_dim,
        subvector_dim,
        kind.signed_max(),
        scale as f32,
    );
    Tensor::from_vec(out, (batch, full_heads, tokens), &candle_core::Device::Cpu)
}

fn cpu_qk_scores_turboquant_packed(
    q: &Tensor,
    packed_codes: &Tensor,
    scales: &Tensor,
    pair_signs: &Tensor,
    packed_residual_signs: Option<&Tensor>,
    residual_scales: Option<&Tensor>,
    kv_heads: usize,
    repeat_factor: usize,
    tokens: usize,
    head_dim: usize,
    subvector_dim: usize,
    scale_block_dim: usize,
    code_bits: usize,
    kind: TurboQuantKind,
    scale: f64,
) -> Result<Tensor> {
    let q = q
        .to_dtype(DType::F32)?
        .to_device(&candle_core::Device::Cpu)?;
    let packed_codes = packed_codes.to_device(&candle_core::Device::Cpu)?;
    let scales = scales
        .to_dtype(DType::F32)?
        .to_device(&candle_core::Device::Cpu)?;
    let pair_signs = pair_signs.to_device(&candle_core::Device::Cpu)?;
    let packed_residual_signs = match packed_residual_signs {
        Some(t) => Some(t.to_device(&candle_core::Device::Cpu)?),
        None => None,
    };
    let residual_scales = match residual_scales {
        Some(t) => Some(
            t.to_dtype(DType::F32)?
                .to_device(&candle_core::Device::Cpu)?,
        ),
        None => None,
    };

    let (batch, full_heads, _) = q.dims3()?;
    let q = q.flatten_all()?.to_vec1::<f32>()?;
    let packed_codes = packed_codes.flatten_all()?.to_vec1::<u8>()?;
    let scales = scales.flatten_all()?.to_vec1::<f32>()?;
    let pair_signs = pair_signs.flatten_all()?.to_vec1::<f32>()?;
    let residual_scales = match residual_scales {
        Some(t) => Some(t.flatten_all()?.to_vec1::<f32>()?),
        None => None,
    };
    let value_count = batch * kv_heads * tokens * head_dim;
    let codes = unpack_packed_u8_values(&packed_codes, code_bits, value_count);
    let residual_signs = match packed_residual_signs {
        Some(t) => Some(unpack_packed_u8_values(
            &t.flatten_all()?.to_vec1::<u8>()?,
            1,
            value_count,
        )),
        None => None,
    };
    let out = cpu_qk_scores_turboquant_impl(
        &q,
        |v| v,
        &codes,
        &scales,
        &pair_signs,
        residual_signs.as_deref(),
        residual_scales.as_deref(),
        batch,
        full_heads,
        kv_heads,
        repeat_factor,
        tokens,
        head_dim,
        subvector_dim,
        scale_block_dim,
        kind.signed_max(),
        scale as f32,
    );
    Tensor::from_vec(out, (batch, full_heads, tokens), &candle_core::Device::Cpu)
}

#[cfg(feature = "metal")]
fn turboquant_kernel_name(
    q_dtype: DType,
    scale_dtype: DType,
    subvector_dim: usize,
    packed: bool,
    code_bits: Option<usize>,
    has_residual: bool,
) -> Result<&'static str> {
    let name = match (
        q_dtype,
        scale_dtype,
        subvector_dim,
        packed,
        code_bits,
        has_residual,
    ) {
        (DType::F16, DType::F16, 4, true, Some(3), false) => {
            "qk_scores_turbo_packed_fast_sv4c3_nr_qf16_sf16"
        }
        (DType::F16, DType::F16, 4, true, Some(3), true) => {
            "qk_scores_turbo_packed_fast_sv4c3_res_qf16_sf16"
        }
        (DType::F16, DType::F16, 4, true, Some(4), false) => {
            "qk_scores_turbo_packed_fast_sv4c4_nr_qf16_sf16"
        }
        (DType::F16, DType::F16, 4, true, Some(4), true) => {
            "qk_scores_turbo_packed_fast_sv4c4_res_qf16_sf16"
        }
        (DType::F16, DType::F16, 8, true, Some(3), false) => {
            "qk_scores_turbo_packed_fast_sv8c3_nr_qf16_sf16"
        }
        (DType::F16, DType::F16, 8, true, Some(3), true) => {
            "qk_scores_turbo_packed_fast_sv8c3_res_qf16_sf16"
        }
        (DType::F16, DType::F16, 8, true, Some(4), false) => {
            "qk_scores_turbo_packed_fast_sv8c4_nr_qf16_sf16"
        }
        (DType::F16, DType::F16, 8, true, Some(4), true) => {
            "qk_scores_turbo_packed_fast_sv8c4_res_qf16_sf16"
        }
        (DType::F32, DType::F32, 4, true, _, _) => "qk_scores_turbo_packed_sv4_qf32_sf32",
        (DType::F32, DType::F16, 4, true, _, _) => "qk_scores_turbo_packed_sv4_qf32_sf16",
        (DType::F16, DType::F32, 4, true, _, _) => "qk_scores_turbo_packed_sv4_qf16_sf32",
        (DType::F16, DType::F16, 4, true, _, _) => "qk_scores_turbo_packed_sv4_qf16_sf16",
        (DType::BF16, DType::F32, 4, true, _, _) => "qk_scores_turbo_packed_sv4_qbf16_sf32",
        (DType::BF16, DType::F16, 4, true, _, _) => "qk_scores_turbo_packed_sv4_qbf16_sf16",
        (DType::F32, DType::F32, 8, true, _, _) => "qk_scores_turbo_packed_sv8_qf32_sf32",
        (DType::F32, DType::F16, 8, true, _, _) => "qk_scores_turbo_packed_sv8_qf32_sf16",
        (DType::F16, DType::F32, 8, true, _, _) => "qk_scores_turbo_packed_sv8_qf16_sf32",
        (DType::F16, DType::F16, 8, true, _, _) => "qk_scores_turbo_packed_sv8_qf16_sf16",
        (DType::BF16, DType::F32, 8, true, _, _) => "qk_scores_turbo_packed_sv8_qbf16_sf32",
        (DType::BF16, DType::F16, 8, true, _, _) => "qk_scores_turbo_packed_sv8_qbf16_sf16",
        (DType::F32, DType::F32, _, true, _, _) => "qk_scores_turbo_packed_qf32_sf32",
        (DType::F32, DType::F16, _, true, _, _) => "qk_scores_turbo_packed_qf32_sf16",
        (DType::F16, DType::F32, _, true, _, _) => "qk_scores_turbo_packed_qf16_sf32",
        (DType::F16, DType::F16, _, true, _, _) => "qk_scores_turbo_packed_qf16_sf16",
        (DType::BF16, DType::F32, _, true, _, _) => "qk_scores_turbo_packed_qbf16_sf32",
        (DType::BF16, DType::F16, _, true, _, _) => "qk_scores_turbo_packed_qbf16_sf16",
        (DType::F32, DType::F32, 4, false, _, _) => "qk_scores_turbo_sv4_qf32_sf32",
        (DType::F32, DType::F16, 4, false, _, _) => "qk_scores_turbo_sv4_qf32_sf16",
        (DType::F16, DType::F32, 4, false, _, _) => "qk_scores_turbo_sv4_qf16_sf32",
        (DType::F16, DType::F16, 4, false, _, _) => "qk_scores_turbo_sv4_qf16_sf16",
        (DType::BF16, DType::F32, 4, false, _, _) => "qk_scores_turbo_sv4_qbf16_sf32",
        (DType::BF16, DType::F16, 4, false, _, _) => "qk_scores_turbo_sv4_qbf16_sf16",
        (DType::F32, DType::F32, 8, false, _, _) => "qk_scores_turbo_sv8_qf32_sf32",
        (DType::F32, DType::F16, 8, false, _, _) => "qk_scores_turbo_sv8_qf32_sf16",
        (DType::F16, DType::F32, 8, false, _, _) => "qk_scores_turbo_sv8_qf16_sf32",
        (DType::F16, DType::F16, 8, false, _, _) => "qk_scores_turbo_sv8_qf16_sf16",
        (DType::BF16, DType::F32, 8, false, _, _) => "qk_scores_turbo_sv8_qbf16_sf32",
        (DType::BF16, DType::F16, 8, false, _, _) => "qk_scores_turbo_sv8_qbf16_sf16",
        (DType::F32, DType::F32, _, false, _, _) => "qk_scores_turbo_qf32_sf32",
        (DType::F32, DType::F16, _, false, _, _) => "qk_scores_turbo_qf32_sf16",
        (DType::F16, DType::F32, _, false, _, _) => "qk_scores_turbo_qf16_sf32",
        (DType::F16, DType::F16, _, false, _, _) => "qk_scores_turbo_qf16_sf16",
        (DType::BF16, DType::F32, _, false, _, _) => "qk_scores_turbo_qbf16_sf32",
        (DType::BF16, DType::F16, _, false, _, _) => "qk_scores_turbo_qbf16_sf16",
        dtypes => {
            candle_core::bail!(
                "unsupported (q, scales, subvector_dim, packed, code_bits, has_residual) for turboquant qk op: {:?}",
                dtypes
            )
        }
    };
    Ok(name)
}

#[cfg(feature = "metal")]
fn turboquant_grouped_threads(max_threads: usize, default_threads: usize) -> usize {
    let parsed = std::env::var("CANDELORA_TURBO_TG_SIZE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0);
    parsed.unwrap_or(default_threads).min(max_threads).max(1)
}

#[cfg(feature = "metal")]
fn rotate_pairwise_query_kernel_name(q_dtype: DType) -> Result<&'static str> {
    let name = match q_dtype {
        DType::F32 => "rotate_pairwise_query_f32",
        DType::F16 => "rotate_pairwise_query_f16",
        DType::BF16 => "rotate_pairwise_query_bf16",
        dtype => {
            candle_core::bail!(
                "unsupported q dtype for turboquant query rotation: {:?}",
                dtype
            )
        }
    };
    Ok(name)
}

#[cfg(feature = "metal")]
fn try_rotate_pairwise_query_metal(q: &Tensor, pair_signs: &Tensor) -> Result<Option<Tensor>> {
    use candle_metal_kernels::BufferOffset;
    use objc2_metal::MTLSize;

    if !q.device().is_metal() {
        return Ok(None);
    }

    let q = q.contiguous()?;
    let pair_signs = pair_signs.contiguous()?;
    let (batch, full_heads, head_dim) = q.dims3()?;
    if head_dim % 2 != 0 {
        candle_core::bail!(
            "turboquant pairwise query rotation requires even head_dim, got {}",
            head_dim
        );
    }
    let pair_dims = pair_signs.dims();
    if pair_dims != [head_dim / 2] {
        candle_core::bail!(
            "turboquant pairwise query rotation pair_signs shape mismatch: got {:?}, expected [{}]",
            pair_dims,
            head_dim / 2
        );
    }

    let out = Tensor::zeros((batch, full_heads, head_dim), q.dtype(), q.device())?;
    if out.elem_count() == 0 {
        return Ok(Some(out));
    }

    {
        let (q_storage, q_layout) = q.storage_and_layout();
        let (pair_storage, pair_layout) = pair_signs.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();
        let (q_metal, pair_metal, out_metal) = match (&*q_storage, &*pair_storage, &*out_storage) {
            (
                candle_core::Storage::Metal(qm),
                candle_core::Storage::Metal(pm),
                candle_core::Storage::Metal(om),
            ) => (qm, pm, om),
            _ => return Ok(None),
        };
        if !q_layout.is_contiguous() || !pair_layout.is_contiguous() || !out_layout.is_contiguous()
        {
            candle_core::bail!("turboquant pairwise query rotation expects contiguous layouts");
        }

        let kernel_name = rotate_pairwise_query_kernel_name(q_metal.dtype())?;
        let bh_full = batch * full_heads;
        let pair_elems = bh_full * (head_dim / 2);
        let bh_u32 =
            u32::try_from(bh_full).map_err(|_| candle_core::Error::msg("bh_full too large"))?;
        let d_u32 =
            u32::try_from(head_dim).map_err(|_| candle_core::Error::msg("head_dim too large"))?;

        let metal = q_metal.device().metal_device();
        let pipeline = get_or_create_kv_attn_pipeline(q_metal.device().id(), metal, kernel_name)?;
        let encoder = q_metal.device().command_encoder()?;
        encoder.set_label("candelora_kv_attn_turbo_rotate_q");
        encoder.set_compute_pipeline_state(&pipeline);

        let q_bo = BufferOffset {
            buffer: q_metal.buffer(),
            offset_in_bytes: q_layout.start_offset() * q_metal.dtype().size_in_bytes(),
        };
        let pair_bo = BufferOffset {
            buffer: pair_metal.buffer(),
            offset_in_bytes: pair_layout.start_offset() * DType::F32.size_in_bytes(),
        };
        let out_bo = BufferOffset {
            buffer: out_metal.buffer(),
            offset_in_bytes: out_layout.start_offset() * q_metal.dtype().size_in_bytes(),
        };
        let encoder_ref = &encoder;
        candle_metal_kernels::set_params!(encoder_ref, (bh_u32, d_u32, &q_bo, &pair_bo, &out_bo));

        let threads = pipeline
            .max_total_threads_per_threadgroup()
            .min(pair_elems.max(1));
        let groups = pair_elems.div_ceil(threads);
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
    Ok(Some(out))
}

#[cfg(feature = "metal")]
fn try_qk_scores_turboquant_metal(
    q: &Tensor,
    codes: &Tensor,
    scales: &Tensor,
    pair_signs: &Tensor,
    residual_signs: Option<&Tensor>,
    residual_scales: Option<&Tensor>,
    kv_heads: usize,
    repeat_factor: usize,
    tokens: usize,
    head_dim: usize,
    subvector_dim: usize,
    kind: TurboQuantKind,
    scale: f64,
) -> Result<Option<Tensor>> {
    use candle_metal_kernels::BufferOffset;
    use objc2_metal::MTLSize;

    if !q.device().is_metal() {
        return Ok(None);
    }

    let dummy_residual_signs;
    let residual_signs = match residual_signs {
        Some(t) => t,
        None => {
            dummy_residual_signs = Tensor::zeros((1,), DType::U8, q.device())?;
            &dummy_residual_signs
        }
    };
    let scale_dtype = scales.dtype();
    let dummy_residual_scales;
    let residual_scales = match residual_scales {
        Some(t) => t,
        None => {
            dummy_residual_scales = Tensor::zeros((1,), scale_dtype, q.device())?;
            &dummy_residual_scales
        }
    };

    let use_grouped_kernel = matches!(subvector_dim, 4 | 8) && head_dim <= 256;
    let q_for_kernel = if use_grouped_kernel {
        q.contiguous()?
    } else {
        match try_rotate_pairwise_query_metal(q, pair_signs)? {
            Some(t) => t,
            None => return Ok(None),
        }
    };

    let (batch, full_heads, _) = q_for_kernel.dims3()?;
    let out = Tensor::zeros(
        (batch, full_heads, tokens),
        DType::F32,
        q_for_kernel.device(),
    )?;
    if out.elem_count() == 0 {
        return Ok(Some(out));
    }

    {
        let (q_storage, q_layout) = q_for_kernel.storage_and_layout();
        let (codes_storage, codes_layout) = codes.storage_and_layout();
        let (scales_storage, scales_layout) = scales.storage_and_layout();
        let (pair_storage, pair_layout) = pair_signs.storage_and_layout();
        let (residual_bits_storage, residual_bits_layout) = residual_signs.storage_and_layout();
        let (residual_scales_storage, residual_scales_layout) =
            residual_scales.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();

        let (
            q_metal,
            codes_metal,
            scales_metal,
            pair_metal,
            residual_bits_metal,
            residual_scales_metal,
            out_metal,
        ) = match (
            &*q_storage,
            &*codes_storage,
            &*scales_storage,
            &*pair_storage,
            &*residual_bits_storage,
            &*residual_scales_storage,
            &*out_storage,
        ) {
            (
                candle_core::Storage::Metal(qm),
                candle_core::Storage::Metal(cm),
                candle_core::Storage::Metal(sm),
                candle_core::Storage::Metal(pm),
                candle_core::Storage::Metal(rbm),
                candle_core::Storage::Metal(rsm),
                candle_core::Storage::Metal(om),
            ) => (qm, cm, sm, pm, rbm, rsm, om),
            _ => return Ok(None),
        };

        if !q_layout.is_contiguous()
            || !codes_layout.is_contiguous()
            || !scales_layout.is_contiguous()
            || !pair_layout.is_contiguous()
            || !residual_bits_layout.is_contiguous()
            || !residual_scales_layout.is_contiguous()
            || !out_layout.is_contiguous()
        {
            candle_core::bail!("turboquant qk expects contiguous layouts");
        }

        let kernel_name = turboquant_kernel_name(
            q_metal.dtype(),
            scales_metal.dtype(),
            subvector_dim,
            false,
            None,
            residual_signs.elem_count() > 1,
        )?;

        let bh_full = batch * full_heads;
        let out_elems = bh_full * tokens;
        let bh_u32 =
            u32::try_from(bh_full).map_err(|_| candle_core::Error::msg("bh_full too large"))?;
        let full_heads_u32 = u32::try_from(full_heads)
            .map_err(|_| candle_core::Error::msg("full_heads too large"))?;
        let kv_heads_u32 =
            u32::try_from(kv_heads).map_err(|_| candle_core::Error::msg("kv_heads too large"))?;
        let repeat_u32 = u32::try_from(repeat_factor)
            .map_err(|_| candle_core::Error::msg("repeat_factor too large"))?;
        let t_u32 =
            u32::try_from(tokens).map_err(|_| candle_core::Error::msg("tokens too large"))?;
        let d_u32 =
            u32::try_from(head_dim).map_err(|_| candle_core::Error::msg("head_dim too large"))?;
        let sub_u32 = u32::try_from(subvector_dim)
            .map_err(|_| candle_core::Error::msg("subvector_dim too large"))?;
        let signed_max = kind.signed_max();
        let use_residual_signs = u32::from(residual_signs.elem_count() > 1);

        let metal = q_metal.device().metal_device();
        let pipeline = get_or_create_kv_attn_pipeline(q_metal.device().id(), metal, kernel_name)?;
        let encoder = q_metal.device().command_encoder()?;
        encoder.set_label("candelora_kv_attn_turbo_qk");
        encoder.set_compute_pipeline_state(&pipeline);

        let q_bo = BufferOffset {
            buffer: q_metal.buffer(),
            offset_in_bytes: q_layout.start_offset() * q_metal.dtype().size_in_bytes(),
        };
        let codes_bo = BufferOffset {
            buffer: codes_metal.buffer(),
            offset_in_bytes: codes_layout.start_offset() * DType::U8.size_in_bytes(),
        };
        let scales_bo = BufferOffset {
            buffer: scales_metal.buffer(),
            offset_in_bytes: scales_layout.start_offset() * scales_metal.dtype().size_in_bytes(),
        };
        let pair_bo = BufferOffset {
            buffer: pair_metal.buffer(),
            offset_in_bytes: pair_layout.start_offset() * DType::F32.size_in_bytes(),
        };
        let residual_bits_bo = BufferOffset {
            buffer: residual_bits_metal.buffer(),
            offset_in_bytes: residual_bits_layout.start_offset() * DType::U8.size_in_bytes(),
        };
        let residual_scales_bo = BufferOffset {
            buffer: residual_scales_metal.buffer(),
            offset_in_bytes: residual_scales_layout.start_offset()
                * residual_scales_metal.dtype().size_in_bytes(),
        };
        let out_bo = BufferOffset {
            buffer: out_metal.buffer(),
            offset_in_bytes: out_layout.start_offset() * DType::F32.size_in_bytes(),
        };
        let encoder_ref = &encoder;
        candle_metal_kernels::set_params!(
            encoder_ref,
            (
                bh_u32,
                full_heads_u32,
                kv_heads_u32,
                repeat_u32,
                t_u32,
                d_u32,
                sub_u32,
                signed_max,
                use_residual_signs,
                scale as f32,
                &q_bo,
                &codes_bo,
                &scales_bo,
                &pair_bo,
                &residual_bits_bo,
                &residual_scales_bo,
                &out_bo
            )
        );

        let (tg_count, tg_size) = if use_grouped_kernel {
            let threads =
                turboquant_grouped_threads(pipeline.max_total_threads_per_threadgroup(), 256);
            (
                MTLSize {
                    width: bh_full,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threads,
                    height: 1,
                    depth: 1,
                },
            )
        } else {
            let threads = pipeline
                .max_total_threads_per_threadgroup()
                .min(out_elems.max(1));
            let groups = out_elems.div_ceil(threads);
            (
                MTLSize {
                    width: groups,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threads,
                    height: 1,
                    depth: 1,
                },
            )
        };
        encoder.dispatch_thread_groups(tg_count, tg_size);
    }
    Ok(Some(out))
}

#[cfg(feature = "metal")]
fn try_qk_scores_turboquant_packed_metal(
    q: &Tensor,
    packed_codes: &Tensor,
    scales: &Tensor,
    pair_signs: &Tensor,
    packed_residual_signs: Option<&Tensor>,
    residual_scales: Option<&Tensor>,
    kv_heads: usize,
    repeat_factor: usize,
    tokens: usize,
    head_dim: usize,
    subvector_dim: usize,
    scale_block_dim: usize,
    code_bits: usize,
    kind: TurboQuantKind,
    scale: f64,
) -> Result<Option<Tensor>> {
    use candle_metal_kernels::BufferOffset;
    use objc2_metal::MTLSize;

    if !q.device().is_metal() {
        return Ok(None);
    }

    let has_residual_signs = packed_residual_signs.is_some();
    let dummy_residual_signs;
    let packed_residual_signs = match packed_residual_signs {
        Some(t) => t,
        None => {
            dummy_residual_signs = Tensor::zeros((1,), DType::U8, q.device())?;
            &dummy_residual_signs
        }
    };
    let scale_dtype = scales.dtype();
    let dummy_residual_scales;
    let residual_scales = match residual_scales {
        Some(t) => t,
        None => {
            dummy_residual_scales = Tensor::zeros((1,), scale_dtype, q.device())?;
            &dummy_residual_scales
        }
    };

    let use_grouped_kernel = matches!(subvector_dim, 4 | 8) && head_dim <= 256;
    let q_for_kernel = if use_grouped_kernel {
        q.contiguous()?
    } else {
        match try_rotate_pairwise_query_metal(q, pair_signs)? {
            Some(t) => t,
            None => return Ok(None),
        }
    };

    let (batch, full_heads, _) = q_for_kernel.dims3()?;
    let out = Tensor::zeros(
        (batch, full_heads, tokens),
        DType::F32,
        q_for_kernel.device(),
    )?;
    if out.elem_count() == 0 {
        return Ok(Some(out));
    }

    {
        let (q_storage, q_layout) = q_for_kernel.storage_and_layout();
        let (codes_storage, codes_layout) = packed_codes.storage_and_layout();
        let (scales_storage, scales_layout) = scales.storage_and_layout();
        let (pair_storage, pair_layout) = pair_signs.storage_and_layout();
        let (residual_bits_storage, residual_bits_layout) =
            packed_residual_signs.storage_and_layout();
        let (residual_scales_storage, residual_scales_layout) =
            residual_scales.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();

        let (
            q_metal,
            codes_metal,
            scales_metal,
            pair_metal,
            residual_bits_metal,
            residual_scales_metal,
            out_metal,
        ) = match (
            &*q_storage,
            &*codes_storage,
            &*scales_storage,
            &*pair_storage,
            &*residual_bits_storage,
            &*residual_scales_storage,
            &*out_storage,
        ) {
            (
                candle_core::Storage::Metal(qm),
                candle_core::Storage::Metal(cm),
                candle_core::Storage::Metal(sm),
                candle_core::Storage::Metal(pm),
                candle_core::Storage::Metal(rbm),
                candle_core::Storage::Metal(rsm),
                candle_core::Storage::Metal(om),
            ) => (qm, cm, sm, pm, rbm, rsm, om),
            _ => return Ok(None),
        };

        if !q_layout.is_contiguous()
            || !codes_layout.is_contiguous()
            || !scales_layout.is_contiguous()
            || !pair_layout.is_contiguous()
            || !residual_bits_layout.is_contiguous()
            || !residual_scales_layout.is_contiguous()
            || !out_layout.is_contiguous()
        {
            candle_core::bail!("turboquant packed qk expects contiguous layouts");
        }

        let kernel_name = turboquant_kernel_name(
            q_metal.dtype(),
            scales_metal.dtype(),
            subvector_dim,
            true,
            Some(code_bits),
            has_residual_signs,
        )?;

        let bh_full = batch * full_heads;
        let out_elems = bh_full * tokens;
        let bh_u32 =
            u32::try_from(bh_full).map_err(|_| candle_core::Error::msg("bh_full too large"))?;
        let full_heads_u32 = u32::try_from(full_heads)
            .map_err(|_| candle_core::Error::msg("full_heads too large"))?;
        let kv_heads_u32 =
            u32::try_from(kv_heads).map_err(|_| candle_core::Error::msg("kv_heads too large"))?;
        let repeat_u32 = u32::try_from(repeat_factor)
            .map_err(|_| candle_core::Error::msg("repeat_factor too large"))?;
        let t_u32 =
            u32::try_from(tokens).map_err(|_| candle_core::Error::msg("tokens too large"))?;
        let d_u32 =
            u32::try_from(head_dim).map_err(|_| candle_core::Error::msg("head_dim too large"))?;
        let sub_u32 = u32::try_from(subvector_dim)
            .map_err(|_| candle_core::Error::msg("subvector_dim too large"))?;
        let scale_block_u32 = u32::try_from(scale_block_dim)
            .map_err(|_| candle_core::Error::msg("scale_block_dim too large"))?;
        let code_bits_u32 =
            u32::try_from(code_bits).map_err(|_| candle_core::Error::msg("code_bits too large"))?;
        let signed_max = kind.signed_max();
        let use_residual_signs = u32::from(has_residual_signs);

        let metal = q_metal.device().metal_device();
        let pipeline = get_or_create_kv_attn_pipeline(q_metal.device().id(), metal, kernel_name)?;
        let encoder = q_metal.device().command_encoder()?;
        encoder.set_label("candelora_kv_attn_turbo_qk_packed");
        encoder.set_compute_pipeline_state(&pipeline);

        let q_bo = BufferOffset {
            buffer: q_metal.buffer(),
            offset_in_bytes: q_layout.start_offset() * q_metal.dtype().size_in_bytes(),
        };
        let codes_bo = BufferOffset {
            buffer: codes_metal.buffer(),
            offset_in_bytes: codes_layout.start_offset() * DType::U8.size_in_bytes(),
        };
        let scales_bo = BufferOffset {
            buffer: scales_metal.buffer(),
            offset_in_bytes: scales_layout.start_offset() * scales_metal.dtype().size_in_bytes(),
        };
        let pair_bo = BufferOffset {
            buffer: pair_metal.buffer(),
            offset_in_bytes: pair_layout.start_offset() * DType::F32.size_in_bytes(),
        };
        let residual_bits_bo = BufferOffset {
            buffer: residual_bits_metal.buffer(),
            offset_in_bytes: residual_bits_layout.start_offset() * DType::U8.size_in_bytes(),
        };
        let residual_scales_bo = BufferOffset {
            buffer: residual_scales_metal.buffer(),
            offset_in_bytes: residual_scales_layout.start_offset()
                * residual_scales_metal.dtype().size_in_bytes(),
        };
        let out_bo = BufferOffset {
            buffer: out_metal.buffer(),
            offset_in_bytes: out_layout.start_offset() * DType::F32.size_in_bytes(),
        };
        let encoder_ref = &encoder;
        candle_metal_kernels::set_params!(
            encoder_ref,
            (
                bh_u32,
                full_heads_u32,
                kv_heads_u32,
                repeat_u32,
                t_u32,
                d_u32,
                sub_u32,
                scale_block_u32,
                code_bits_u32,
                signed_max,
                use_residual_signs,
                scale as f32,
                &q_bo,
                &codes_bo,
                &scales_bo,
                &pair_bo,
                &residual_bits_bo,
                &residual_scales_bo,
                &out_bo
            )
        );

        let (tg_count, tg_size) = if use_grouped_kernel {
            let threads =
                turboquant_grouped_threads(pipeline.max_total_threads_per_threadgroup(), 256);
            (
                MTLSize {
                    width: bh_full,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threads,
                    height: 1,
                    depth: 1,
                },
            )
        } else {
            let threads = pipeline
                .max_total_threads_per_threadgroup()
                .min(out_elems.max(1));
            let groups = out_elems.div_ceil(threads);
            (
                MTLSize {
                    width: groups,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threads,
                    height: 1,
                    depth: 1,
                },
            )
        };
        encoder.dispatch_thread_groups(tg_count, tg_size);
    }
    Ok(Some(out))
}

/// Computes q·k^T for experimental TurboQuant cold keys.
///
/// Shapes:
/// - `q`: `[batch, full_heads, head_dim]`
/// - `codes`: `[rows, head_dim]` with `rows = batch * kv_heads * tokens`
/// - `scales`: `[rows, num_subvectors]` in `F32` or `F16`
/// - `pair_signs`: `[head_dim / 2]`
/// - `residual_signs`: optional `[rows, head_dim]` in `U8` with values `0|1`
/// - `residual_scales`: optional `[rows, num_subvectors]` in `F32` or `F16`
///
/// Returns:
/// - `[batch, full_heads, tokens]` in `F32`.
pub fn qk_scores_turboquant(
    q: &Tensor,
    codes: &Tensor,
    scales: &Tensor,
    pair_signs: &Tensor,
    residual_signs: Option<&Tensor>,
    residual_scales: Option<&Tensor>,
    kv_heads: usize,
    repeat_factor: usize,
    tokens: usize,
    head_dim: usize,
    subvector_dim: usize,
    kind: TurboQuantKind,
    scale: f64,
) -> Result<Tensor> {
    if repeat_factor == 0 {
        candle_core::bail!("turboquant qk requires repeat_factor > 0")
    }
    if head_dim == 0 {
        candle_core::bail!("turboquant qk requires head_dim > 0")
    }
    if subvector_dim == 0 {
        candle_core::bail!("turboquant qk requires subvector_dim > 0")
    }
    if head_dim % 2 != 0 {
        candle_core::bail!(
            "turboquant qk requires even head_dim for pairwise rotation, got {}",
            head_dim
        )
    }
    if head_dim % subvector_dim != 0 {
        candle_core::bail!(
            "turboquant qk requires head_dim divisible by subvector_dim ({} % {} != 0)",
            head_dim,
            subvector_dim
        )
    }
    if residual_signs.is_some() ^ residual_scales.is_some() {
        candle_core::bail!("turboquant qk requires residual_signs and residual_scales together")
    }

    let q = q.contiguous()?;
    let codes = codes.contiguous()?;
    let scales = scales.contiguous()?;
    let pair_signs = pair_signs.contiguous()?;
    let residual_signs = match residual_signs {
        Some(t) => Some(t.contiguous()?),
        None => None,
    };
    let residual_scales = match residual_scales {
        Some(t) => Some(t.contiguous()?),
        None => None,
    };

    let qd = q.dims();
    if qd.len() != 3 {
        candle_core::bail!("turboquant qk expects q rank-3 [b,h,d], got {:?}", qd)
    }
    let batch = qd[0];
    let full_heads = qd[1];
    if qd[2] != head_dim {
        candle_core::bail!(
            "turboquant qk head_dim mismatch: q={} arg={}",
            qd[2],
            head_dim
        )
    }
    if full_heads % repeat_factor != 0 || full_heads / repeat_factor != kv_heads {
        candle_core::bail!(
            "turboquant qk invalid head config: full_heads={} kv_heads={} repeat_factor={}",
            full_heads,
            kv_heads,
            repeat_factor
        )
    }

    let rows = batch * kv_heads * tokens;
    let num_subvectors = head_dim / subvector_dim;
    let codes_dims = codes.dims();
    if codes_dims != [rows, head_dim] {
        candle_core::bail!(
            "turboquant qk codes shape mismatch: got {:?}, expected [{}, {}]",
            codes_dims,
            rows,
            head_dim
        )
    }
    if codes.dtype() != DType::U8 {
        candle_core::bail!("turboquant qk expects U8 codes, got {:?}", codes.dtype())
    }
    let scales_dims = scales.dims();
    if scales_dims != [rows, num_subvectors] {
        candle_core::bail!(
            "turboquant qk scales shape mismatch: got {:?}, expected [{}, {}]",
            scales_dims,
            rows,
            num_subvectors
        )
    }
    if !matches!(scales.dtype(), DType::F32 | DType::F16) {
        candle_core::bail!(
            "turboquant qk expects F32 or F16 scales, got {:?}",
            scales.dtype()
        )
    }
    let pair_dims = pair_signs.dims();
    if pair_dims != [head_dim / 2] {
        candle_core::bail!(
            "turboquant qk pair_signs shape mismatch: got {:?}, expected [{}]",
            pair_dims,
            head_dim / 2
        )
    }
    if pair_signs.dtype() != DType::F32 {
        candle_core::bail!(
            "turboquant qk expects F32 pair_signs, got {:?}",
            pair_signs.dtype()
        )
    }
    if !q.device().same_device(codes.device())
        || !q.device().same_device(scales.device())
        || !q.device().same_device(pair_signs.device())
    {
        candle_core::bail!(
            "turboquant qk device mismatch: q={:?}, codes={:?}, scales={:?}, pair_signs={:?}",
            q.device(),
            codes.device(),
            scales.device(),
            pair_signs.device()
        )
    }
    if let (Some(bits), Some(rscales)) = (&residual_signs, &residual_scales) {
        let bits_dims = bits.dims();
        if bits_dims != [rows, head_dim] {
            candle_core::bail!(
                "turboquant qk residual_signs shape mismatch: got {:?}, expected [{}, {}]",
                bits_dims,
                rows,
                head_dim
            )
        }
        if bits.dtype() != DType::U8 {
            candle_core::bail!(
                "turboquant qk expects U8 residual_signs, got {:?}",
                bits.dtype()
            )
        }
        let rscale_dims = rscales.dims();
        if rscale_dims != [rows, num_subvectors] {
            candle_core::bail!(
                "turboquant qk residual_scales shape mismatch: got {:?}, expected [{}, {}]",
                rscale_dims,
                rows,
                num_subvectors
            )
        }
        if !matches!(rscales.dtype(), DType::F32 | DType::F16) {
            candle_core::bail!(
                "turboquant qk expects F32 or F16 residual_scales, got {:?}",
                rscales.dtype()
            )
        }
        if rscales.dtype() != scales.dtype() {
            candle_core::bail!(
                "turboquant qk expects residual_scales dtype to match scales dtype (scales={:?}, residual_scales={:?})",
                scales.dtype(),
                rscales.dtype()
            )
        }
        if !q.device().same_device(bits.device()) || !q.device().same_device(rscales.device()) {
            candle_core::bail!(
                "turboquant qk device mismatch for residuals: q={:?}, residual_signs={:?}, residual_scales={:?}",
                q.device(),
                bits.device(),
                rscales.device()
            )
        }
    }

    #[cfg(feature = "metal")]
    if let Some(out) = try_qk_scores_turboquant_metal(
        &q,
        &codes,
        &scales,
        &pair_signs,
        residual_signs.as_ref(),
        residual_scales.as_ref(),
        kv_heads,
        repeat_factor,
        tokens,
        head_dim,
        subvector_dim,
        kind,
        scale,
    )? {
        return Ok(out);
    }

    cpu_qk_scores_turboquant(
        &q,
        &codes,
        &scales,
        &pair_signs,
        residual_signs.as_ref(),
        residual_scales.as_ref(),
        kv_heads,
        repeat_factor,
        tokens,
        head_dim,
        subvector_dim,
        kind,
        scale,
    )
}

/// Computes q·k^T for TurboQuant cold keys stored in packed-code form.
///
/// Shapes:
/// - `q`: `[batch, full_heads, head_dim]`
/// - `packed_codes`: `[packed_code_bytes]`
/// - `scales`: `[rows, num_scale_blocks]` in `F32` or `F16`
/// - `pair_signs`: `[head_dim / 2]`
/// - `packed_residual_signs`: optional `[packed_sign_bytes]`
/// - `residual_scales`: optional `[rows, num_scale_blocks]` in `F32` or `F16`
///
/// Where:
/// - `rows = batch * kv_heads * tokens`
/// - `packed_code_bytes = ceil(rows * head_dim * code_bits / 8)`
/// - `packed_sign_bytes = ceil(rows * head_dim / 8)`
///
/// Returns:
/// - `[batch, full_heads, tokens]` in `F32`.
pub fn qk_scores_turboquant_packed(
    q: &Tensor,
    packed_codes: &Tensor,
    scales: &Tensor,
    pair_signs: &Tensor,
    packed_residual_signs: Option<&Tensor>,
    residual_scales: Option<&Tensor>,
    kv_heads: usize,
    repeat_factor: usize,
    tokens: usize,
    head_dim: usize,
    subvector_dim: usize,
    scale_block_dim: usize,
    code_bits: usize,
    kind: TurboQuantKind,
    scale: f64,
) -> Result<Tensor> {
    if repeat_factor == 0 {
        candle_core::bail!("turboquant packed qk requires repeat_factor > 0")
    }
    if head_dim == 0 {
        candle_core::bail!("turboquant packed qk requires head_dim > 0")
    }
    if subvector_dim == 0 {
        candle_core::bail!("turboquant packed qk requires subvector_dim > 0")
    }
    if scale_block_dim == 0 {
        candle_core::bail!("turboquant packed qk requires scale_block_dim > 0")
    }
    if code_bits == 0 || code_bits > 8 {
        candle_core::bail!(
            "turboquant packed qk requires code_bits in 1..=8, got {}",
            code_bits
        )
    }
    if head_dim % 2 != 0 {
        candle_core::bail!(
            "turboquant packed qk requires even head_dim for pairwise rotation, got {}",
            head_dim
        )
    }
    if head_dim % subvector_dim != 0 {
        candle_core::bail!(
            "turboquant packed qk requires head_dim divisible by subvector_dim ({} % {} != 0)",
            head_dim,
            subvector_dim
        )
    }
    if head_dim % scale_block_dim != 0 {
        candle_core::bail!(
            "turboquant packed qk requires head_dim divisible by scale_block_dim ({} % {} != 0)",
            head_dim,
            scale_block_dim
        )
    }
    if scale_block_dim % subvector_dim != 0 {
        candle_core::bail!(
            "turboquant packed qk requires scale_block_dim divisible by subvector_dim ({} % {} != 0)",
            scale_block_dim,
            subvector_dim
        )
    }
    if packed_residual_signs.is_some() ^ residual_scales.is_some() {
        candle_core::bail!(
            "turboquant packed qk requires packed_residual_signs and residual_scales together"
        )
    }

    let q = q.contiguous()?;
    let packed_codes = packed_codes.contiguous()?;
    let scales = scales.contiguous()?;
    let pair_signs = pair_signs.contiguous()?;
    let packed_residual_signs = match packed_residual_signs {
        Some(t) => Some(t.contiguous()?),
        None => None,
    };
    let residual_scales = match residual_scales {
        Some(t) => Some(t.contiguous()?),
        None => None,
    };

    let qd = q.dims();
    if qd.len() != 3 {
        candle_core::bail!(
            "turboquant packed qk expects q rank-3 [b,h,d], got {:?}",
            qd
        )
    }
    let batch = qd[0];
    let full_heads = qd[1];
    if qd[2] != head_dim {
        candle_core::bail!(
            "turboquant packed qk head_dim mismatch: q={} arg={}",
            qd[2],
            head_dim
        )
    }
    if full_heads % repeat_factor != 0 || full_heads / repeat_factor != kv_heads {
        candle_core::bail!(
            "turboquant packed qk invalid head config: full_heads={} kv_heads={} repeat_factor={}",
            full_heads,
            kv_heads,
            repeat_factor
        )
    }

    let rows = batch * kv_heads * tokens;
    let num_scale_blocks = head_dim / scale_block_dim;
    let code_value_count = rows * head_dim;
    let packed_code_bytes = (code_value_count * code_bits).div_ceil(8);
    if packed_codes.elem_count() != packed_code_bytes {
        candle_core::bail!(
            "turboquant packed qk codes size mismatch: got {} bytes, expected {}",
            packed_codes.elem_count(),
            packed_code_bytes
        )
    }
    if packed_codes.dtype() != DType::U8 {
        candle_core::bail!(
            "turboquant packed qk expects U8 packed_codes, got {:?}",
            packed_codes.dtype()
        )
    }
    let scales_dims = scales.dims();
    if scales_dims != [rows, num_scale_blocks] {
        candle_core::bail!(
            "turboquant packed qk scales shape mismatch: got {:?}, expected [{}, {}]",
            scales_dims,
            rows,
            num_scale_blocks
        )
    }
    if !matches!(scales.dtype(), DType::F32 | DType::F16) {
        candle_core::bail!(
            "turboquant packed qk expects F32 or F16 scales, got {:?}",
            scales.dtype()
        )
    }
    let pair_dims = pair_signs.dims();
    if pair_dims != [head_dim / 2] {
        candle_core::bail!(
            "turboquant packed qk pair_signs shape mismatch: got {:?}, expected [{}]",
            pair_dims,
            head_dim / 2
        )
    }
    if pair_signs.dtype() != DType::F32 {
        candle_core::bail!(
            "turboquant packed qk expects F32 pair_signs, got {:?}",
            pair_signs.dtype()
        )
    }
    if !q.device().same_device(packed_codes.device())
        || !q.device().same_device(scales.device())
        || !q.device().same_device(pair_signs.device())
    {
        candle_core::bail!(
            "turboquant packed qk device mismatch: q={:?}, packed_codes={:?}, scales={:?}, pair_signs={:?}",
            q.device(),
            packed_codes.device(),
            scales.device(),
            pair_signs.device()
        )
    }
    if let (Some(bits), Some(rscales)) = (&packed_residual_signs, &residual_scales) {
        let packed_sign_bytes = code_value_count.div_ceil(8);
        if bits.elem_count() != packed_sign_bytes {
            candle_core::bail!(
                "turboquant packed qk residual_signs size mismatch: got {} bytes, expected {}",
                bits.elem_count(),
                packed_sign_bytes
            )
        }
        if bits.dtype() != DType::U8 {
            candle_core::bail!(
                "turboquant packed qk expects U8 packed_residual_signs, got {:?}",
                bits.dtype()
            )
        }
        let rscale_dims = rscales.dims();
        if rscale_dims != [rows, num_scale_blocks] {
            candle_core::bail!(
                "turboquant packed qk residual_scales shape mismatch: got {:?}, expected [{}, {}]",
                rscale_dims,
                rows,
                num_scale_blocks
            )
        }
        if !matches!(rscales.dtype(), DType::F32 | DType::F16) {
            candle_core::bail!(
                "turboquant packed qk expects F32 or F16 residual_scales, got {:?}",
                rscales.dtype()
            )
        }
        if rscales.dtype() != scales.dtype() {
            candle_core::bail!(
                "turboquant packed qk expects residual_scales dtype to match scales dtype (scales={:?}, residual_scales={:?})",
                scales.dtype(),
                rscales.dtype()
            )
        }
        if !q.device().same_device(bits.device()) || !q.device().same_device(rscales.device()) {
            candle_core::bail!(
                "turboquant packed qk device mismatch for residuals: q={:?}, packed_residual_signs={:?}, residual_scales={:?}",
                q.device(),
                bits.device(),
                rscales.device()
            )
        }
    }

    #[cfg(feature = "metal")]
    if let Some(out) = try_qk_scores_turboquant_packed_metal(
        &q,
        &packed_codes,
        &scales,
        &pair_signs,
        packed_residual_signs.as_ref(),
        residual_scales.as_ref(),
        kv_heads,
        repeat_factor,
        tokens,
        head_dim,
        subvector_dim,
        scale_block_dim,
        code_bits,
        kind,
        scale,
    )? {
        return Ok(out);
    }

    cpu_qk_scores_turboquant_packed(
        &q,
        &packed_codes,
        &scales,
        &pair_signs,
        packed_residual_signs.as_ref(),
        residual_scales.as_ref(),
        kv_heads,
        repeat_factor,
        tokens,
        head_dim,
        subvector_dim,
        scale_block_dim,
        code_bits,
        kind,
        scale,
    )
}

#[cfg(test)]
mod tests {
    use super::{
        attn_weighted_sum, qk_scores, qk_scores_turboquant, qk_scores_turboquant_packed,
        TurboQuantKind,
    };
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
        // b=1,h=1,d=4
        let q = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], device)?
            .reshape((1, 1, 4))?
            .to_dtype(dtype)?;
        // t=3
        let k = Tensor::new(
            &[
                1.0f32, 0.0, 0.0, 0.0, // dot=1
                0.0, 1.0, 1.0, 0.0, // dot=5
                1.0, 1.0, 1.0, 1.0, // dot=10
            ],
            device,
        )?
        .reshape((1, 1, 3, 4))?
        .to_dtype(dtype)?;
        let out = qk_scores(&q, &k, 0.5)?;
        let out = out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let expected = [0.5f32, 2.5, 5.0];
        assert_close(&out, &expected, 1e-3, 1e-3);
        Ok(())
    }

    fn run_weighted_sum_case(device: &Device, dtype: DType) -> Result<()> {
        // b=1,h=1,t=3
        let attn = Tensor::new(&[0.2f32, 0.3, 0.5], device)?.reshape((1, 1, 3))?;
        // v: t=3,d=2
        let v = Tensor::new(
            &[
                1.0f32, 2.0, //
                3.0, 4.0, //
                5.0, 6.0, //
            ],
            device,
        )?
        .reshape((1, 1, 3, 2))?
        .to_dtype(dtype)?;
        let out = attn_weighted_sum(&attn, &v)?;
        let out = out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        // 0.2*[1,2] + 0.3*[3,4] + 0.5*[5,6] = [3.6, 4.6]
        let expected = [3.6f32, 4.6];
        assert_close(&out, &expected, 1e-3, 1e-3);
        Ok(())
    }

    #[test]
    fn qk_scores_cpu_all_dtypes() -> Result<()> {
        let device = Device::Cpu;
        run_case(&device, DType::F32)?;
        run_case(&device, DType::F16)?;
        run_case(&device, DType::BF16)?;
        run_weighted_sum_case(&device, DType::F32)?;
        run_weighted_sum_case(&device, DType::F16)?;
        run_weighted_sum_case(&device, DType::BF16)?;
        Ok(())
    }

    fn turbo_case_tensors(
        device: &Device,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let q = Tensor::new(
            &[
                1.0f32, 2.0, 3.0, 4.0, // head 0
                4.0, 3.0, 2.0, 1.0, // head 1
            ],
            device,
        )?
        .reshape((1, 2, 4))?;
        let codes = Tensor::new(
            &[
                6u8, 3, 5, 2, // row 0 => [3,0,2,-1]
                1u8, 4, 0, 6, // row 1 => [-2,1,-3,3]
            ],
            device,
        )?
        .reshape((2, 4))?;
        let scales = Tensor::new(&[0.5f32, 0.25, 0.75, 0.4], device)?.reshape((2, 2))?;
        let pair_signs = Tensor::new(&[1.0f32, -1.0], device)?;
        let residual_signs = Tensor::new(
            &[
                1u8, 0, 1, 0, // row 0
                0u8, 1, 0, 1, // row 1
            ],
            device,
        )?
        .reshape((2, 4))?;
        let residual_scales = Tensor::new(&[0.1f32, 0.05, 0.2, 0.07], device)?.reshape((2, 2))?;
        Ok((
            q,
            codes,
            scales,
            pair_signs,
            residual_signs,
            residual_scales,
        ))
    }

    fn pack_values(values: &[u8], bits_per_value: usize) -> Vec<u8> {
        let total_bits = values.len() * bits_per_value;
        let mut out = vec![0u8; total_bits.div_ceil(8)];
        let mask = (1u16 << bits_per_value) - 1;
        let mut bit_cursor = 0usize;
        for &value in values {
            let byte_idx = bit_cursor / 8;
            let bit_offset = bit_cursor % 8;
            let val = u16::from(value) & mask;
            out[byte_idx] |= (val << bit_offset) as u8;
            if bit_offset + bits_per_value > 8 {
                out[byte_idx + 1] |= (val >> (8 - bit_offset)) as u8;
            }
            bit_cursor += bits_per_value;
        }
        out
    }

    #[test]
    fn qk_scores_turboquant_cpu_matches_reference() -> Result<()> {
        let device = Device::Cpu;
        let (q, codes, scales, pair_signs, residual_signs, residual_scales) =
            turbo_case_tensors(&device)?;
        let out = qk_scores_turboquant(
            &q,
            &codes,
            &scales,
            &pair_signs,
            Some(&residual_signs),
            Some(&residual_scales),
            1,
            2,
            2,
            4,
            2,
            TurboQuantKind::Turbo3,
            0.5,
        )?;
        let out = out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let q_vec = q.flatten_all()?.to_vec1::<f32>()?;
        let codes_vec = codes.flatten_all()?.to_vec1::<u8>()?;
        let scales_vec = scales.flatten_all()?.to_vec1::<f32>()?;
        let pair_vec = pair_signs.flatten_all()?.to_vec1::<f32>()?;
        let residual_bits_vec = residual_signs.flatten_all()?.to_vec1::<u8>()?;
        let residual_scales_vec = residual_scales.flatten_all()?.to_vec1::<f32>()?;
        let expected = super::cpu_qk_scores_turboquant_impl(
            &q_vec,
            |v| v,
            &codes_vec,
            &scales_vec,
            &pair_vec,
            Some(&residual_bits_vec),
            Some(&residual_scales_vec),
            1,
            2,
            1,
            2,
            2,
            4,
            2,
            2,
            TurboQuantKind::Turbo3.signed_max(),
            0.5,
        );
        assert_close(&out, &expected, 1e-5, 1e-5);
        Ok(())
    }

    #[test]
    fn qk_scores_turboquant_packed_cpu_matches_reference() -> Result<()> {
        let device = Device::Cpu;
        let (q, codes, scales, pair_signs, residual_signs, residual_scales) =
            turbo_case_tensors(&device)?;
        let packed_codes = Tensor::from_vec(
            pack_values(&codes.flatten_all()?.to_vec1::<u8>()?, 3),
            (3usize,),
            &device,
        )?;
        let packed_residual_signs = Tensor::from_vec(
            pack_values(&residual_signs.flatten_all()?.to_vec1::<u8>()?, 1),
            (1usize,),
            &device,
        )?;
        let out = qk_scores_turboquant_packed(
            &q,
            &packed_codes,
            &scales,
            &pair_signs,
            Some(&packed_residual_signs),
            Some(&residual_scales),
            1,
            2,
            2,
            4,
            2,
            2,
            3,
            TurboQuantKind::Turbo3,
            0.5,
        )?;
        let out = out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let q_vec = q.flatten_all()?.to_vec1::<f32>()?;
        let codes_vec = codes.flatten_all()?.to_vec1::<u8>()?;
        let scales_vec = scales.flatten_all()?.to_vec1::<f32>()?;
        let pair_vec = pair_signs.flatten_all()?.to_vec1::<f32>()?;
        let residual_bits_vec = residual_signs.flatten_all()?.to_vec1::<u8>()?;
        let residual_scales_vec = residual_scales.flatten_all()?.to_vec1::<f32>()?;
        let expected = super::cpu_qk_scores_turboquant_impl(
            &q_vec,
            |v| v,
            &codes_vec,
            &scales_vec,
            &pair_vec,
            Some(&residual_bits_vec),
            Some(&residual_scales_vec),
            1,
            2,
            1,
            2,
            2,
            4,
            2,
            2,
            TurboQuantKind::Turbo3.signed_max(),
            0.5,
        );
        assert_close(&out, &expected, 1e-5, 1e-5);
        Ok(())
    }

    #[test]
    fn qk_scores_turboquant_cpu_accepts_f16_scales() -> Result<()> {
        let device = Device::Cpu;
        let (q, codes, scales, pair_signs, residual_signs, residual_scales) =
            turbo_case_tensors(&device)?;
        let expected = qk_scores_turboquant(
            &q,
            &codes,
            &scales,
            &pair_signs,
            Some(&residual_signs),
            Some(&residual_scales),
            1,
            2,
            2,
            4,
            2,
            TurboQuantKind::Turbo3,
            0.5,
        )?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
        let got = qk_scores_turboquant(
            &q,
            &codes,
            &scales.to_dtype(DType::F16)?,
            &pair_signs,
            Some(&residual_signs),
            Some(&residual_scales.to_dtype(DType::F16)?),
            1,
            2,
            2,
            4,
            2,
            TurboQuantKind::Turbo3,
            0.5,
        )?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
        assert_close(&got, &expected, 1e-3, 1e-3);
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn qk_scores_turboquant_metal_matches_cpu() -> Result<()> {
        let metal = match Device::metal_if_available(0) {
            Ok(d) if !d.is_cpu() => d,
            _ => return Ok(()),
        };
        let cpu = Device::Cpu;
        let (q_cpu, codes_cpu, scales_cpu, pair_cpu, residual_bits_cpu, residual_scales_cpu) =
            turbo_case_tensors(&cpu)?;
        let expected = qk_scores_turboquant(
            &q_cpu,
            &codes_cpu,
            &scales_cpu,
            &pair_cpu,
            Some(&residual_bits_cpu),
            Some(&residual_scales_cpu),
            1,
            2,
            2,
            4,
            2,
            TurboQuantKind::Turbo3,
            0.5,
        )?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
        let (q, codes, scales, pair_signs, residual_signs, residual_scales) =
            turbo_case_tensors(&metal)?;
        let got = qk_scores_turboquant(
            &q.to_dtype(DType::F16)?,
            &codes,
            &scales.to_dtype(DType::F16)?,
            &pair_signs,
            Some(&residual_signs),
            Some(&residual_scales.to_dtype(DType::F16)?),
            1,
            2,
            2,
            4,
            2,
            TurboQuantKind::Turbo3,
            0.5,
        )?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
        assert_close(&got, &expected, 1e-3, 1e-3);
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn qk_scores_turboquant_packed_metal_matches_cpu() -> Result<()> {
        let metal = match Device::metal_if_available(0) {
            Ok(d) if !d.is_cpu() => d,
            _ => return Ok(()),
        };
        let cpu = Device::Cpu;
        let (q_cpu, codes_cpu, scales_cpu, pair_cpu, residual_bits_cpu, residual_scales_cpu) =
            turbo_case_tensors(&cpu)?;
        let packed_codes_cpu = Tensor::from_vec(
            pack_values(&codes_cpu.flatten_all()?.to_vec1::<u8>()?, 3),
            (3usize,),
            &cpu,
        )?;
        let packed_residuals_cpu = Tensor::from_vec(
            pack_values(&residual_bits_cpu.flatten_all()?.to_vec1::<u8>()?, 1),
            (1usize,),
            &cpu,
        )?;
        let expected = qk_scores_turboquant_packed(
            &q_cpu,
            &packed_codes_cpu,
            &scales_cpu,
            &pair_cpu,
            Some(&packed_residuals_cpu),
            Some(&residual_scales_cpu),
            1,
            2,
            2,
            4,
            2,
            2,
            3,
            TurboQuantKind::Turbo3,
            0.5,
        )?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
        let (q, codes, scales, pair_signs, residual_signs, residual_scales) =
            turbo_case_tensors(&metal)?;
        let packed_codes = Tensor::from_vec(
            pack_values(&codes.flatten_all()?.to_vec1::<u8>()?, 3),
            (3usize,),
            &metal,
        )?;
        let packed_residuals = Tensor::from_vec(
            pack_values(&residual_signs.flatten_all()?.to_vec1::<u8>()?, 1),
            (1usize,),
            &metal,
        )?;
        let got = qk_scores_turboquant_packed(
            &q.to_dtype(DType::F16)?,
            &packed_codes,
            &scales.to_dtype(DType::F16)?,
            &pair_signs,
            Some(&packed_residuals),
            Some(&residual_scales.to_dtype(DType::F16)?),
            1,
            2,
            2,
            4,
            2,
            2,
            3,
            TurboQuantKind::Turbo3,
            0.5,
        )?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
        assert_close(&got, &expected, 1e-3, 1e-3);
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn qk_scores_metal_f32_f16() -> Result<()> {
        let device = match Device::metal_if_available(0) {
            Ok(d) if !d.is_cpu() => d,
            _ => return Ok(()),
        };
        run_case(&device, DType::F32)?;
        run_case(&device, DType::F16)?;
        run_weighted_sum_case(&device, DType::F32)?;
        run_weighted_sum_case(&device, DType::F16)?;
        Ok(())
    }
}
