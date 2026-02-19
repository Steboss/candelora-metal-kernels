#[cfg(feature = "metal")]
use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp1, CustomOp2, CustomOp3, DType, Layout, Module, Result, Shape, Tensor};
use candle_core::quantized::{QMatMul, QTensor};
use half::{bf16, f16};
#[cfg(feature = "metal")]
use std::collections::HashMap;
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
            .map_err(|e| candle_core::Error::msg(format!("failed loading kv-attn metal function: {e}")))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| candle_core::Error::msg(format!("failed creating kv-attn metal pipeline: {e}")))?;
        dev_cache.pipelines.insert(kernel_name, pipeline.clone());
        return Ok(pipeline);
    }

    let options = objc2_metal::MTLCompileOptions::new();
    let library = device
        .new_library_with_source(KV_ATTN_METAL, Some(&options))
        .map_err(|e| candle_core::Error::msg(format!("failed compiling kv-attn metal source: {e}")))?;
    let func = library
        .get_function(kernel_name, None)
        .map_err(|e| candle_core::Error::msg(format!("failed loading kv-attn metal function: {e}")))?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| candle_core::Error::msg(format!("failed creating kv-attn metal pipeline: {e}")))?;

    let mut pipelines = HashMap::new();
    pipelines.insert(kernel_name, pipeline.clone());
    cache.insert(device_id, KvAttnMetalDeviceCache { library, pipelines });
    Ok(pipeline)
}

fn contiguous_slice<'a, T>(values: &'a [T], layout: &Layout, name: &'static str) -> Result<&'a [T]> {
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
        candle_core::bail!(
            "q/k shape mismatch: q=[{b},{h},{d}], k=[{bk},{hk},{t},{dk}]"
        )
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
        candle_core::bail!(
            "attn/v shape mismatch: attn=[{b},{h},{t}], v=[{bv},{hv},{tv},{d}]"
        )
    }
    Ok((b, h, t, d))
}

fn cpu_qk_scores_f32(q: &[f32], k: &[f32], b: usize, h: usize, t: usize, d: usize, scale: f32) -> Vec<f32> {
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

fn cpu_qk_scores_f16(q: &[f16], k: &[f16], b: usize, h: usize, t: usize, d: usize, scale: f32) -> Vec<f32> {
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

fn cpu_attn_weighted_sum_f32(attn: &[f32], v: &[f32], b: usize, h: usize, t: usize, d: usize) -> Vec<f32> {
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

fn cpu_attn_weighted_sum_f16(attn: &[f32], v: &[f16], b: usize, h: usize, t: usize, d: usize) -> Vec<f32> {
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

fn dequant_row_elem(data: &[u8], row: usize, col: usize, head_dim: usize, kind: RowwiseQuantKind) -> f32 {
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

fn cpu_pack_scales_rowwise_f32(x: &[f32], rows: usize, head_dim: usize, kind: RowwiseQuantKind) -> Vec<f32> {
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

fn cpu_pack_scales_rowwise_f16(x: &[f16], rows: usize, head_dim: usize, kind: RowwiseQuantKind) -> Vec<f32> {
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
            let out = s1.device().new_buffer(0, DType::F32, "kv-attn-pack-scales-out")?;
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

        let rows_u32 = u32::try_from(rows).map_err(|_| candle_core::Error::msg("rows too large"))?;
        let d_u32 =
            u32::try_from(self.head_dim).map_err(|_| candle_core::Error::msg("head_dim too large"))?;
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

        let threads = pipeline.max_total_threads_per_threadgroup().min(rows.max(1));
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
            CpuStorage::F32(x) => {
                cpu_pack_data_rowwise_f32(contiguous_slice(x, l1, "x")?, scales, rows, self.head_dim, self.kind)
            }
            CpuStorage::F16(x) => {
                cpu_pack_data_rowwise_f16(contiguous_slice(x, l1, "x")?, scales, rows, self.head_dim, self.kind)
            }
            CpuStorage::BF16(x) => {
                cpu_pack_data_rowwise_bf16(contiguous_slice(x, l1, "x")?, scales, rows, self.head_dim, self.kind)
            }
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
            let out = s1.device().new_buffer(0, DType::U8, "kv-attn-pack-data-out")?;
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
        let rows_u32 = u32::try_from(rows).map_err(|_| candle_core::Error::msg("rows too large"))?;
        let d_u32 =
            u32::try_from(self.head_dim).map_err(|_| candle_core::Error::msg("head_dim too large"))?;
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

        let threads = pipeline.max_total_threads_per_threadgroup().min(rows.max(1));
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
                Ok((CpuStorage::F32(cpu_qk_scores_f32(q, k, b, h, t, d, self.scale)), out_shape))
            }
            (CpuStorage::F16(q), CpuStorage::F16(k)) => {
                let q = contiguous_slice(q, l1, "q")?;
                let k = contiguous_slice(k, l2, "k")?;
                Ok((CpuStorage::F32(cpu_qk_scores_f16(q, k, b, h, t, d, self.scale)), out_shape))
            }
            (CpuStorage::BF16(q), CpuStorage::BF16(k)) => {
                let q = contiguous_slice(q, l1, "q")?;
                let k = contiguous_slice(k, l2, "k")?;
                Ok((CpuStorage::F32(cpu_qk_scores_bf16(q, k, b, h, t, d, self.scale)), out_shape))
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
            candle_core::bail!("dtype mismatch in fused kv-attn qk op: q={:?}, k={:?}", dtype, s2.dtype());
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
        candle_metal_kernels::set_params!(encoder_ref, (bh_u32, t_u32, d_u32, self.scale, &q, &k, &out));

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
        let scales = match s3 {
            CpuStorage::F32(v) => contiguous_slice(v, l3, "k_scales")?,
            _ => candle_core::bail!("rowwise qk expects F32 scales"),
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
        if full_heads % self.repeat_factor != 0 || full_heads / self.repeat_factor != self.kv_heads {
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
        if s3.dtype() != DType::F32 {
            candle_core::bail!("rowwise qk expects F32 scales, got {:?}", s3.dtype())
        }

        let kernel_name = match (self.kind, s1.dtype()) {
            (RowwiseQuantKind::Int8, DType::F32) => "qk_scores_rowwise_q8_f32",
            (RowwiseQuantKind::Int8, DType::F16) => "qk_scores_rowwise_q8_f16",
            (RowwiseQuantKind::Int8, DType::BF16) => "qk_scores_rowwise_q8_bf16",
            (RowwiseQuantKind::Int4, DType::F32) => "qk_scores_rowwise_q4_f32",
            (RowwiseQuantKind::Int4, DType::F16) => "qk_scores_rowwise_q4_f16",
            (RowwiseQuantKind::Int4, DType::BF16) => "qk_scores_rowwise_q4_bf16",
            (_, dt) => candle_core::bail!("unsupported q dtype for rowwise qk op: {:?}", dt),
        };

        let bh_full = b * full_heads;
        let out_elems = bh_full * self.tokens;
        let out_shape = Shape::from((b, full_heads, self.tokens));
        if out_elems == 0 {
            let out = s1.device().new_buffer(0, DType::F32, "kv-attn-rowwise-qk-out")?;
            return Ok((
                candle_core::MetalStorage::new(out, s1.device().clone(), 0, DType::F32),
                out_shape,
            ));
        }

        let bh_u32 = u32::try_from(bh_full).map_err(|_| candle_core::Error::msg("bh_full too large"))?;
        let full_heads_u32 =
            u32::try_from(full_heads).map_err(|_| candle_core::Error::msg("full_heads too large"))?;
        let kv_heads_u32 =
            u32::try_from(self.kv_heads).map_err(|_| candle_core::Error::msg("kv_heads too large"))?;
        let repeat_u32 = u32::try_from(self.repeat_factor)
            .map_err(|_| candle_core::Error::msg("repeat_factor too large"))?;
        let t_u32 = u32::try_from(self.tokens).map_err(|_| candle_core::Error::msg("tokens too large"))?;
        let d_u32 = u32::try_from(self.head_dim).map_err(|_| candle_core::Error::msg("head_dim too large"))?;

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
            candle_core::bail!("rowwise weighted-sum expects attn rank-3 [b,h,t], got {:?}", ad)
        }
        let (b, full_heads, t) = (ad[0], ad[1], ad[2]);
        if t != self.tokens {
            candle_core::bail!("rowwise weighted-sum token mismatch: attn={} op={}", t, self.tokens)
        }
        if full_heads % self.repeat_factor != 0 || full_heads / self.repeat_factor != self.kv_heads {
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
        let scales = match s3 {
            CpuStorage::F32(v) => contiguous_slice(v, l3, "v_scales")?,
            _ => candle_core::bail!("rowwise weighted-sum expects F32 scales"),
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
        Ok((CpuStorage::F32(out), Shape::from((b, full_heads, self.head_dim))))
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
            candle_core::bail!("rowwise weighted-sum expects F32 attn probs, got {:?}", s1.dtype())
        }
        if s2.dtype() != DType::U8 {
            candle_core::bail!("rowwise weighted-sum expects U8 packed data, got {:?}", s2.dtype())
        }
        if s3.dtype() != DType::F32 {
            candle_core::bail!("rowwise weighted-sum expects F32 scales, got {:?}", s3.dtype())
        }

        let ad = l1.shape().dims();
        if ad.len() != 3 {
            candle_core::bail!("rowwise weighted-sum expects attn rank-3 [b,h,t], got {:?}", ad)
        }
        let (b, full_heads, t) = (ad[0], ad[1], ad[2]);
        if t != self.tokens {
            candle_core::bail!("rowwise weighted-sum token mismatch: attn={} op={}", t, self.tokens)
        }
        if full_heads % self.repeat_factor != 0 || full_heads / self.repeat_factor != self.kv_heads {
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

        let kernel_name = match self.kind {
            RowwiseQuantKind::Int8 => "attn_weighted_sum_rowwise_q8",
            RowwiseQuantKind::Int4 => "attn_weighted_sum_rowwise_q4",
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

        let bh_u32 = u32::try_from(bh_full).map_err(|_| candle_core::Error::msg("bh_full too large"))?;
        let full_heads_u32 =
            u32::try_from(full_heads).map_err(|_| candle_core::Error::msg("full_heads too large"))?;
        let kv_heads_u32 =
            u32::try_from(self.kv_heads).map_err(|_| candle_core::Error::msg("kv_heads too large"))?;
        let repeat_u32 = u32::try_from(self.repeat_factor)
            .map_err(|_| candle_core::Error::msg("repeat_factor too large"))?;
        let t_u32 = u32::try_from(self.tokens).map_err(|_| candle_core::Error::msg("tokens too large"))?;
        let d_u32 = u32::try_from(self.head_dim).map_err(|_| candle_core::Error::msg("head_dim too large"))?;

        let metal = s1.device().metal_device();
        let pipeline = get_or_create_kv_attn_pipeline(s1.device().id(), metal, kernel_name)?;
        let output = s1
            .device()
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
        candle_core::bail!("device mismatch in fused kv-attn qk op: q={:?}, k={:?}", q.device(), k.device());
    }
    if q.dtype() != k.dtype() {
        candle_core::bail!("dtype mismatch in fused kv-attn qk op: q={:?}, k={:?}", q.dtype(), k.dtype());
    }
    if q.rank() != 3 || k.rank() != 4 {
        candle_core::bail!(
            "fused kv-attn qk op expects q rank-3 and k rank-4, got q={:?}, k={:?}",
            q.dims(),
            k.dims()
        );
    }
    q.apply_op2_no_bwd(k, &QkScoresOp {
        scale: scale as f32,
    })
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

        let (x_metal, packed_metal, scales_metal) = match (&*x_storage, &*packed_storage, &*scales_storage) {
            (candle_core::Storage::Metal(xm), candle_core::Storage::Metal(pm), candle_core::Storage::Metal(sm)) => (xm, pm, sm),
            _ => return Ok(None),
        };
        if !x_layout.is_contiguous() || !packed_layout.is_contiguous() || !scales_layout.is_contiguous() {
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
        let rows_u32 = u32::try_from(rows).map_err(|_| candle_core::Error::msg("rows too large"))?;
        let d_u32 = u32::try_from(head_dim).map_err(|_| candle_core::Error::msg("head_dim too large"))?;
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
        candle_metal_kernels::set_params!(encoder_ref, (rows_u32, d_u32, &x_bo, &packed_bo, &scales_bo));

        let threads = pipeline.max_total_threads_per_threadgroup().min(rows.max(1));
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
        candle_core::bail!("pack_rowwise_quantized expects rank >= 1 tensor, got {:?}", x.dims())
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

#[cfg(test)]
mod tests {
    use super::{attn_weighted_sum, qk_scores};
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
