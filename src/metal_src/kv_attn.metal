#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void qk_scores_kernel(
    constant uint& bh_count [[buffer(0)]],
    constant uint& t [[buffer(1)]],
    constant uint& d [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    const device T* q [[buffer(4)]],
    const device T* k [[buffer(5)]],
    device float* out [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint n = bh_count * t;
    if (gid >= n) {
        return;
    }

    uint bh_idx = gid / t;
    uint tok = gid % t;
    uint q_base = bh_idx * d;
    uint k_base = (bh_idx * t + tok) * d;

    float acc = 0.0f;
    for (uint i = 0; i < d; ++i) {
        acc += float(q[q_base + i]) * float(k[k_base + i]);
    }
    out[gid] = acc * scale;
}

template [[host_name("qk_scores_f32")]] [[kernel]] decltype(qk_scores_kernel<float>) qk_scores_kernel<float>;
template [[host_name("qk_scores_f16")]] [[kernel]] decltype(qk_scores_kernel<half>) qk_scores_kernel<half>;
template [[host_name("qk_scores_bf16")]] [[kernel]] decltype(qk_scores_kernel<bfloat>) qk_scores_kernel<bfloat>;

template <typename T>
[[kernel]] void attn_weighted_sum_kernel(
    constant uint& bh_count [[buffer(0)]],
    constant uint& t [[buffer(1)]],
    constant uint& d [[buffer(2)]],
    const device float* attn [[buffer(3)]],
    const device T* v [[buffer(4)]],
    device float* out [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint n = bh_count * d;
    if (gid >= n) {
        return;
    }

    uint bh_idx = gid / d;
    uint dim = gid % d;
    uint attn_base = bh_idx * t;

    float acc = 0.0f;
    for (uint tok = 0; tok < t; ++tok) {
        uint v_idx = (bh_idx * t + tok) * d + dim;
        acc += attn[attn_base + tok] * float(v[v_idx]);
    }
    out[gid] = acc;
}

template [[host_name("attn_weighted_sum_v_f32")]] [[kernel]] decltype(attn_weighted_sum_kernel<float>) attn_weighted_sum_kernel<float>;
template [[host_name("attn_weighted_sum_v_f16")]] [[kernel]] decltype(attn_weighted_sum_kernel<half>) attn_weighted_sum_kernel<half>;
template [[host_name("attn_weighted_sum_v_bf16")]] [[kernel]] decltype(attn_weighted_sum_kernel<bfloat>) attn_weighted_sum_kernel<bfloat>;

template <typename T, bool IS_Q4>
[[kernel]] void qk_scores_rowwise_quant_kernel(
    constant uint& bh_full [[buffer(0)]],
    constant uint& full_heads [[buffer(1)]],
    constant uint& kv_heads [[buffer(2)]],
    constant uint& repeat_factor [[buffer(3)]],
    constant uint& t [[buffer(4)]],
    constant uint& d [[buffer(5)]],
    constant float& attn_scale [[buffer(6)]],
    const device T* q [[buffer(7)]],
    const device uchar* kq [[buffer(8)]],
    const device float* k_scales [[buffer(9)]],
    device float* out [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    uint n = bh_full * t;
    if (gid >= n) {
        return;
    }

    uint bh_idx = gid / t;
    uint tok = gid % t;
    uint b = bh_idx / full_heads;
    uint h = bh_idx % full_heads;
    uint kv_h = h / repeat_factor;
    uint row = (b * kv_heads + kv_h) * t + tok;
    float row_scale = k_scales[row];

    uint q_base = bh_idx * d;
    float acc = 0.0f;
    if constexpr (!IS_Q4) {
        uint k_base = row * d;
        for (uint i = 0; i < d; ++i) {
            int qv = int(kq[k_base + i]) - 128;
            acc += float(q[q_base + i]) * (float(qv) * row_scale);
        }
    } else {
        uint packed_cols = (d + 1) / 2;
        uint k_base = row * packed_cols;
        for (uint i = 0; i < d; ++i) {
            uchar byte = kq[k_base + (i >> 1)];
            uchar nib = ((i & 1) == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
            int qv = (nib >= 8) ? (int(nib) - 16) : int(nib);
            acc += float(q[q_base + i]) * (float(qv) * row_scale);
        }
    }
    out[gid] = acc * attn_scale;
}

template [[host_name("qk_scores_rowwise_q8_f32")]] [[kernel]] decltype(qk_scores_rowwise_quant_kernel<float, false>) qk_scores_rowwise_quant_kernel<float, false>;
template [[host_name("qk_scores_rowwise_q8_f16")]] [[kernel]] decltype(qk_scores_rowwise_quant_kernel<half, false>) qk_scores_rowwise_quant_kernel<half, false>;
template [[host_name("qk_scores_rowwise_q8_bf16")]] [[kernel]] decltype(qk_scores_rowwise_quant_kernel<bfloat, false>) qk_scores_rowwise_quant_kernel<bfloat, false>;
template [[host_name("qk_scores_rowwise_q4_f32")]] [[kernel]] decltype(qk_scores_rowwise_quant_kernel<float, true>) qk_scores_rowwise_quant_kernel<float, true>;
template [[host_name("qk_scores_rowwise_q4_f16")]] [[kernel]] decltype(qk_scores_rowwise_quant_kernel<half, true>) qk_scores_rowwise_quant_kernel<half, true>;
template [[host_name("qk_scores_rowwise_q4_bf16")]] [[kernel]] decltype(qk_scores_rowwise_quant_kernel<bfloat, true>) qk_scores_rowwise_quant_kernel<bfloat, true>;

template <bool IS_Q4>
[[kernel]] void attn_weighted_sum_rowwise_quant_kernel(
    constant uint& bh_full [[buffer(0)]],
    constant uint& full_heads [[buffer(1)]],
    constant uint& kv_heads [[buffer(2)]],
    constant uint& repeat_factor [[buffer(3)]],
    constant uint& t [[buffer(4)]],
    constant uint& d [[buffer(5)]],
    const device float* attn [[buffer(6)]],
    const device uchar* vq [[buffer(7)]],
    const device float* v_scales [[buffer(8)]],
    device float* out [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    uint n = bh_full * d;
    if (gid >= n) {
        return;
    }

    uint bh_idx = gid / d;
    uint dim = gid % d;
    uint b = bh_idx / full_heads;
    uint h = bh_idx % full_heads;
    uint kv_h = h / repeat_factor;
    uint attn_base = bh_idx * t;

    float acc = 0.0f;
    if constexpr (!IS_Q4) {
        for (uint tok = 0; tok < t; ++tok) {
            uint row = (b * kv_heads + kv_h) * t + tok;
            uint v_idx = row * d + dim;
            int qv = int(vq[v_idx]) - 128;
            acc += attn[attn_base + tok] * (float(qv) * v_scales[row]);
        }
    } else {
        uint packed_cols = (d + 1) / 2;
        for (uint tok = 0; tok < t; ++tok) {
            uint row = (b * kv_heads + kv_h) * t + tok;
            uint byte_idx = row * packed_cols + (dim >> 1);
            uchar byte = vq[byte_idx];
            uchar nib = ((dim & 1) == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
            int qv = (nib >= 8) ? (int(nib) - 16) : int(nib);
            acc += attn[attn_base + tok] * (float(qv) * v_scales[row]);
        }
    }
    out[gid] = acc;
}

template [[host_name("attn_weighted_sum_rowwise_q8")]] [[kernel]] decltype(attn_weighted_sum_rowwise_quant_kernel<false>) attn_weighted_sum_rowwise_quant_kernel<false>;
template [[host_name("attn_weighted_sum_rowwise_q4")]] [[kernel]] decltype(attn_weighted_sum_rowwise_quant_kernel<true>) attn_weighted_sum_rowwise_quant_kernel<true>;
