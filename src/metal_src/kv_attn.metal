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



template <typename T>
[[kernel]] void qk_scores_turboquant_kernel(
    constant uint& bh_full [[buffer(0)]],
    constant uint& full_heads [[buffer(1)]],
    constant uint& kv_heads [[buffer(2)]],
    constant uint& repeat_factor [[buffer(3)]],
    constant uint& t [[buffer(4)]],
    constant uint& d [[buffer(5)]],
    constant uint& subvector_dim [[buffer(6)]],
    constant int& signed_max [[buffer(7)]],
    constant uint& use_residual_signs [[buffer(8)]],
    constant float& attn_scale [[buffer(9)]],
    const device T* q [[buffer(10)]],
    const device uchar* codes [[buffer(11)]],
    const device float* scales [[buffer(12)]],
    const device float* pair_signs [[buffer(13)]],
    const device uchar* residual_signs [[buffer(14)]],
    const device float* residual_scales [[buffer(15)]],
    device float* out [[buffer(16)]],
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
    uint q_base = bh_idx * d;
    uint row_base = row * d;
    uint num_subvectors = d / subvector_dim;

    float acc = 0.0f;
    for (uint pair_idx = 0; pair_idx < (d / 2); ++pair_idx) {
        uint i = pair_idx * 2;
        uint j = i + 1;
        float sign = pair_signs[pair_idx];
        float qa = float(q[q_base + i]);
        float qb = float(q[q_base + j]);
        float q_rot_i = (qa + sign * qb) * 0.7071067811865475244f;
        float q_rot_j = (-sign * qa + qb) * 0.7071067811865475244f;

        uint sub_i = i / subvector_dim;
        uint scale_idx_i = row * num_subvectors + sub_i;
        float sub_scale_i = scales[scale_idx_i];
        int code_i = int(codes[row_base + i]) - signed_max;
        acc += q_rot_i * (float(code_i) * sub_scale_i);
        if (use_residual_signs != 0) {
            float residual_scale_i = residual_scales[scale_idx_i];
            float residual_sign_i = residual_signs[row_base + i] == 0 ? -1.0f : 1.0f;
            acc += q_rot_i * residual_sign_i * residual_scale_i;
        }

        uint sub_j = j / subvector_dim;
        uint scale_idx_j = row * num_subvectors + sub_j;
        float sub_scale_j = scales[scale_idx_j];
        int code_j = int(codes[row_base + j]) - signed_max;
        acc += q_rot_j * (float(code_j) * sub_scale_j);
        if (use_residual_signs != 0) {
            float residual_scale_j = residual_scales[scale_idx_j];
            float residual_sign_j = residual_signs[row_base + j] == 0 ? -1.0f : 1.0f;
            acc += q_rot_j * residual_sign_j * residual_scale_j;
        }
    }

    out[gid] = acc * attn_scale;
}

template [[host_name("qk_scores_turbo_f32")]] [[kernel]] decltype(qk_scores_turboquant_kernel<float>) qk_scores_turboquant_kernel<float>;
template [[host_name("qk_scores_turbo_f16")]] [[kernel]] decltype(qk_scores_turboquant_kernel<half>) qk_scores_turboquant_kernel<half>;
template [[host_name("qk_scores_turbo_bf16")]] [[kernel]] decltype(qk_scores_turboquant_kernel<bfloat>) qk_scores_turboquant_kernel<bfloat>;

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

template <typename T>
[[kernel]] void pack_scales_rowwise_kernel(
    constant uint& rows [[buffer(0)]],
    constant uint& d [[buffer(1)]],
    constant float& denom [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device float* scales [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= rows) {
        return;
    }
    uint base = gid * d;
    float max_abs = 0.0f;
    for (uint i = 0; i < d; ++i) {
        float v = fabs(float(x[base + i]));
        max_abs = max(max_abs, v);
    }
    float s = (max_abs == 0.0f) ? 1e-8f : max(max_abs / denom, 1e-8f);
    scales[gid] = s;
}

template [[host_name("pack_scales_rowwise_q8_f32")]] [[kernel]] decltype(pack_scales_rowwise_kernel<float>) pack_scales_rowwise_kernel<float>;
template [[host_name("pack_scales_rowwise_q8_f16")]] [[kernel]] decltype(pack_scales_rowwise_kernel<half>) pack_scales_rowwise_kernel<half>;
template [[host_name("pack_scales_rowwise_q8_bf16")]] [[kernel]] decltype(pack_scales_rowwise_kernel<bfloat>) pack_scales_rowwise_kernel<bfloat>;

template <typename T, bool IS_Q4>
[[kernel]] void pack_data_rowwise_kernel(
    constant uint& rows [[buffer(0)]],
    constant uint& d [[buffer(1)]],
    const device T* x [[buffer(2)]],
    const device float* scales [[buffer(3)]],
    device uchar* out [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= rows) {
        return;
    }
    float scale = max(scales[gid], 1e-8f);
    uint x_base = gid * d;
    if constexpr (!IS_Q4) {
        uint out_base = gid * d;
        for (uint i = 0; i < d; ++i) {
            int q = int(rint(float(x[x_base + i]) / scale));
            q = clamp(q, -127, 127);
            out[out_base + i] = uchar(q + 128);
        }
    } else {
        uint packed_cols = (d + 1) / 2;
        uint out_base = gid * packed_cols;
        for (uint col = 0; col < packed_cols; ++col) {
            uint i0 = col * 2;
            uint i1 = i0 + 1;

            int q0 = int(rint(float(x[x_base + i0]) / scale));
            q0 = clamp(q0, -8, 7);
            uchar n0 = uchar(q0 & 0x0F);

            uchar n1 = 0;
            if (i1 < d) {
                int q1 = int(rint(float(x[x_base + i1]) / scale));
                q1 = clamp(q1, -8, 7);
                n1 = uchar(q1 & 0x0F);
            }
            out[out_base + col] = uchar(n0 | (n1 << 4));
        }
    }
}

template [[host_name("pack_data_rowwise_q8_f32")]] [[kernel]] decltype(pack_data_rowwise_kernel<float, false>) pack_data_rowwise_kernel<float, false>;
template [[host_name("pack_data_rowwise_q8_f16")]] [[kernel]] decltype(pack_data_rowwise_kernel<half, false>) pack_data_rowwise_kernel<half, false>;
template [[host_name("pack_data_rowwise_q8_bf16")]] [[kernel]] decltype(pack_data_rowwise_kernel<bfloat, false>) pack_data_rowwise_kernel<bfloat, false>;
template [[host_name("pack_data_rowwise_q4_f32")]] [[kernel]] decltype(pack_data_rowwise_kernel<float, true>) pack_data_rowwise_kernel<float, true>;
template [[host_name("pack_data_rowwise_q4_f16")]] [[kernel]] decltype(pack_data_rowwise_kernel<half, true>) pack_data_rowwise_kernel<half, true>;
template [[host_name("pack_data_rowwise_q4_bf16")]] [[kernel]] decltype(pack_data_rowwise_kernel<bfloat, true>) pack_data_rowwise_kernel<bfloat, true>;

template <typename T, bool IS_Q4>
[[kernel]] void pack_rowwise_fused_kernel(
    constant uint& rows [[buffer(0)]],
    constant uint& d [[buffer(1)]],
    const device T* x [[buffer(2)]],
    device uchar* out [[buffer(3)]],
    device float* scales [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= rows) {
        return;
    }

    uint x_base = gid * d;
    float max_abs = 0.0f;
    for (uint i = 0; i < d; ++i) {
        float v = fabs(float(x[x_base + i]));
        max_abs = max(max_abs, v);
    }
    float denom = IS_Q4 ? 7.0f : 127.0f;
    float scale = (max_abs == 0.0f) ? 1e-8f : max(max_abs / denom, 1e-8f);
    scales[gid] = scale;

    if constexpr (!IS_Q4) {
        uint out_base = gid * d;
        for (uint i = 0; i < d; ++i) {
            int q = int(rint(float(x[x_base + i]) / scale));
            q = clamp(q, -127, 127);
            out[out_base + i] = uchar(q + 128);
        }
    } else {
        uint packed_cols = (d + 1) / 2;
        uint out_base = gid * packed_cols;
        for (uint col = 0; col < packed_cols; ++col) {
            uint i0 = col * 2;
            uint i1 = i0 + 1;
            int q0 = int(rint(float(x[x_base + i0]) / scale));
            q0 = clamp(q0, -8, 7);
            uchar n0 = uchar(q0 & 0x0F);

            uchar n1 = 0;
            if (i1 < d) {
                int q1 = int(rint(float(x[x_base + i1]) / scale));
                q1 = clamp(q1, -8, 7);
                n1 = uchar(q1 & 0x0F);
            }
            out[out_base + col] = uchar(n0 | (n1 << 4));
        }
    }
}

template [[host_name("pack_rowwise_fused_q8_f32")]] [[kernel]] decltype(pack_rowwise_fused_kernel<float, false>) pack_rowwise_fused_kernel<float, false>;
template [[host_name("pack_rowwise_fused_q8_f16")]] [[kernel]] decltype(pack_rowwise_fused_kernel<half, false>) pack_rowwise_fused_kernel<half, false>;
template [[host_name("pack_rowwise_fused_q8_bf16")]] [[kernel]] decltype(pack_rowwise_fused_kernel<bfloat, false>) pack_rowwise_fused_kernel<bfloat, false>;
template [[host_name("pack_rowwise_fused_q4_f32")]] [[kernel]] decltype(pack_rowwise_fused_kernel<float, true>) pack_rowwise_fused_kernel<float, true>;
template [[host_name("pack_rowwise_fused_q4_f16")]] [[kernel]] decltype(pack_rowwise_fused_kernel<half, true>) pack_rowwise_fused_kernel<half, true>;
template [[host_name("pack_rowwise_fused_q4_bf16")]] [[kernel]] decltype(pack_rowwise_fused_kernel<bfloat, true>) pack_rowwise_fused_kernel<bfloat, true>;
