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
[[kernel]] void rotate_pairwise_query_kernel(
    constant uint& bh_full [[buffer(0)]],
    constant uint& d [[buffer(1)]],
    const device T* q [[buffer(2)]],
    const device float* pair_signs [[buffer(3)]],
    device T* out [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint pair_count = d / 2;
    uint n = bh_full * pair_count;
    if (gid >= n) {
        return;
    }

    uint bh_idx = gid / pair_count;
    uint pair_idx = gid % pair_count;
    uint base = bh_idx * d;
    uint i = base + pair_idx * 2;
    uint j = i + 1;
    float sign = pair_signs[pair_idx];
    float qa = float(q[i]);
    float qb = float(q[j]);
    out[i] = T((qa + sign * qb) * 0.7071067811865475244f);
    out[j] = T((-sign * qa + qb) * 0.7071067811865475244f);
}

inline uint unpack_packed_u8_value(
    const device uchar* packed,
    uint bits_per_value,
    uint idx
) {
    uint start_bit = idx * bits_per_value;
    uint byte_idx = start_bit / 8;
    uint bit_offset = start_bit % 8;
    uint word = uint(packed[byte_idx]);
    if (bit_offset + bits_per_value > 8) {
        word |= uint(packed[byte_idx + 1]) << 8;
    }
    uint mask = (1u << bits_per_value) - 1u;
    return (word >> bit_offset) & mask;
}

template <uint CODE_BITS>
inline uint4 unpack4_packed_codes(const device uchar* packed, uint idx_base);

template <>
inline uint4 unpack4_packed_codes<3>(const device uchar* packed, uint idx_base) {
    uint start_bit = idx_base * 3;
    uint byte_idx = start_bit / 8;
    uint bit_offset = start_bit % 8;
    uint word = uint(packed[byte_idx]);
    word |= uint(packed[byte_idx + 1]) << 8;
    word |= uint(packed[byte_idx + 2]) << 16;
    word >>= bit_offset;
    return uint4(word & 0x7u, (word >> 3) & 0x7u, (word >> 6) & 0x7u, (word >> 9) & 0x7u);
}

template <>
inline uint4 unpack4_packed_codes<4>(const device uchar* packed, uint idx_base) {
    uint byte_idx = (idx_base * 4) / 8;
    uint word = uint(packed[byte_idx]) | (uint(packed[byte_idx + 1]) << 8);
    return uint4(word & 0xFu, (word >> 4) & 0xFu, (word >> 8) & 0xFu, (word >> 12) & 0xFu);
}

inline float4 unpack4_packed_signs_pm1(const device uchar* packed, uint idx_base) {
    uint start_bit = idx_base;
    uint byte_idx = start_bit / 8;
    uint bit_offset = start_bit % 8;
    uint word = uint(packed[byte_idx]) >> bit_offset;
    return float4(
        (word & 0x1u) == 0u ? -1.0f : 1.0f,
        ((word >> 1) & 0x1u) == 0u ? -1.0f : 1.0f,
        ((word >> 2) & 0x1u) == 0u ? -1.0f : 1.0f,
        ((word >> 3) & 0x1u) == 0u ? -1.0f : 1.0f
    );
}

inline float4 centered_codes_to_float4(uint4 codes, int signed_max, float scale) {
    return float4(
        float(int(codes.x) - signed_max),
        float(int(codes.y) - signed_max),
        float(int(codes.z) - signed_max),
        float(int(codes.w) - signed_max)
    ) * scale;
}

template <typename T, typename S>
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
    const device S* scales [[buffer(12)]],
    const device float* pair_signs [[buffer(13)]],
    const device uchar* residual_signs [[buffer(14)]],
    const device S* residual_scales [[buffer(15)]],
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
        float q_rot_i = float(q[q_base + i]);
        float q_rot_j = float(q[q_base + j]);

        uint sub_i = i / subvector_dim;
        uint scale_idx_i = row * num_subvectors + sub_i;
        float sub_scale_i = float(scales[scale_idx_i]);
        int code_i = int(codes[row_base + i]) - signed_max;
        acc += q_rot_i * (float(code_i) * sub_scale_i);
        if (use_residual_signs != 0) {
            float residual_scale_i = float(residual_scales[scale_idx_i]);
            float residual_sign_i = residual_signs[row_base + i] == 0 ? -1.0f : 1.0f;
            acc += q_rot_i * residual_sign_i * residual_scale_i;
        }

        uint sub_j = j / subvector_dim;
        uint scale_idx_j = row * num_subvectors + sub_j;
        float sub_scale_j = float(scales[scale_idx_j]);
        int code_j = int(codes[row_base + j]) - signed_max;
        acc += q_rot_j * (float(code_j) * sub_scale_j);
        if (use_residual_signs != 0) {
            float residual_scale_j = float(residual_scales[scale_idx_j]);
            float residual_sign_j = residual_signs[row_base + j] == 0 ? -1.0f : 1.0f;
            acc += q_rot_j * residual_sign_j * residual_scale_j;
        }
    }

    out[gid] = acc * attn_scale;
}

template <typename T, typename S>
[[kernel]] void qk_scores_turboquant_packed_kernel(
    constant uint& bh_full [[buffer(0)]],
    constant uint& full_heads [[buffer(1)]],
    constant uint& kv_heads [[buffer(2)]],
    constant uint& repeat_factor [[buffer(3)]],
    constant uint& t [[buffer(4)]],
    constant uint& d [[buffer(5)]],
    constant uint& subvector_dim [[buffer(6)]],
    constant uint& scale_block_dim [[buffer(7)]],
    constant uint& code_bits [[buffer(8)]],
    constant int& signed_max [[buffer(9)]],
    constant uint& use_residual_signs [[buffer(10)]],
    constant float& attn_scale [[buffer(11)]],
    const device T* q [[buffer(12)]],
    const device uchar* packed_codes [[buffer(13)]],
    const device S* scales [[buffer(14)]],
    const device float* pair_signs [[buffer(15)]],
    const device uchar* packed_residual_signs [[buffer(16)]],
    const device S* residual_scales [[buffer(17)]],
    device float* out [[buffer(18)]],
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
    uint num_scale_blocks = d / scale_block_dim;

    float acc = 0.0f;
    for (uint pair_idx = 0; pair_idx < (d / 2); ++pair_idx) {
        uint i = pair_idx * 2;
        uint j = i + 1;
        float q_rot_i = float(q[q_base + i]);
        float q_rot_j = float(q[q_base + j]);

        uint block_i = i / scale_block_dim;
        uint scale_idx_i = row * num_scale_blocks + block_i;
        float sub_scale_i = float(scales[scale_idx_i]);
        int code_i = int(unpack_packed_u8_value(packed_codes, code_bits, row_base + i)) - signed_max;
        acc += q_rot_i * (float(code_i) * sub_scale_i);
        if (use_residual_signs != 0) {
            float residual_scale_i = float(residual_scales[scale_idx_i]);
            float residual_sign_i =
                unpack_packed_u8_value(packed_residual_signs, 1, row_base + i) == 0 ? -1.0f : 1.0f;
            acc += q_rot_i * residual_sign_i * residual_scale_i;
        }

        uint block_j = j / scale_block_dim;
        uint scale_idx_j = row * num_scale_blocks + block_j;
        float sub_scale_j = float(scales[scale_idx_j]);
        int code_j = int(unpack_packed_u8_value(packed_codes, code_bits, row_base + j)) - signed_max;
        acc += q_rot_j * (float(code_j) * sub_scale_j);
        if (use_residual_signs != 0) {
            float residual_scale_j = float(residual_scales[scale_idx_j]);
            float residual_sign_j =
                unpack_packed_u8_value(packed_residual_signs, 1, row_base + j) == 0 ? -1.0f : 1.0f;
            acc += q_rot_j * residual_sign_j * residual_scale_j;
        }
    }

    out[gid] = acc * attn_scale;
}

template <typename T, typename S, uint SUBVECTOR_DIM>
[[kernel]] void qk_scores_turboquant_kernel_fixed_subvector(
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
    const device S* scales [[buffer(12)]],
    const device float* pair_signs [[buffer(13)]],
    const device uchar* residual_signs [[buffer(14)]],
    const device S* residual_scales [[buffer(15)]],
    device float* out [[buffer(16)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 threads_per_tg [[threads_per_threadgroup]]
) {
    (void)subvector_dim;
    constexpr uint MAX_TURBO_HEAD_DIM = 256;
    threadgroup float q_rot_shared[MAX_TURBO_HEAD_DIM];

    if (d > MAX_TURBO_HEAD_DIM) {
        return;
    }
    uint bh_idx = tgpig.x;
    if (bh_idx >= bh_full) {
        return;
    }
    uint b = bh_idx / full_heads;
    uint h = bh_idx % full_heads;
    uint kv_h = h / repeat_factor;
    uint q_base = bh_idx * d;
    uint num_subvectors = d / SUBVECTOR_DIM;
    uint thread_count = max(uint(1), threads_per_tg.x);

    for (uint pair_idx = tid; pair_idx < (d / 2); pair_idx += thread_count) {
        uint i = pair_idx * 2;
        uint j = i + 1;
        float sign = pair_signs[pair_idx];
        float qa = float(q[q_base + i]);
        float qb = float(q[q_base + j]);
        q_rot_shared[i] = (qa + sign * qb) * 0.7071067811865475244f;
        q_rot_shared[j] = (-sign * qa + qb) * 0.7071067811865475244f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tok = tid; tok < t; tok += thread_count) {
        uint row = (b * kv_heads + kv_h) * t + tok;
        uint row_base = row * d;
        float acc = 0.0f;
        for (uint sub = 0; sub < num_subvectors; ++sub) {
            uint scale_idx = row * num_subvectors + sub;
            float sub_scale = float(scales[scale_idx]);
            float residual_scale = use_residual_signs != 0 ? float(residual_scales[scale_idx]) : 0.0f;
            uint sub_base = sub * SUBVECTOR_DIM;

            for (uint pair_local = 0; pair_local < (SUBVECTOR_DIM / 2); ++pair_local) {
                uint i = sub_base + pair_local * 2;
                uint j = i + 1;
                float q_rot_i = q_rot_shared[i];
                float q_rot_j = q_rot_shared[j];

                int code_i = int(codes[row_base + i]) - signed_max;
                acc += q_rot_i * (float(code_i) * sub_scale);
                if (use_residual_signs != 0) {
                    float residual_sign_i = residual_signs[row_base + i] == 0 ? -1.0f : 1.0f;
                    acc += q_rot_i * residual_sign_i * residual_scale;
                }

                int code_j = int(codes[row_base + j]) - signed_max;
                acc += q_rot_j * (float(code_j) * sub_scale);
                if (use_residual_signs != 0) {
                    float residual_sign_j = residual_signs[row_base + j] == 0 ? -1.0f : 1.0f;
                    acc += q_rot_j * residual_sign_j * residual_scale;
                }
            }
        }
        out[bh_idx * t + tok] = acc * attn_scale;
    }
}

template <typename T, typename S, uint SUBVECTOR_DIM>
[[kernel]] void qk_scores_turboquant_packed_kernel_fixed_subvector(
    constant uint& bh_full [[buffer(0)]],
    constant uint& full_heads [[buffer(1)]],
    constant uint& kv_heads [[buffer(2)]],
    constant uint& repeat_factor [[buffer(3)]],
    constant uint& t [[buffer(4)]],
    constant uint& d [[buffer(5)]],
    constant uint& subvector_dim [[buffer(6)]],
    constant uint& scale_block_dim [[buffer(7)]],
    constant uint& code_bits [[buffer(8)]],
    constant int& signed_max [[buffer(9)]],
    constant uint& use_residual_signs [[buffer(10)]],
    constant float& attn_scale [[buffer(11)]],
    const device T* q [[buffer(12)]],
    const device uchar* packed_codes [[buffer(13)]],
    const device S* scales [[buffer(14)]],
    const device float* pair_signs [[buffer(15)]],
    const device uchar* packed_residual_signs [[buffer(16)]],
    const device S* residual_scales [[buffer(17)]],
    device float* out [[buffer(18)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 threads_per_tg [[threads_per_threadgroup]]
) {
    (void)subvector_dim;
    constexpr uint MAX_TURBO_HEAD_DIM = 256;
    threadgroup float q_rot_shared[MAX_TURBO_HEAD_DIM];

    if (d > MAX_TURBO_HEAD_DIM) {
        return;
    }
    uint bh_idx = tgpig.x;
    if (bh_idx >= bh_full) {
        return;
    }
    uint b = bh_idx / full_heads;
    uint h = bh_idx % full_heads;
    uint kv_h = h / repeat_factor;
    uint q_base = bh_idx * d;
    uint num_scale_blocks = d / scale_block_dim;
    uint subvectors_per_block = scale_block_dim / SUBVECTOR_DIM;
    uint thread_count = max(uint(1), threads_per_tg.x);

    for (uint pair_idx = tid; pair_idx < (d / 2); pair_idx += thread_count) {
        uint i = pair_idx * 2;
        uint j = i + 1;
        float sign = pair_signs[pair_idx];
        float qa = float(q[q_base + i]);
        float qb = float(q[q_base + j]);
        q_rot_shared[i] = (qa + sign * qb) * 0.7071067811865475244f;
        q_rot_shared[j] = (-sign * qa + qb) * 0.7071067811865475244f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tok = tid; tok < t; tok += thread_count) {
        uint row = (b * kv_heads + kv_h) * t + tok;
        uint row_base = row * d;
        float acc = 0.0f;
        for (uint block = 0; block < num_scale_blocks; ++block) {
            uint scale_idx = row * num_scale_blocks + block;
            float sub_scale = float(scales[scale_idx]);
            float residual_scale = use_residual_signs != 0 ? float(residual_scales[scale_idx]) : 0.0f;
            uint block_base = block * scale_block_dim;

            for (uint local_sub = 0; local_sub < subvectors_per_block; ++local_sub) {
                uint sub_base = block_base + local_sub * SUBVECTOR_DIM;
                for (uint pair_local = 0; pair_local < (SUBVECTOR_DIM / 2); ++pair_local) {
                    uint i = sub_base + pair_local * 2;
                    uint j = i + 1;
                    float q_rot_i = q_rot_shared[i];
                    float q_rot_j = q_rot_shared[j];

                    int code_i = int(unpack_packed_u8_value(packed_codes, code_bits, row_base + i)) - signed_max;
                    acc += q_rot_i * (float(code_i) * sub_scale);
                    if (use_residual_signs != 0) {
                        float residual_sign_i =
                            unpack_packed_u8_value(packed_residual_signs, 1, row_base + i) == 0 ? -1.0f : 1.0f;
                        acc += q_rot_i * residual_sign_i * residual_scale;
                    }

                    int code_j = int(unpack_packed_u8_value(packed_codes, code_bits, row_base + j)) - signed_max;
                    acc += q_rot_j * (float(code_j) * sub_scale);
                    if (use_residual_signs != 0) {
                        float residual_sign_j =
                            unpack_packed_u8_value(packed_residual_signs, 1, row_base + j) == 0 ? -1.0f : 1.0f;
                        acc += q_rot_j * residual_sign_j * residual_scale;
                    }
                }
            }
        }
        out[bh_idx * t + tok] = acc * attn_scale;
    }
}

template <uint SUBVECTOR_DIM, uint CODE_BITS, bool HAS_RESIDUAL>
[[kernel]] void qk_scores_turboquant_packed_kernel_fixed_subvector_f16_f16(
    constant uint& bh_full [[buffer(0)]],
    constant uint& full_heads [[buffer(1)]],
    constant uint& kv_heads [[buffer(2)]],
    constant uint& repeat_factor [[buffer(3)]],
    constant uint& t [[buffer(4)]],
    constant uint& d [[buffer(5)]],
    constant uint& subvector_dim [[buffer(6)]],
    constant uint& scale_block_dim [[buffer(7)]],
    constant uint& code_bits [[buffer(8)]],
    constant int& signed_max [[buffer(9)]],
    constant uint& use_residual_signs [[buffer(10)]],
    constant float& attn_scale [[buffer(11)]],
    const device half* q [[buffer(12)]],
    const device uchar* packed_codes [[buffer(13)]],
    const device half* scales [[buffer(14)]],
    const device float* pair_signs [[buffer(15)]],
    const device uchar* packed_residual_signs [[buffer(16)]],
    const device half* residual_scales [[buffer(17)]],
    device float* out [[buffer(18)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 threads_per_tg [[threads_per_threadgroup]]
) {
    (void)subvector_dim;
    (void)code_bits;
    (void)use_residual_signs;
    constexpr uint MAX_TURBO_HEAD_DIM = 256;
    threadgroup float q_rot_shared[MAX_TURBO_HEAD_DIM];

    if (d > MAX_TURBO_HEAD_DIM) {
        return;
    }
    uint bh_idx = tgpig.x;
    if (bh_idx >= bh_full) {
        return;
    }
    uint b = bh_idx / full_heads;
    uint h = bh_idx % full_heads;
    uint kv_h = h / repeat_factor;
    uint q_base = bh_idx * d;
    uint num_scale_blocks = d / scale_block_dim;
    uint subvectors_per_block = scale_block_dim / SUBVECTOR_DIM;
    uint thread_count = max(uint(1), threads_per_tg.x);

    for (uint pair_idx = tid; pair_idx < (d / 2); pair_idx += thread_count) {
        uint i = pair_idx * 2;
        uint j = i + 1;
        float sign = pair_signs[pair_idx];
        float qa = float(q[q_base + i]);
        float qb = float(q[q_base + j]);
        q_rot_shared[i] = (qa + sign * qb) * 0.7071067811865475244f;
        q_rot_shared[j] = (-sign * qa + qb) * 0.7071067811865475244f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tok = tid; tok < t; tok += thread_count) {
        uint row = (b * kv_heads + kv_h) * t + tok;
        uint row_base = row * d;
        float acc = 0.0f;
        for (uint block = 0; block < num_scale_blocks; ++block) {
            uint scale_idx = row * num_scale_blocks + block;
            float sub_scale = float(scales[scale_idx]);
            uint block_base = block * scale_block_dim;

            if constexpr (SUBVECTOR_DIM == 4) {
                for (uint local_sub = 0; local_sub < subvectors_per_block; ++local_sub) {
                    uint sub_base = block_base + local_sub * SUBVECTOR_DIM;
                    float4 qv = float4(
                        q_rot_shared[sub_base + 0],
                        q_rot_shared[sub_base + 1],
                        q_rot_shared[sub_base + 2],
                        q_rot_shared[sub_base + 3]
                    );
                    uint4 codes4 = unpack4_packed_codes<CODE_BITS>(packed_codes, row_base + sub_base);
                    acc += dot(qv, centered_codes_to_float4(codes4, signed_max, sub_scale));
                    if constexpr (HAS_RESIDUAL) {
                        float residual_scale = float(residual_scales[scale_idx]);
                        float4 signs4 = unpack4_packed_signs_pm1(packed_residual_signs, row_base + sub_base);
                        acc += dot(qv, signs4 * residual_scale);
                    }
                }
            } else {
                for (uint local_sub = 0; local_sub < subvectors_per_block; ++local_sub) {
                    uint sub_base = block_base + local_sub * SUBVECTOR_DIM;
                    float4 qv0 = float4(
                        q_rot_shared[sub_base + 0],
                        q_rot_shared[sub_base + 1],
                        q_rot_shared[sub_base + 2],
                        q_rot_shared[sub_base + 3]
                    );
                    float4 qv1 = float4(
                        q_rot_shared[sub_base + 4],
                        q_rot_shared[sub_base + 5],
                        q_rot_shared[sub_base + 6],
                        q_rot_shared[sub_base + 7]
                    );
                    uint4 codes0 = unpack4_packed_codes<CODE_BITS>(packed_codes, row_base + sub_base);
                    uint4 codes1 = unpack4_packed_codes<CODE_BITS>(packed_codes, row_base + sub_base + 4);
                    acc += dot(qv0, centered_codes_to_float4(codes0, signed_max, sub_scale));
                    acc += dot(qv1, centered_codes_to_float4(codes1, signed_max, sub_scale));
                    if constexpr (HAS_RESIDUAL) {
                        float residual_scale = float(residual_scales[scale_idx]);
                        float4 signs0 = unpack4_packed_signs_pm1(packed_residual_signs, row_base + sub_base);
                        float4 signs1 = unpack4_packed_signs_pm1(packed_residual_signs, row_base + sub_base + 4);
                        acc += dot(qv0, signs0 * residual_scale);
                        acc += dot(qv1, signs1 * residual_scale);
                    }
                }
            }
        }
        out[bh_idx * t + tok] = acc * attn_scale;
    }
}

template [[host_name("rotate_pairwise_query_f32")]] [[kernel]] decltype(rotate_pairwise_query_kernel<float>) rotate_pairwise_query_kernel<float>;
template [[host_name("rotate_pairwise_query_f16")]] [[kernel]] decltype(rotate_pairwise_query_kernel<half>) rotate_pairwise_query_kernel<half>;
template [[host_name("rotate_pairwise_query_bf16")]] [[kernel]] decltype(rotate_pairwise_query_kernel<bfloat>) rotate_pairwise_query_kernel<bfloat>;
template [[host_name("qk_scores_turbo_qf32_sf32")]] [[kernel]] decltype(qk_scores_turboquant_kernel<float, float>) qk_scores_turboquant_kernel<float, float>;
template [[host_name("qk_scores_turbo_qf32_sf16")]] [[kernel]] decltype(qk_scores_turboquant_kernel<float, half>) qk_scores_turboquant_kernel<float, half>;
template [[host_name("qk_scores_turbo_qf16_sf32")]] [[kernel]] decltype(qk_scores_turboquant_kernel<half, float>) qk_scores_turboquant_kernel<half, float>;
template [[host_name("qk_scores_turbo_qf16_sf16")]] [[kernel]] decltype(qk_scores_turboquant_kernel<half, half>) qk_scores_turboquant_kernel<half, half>;
template [[host_name("qk_scores_turbo_qbf16_sf32")]] [[kernel]] decltype(qk_scores_turboquant_kernel<bfloat, float>) qk_scores_turboquant_kernel<bfloat, float>;
template [[host_name("qk_scores_turbo_qbf16_sf16")]] [[kernel]] decltype(qk_scores_turboquant_kernel<bfloat, half>) qk_scores_turboquant_kernel<bfloat, half>;
template [[host_name("qk_scores_turbo_sv4_qf32_sf32")]] [[kernel]] decltype(qk_scores_turboquant_kernel_fixed_subvector<float, float, 4>) qk_scores_turboquant_kernel_fixed_subvector<float, float, 4>;
template [[host_name("qk_scores_turbo_sv4_qf32_sf16")]] [[kernel]] decltype(qk_scores_turboquant_kernel_fixed_subvector<float, half, 4>) qk_scores_turboquant_kernel_fixed_subvector<float, half, 4>;
template [[host_name("qk_scores_turbo_sv4_qf16_sf32")]] [[kernel]] decltype(qk_scores_turboquant_kernel_fixed_subvector<half, float, 4>) qk_scores_turboquant_kernel_fixed_subvector<half, float, 4>;
template [[host_name("qk_scores_turbo_sv4_qf16_sf16")]] [[kernel]] decltype(qk_scores_turboquant_kernel_fixed_subvector<half, half, 4>) qk_scores_turboquant_kernel_fixed_subvector<half, half, 4>;
template [[host_name("qk_scores_turbo_sv4_qbf16_sf32")]] [[kernel]] decltype(qk_scores_turboquant_kernel_fixed_subvector<bfloat, float, 4>) qk_scores_turboquant_kernel_fixed_subvector<bfloat, float, 4>;
template [[host_name("qk_scores_turbo_sv4_qbf16_sf16")]] [[kernel]] decltype(qk_scores_turboquant_kernel_fixed_subvector<bfloat, half, 4>) qk_scores_turboquant_kernel_fixed_subvector<bfloat, half, 4>;
template [[host_name("qk_scores_turbo_sv8_qf32_sf32")]] [[kernel]] decltype(qk_scores_turboquant_kernel_fixed_subvector<float, float, 8>) qk_scores_turboquant_kernel_fixed_subvector<float, float, 8>;
template [[host_name("qk_scores_turbo_sv8_qf32_sf16")]] [[kernel]] decltype(qk_scores_turboquant_kernel_fixed_subvector<float, half, 8>) qk_scores_turboquant_kernel_fixed_subvector<float, half, 8>;
template [[host_name("qk_scores_turbo_sv8_qf16_sf32")]] [[kernel]] decltype(qk_scores_turboquant_kernel_fixed_subvector<half, float, 8>) qk_scores_turboquant_kernel_fixed_subvector<half, float, 8>;
template [[host_name("qk_scores_turbo_sv8_qf16_sf16")]] [[kernel]] decltype(qk_scores_turboquant_kernel_fixed_subvector<half, half, 8>) qk_scores_turboquant_kernel_fixed_subvector<half, half, 8>;
template [[host_name("qk_scores_turbo_sv8_qbf16_sf32")]] [[kernel]] decltype(qk_scores_turboquant_kernel_fixed_subvector<bfloat, float, 8>) qk_scores_turboquant_kernel_fixed_subvector<bfloat, float, 8>;
template [[host_name("qk_scores_turbo_sv8_qbf16_sf16")]] [[kernel]] decltype(qk_scores_turboquant_kernel_fixed_subvector<bfloat, half, 8>) qk_scores_turboquant_kernel_fixed_subvector<bfloat, half, 8>;
template [[host_name("qk_scores_turbo_packed_qf32_sf32")]] [[kernel]] decltype(qk_scores_turboquant_packed_kernel<float, float>) qk_scores_turboquant_packed_kernel<float, float>;
template [[host_name("qk_scores_turbo_packed_qf32_sf16")]] [[kernel]] decltype(qk_scores_turboquant_packed_kernel<float, half>) qk_scores_turboquant_packed_kernel<float, half>;
template [[host_name("qk_scores_turbo_packed_qf16_sf32")]] [[kernel]] decltype(qk_scores_turboquant_packed_kernel<half, float>) qk_scores_turboquant_packed_kernel<half, float>;
template [[host_name("qk_scores_turbo_packed_qf16_sf16")]] [[kernel]] decltype(qk_scores_turboquant_packed_kernel<half, half>) qk_scores_turboquant_packed_kernel<half, half>;
template [[host_name("qk_scores_turbo_packed_qbf16_sf32")]] [[kernel]] decltype(qk_scores_turboquant_packed_kernel<bfloat, float>) qk_scores_turboquant_packed_kernel<bfloat, float>;
template [[host_name("qk_scores_turbo_packed_qbf16_sf16")]] [[kernel]] decltype(qk_scores_turboquant_packed_kernel<bfloat, half>) qk_scores_turboquant_packed_kernel<bfloat, half>;
template [[host_name("qk_scores_turbo_packed_sv4_qf32_sf32")]] [[kernel]] decltype(qk_scores_turboquant_packed_kernel_fixed_subvector<float, float, 4>) qk_scores_turboquant_packed_kernel_fixed_subvector<float, float, 4>;
template [[host_name("qk_scores_turbo_packed_sv4_qf32_sf16")]] [[kernel]] decltype(qk_scores_turboquant_packed_kernel_fixed_subvector<float, half, 4>) qk_scores_turboquant_packed_kernel_fixed_subvector<float, half, 4>;
template [[host_name("qk_scores_turbo_packed_sv4_qf16_sf32")]] [[kernel]] decltype(qk_scores_turboquant_packed_kernel_fixed_subvector<half, float, 4>) qk_scores_turboquant_packed_kernel_fixed_subvector<half, float, 4>;
template [[host_name("qk_scores_turbo_packed_sv4_qf16_sf16")]] [[kernel]] decltype(qk_scores_turboquant_packed_kernel_fixed_subvector<half, half, 4>) qk_scores_turboquant_packed_kernel_fixed_subvector<half, half, 4>;
template [[host_name("qk_scores_turbo_packed_sv4_qbf16_sf32")]] [[kernel]] decltype(qk_scores_turboquant_packed_kernel_fixed_subvector<bfloat, float, 4>) qk_scores_turboquant_packed_kernel_fixed_subvector<bfloat, float, 4>;
template [[host_name("qk_scores_turbo_packed_sv4_qbf16_sf16")]] [[kernel]] decltype(qk_scores_turboquant_packed_kernel_fixed_subvector<bfloat, half, 4>) qk_scores_turboquant_packed_kernel_fixed_subvector<bfloat, half, 4>;
template [[host_name("qk_scores_turbo_packed_sv8_qf32_sf32")]] [[kernel]] decltype(qk_scores_turboquant_packed_kernel_fixed_subvector<float, float, 8>) qk_scores_turboquant_packed_kernel_fixed_subvector<float, float, 8>;
template [[host_name("qk_scores_turbo_packed_sv8_qf32_sf16")]] [[kernel]] decltype(qk_scores_turboquant_packed_kernel_fixed_subvector<float, half, 8>) qk_scores_turboquant_packed_kernel_fixed_subvector<float, half, 8>;
template [[host_name("qk_scores_turbo_packed_sv8_qf16_sf32")]] [[kernel]] decltype(qk_scores_turboquant_packed_kernel_fixed_subvector<half, float, 8>) qk_scores_turboquant_packed_kernel_fixed_subvector<half, float, 8>;
template [[host_name("qk_scores_turbo_packed_sv8_qf16_sf16")]] [[kernel]] decltype(qk_scores_turboquant_packed_kernel_fixed_subvector<half, half, 8>) qk_scores_turboquant_packed_kernel_fixed_subvector<half, half, 8>;
template [[host_name("qk_scores_turbo_packed_sv8_qbf16_sf32")]] [[kernel]] decltype(qk_scores_turboquant_packed_kernel_fixed_subvector<bfloat, float, 8>) qk_scores_turboquant_packed_kernel_fixed_subvector<bfloat, float, 8>;
template [[host_name("qk_scores_turbo_packed_sv8_qbf16_sf16")]] [[kernel]] decltype(qk_scores_turboquant_packed_kernel_fixed_subvector<bfloat, half, 8>) qk_scores_turboquant_packed_kernel_fixed_subvector<bfloat, half, 8>;
template [[host_name("qk_scores_turbo_packed_fast_sv4c3_nr_qf16_sf16")]] [[kernel]] decltype(qk_scores_turboquant_packed_kernel_fixed_subvector_f16_f16<4, 3, false>) qk_scores_turboquant_packed_kernel_fixed_subvector_f16_f16<4, 3, false>;
template [[host_name("qk_scores_turbo_packed_fast_sv4c3_res_qf16_sf16")]] [[kernel]] decltype(qk_scores_turboquant_packed_kernel_fixed_subvector_f16_f16<4, 3, true>) qk_scores_turboquant_packed_kernel_fixed_subvector_f16_f16<4, 3, true>;
template [[host_name("qk_scores_turbo_packed_fast_sv4c4_nr_qf16_sf16")]] [[kernel]] decltype(qk_scores_turboquant_packed_kernel_fixed_subvector_f16_f16<4, 4, false>) qk_scores_turboquant_packed_kernel_fixed_subvector_f16_f16<4, 4, false>;
template [[host_name("qk_scores_turbo_packed_fast_sv4c4_res_qf16_sf16")]] [[kernel]] decltype(qk_scores_turboquant_packed_kernel_fixed_subvector_f16_f16<4, 4, true>) qk_scores_turboquant_packed_kernel_fixed_subvector_f16_f16<4, 4, true>;
template [[host_name("qk_scores_turbo_packed_fast_sv8c3_nr_qf16_sf16")]] [[kernel]] decltype(qk_scores_turboquant_packed_kernel_fixed_subvector_f16_f16<8, 3, false>) qk_scores_turboquant_packed_kernel_fixed_subvector_f16_f16<8, 3, false>;
template [[host_name("qk_scores_turbo_packed_fast_sv8c3_res_qf16_sf16")]] [[kernel]] decltype(qk_scores_turboquant_packed_kernel_fixed_subvector_f16_f16<8, 3, true>) qk_scores_turboquant_packed_kernel_fixed_subvector_f16_f16<8, 3, true>;
template [[host_name("qk_scores_turbo_packed_fast_sv8c4_nr_qf16_sf16")]] [[kernel]] decltype(qk_scores_turboquant_packed_kernel_fixed_subvector_f16_f16<8, 4, false>) qk_scores_turboquant_packed_kernel_fixed_subvector_f16_f16<8, 4, false>;
template [[host_name("qk_scores_turbo_packed_fast_sv8c4_res_qf16_sf16")]] [[kernel]] decltype(qk_scores_turboquant_packed_kernel_fixed_subvector_f16_f16<8, 4, true>) qk_scores_turboquant_packed_kernel_fixed_subvector_f16_f16<8, 4, true>;

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
