#include <metal_stdlib>
using namespace metal;

inline ushort read_u16_le(const device uchar* p) {
    return ushort(p[0]) | (ushort(p[1]) << 8);
}

template <typename T>
inline void accum_iq2_xxs_word(
    ulong grid_pack,
    uchar signs,
    float dl,
    uint base_col,
    uint in_dim,
    uint x_base,
    const device T* x,
    thread float& acc
) {
    for (uint j = 0; j < 8; ++j) {
        uint col = base_col + j;
        if (col >= in_dim) {
            break;
        }
        uchar gv = uchar((grid_pack >> (8 * j)) & 0xFF);
        float sgn = (signs & kmask_iq2xs[j]) ? -1.0f : 1.0f;
        acc += float(x[x_base + col]) * (dl * float(gv) * sgn);
    }
}

template <typename T>
inline void accum_iq2_xs_word(
    ushort qword,
    float dl,
    uint base_col,
    uint in_dim,
    uint x_base,
    const device T* x,
    thread float& acc
) {
    uint grid_idx = uint(qword) & 0x1FF;
    uchar signs = ksigns_iq2xs[(uint(qword) >> 9) & 0x7F];
    ulong grid_pack = iq2xs_grid[grid_idx];
    for (uint j = 0; j < 8; ++j) {
        uint col = base_col + j;
        if (col >= in_dim) {
            break;
        }
        uchar gv = uchar((grid_pack >> (8 * j)) & 0xFF);
        float sgn = (signs & kmask_iq2xs[j]) ? -1.0f : 1.0f;
        acc += float(x[x_base + col]) * (dl * float(gv) * sgn);
    }
}

template <typename T>
inline void accum_iq2_s_grids(
    uint grid_idx0,
    uint grid_idx1,
    uchar sign0,
    uchar sign1,
    float dl,
    uint base_col,
    uint in_dim,
    uint x_base,
    const device T* x,
    thread float& acc
) {
    ulong grid0 = iq2s_grid[grid_idx0];
    ulong grid1 = iq2s_grid[grid_idx1];
    for (uint j = 0; j < 8; ++j) {
        uint col0 = base_col + j;
        if (col0 < in_dim) {
            uchar gv0 = uchar((grid0 >> (8 * j)) & 0xFF);
            float sgn0 = (sign0 & kmask_iq2xs[j]) ? -1.0f : 1.0f;
            acc += float(x[x_base + col0]) * (dl * float(gv0) * sgn0);
        }
        uint col1 = base_col + 8 + j;
        if (col1 < in_dim) {
            uchar gv1 = uchar((grid1 >> (8 * j)) & 0xFF);
            float sgn1 = (sign1 & kmask_iq2xs[j]) ? -1.0f : 1.0f;
            acc += float(x[x_base + col1]) * (dl * float(gv1) * sgn1);
        }
    }
}

template <typename T>
[[kernel]] void iq2_xxs_matmul_kernel(
    constant uint& m [[buffer(0)]],
    constant uint& out_dim [[buffer(1)]],
    constant uint& in_dim [[buffer(2)]],
    constant uint& blocks_per_row [[buffer(3)]],
    const device T* x [[buffer(4)]],
    const device uchar* w_bytes [[buffer(5)]],
    const device uchar* w_scales [[buffer(6)]],
    device float* out [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    uint n = m * out_dim;
    if (gid >= n) {
        return;
    }

    uint row = gid / out_dim;
    uint out_idx = gid % out_dim;
    uint x_base = row * in_dim;

    float acc = 0.0f;
    for (uint blk = 0; blk < blocks_per_row; ++blk) {
        uint block_idx = out_idx * blocks_per_row + blk;
        uint byte_base = block_idx * 64;
        uint scale_base = block_idx * 2;

        ushort d_bits = read_u16_le(w_scales + scale_base);
        float d = float(as_type<half>(d_bits));

        for (uint ib32 = 0; ib32 < 8; ++ib32) {
            uint q2_base = byte_base + ib32 * 8;
            ushort q20 = read_u16_le(w_bytes + q2_base + 0);
            ushort q21 = read_u16_le(w_bytes + q2_base + 2);
            ushort q22 = read_u16_le(w_bytes + q2_base + 4);
            ushort q23 = read_u16_le(w_bytes + q2_base + 6);

            uint aux32_g = uint(q20) | (uint(q21) << 16);
            uint aux32_s = uint(q22) | (uint(q23) << 16);
            float dl = d * (0.5f + float((aux32_s >> 28) & 0xF)) * 0.25f;

            for (uint g = 0; g < 4; ++g) {
                uint grid_idx = (aux32_g >> (8 * g)) & 0xFF;
                ulong grid_pack = iq2xxs_grid[grid_idx];
                uchar signs = ksigns_iq2xs[(aux32_s >> (7 * g)) & 0x7F];
                uint base_col = blk * 256 + ib32 * 32 + g * 8;
                accum_iq2_xxs_word(grid_pack, signs, dl, base_col, in_dim, x_base, x, acc);
            }
        }
    }

    out[gid] = acc;
}

template <typename T>
[[kernel]] void iq2_xs_matmul_kernel(
    constant uint& m [[buffer(0)]],
    constant uint& out_dim [[buffer(1)]],
    constant uint& in_dim [[buffer(2)]],
    constant uint& blocks_per_row [[buffer(3)]],
    const device T* x [[buffer(4)]],
    const device uchar* w_bytes [[buffer(5)]],
    const device uchar* w_scales [[buffer(6)]],
    device float* out [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    uint n = m * out_dim;
    if (gid >= n) {
        return;
    }

    uint row = gid / out_dim;
    uint out_idx = gid % out_dim;
    uint x_base = row * in_dim;

    float acc = 0.0f;
    for (uint blk = 0; blk < blocks_per_row; ++blk) {
        uint block_idx = out_idx * blocks_per_row + blk;
        uint byte_base = block_idx * 64;
        uint scale_base = block_idx * 10;

        ushort d_bits = read_u16_le(w_scales + scale_base);
        float d = float(as_type<half>(d_bits));

        for (uint ib32 = 0; ib32 < 8; ++ib32) {
            uint q2_base = byte_base + ib32 * 8;
            ushort q20 = read_u16_le(w_bytes + q2_base + 0);
            ushort q21 = read_u16_le(w_bytes + q2_base + 2);
            ushort q22 = read_u16_le(w_bytes + q2_base + 4);
            ushort q23 = read_u16_le(w_bytes + q2_base + 6);

            uchar scale_byte = w_scales[scale_base + 2 + ib32];

            float dl0 = d * (0.5f + float(scale_byte & 0x0F)) * 0.25f;
            uint base0 = blk * 256 + ib32 * 32;
            accum_iq2_xs_word(q20, dl0, base0 + 0, in_dim, x_base, x, acc);
            accum_iq2_xs_word(q21, dl0, base0 + 8, in_dim, x_base, x, acc);

            float dl1 = d * (0.5f + float((scale_byte >> 4) & 0x0F)) * 0.25f;
            accum_iq2_xs_word(q22, dl1, base0 + 16, in_dim, x_base, x, acc);
            accum_iq2_xs_word(q23, dl1, base0 + 24, in_dim, x_base, x, acc);
        }
    }

    out[gid] = acc;
}

template <typename T>
[[kernel]] void iq2_s_matmul_kernel(
    constant uint& m [[buffer(0)]],
    constant uint& out_dim [[buffer(1)]],
    constant uint& in_dim [[buffer(2)]],
    constant uint& blocks_per_row [[buffer(3)]],
    const device T* x [[buffer(4)]],
    const device uchar* w_bytes [[buffer(5)]],
    const device uchar* w_scales [[buffer(6)]],
    device float* out [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    uint n = m * out_dim;
    if (gid >= n) {
        return;
    }

    uint row = gid / out_dim;
    uint out_idx = gid % out_dim;
    uint x_base = row * in_dim;

    float acc = 0.0f;
    for (uint blk = 0; blk < blocks_per_row; ++blk) {
        uint block_idx = out_idx * blocks_per_row + blk;
        uint byte_base = block_idx * 72;
        uint scale_base = block_idx * 10;

        ushort d_bits = read_u16_le(w_scales + scale_base);
        float d = float(as_type<half>(d_bits));

        const device uchar* qs = w_bytes + byte_base;
        const device uchar* qh = w_bytes + byte_base + 64;

        for (uint ib32 = 0; ib32 < 8; ++ib32) {
            uchar qh_byte = qh[ib32];
            uchar scale_byte = w_scales[scale_base + 2 + ib32];

            for (uint il = 0; il < 2; ++il) {
                uint qs_off = 4 * ib32 + 2 * il;
                uchar qs0 = qs[qs_off + 0];
                uchar qs1 = qs[qs_off + 1];
                uchar sign0 = qs[32 + qs_off + 0];
                uchar sign1 = qs[32 + qs_off + 1];

                uchar qh_nib = (qh_byte >> (4 * il));
                float dl = d * (0.5f + float((scale_byte >> (4 * il)) & 0x0F)) * 0.25f;

                uint grid_idx0 = uint(qs0) | ((uint(qh_nib) << 8) & 0x300);
                uint grid_idx1 = uint(qs1) | ((uint(qh_nib) << 6) & 0x300);
                uint base_col = blk * 256 + ib32 * 32 + il * 16;
                accum_iq2_s_grids(
                    grid_idx0,
                    grid_idx1,
                    sign0,
                    sign1,
                    dl,
                    base_col,
                    in_dim,
                    x_base,
                    x,
                    acc
                );
            }
        }
    }

    out[gid] = acc;
}

template <typename T>
[[kernel]] void iq3_s_matmul_kernel(
    constant uint& m [[buffer(0)]],
    constant uint& out_dim [[buffer(1)]],
    constant uint& in_dim [[buffer(2)]],
    constant uint& blocks_per_row [[buffer(3)]],
    const device T* x [[buffer(4)]],
    const device uchar* w_bytes [[buffer(5)]],
    const device uchar* w_scales [[buffer(6)]],
    device float* out [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    uint n = m * out_dim;
    if (gid >= n) {
        return;
    }

    uint row = gid / out_dim;
    uint out_idx = gid % out_dim;
    uint x_base = row * in_dim;

    float acc = 0.0f;
    for (uint blk = 0; blk < blocks_per_row; ++blk) {
        uint block_idx = out_idx * blocks_per_row + blk;
        uint byte_base = block_idx * 104;
        uint scale_base = block_idx * 6;

        ushort d_bits = read_u16_le(w_scales + scale_base);
        float d = float(as_type<half>(d_bits));

        const device uchar* qs = w_bytes + byte_base;
        const device uchar* qh = w_bytes + byte_base + 64;
        const device uchar* signs = w_bytes + byte_base + 72;

        for (uint ib32 = 0; ib32 < 8; ++ib32) {
            uchar qh_byte = qh[ib32];
            uchar scale_nib = (w_scales[scale_base + 2 + (ib32 / 2)] >> (4 * (ib32 % 2))) & 0x0F;
            float dl = d * (1.0f + 2.0f * float(scale_nib));

            uint qs_off = 8 * ib32;
            uint signs_off = 4 * ib32;
            for (uint l = 0; l < 4; ++l) {
                uint qh_sel0 = (qh_byte & kmask_iq2xs[2 * l + 0]) ? 256u : 0u;
                uint qh_sel1 = (qh_byte & kmask_iq2xs[2 * l + 1]) ? 256u : 0u;
                uint idx1 = uint(qs[qs_off + 2 * l + 0]) | qh_sel0;
                uint idx2 = uint(qs[qs_off + 2 * l + 1]) | qh_sel1;
                uint grid1 = iq3s_grid[idx1];
                uint grid2 = iq3s_grid[idx2];
                uchar sign_byte = signs[signs_off + l];
                uint base_col = blk * 256 + ib32 * 32 + l * 8;

                for (uint j = 0; j < 4; ++j) {
                    uint col = base_col + j;
                    if (col < in_dim) {
                        uchar gv = uchar((grid1 >> (8 * j)) & 0xFF);
                        float sgn = (sign_byte & kmask_iq2xs[j]) ? -1.0f : 1.0f;
                        acc += float(x[x_base + col]) * (dl * float(gv) * sgn);
                    }
                    col = base_col + 4 + j;
                    if (col < in_dim) {
                        uchar gv = uchar((grid2 >> (8 * j)) & 0xFF);
                        float sgn = (sign_byte & kmask_iq2xs[4 + j]) ? -1.0f : 1.0f;
                        acc += float(x[x_base + col]) * (dl * float(gv) * sgn);
                    }
                }
            }
        }
    }

    out[gid] = acc;
}

template [[host_name("iq2_xxs_matmul_f32")]] [[kernel]]
decltype(iq2_xxs_matmul_kernel<float>) iq2_xxs_matmul_kernel<float>;
template [[host_name("iq2_xxs_matmul_f16")]] [[kernel]]
decltype(iq2_xxs_matmul_kernel<half>) iq2_xxs_matmul_kernel<half>;
template [[host_name("iq2_xxs_matmul_bf16")]] [[kernel]]
decltype(iq2_xxs_matmul_kernel<bfloat>) iq2_xxs_matmul_kernel<bfloat>;

template [[host_name("iq2_xs_matmul_f32")]] [[kernel]]
decltype(iq2_xs_matmul_kernel<float>) iq2_xs_matmul_kernel<float>;
template [[host_name("iq2_xs_matmul_f16")]] [[kernel]]
decltype(iq2_xs_matmul_kernel<half>) iq2_xs_matmul_kernel<half>;
template [[host_name("iq2_xs_matmul_bf16")]] [[kernel]]
decltype(iq2_xs_matmul_kernel<bfloat>) iq2_xs_matmul_kernel<bfloat>;

template [[host_name("iq2_s_matmul_f32")]] [[kernel]]
decltype(iq2_s_matmul_kernel<float>) iq2_s_matmul_kernel<float>;
template [[host_name("iq2_s_matmul_f16")]] [[kernel]]
decltype(iq2_s_matmul_kernel<half>) iq2_s_matmul_kernel<half>;
template [[host_name("iq2_s_matmul_bf16")]] [[kernel]]
decltype(iq2_s_matmul_kernel<bfloat>) iq2_s_matmul_kernel<bfloat>;

template [[host_name("iq3_s_matmul_f32")]] [[kernel]]
decltype(iq3_s_matmul_kernel<float>) iq3_s_matmul_kernel<float>;
template [[host_name("iq3_s_matmul_f16")]] [[kernel]]
decltype(iq3_s_matmul_kernel<half>) iq3_s_matmul_kernel<half>;
template [[host_name("iq3_s_matmul_bf16")]] [[kernel]]
decltype(iq3_s_matmul_kernel<bfloat>) iq3_s_matmul_kernel<bfloat>;
