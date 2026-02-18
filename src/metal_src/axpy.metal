#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void axpy_kernel(
    constant uint& n [[buffer(0)]],
    constant float& alpha [[buffer(1)]],
    const device T* y [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* out [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) {
        return;
    }
    out[gid] = y[gid] + T(alpha) * x[gid];
}

template [[host_name("axpy_f32")]] [[kernel]] decltype(axpy_kernel<float>) axpy_kernel<float>;
template [[host_name("axpy_f16")]] [[kernel]] decltype(axpy_kernel<half>) axpy_kernel<half>;
template [[host_name("axpy_bf16")]] [[kernel]] decltype(axpy_kernel<bfloat>) axpy_kernel<bfloat>;

