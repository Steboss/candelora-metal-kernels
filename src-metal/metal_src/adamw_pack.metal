#include <metal_stdlib>
using namespace metal;

template<typename T>
[[kernel]] void adamw_pack_kernel(
    constant size_t &dim,
    constant float &beta1,
    constant float &beta2,
    constant float &scale_m,
    constant float &scale_v,
    constant float &eps,
    constant float &lr,
    constant float &lr_lambda,
    device const T *theta,
    device const T *first_moment,
    device const T *second_moment,
    device const T *grad,
    device T *out,
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) {
        return;
    }
    const float one_minus_beta1 = 1.0f - beta1;
    const float one_minus_beta2 = 1.0f - beta2;
    const float one_minus_lr_lambda = 1.0f - lr_lambda;

    const float g = static_cast<float>(grad[tid]);
    const float next_m = beta1 * static_cast<float>(first_moment[tid]) + one_minus_beta1 * g;
    const float next_v = beta2 * static_cast<float>(second_moment[tid]) + one_minus_beta2 * g * g;
    const float m_hat = next_m * scale_m;
    const float v_hat = next_v * scale_v;
    const float next_theta = one_minus_lr_lambda * static_cast<float>(theta[tid])
        - lr * (m_hat / (sqrt(v_hat) + eps));

    out[tid] = static_cast<T>(next_m);
    out[tid + dim] = static_cast<T>(next_v);
    out[tid + 2 * dim] = static_cast<T>(next_theta);
}

template [[host_name("adamw_pack_f32")]] [[kernel]] decltype(adamw_pack_kernel<float>)
adamw_pack_kernel<float>;
template [[host_name("adamw_pack_f16")]] [[kernel]] decltype(adamw_pack_kernel<half>)
adamw_pack_kernel<half>;
#if defined(__HAVE_BFLOAT__)
template [[host_name("adamw_pack_bf16")]] [[kernel]] decltype(adamw_pack_kernel<bfloat>)
adamw_pack_kernel<bfloat>;
#endif
