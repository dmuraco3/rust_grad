#include <metal_stdlib>

using namespace metal;

kernel void relu(
    device const float* src,
    device float* out,
    uint2 pos [[thread_position_in_grid]],
    uint2 grid_size [[grid_size]],
    uint2 thread_grid_size [[threads_per_grid]]
) {
    uint idx = pos.y * thread_grid_size.x + pos.x;

    out[idx] = max(src[idx], 0.0);
}

kernel void relu_backward(
    device const float* out,
    device const float* grad_out,
    device float* grad_src,
    uint pos [[thread_position_in_grid]]
) {
    grad_src[pos] = grad_out[pos] * (out[pos] > 0.0);
}