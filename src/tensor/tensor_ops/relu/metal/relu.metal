#include <metal_stdlib>

using namespace metal;

kernel void relu(
    device const float* src,
    device float* result,
    uint2 pos [[thread_position_in_grid]],
    uint2 grid_size [[grid_size]],
    uint2 thread_grid_size [[threads_per_grid]]
) {
    uint idx = pos.y * thread_grid_size.x + pos.x;

    result[idx] = max(src[idx], 0.0);
}