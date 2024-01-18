#include <metal_stdlib>

using namespace metal;

kernel void add_matrices(
    device const float* in_a [[ buffer(0) ]],
    device const float* in_b [[ buffer(1) ]],
    device float* result [[ buffer(2) ]],
    uint2 pos [[thread_position_in_grid]],
    uint2 thread_grid_size [[threads_per_grid]]
) {
    uint idx = pos.y * thread_grid_size.x + pos.x;

    result[idx] = in_a[idx] + in_b[idx];
}