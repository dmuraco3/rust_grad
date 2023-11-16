#include <metal_stdlib>

using namespace metal;

kernel void add_matrices(
    device const float* in_a,
    device const float* in_b,
    device float* result,
    uint2 pos [[thread_position_in_grid]],
    uint2 grid_size [[grid_size]],
    uint2 thread_grid_size [[threads_per_grid]],
) {
    
}