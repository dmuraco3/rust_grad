#include <metal_stdlib>
using namespace metal;

// constant int THREADGROUP_SIZE = 256;

// kernel void reduce(
//     const device float *array [[ buffer(0) ]],
//     volatile device atomic_float *result [[ buffer(1) ]],
//     uint id [[ thread_position_in_grid ]],
//     uint tid [[ thread_index_in_threadgroup ]],
//     uint bid [[ threadgroup_position_in_grid ]],
//     uint blockDim [[ threads_per_threadgroup ]]
// ) {
//     threadgroup int shared_memory[THREADGROUP_SIZE];
    
//     uint i = bid * (blockDim * 2) + tid;
    
//     shared_memory[tid] = array[i] + array[i + blockDim];
    
//     threadgroup_barrier(mem_flags::mem_none);
    
//     // reduction in shared memory
//     for (uint s = blockDim / 2; s > 0; s >>= 1) {
//         if (tid < s) {
//             shared_memory[tid] += shared_memory[tid + s];
//         }
//         threadgroup_barrier(mem_flags::mem_none);
//     }
    
//     // it's not recommended (just to show atomic operation capability)!
//     if (0 == tid) {
//         atomic_fetch_add_explicit(result, shared_memory[0], memory_order_relaxed);
//     }
// }

kernel void softmax(
    device const float *src [[ buffer(0) ]],
    device float *out [[ buffer(1) ]],
    volatile device atomic_float *exp_sum [[ buffer(2) ]],
    uint pos [[thread_position_in_grid]]
) {
    atomic_fetch_add_explicit(exp_sum, exp(src[pos]), memory_order_relaxed);

    threadgroup_barrier(mem_flags::mem_none);

    out[pos] = exp(src[pos]) / atomic_load_explicit(exp_sum, memory_order_relaxed);
}