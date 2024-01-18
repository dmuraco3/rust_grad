#include <metal_stdlib>
using namespace metal;

kernel void cross_entropy(
    device const float *src [[ buffer(0) ]],
    device const float *labels [[ buffer(1) ]],
    volatile device atomic_float *loss [[ buffer(2) ]],
    uint pos [[thread_position_in_grid]],
    uint num_ele [[ threads_per_threadgroup ]]
) {
    float inner = labels[pos] * log(src[pos]);

    atomic_fetch_sub_explicit(loss, inner, memory_order_relaxed);
}