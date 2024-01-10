#include <metal_stdlib>
using namespace metal;

kernel void softmax_crossentropy(
    device const float *src [[ buffer(0) ]],
    device const float *labels [[ buffer(1) ]],
    volatile device atomic_float *loss [[ buffer(2) ]],
    uint pos [[thread_position_in_grid]],
    uint num_ele [[ threads_per_threadgroup ]]
) {
    threadgroup float softmax[]
}