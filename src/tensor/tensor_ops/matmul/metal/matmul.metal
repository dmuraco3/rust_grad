#include <metal_stdlib>

using namespace metal;

kernel void mul_matrices(
  device const float* in_a,
  device const float* in_b,
  device float* result,
  uint2 pos [[thread_position_in_grid]],
  uint2 grid_size [[grid_size]],
  uint2 thread_grid_size [[threads_per_grid]]
)
{
  
  uint n = thread_grid_size.x;

  // result[pos.y * n + pos.x] += in_b[pos.y * n + pos.z] * in_a[pos.y * n + pos.x];

  for (uint i=0; i<n; i++) {
    result[pos.y * n + pos.x] += in_b[pos.y * n + i] * in_a[i * n + pos.x];
  }

}  

kernel void matvec_backward(
  device const float *lhs [[ buffer(0) ]],
  device const float *rhs [[ buffer(1) ]],
  device const float *out [[ buffer(2) ]],
  volatile device atomic_float *grad_lhs [[ buffer(3) ]],
  volatile device atomic_float *grad_rhs [[ buffer(4) ]],
  device const float *grad_out [[ buffer(5) ]],
  uint2 pos [[thread_position_in_grid]],
  uint2 grid_size [[grid_size]],
  uint2 thread_grid_size [[threads_per_grid]]
)
{
  uint n = thread_grid_size.x;

  // grad_lhs[y][x] = grad_out[x] * rhs[x];
  atomic_store_explicit(
    &grad_lhs[pos.y * n + pos.x],
    grad_out[pos.x] * rhs[pos.x],
    memory_order_relaxed
  );

  // grad_rhs[x] = grad_out[x] * lhs[y][x];
  atomic_fetch_add_explicit(
    &grad_rhs[pos.x],
    grad_out[pos.x] * lhs[pos.y * n + pos.x],
    memory_order_relaxed
  );
}