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