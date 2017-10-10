#include <stdio.h> 
#define DEVICECODE true 
#include "bicg_kernel.hu"
__global__ void kernel0(float A[124][116], float r[124], float s[116])
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ float shared_r[32];
    __shared__ float shared_s[32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    {
      for (int c1 = 0; c1 <= 123; c1 += 32) {
        if (t0 + c1 <= 123)
          shared_r[t0] = r[t0 + c1];
        __syncthreads();
        if (32 * b0 + t0 <= 115 && c1 == 0)
          shared_s[t0] = 0;
        if (32 * b0 + t0 <= 115)
          for (int c3 = 0; c3 <= ppcg_min(31, -c1 + 123); c3 += 1)
            shared_s[t0] = (shared_s[t0] + (shared_r[c3] * A[c1 + c3][32 * b0 + t0]));
        __syncthreads();
      }
      if (32 * b0 + t0 <= 115)
        s[32 * b0 + t0] = shared_s[t0];
    }
}
__global__ void kernel1(float A[124][116], float p[116], float q[124])
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ float shared_A[32][32];
    __shared__ float shared_p[32];
    __shared__ float shared_q[32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    {
      for (int c1 = 0; c1 <= 115; c1 += 32) {
        if (t0 + c1 <= 115) {
          for (int c2 = 0; c2 <= ppcg_min(31, -32 * b0 + 123); c2 += 1)
            shared_A[c2][t0] = A[32 * b0 + c2][t0 + c1];
          shared_p[t0] = p[t0 + c1];
        }
        __syncthreads();
        if (32 * b0 + t0 <= 123 && c1 == 0)
          shared_q[t0] = 0.F;
        if (32 * b0 + t0 <= 123)
          for (int c3 = 0; c3 <= ppcg_min(31, -c1 + 115); c3 += 1)
            shared_q[t0] = (shared_q[t0] + (shared_A[t0][c3] * shared_p[c3]));
        __syncthreads();
      }
      if (32 * b0 + t0 <= 123)
        q[32 * b0 + t0] = shared_q[t0];
    }
}
