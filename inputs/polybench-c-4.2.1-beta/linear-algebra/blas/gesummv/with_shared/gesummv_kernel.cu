#include <stdio.h> 
#define DEVICECODE true 
#include "gesummv_kernel.hu"
__global__ void kernel0(float A[90][90], float B[90][90], float alpha, float beta, float tmp[90], float x[90], float y[90])
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ float shared_A[32][32];
    __shared__ float shared_B[32][32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c1 = 0; c1 <= 89; c1 += 32) {
      if (t0 + c1 <= 89) {
        for (int c2 = 0; c2 <= ppcg_min(31, -32 * b0 + 89); c2 += 1)
          shared_A[c2][t0] = A[32 * b0 + c2][t0 + c1];
        for (int c2 = 0; c2 <= ppcg_min(31, -32 * b0 + 89); c2 += 1)
          shared_B[c2][t0] = B[32 * b0 + c2][t0 + c1];
      }
      __syncthreads();
      if (32 * b0 + t0 <= 89 && c1 == 0)
        y[32 * b0 + t0] = 0.F;
      if (32 * b0 + t0 <= 89) {
        for (int c3 = 0; c3 <= ppcg_min(31, -c1 + 89); c3 += 1) {
          y[32 * b0 + t0] = ((shared_B[t0][c3] * x[c1 + c3]) + y[32 * b0 + t0]);
          if (c1 == 0 && c3 == 0)
            tmp[32 * b0 + t0] = 0.F;
          tmp[32 * b0 + t0] = ((shared_A[t0][c3] * x[c1 + c3]) + tmp[32 * b0 + t0]);
        }
        if (c1 == 64)
          y[32 * b0 + t0] = ((alpha * tmp[32 * b0 + t0]) + (beta * y[32 * b0 + t0]));
      }
      __syncthreads();
    }
}
