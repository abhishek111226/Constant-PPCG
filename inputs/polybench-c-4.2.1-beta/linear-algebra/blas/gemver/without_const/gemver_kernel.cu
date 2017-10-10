#include <stdio.h> 
#define DEVICECODE true 
#include "gemver_kernel.hu"
__global__ void kernel0(double A[120][120], double u1[120], double u2[120], double v1[120], double v2[120])
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    if (32 * b0 + t0 <= 119)
      for (int c3 = t1; c3 <= ppcg_min(31, -32 * b1 + 119); c3 += 16)
        A[32 * b0 + t0][32 * b1 + c3] = ((A[32 * b0 + t0][32 * b1 + c3] + (u1[32 * b0 + t0] * v1[32 * b1 + c3])) + (u2[32 * b0 + t0] * v2[32 * b1 + c3]));
}
__global__ void kernel1(double A[120][120], double beta, double x[120], double y[120], double z[120])
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c1 = 0; c1 <= 119; c1 += 32) {
      if (32 * b0 + t0 <= 119) {
        for (int c3 = 0; c3 <= ppcg_min(31, -c1 + 119); c3 += 1)
          x[32 * b0 + t0] = (x[32 * b0 + t0] + ((beta * A[c1 + c3][32 * b0 + t0]) * y[c1 + c3]));
        if (c1 == 96)
          x[32 * b0 + t0] = (x[32 * b0 + t0] + z[32 * b0 + t0]);
      }
      __syncthreads();
    }
}
__global__ void kernel2(double A[120][120], double alpha, double w[120], double x[120])
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c1 = 0; c1 <= 119; c1 += 32) {
      if (32 * b0 + t0 <= 119)
        for (int c3 = 0; c3 <= ppcg_min(31, -c1 + 119); c3 += 1)
          w[32 * b0 + t0] = (w[32 * b0 + t0] + ((alpha * A[32 * b0 + t0][c1 + c3]) * x[c1 + c3]));
      __syncthreads();
    }
}
