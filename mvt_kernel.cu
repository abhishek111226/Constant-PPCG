#include <stdio.h> 
#define DEVICECODE true 
#include "mvt_kernel.hu"
__global__ void kernel0(int *A, int *x1)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c1 = 0; c1 <= 99; c1 += 32) {
      if (32 * b0 + t0 <= 99)
        for (int c3 = 0; c3 <= ppcg_min(31, -c1 + 99); c3 += 1)
          x1[32 * b0 + t0] = (x1[32 * b0 + t0] + (A[(32 * b0 + t0) * 100 + (c1 + c3)] * const_y_1[c1 + c3]));
      __syncthreads();
    }
}
__global__ void kernel1(int *A, int *x2)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c1 = 0; c1 <= 99; c1 += 32) {
      if (32 * b0 + t0 <= 99)
        for (int c3 = 0; c3 <= ppcg_min(31, -c1 + 99); c3 += 1)
          x2[32 * b0 + t0] = (x2[32 * b0 + t0] + (A[(c1 + c3) * 100 + (32 * b0 + t0)] * const_y_2[c1 + c3]));
      __syncthreads();
    }
}
