#include <stdio.h> 
#define DEVICECODE true 
#include "gemm_kernel.hu"
__global__ void kernel0(double A[60][80], double B[80][70], double C[60][70], double alpha, double beta)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c2 = 0; c2 <= 79; c2 += 32) {
      if (32 * b0 + t0 <= 59)
        for (int c4 = t1; c4 <= ppcg_min(31, -32 * b1 + 69); c4 += 16) {
          if (c2 == 0)
            C[32 * b0 + t0][32 * b1 + c4] *= beta;
          for (int c5 = 0; c5 <= ppcg_min(31, -c2 + 79); c5 += 1)
            C[32 * b0 + t0][32 * b1 + c4] += ((alpha * A[32 * b0 + t0][c2 + c5]) * B[c2 + c5][32 * b1 + c4]);
        }
      __syncthreads();
    }
}
