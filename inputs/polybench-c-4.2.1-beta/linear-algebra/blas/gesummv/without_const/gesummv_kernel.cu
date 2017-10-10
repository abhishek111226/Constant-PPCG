#include <stdio.h> 
#define DEVICECODE true 
#include "gesummv_kernel.hu"
__global__ void kernel0(double A[90][90], double B[90][90], double alpha, double beta, double tmp[90], double x[90], double y[90])
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c1 = 0; c1 <= 89; c1 += 32) {
      if (32 * b0 + t0 <= 89 && c1 == 0)
        y[32 * b0 + t0] = 0.;
      if (32 * b0 + t0 <= 89) {
        for (int c3 = 0; c3 <= ppcg_min(31, -c1 + 89); c3 += 1) {
          y[32 * b0 + t0] = ((B[32 * b0 + t0][c1 + c3] * x[c1 + c3]) + y[32 * b0 + t0]);
          if (c1 == 0 && c3 == 0)
            tmp[32 * b0 + t0] = 0.;
          tmp[32 * b0 + t0] = ((A[32 * b0 + t0][c1 + c3] * x[c1 + c3]) + tmp[32 * b0 + t0]);
        }
        if (c1 == 64)
          y[32 * b0 + t0] = ((alpha * tmp[32 * b0 + t0]) + (beta * y[32 * b0 + t0]));
      }
      __syncthreads();
    }
}
