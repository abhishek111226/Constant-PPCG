#include <stdio.h> 
#define DEVICECODE true 
#include "fdtd-2d_kernel.hu"
__global__ void kernel0(double _fict_[40], double ey[60][80], int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    if (32 * b0 + t0 <= 79)
      ey[0][32 * b0 + t0] = _fict_[c0];
}
__global__ void kernel1(double ey[60][80], double hz[60][80], int c0)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    if (32 * b0 + t0 >= 1 && 32 * b0 + t0 <= 59)
      for (int c4 = t1; c4 <= ppcg_min(31, -32 * b1 + 79); c4 += 16)
        ey[32 * b0 + t0][32 * b1 + c4] = (ey[32 * b0 + t0][32 * b1 + c4] - (0.5 * (hz[32 * b0 + t0][32 * b1 + c4] - hz[32 * b0 + t0 - 1][32 * b1 + c4])));
}
__global__ void kernel2(double ex[60][80], double hz[60][80], int c0)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    if (32 * b0 + t0 <= 59)
      for (int c4 = ppcg_max(t1, ((t1 + 15) % 16) - 32 * b1 + 1); c4 <= ppcg_min(31, -32 * b1 + 79); c4 += 16)
        ex[32 * b0 + t0][32 * b1 + c4] = (ex[32 * b0 + t0][32 * b1 + c4] - (0.5 * (hz[32 * b0 + t0][32 * b1 + c4] - hz[32 * b0 + t0][32 * b1 + c4 - 1])));
}
__global__ void kernel3(double ex[60][80], double ey[60][80], double hz[60][80], int c0)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    if (32 * b0 + t0 <= 58)
      for (int c4 = t1; c4 <= ppcg_min(31, -32 * b1 + 78); c4 += 16)
        hz[32 * b0 + t0][32 * b1 + c4] = (hz[32 * b0 + t0][32 * b1 + c4] - (0.69999999999999996 * (((ex[32 * b0 + t0][32 * b1 + c4 + 1] - ex[32 * b0 + t0][32 * b1 + c4]) + ey[32 * b0 + t0 + 1][32 * b1 + c4]) - ey[32 * b0 + t0][32 * b1 + c4])));
}
