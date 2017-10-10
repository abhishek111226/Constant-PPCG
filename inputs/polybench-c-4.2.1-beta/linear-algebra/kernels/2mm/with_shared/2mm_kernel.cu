#include <stdio.h> 
#define DEVICECODE true 
#include "2mm_kernel.hu"
__global__ void kernel0(float A[40][70], float B[70][50], float alpha, float tmp[40][50])
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    __shared__ float shared_A[32][32];
    __shared__ float shared_B[32][32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c2 = 0; c2 <= 69; c2 += 32) {
      if (32 * b0 + t0 <= 39)
        for (int c4 = t1; c4 <= ppcg_min(31, -c2 + 69); c4 += 16)
          shared_A[t0][c4] = A[32 * b0 + t0][c2 + c4];
      if (t0 + c2 <= 69)
        for (int c4 = t1; c4 <= ppcg_min(31, -32 * b1 + 49); c4 += 16)
          shared_B[t0][c4] = B[t0 + c2][32 * b1 + c4];
      __syncthreads();
      if (32 * b0 + t0 <= 39)
        for (int c4 = t1; c4 <= ppcg_min(31, -32 * b1 + 49); c4 += 16) {
          if (c2 == 0)
            tmp[32 * b0 + t0][32 * b1 + c4] = 0.F;
          for (int c5 = 0; c5 <= ppcg_min(31, -c2 + 69); c5 += 1)
            tmp[32 * b0 + t0][32 * b1 + c4] += ((alpha * shared_A[t0][c5]) * shared_B[c5][c4]);
        }
      __syncthreads();
    }
}
__global__ void kernel1(float C[50][80], float D[40][80], float beta, float tmp[40][50])
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    __shared__ float shared_C[32][32];
    __shared__ float shared_D[32][32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    {
      if (32 * b0 + t0 <= 39)
        for (int c1 = t1; c1 <= ppcg_min(31, -32 * b1 + 79); c1 += 16)
          shared_D[t0][c1] = D[32 * b0 + t0][32 * b1 + c1];
      for (int c2 = 0; c2 <= 49; c2 += 32) {
        if (t0 + c2 <= 49)
          for (int c4 = t1; c4 <= ppcg_min(31, -32 * b1 + 79); c4 += 16)
            shared_C[t0][c4] = C[t0 + c2][32 * b1 + c4];
        __syncthreads();
        if (32 * b0 + t0 <= 39)
          for (int c4 = t1; c4 <= ppcg_min(31, -32 * b1 + 79); c4 += 16) {
            if (c2 == 0)
              shared_D[t0][c4] *= beta;
            for (int c5 = 0; c5 <= ppcg_min(31, -c2 + 49); c5 += 1)
              shared_D[t0][c4] += (tmp[32 * b0 + t0][c2 + c5] * shared_C[c5][c4]);
          }
        __syncthreads();
      }
      if (32 * b0 + t0 <= 39)
        for (int c1 = t1; c1 <= ppcg_min(31, -32 * b1 + 79); c1 += 16)
          D[32 * b0 + t0][32 * b1 + c1] = shared_D[t0][c1];
    }
}
