#include <stdio.h> 
#define DEVICECODE true 
#include "3mm_kernel.hu"
__global__ void kernel0(float A[40][60], float B[60][50], float E[40][50])
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    __shared__ float shared_A[32][32];
    __shared__ float shared_B[32][32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c2 = 0; c2 <= 59; c2 += 32) {
      if (32 * b0 + t0 <= 39)
        for (int c4 = t1; c4 <= ppcg_min(31, -c2 + 59); c4 += 16)
          shared_A[t0][c4] = A[32 * b0 + t0][c2 + c4];
      if (t0 + c2 <= 59)
        for (int c4 = t1; c4 <= ppcg_min(31, -32 * b1 + 49); c4 += 16)
          shared_B[t0][c4] = B[t0 + c2][32 * b1 + c4];
      __syncthreads();
      if (32 * b0 + t0 <= 39)
        for (int c4 = t1; c4 <= ppcg_min(31, -32 * b1 + 49); c4 += 16) {
          if (c2 == 0)
            E[32 * b0 + t0][32 * b1 + c4] = 0.F;
          for (int c5 = 0; c5 <= ppcg_min(31, -c2 + 59); c5 += 1)
            E[32 * b0 + t0][32 * b1 + c4] += (shared_A[t0][c5] * shared_B[c5][c4]);
        }
      __syncthreads();
    }
}
__global__ void kernel1(float C[50][80], float D[80][70], float F[50][70])
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    __shared__ float shared_C[32][32];
    __shared__ float shared_D[32][32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c2 = 0; c2 <= 79; c2 += 32) {
      if (32 * b0 + t0 <= 49)
        for (int c4 = t1; c4 <= ppcg_min(31, -c2 + 79); c4 += 16)
          shared_C[t0][c4] = C[32 * b0 + t0][c2 + c4];
      if (t0 + c2 <= 79)
        for (int c4 = t1; c4 <= ppcg_min(31, -32 * b1 + 69); c4 += 16)
          shared_D[t0][c4] = D[t0 + c2][32 * b1 + c4];
      __syncthreads();
      if (32 * b0 + t0 <= 49)
        for (int c4 = t1; c4 <= ppcg_min(31, -32 * b1 + 69); c4 += 16) {
          if (c2 == 0)
            F[32 * b0 + t0][32 * b1 + c4] = 0.F;
          for (int c5 = 0; c5 <= ppcg_min(31, -c2 + 79); c5 += 1)
            F[32 * b0 + t0][32 * b1 + c4] += (shared_C[t0][c5] * shared_D[c5][c4]);
        }
      __syncthreads();
    }
}
__global__ void kernel2(float E[40][50], float F[50][70], float G[40][70])
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    __shared__ float shared_E[32][32];
    __shared__ float shared_F[32][32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c2 = 0; c2 <= 49; c2 += 32) {
      if (32 * b0 + t0 <= 39)
        for (int c4 = t1; c4 <= ppcg_min(31, -c2 + 49); c4 += 16)
          shared_E[t0][c4] = E[32 * b0 + t0][c2 + c4];
      if (t0 + c2 <= 49)
        for (int c4 = t1; c4 <= ppcg_min(31, -32 * b1 + 69); c4 += 16)
          shared_F[t0][c4] = F[t0 + c2][32 * b1 + c4];
      __syncthreads();
      if (32 * b0 + t0 <= 39)
        for (int c4 = t1; c4 <= ppcg_min(31, -32 * b1 + 69); c4 += 16) {
          if (c2 == 0)
            G[32 * b0 + t0][32 * b1 + c4] = 0.F;
          for (int c5 = 0; c5 <= ppcg_min(31, -c2 + 49); c5 += 1)
            G[32 * b0 + t0][32 * b1 + c4] += (shared_E[t0][c5] * shared_F[c5][c4]);
        }
      __syncthreads();
    }
}
