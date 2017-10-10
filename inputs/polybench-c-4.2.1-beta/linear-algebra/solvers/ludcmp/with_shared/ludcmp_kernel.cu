#include <stdio.h> 
#define DEVICECODE true 
#include "ludcmp_kernel.hu"
__global__ void kernel0(float A[120][120], int c0)
{
    float private_w;

    for (int c1 = 0; c1 < c0; c1 += 1) {
      private_w = A[c0][c1];
      for (int c2 = 0; c2 < c1; c2 += 1)
        private_w -= (A[c0][c2] * A[c2][c1]);
      A[c0][c1] = (private_w / A[c1][c1]);
    }
}
__global__ void kernel1(float A[120][120], int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ float shared_A_0[1][32];
    __shared__ float shared_A_1[1][119];
    float private_w;

    {
      if (32 * b0 + t0 <= 119)
        shared_A_0[0][t0] = A[c0][32 * b0 + t0];
      for (int c2 = t0; c2 <= 118; c2 += 32)
        shared_A_1[0][c2] = A[c0][c2];
      __syncthreads();
      if (32 * b0 + t0 <= 119 && 32 * b0 + t0 >= c0) {
        private_w = shared_A_0[0][t0];
        for (int c3 = 0; c3 < c0; c3 += 1)
          private_w -= (shared_A_1[0][c3] * A[c3][32 * b0 + t0]);
        shared_A_0[0][t0] = private_w;
      }
      __syncthreads();
      if (32 * b0 + t0 <= 119 && 32 * b0 + t0 >= c0)
        A[c0][32 * b0 + t0] = shared_A_0[0][t0];
    }
}
__global__ void kernel2(float A[120][120], float b[120], float y[120])
{
    float private_w;

    for (int c0 = 0; c0 <= 119; c0 += 1) {
      private_w = b[c0];
      for (int c1 = 0; c1 < c0; c1 += 1)
        private_w -= (A[c0][c1] * y[c1]);
      y[c0] = private_w;
    }
}
__global__ void kernel3(float A[120][120], float x[120], float y[120])
{
    float private_w;

    for (int c0 = -119; c0 <= 0; c0 += 1) {
      private_w = y[-c0];
      for (int c1 = -c0 + 1; c1 <= 119; c1 += 1)
        private_w -= (A[-c0][c1] * x[c1]);
      x[-c0] = (private_w / A[-c0][-c0]);
    }
}
