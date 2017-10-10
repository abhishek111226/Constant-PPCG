#include <stdio.h> 
#define DEVICECODE true 
#include "gemver_kernel.hu"
__global__ void kernel0(float A[120][120], float u1[120], float u2[120], float v1[120], float v2[120])
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    __shared__ float shared_u1[32];
    __shared__ float shared_u2[32];
    __shared__ float shared_v1[32];
    __shared__ float shared_v2[32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    {
      if (t0 == 0) {
        for (int c0 = t1; c0 <= ppcg_min(31, -32 * b0 + 119); c0 += 16)
          shared_u1[c0] = u1[32 * b0 + c0];
        for (int c0 = t1; c0 <= ppcg_min(31, -32 * b0 + 119); c0 += 16)
          shared_u2[c0] = u2[32 * b0 + c0];
        for (int c0 = t1; c0 <= ppcg_min(31, -32 * b1 + 119); c0 += 16)
          shared_v1[c0] = v1[32 * b1 + c0];
        for (int c0 = t1; c0 <= ppcg_min(31, -32 * b1 + 119); c0 += 16)
          shared_v2[c0] = v2[32 * b1 + c0];
      }
      __syncthreads();
      if (32 * b0 + t0 <= 119)
        for (int c3 = t1; c3 <= ppcg_min(31, -32 * b1 + 119); c3 += 16)
          A[32 * b0 + t0][32 * b1 + c3] = ((A[32 * b0 + t0][32 * b1 + c3] + (shared_u1[t0] * shared_v1[c3])) + (shared_u2[t0] * shared_v2[c3]));
    }
}
__global__ void kernel1(float A[120][120], float beta, float x[120], float y[120], float z[120])
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ float shared_x[32];
    __shared__ float shared_y[32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    {
      if (32 * b0 + t0 <= 119)
        shared_x[t0] = x[32 * b0 + t0];
      __syncthreads();
      for (int c1 = 0; c1 <= 119; c1 += 32) {
        if (t0 + c1 <= 119)
          shared_y[t0] = y[t0 + c1];
        __syncthreads();
        if (32 * b0 + t0 <= 119) {
          for (int c3 = 0; c3 <= ppcg_min(31, -c1 + 119); c3 += 1)
            shared_x[t0] = (shared_x[t0] + ((beta * A[c1 + c3][32 * b0 + t0]) * shared_y[c3]));
          if (c1 == 96)
            shared_x[t0] = (shared_x[t0] + z[32 * b0 + t0]);
        }
        __syncthreads();
      }
      if (32 * b0 + t0 <= 119)
        x[32 * b0 + t0] = shared_x[t0];
    }
}
__global__ void kernel2(float A[120][120], float alpha, float w[120], float x[120])
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ float shared_A[32][32];
    __shared__ float shared_w[32];
    __shared__ float shared_x[32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    {
      if (32 * b0 + t0 <= 119)
        shared_w[t0] = w[32 * b0 + t0];
      for (int c1 = 0; c1 <= 119; c1 += 32) {
        if (t0 + c1 <= 119) {
          for (int c2 = 0; c2 <= ppcg_min(31, -32 * b0 + 119); c2 += 1)
            shared_A[c2][t0] = A[32 * b0 + c2][t0 + c1];
          shared_x[t0] = x[t0 + c1];
        }
        __syncthreads();
        if (32 * b0 + t0 <= 119)
          for (int c3 = 0; c3 <= ppcg_min(31, -c1 + 119); c3 += 1)
            shared_w[t0] = (shared_w[t0] + ((alpha * shared_A[t0][c3]) * shared_x[c3]));
        __syncthreads();
      }
      if (32 * b0 + t0 <= 119)
        w[32 * b0 + t0] = shared_w[t0];
    }
}
