#include <stdio.h> 
#define DEVICECODE true 
#include "3mm_kernel.hu"
__global__ void kernel0(double A[40][60], double E[40][50])
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c2 = 0; c2 <= 59; c2 += 32) {
      if (32 * b0 + t0 <= 39)
        for (int c4 = t1; c4 <= ppcg_min(31, -32 * b1 + 49); c4 += 16) {
          if (c2 == 0)
            E[32 * b0 + t0][32 * b1 + c4] = 0.;
          for (int c5 = 0; c5 <= ppcg_min(31, -c2 + 59); c5 += 1)
            E[32 * b0 + t0][32 * b1 + c4] += (A[32 * b0 + t0][c2 + c5] * const_B[c2 + c5][32 * b1 + c4]);
        }
      __syncthreads();
    }
}
__global__ void kernel1(double C[50][80], double F[50][70])
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c2 = 0; c2 <= 79; c2 += 32) {
      if (32 * b0 + t0 <= 49)
        for (int c4 = t1; c4 <= ppcg_min(31, -32 * b1 + 69); c4 += 16) {
          if (c2 == 0)
            F[32 * b0 + t0][32 * b1 + c4] = 0.;
          for (int c5 = 0; c5 <= ppcg_min(31, -c2 + 79); c5 += 1)
            F[32 * b0 + t0][32 * b1 + c4] += (C[32 * b0 + t0][c2 + c5] * const_D[c2 + c5][32 * b1 + c4]);
        }
      __syncthreads();
    }
}
__global__ void kernel2(double E[40][50], double F[50][70], double G[40][70])
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c2 = 0; c2 <= 49; c2 += 32) {
      if (32 * b0 + t0 <= 39)
        for (int c4 = t1; c4 <= ppcg_min(31, -32 * b1 + 69); c4 += 16) {
          if (c2 == 0)
            G[32 * b0 + t0][32 * b1 + c4] = 0.;
          for (int c5 = 0; c5 <= ppcg_min(31, -c2 + 49); c5 += 1)
            G[32 * b0 + t0][32 * b1 + c4] += (E[32 * b0 + t0][c2 + c5] * F[c2 + c5][32 * b1 + c4]);
        }
      __syncthreads();
    }
}
void init_const_scop_0(double B[60][50],double D[80][70])
{
#define cudaCheckReturn(ret) \
  do { \
    cudaError_t cudaCheckReturn_e = (ret); \
    if (cudaCheckReturn_e != cudaSuccess) { \
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaCheckReturn_e)); \
      fflush(stderr); \
    } \
    assert(cudaCheckReturn_e == cudaSuccess); \
  } while(0)
#define cudaCheckKernel() \
  do { \
    cudaCheckReturn(cudaGetLastError()); \
  } while(0)


cudaCheckReturn(cudaMemcpyToSymbol(const_B, B, (60) * (50) * sizeof(double)));
cudaCheckReturn(cudaMemcpyToSymbol(const_D, D, (80) * (70) * sizeof(double)));


}

