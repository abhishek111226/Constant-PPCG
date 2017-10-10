#include <stdio.h> 
#define DEVICECODE true 
#include "trmm_kernel.hu"
__global__ void kernel0(double B[60][80])
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    for (int c1 = 0; c1 <= 58; c1 += 32)
      for (int c2 = c1; c2 <= 59; c2 += 32) {
        if (32 * b0 + t0 <= 79)
          for (int c4 = 0; c4 <= ppcg_min(ppcg_min(31, -c1 + 58), -c1 + c2 + 30); c4 += 1)
            for (int c5 = ppcg_max(0, c1 - c2 + c4 + 1); c5 <= ppcg_min(31, -c2 + 59); c5 += 1)
              B[c1 + c4][32 * b0 + t0] += (const_A[c2 + c5][c1 + c4] * B[c2 + c5][32 * b0 + t0]);
        __syncthreads();
      }
}
__global__ void kernel1(double B[60][80], double alpha)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    if (32 * b0 + t0 <= 59)
      for (int c3 = t1; c3 <= ppcg_min(31, -32 * b1 + 79); c3 += 16)
        B[32 * b0 + t0][32 * b1 + c3] = (alpha * B[32 * b0 + t0][32 * b1 + c3]);
}
void init_const_scop_0(double A[60][60])
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


cudaCheckReturn(cudaMemcpyToSymbol(const_A, A, (60) * (60) * sizeof(double)));


}

