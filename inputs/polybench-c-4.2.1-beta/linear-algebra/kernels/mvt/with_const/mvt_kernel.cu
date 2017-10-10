#include <stdio.h> 
#define DEVICECODE true 
#include "mvt_kernel.hu"
__global__ void kernel0(double A[120][120], double x1[120])
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c1 = 0; c1 <= 119; c1 += 32) {
      if (32 * b0 + t0 <= 119)
        for (int c3 = 0; c3 <= ppcg_min(31, -c1 + 119); c3 += 1)
          x1[32 * b0 + t0] = (x1[32 * b0 + t0] + (A[32 * b0 + t0][c1 + c3] * const_y_1[c1 + c3]));
      __syncthreads();
    }
}
__global__ void kernel1(double A[120][120], double x2[120])
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c1 = 0; c1 <= 119; c1 += 32) {
      if (32 * b0 + t0 <= 119)
        for (int c3 = 0; c3 <= ppcg_min(31, -c1 + 119); c3 += 1)
          x2[32 * b0 + t0] = (x2[32 * b0 + t0] + (A[c1 + c3][32 * b0 + t0] * const_y_2[c1 + c3]));
      __syncthreads();
    }
}
void init_const_scop_0(double y_1[120],double y_2[120])
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


cudaCheckReturn(cudaMemcpyToSymbol(const_y_1, y_1, (120) * sizeof(double)));
cudaCheckReturn(cudaMemcpyToSymbol(const_y_2, y_2, (120) * sizeof(double)));


}

