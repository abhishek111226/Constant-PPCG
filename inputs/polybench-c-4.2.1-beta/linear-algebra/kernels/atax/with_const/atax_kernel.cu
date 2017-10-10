#include <stdio.h> 
#define DEVICECODE true 
#include "atax_kernel.hu"
__global__ void kernel0(double A[116][124], double tmp[116])
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c1 = 0; c1 <= 123; c1 += 32) {
      if (32 * b0 + t0 <= 115 && c1 == 0)
        tmp[32 * b0 + t0] = 0.;
      if (32 * b0 + t0 <= 115)
        for (int c3 = 0; c3 <= ppcg_min(31, -c1 + 123); c3 += 1)
          tmp[32 * b0 + t0] = (tmp[32 * b0 + t0] + (A[32 * b0 + t0][c1 + c3] * const_x[c1 + c3]));
      __syncthreads();
    }
}
__global__ void kernel1(double A[116][124], double tmp[116], double y[124])
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c1 = 0; c1 <= 115; c1 += 32) {
      if (32 * b0 + t0 <= 123 && c1 == 0)
        y[32 * b0 + t0] = 0;
      if (32 * b0 + t0 <= 123)
        for (int c3 = 0; c3 <= ppcg_min(31, -c1 + 115); c3 += 1)
          y[32 * b0 + t0] = (y[32 * b0 + t0] + (A[c1 + c3][32 * b0 + t0] * tmp[c1 + c3]));
      __syncthreads();
    }
}
void init_const_scop_0(double x[124])
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


cudaCheckReturn(cudaMemcpyToSymbol(const_x, x, (124) * sizeof(double)));


}

