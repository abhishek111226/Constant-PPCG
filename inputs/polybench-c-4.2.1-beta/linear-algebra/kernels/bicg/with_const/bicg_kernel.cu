#include <stdio.h> 
#define DEVICECODE true 
#include "bicg_kernel.hu"
__global__ void kernel0(double A[124][116], double s[116])
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c1 = 0; c1 <= 123; c1 += 32) {
      if (32 * b0 + t0 <= 115 && c1 == 0)
        s[32 * b0 + t0] = 0;
      if (32 * b0 + t0 <= 115)
        for (int c3 = 0; c3 <= ppcg_min(31, -c1 + 123); c3 += 1)
          s[32 * b0 + t0] = (s[32 * b0 + t0] + (const_r[c1 + c3] * A[c1 + c3][32 * b0 + t0]));
      __syncthreads();
    }
}
__global__ void kernel1(double A[124][116], double q[124])
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c1 = 0; c1 <= 115; c1 += 32) {
      if (32 * b0 + t0 <= 123 && c1 == 0)
        q[32 * b0 + t0] = 0.;
      if (32 * b0 + t0 <= 123)
        for (int c3 = 0; c3 <= ppcg_min(31, -c1 + 115); c3 += 1)
          q[32 * b0 + t0] = (q[32 * b0 + t0] + (A[32 * b0 + t0][c1 + c3] * const_p[c1 + c3]));
      __syncthreads();
    }
}
void init_const_scop_0(double p[116],double r[124])
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


cudaCheckReturn(cudaMemcpyToSymbol(const_p, p, (116) * sizeof(double)));
cudaCheckReturn(cudaMemcpyToSymbol(const_r, r, (124) * sizeof(double)));


}

