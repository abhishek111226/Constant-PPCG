#include <stdio.h> 
#define DEVICECODE true 
#include "ludcmp_kernel.hu"
__global__ void kernel0(double A[120][120], int c0)
{
    double private_w;

    for (int c1 = 0; c1 < c0; c1 += 1) {
      private_w = A[c0][c1];
      for (int c2 = 0; c2 < c1; c2 += 1)
        private_w -= (A[c0][c2] * A[c2][c1]);
      A[c0][c1] = (private_w / A[c1][c1]);
    }
}
__global__ void kernel1(double A[120][120], int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    double private_w;

    if (32 * b0 + t0 <= 119 && 32 * b0 + t0 >= c0) {
      private_w = A[c0][32 * b0 + t0];
      for (int c3 = 0; c3 < c0; c3 += 1)
        private_w -= (A[c0][c3] * A[c3][32 * b0 + t0]);
      A[c0][32 * b0 + t0] = private_w;
    }
}
__global__ void kernel2(double A[120][120], double y[120])
{
    double private_w;

    for (int c0 = 0; c0 <= 119; c0 += 1) {
      private_w = const_b[c0];
      for (int c1 = 0; c1 < c0; c1 += 1)
        private_w -= (A[c0][c1] * y[c1]);
      y[c0] = private_w;
    }
}
__global__ void kernel3(double A[120][120], double x[120], double y[120])
{
    double private_w;

    for (int c0 = -119; c0 <= 0; c0 += 1) {
      private_w = y[-c0];
      for (int c1 = -c0 + 1; c1 <= 119; c1 += 1)
        private_w -= (A[-c0][c1] * x[c1]);
      x[-c0] = (private_w / A[-c0][-c0]);
    }
}
void init_const_scop_0(double b[120])
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


cudaCheckReturn(cudaMemcpyToSymbol(const_b, b, (120) * sizeof(double)));


}

