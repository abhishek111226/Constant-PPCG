#include <stdio.h> 
#define DEVICECODE true 
#include "chol_kernel.hu"
__global__ void kernel0(int *A, int c0)
{

    {
      for (int c1 = 0; c1 < c0; c1 += 1)
        A[c0 * 3 + c0] -= (A[c0 * 3 + c1] * A[c0 * 3 + c1]);
      if (c0 == 1)
        A[2 * 3 + 1] -= (A[2 * 3 + 0] * A[1 * 3 + 0]);
    }
}
__global__ void kernel1(int *A, int c0)
{

    A[c0 * 3 + c0] = A[c0 * 3 + c0];
}
__global__ void kernel2(int *A, int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    if (t0 >= c0 + 1)
      A[t0 * 3 + c0] /= A[c0 * 3 + c0];
}
void init_const_scop_0()
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




}

