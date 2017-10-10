#include <stdio.h> 
#define DEVICECODE true 
#include "durbin_kernel.hu"
__global__ void kernel0(double *sum, double y[120], int c0)
{
    double private_sum;

    {
      private_sum = *sum;
      for (int c1 = 0; c1 < (c0 - 117) / 472; c1 += 1)
        private_sum += (const_r[((c0 - 589) / 472) - c1] * y[c1]);
      *sum = private_sum;
    }
}
__global__ void kernel1(double *alpha, double *beta, int c0)
{

    beta[0] = ((1 - (alpha[0] * alpha[0])) * beta[0]);
}
__global__ void kernel2(double *alpha, double y[120], int c0)
{

    y[(c0 - 472) / 471] = alpha[0];
}
__global__ void kernel3(double *alpha, double *beta, double *sum, int c0)
{

    alpha[0] = ((-(const_r[(c0 - 235) / 472] + sum[0])) / beta[0]);
}
__global__ void kernel4(double *sum, int c0)
{

    sum[0] = 0.;
}
__global__ void kernel5(double y[120], double z[119], int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    if (c0 >= 15104 * b0 + 472 * t0 + 943)
      y[32 * b0 + t0] = z[32 * b0 + t0];
}
__global__ void kernel6(double *alpha, double y[120], double z[119], int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    if (c0 >= 15104 * b0 + 472 * t0 + 825)
      z[32 * b0 + t0] = (y[32 * b0 + t0] + (alpha[0] * y[((c0 - 825) / 472) - 32 * b0 - t0]));
}
void init_const_scop_0(double r[120])
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


cudaCheckReturn(cudaMemcpyToSymbol(const_r, r, (120) * sizeof(double)));


}

