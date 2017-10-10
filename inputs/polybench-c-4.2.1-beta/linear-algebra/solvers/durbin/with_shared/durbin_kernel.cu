#include <stdio.h> 
#define DEVICECODE true 
#include "durbin_kernel.hu"
__global__ void kernel0(float r[120], float *sum, float y[120], int c0)
{
    float private_sum;

    {
      private_sum = *sum;
      for (int c1 = 0; c1 < (c0 - 117) / 472; c1 += 1)
        private_sum += (r[((c0 - 589) / 472) - c1] * y[c1]);
      *sum = private_sum;
    }
}
__global__ void kernel1(float *alpha, float *beta, int c0)
{

    beta[0] = ((1 - (alpha[0] * alpha[0])) * beta[0]);
}
__global__ void kernel2(float *alpha, float y[120], int c0)
{

    y[(c0 - 472) / 471] = alpha[0];
}
__global__ void kernel3(float *alpha, float *beta, float r[120], float *sum, int c0)
{

    alpha[0] = ((-(r[(c0 - 235) / 472] + sum[0])) / beta[0]);
}
__global__ void kernel4(float *sum, int c0)
{

    sum[0] = 0.F;
}
__global__ void kernel5(float y[120], float z[119], int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    if (c0 >= 15104 * b0 + 472 * t0 + 943)
      y[32 * b0 + t0] = z[32 * b0 + t0];
}
__global__ void kernel6(float *alpha, float y[120], float z[119], int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ float shared_alpha;
    __shared__ float shared_y_1[32];

    {
      if (t0 == 0 && c0 >= 15104 * b0 + 825)
        shared_alpha = *alpha;
      if (472 * t0 + c0 >= 15104 * b0 + 15457)
        shared_y_1[t0] = y[((c0 - 15457) / 472) - 32 * b0 + t0];
      __syncthreads();
      if (c0 >= 15104 * b0 + 472 * t0 + 825)
        z[32 * b0 + t0] = (y[32 * b0 + t0] + (shared_alpha * shared_y_1[-t0 + 31]));
    }
}
