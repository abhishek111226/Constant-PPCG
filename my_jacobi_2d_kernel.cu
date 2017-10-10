#include <stdio.h> 
#define DEVICECODE true 
#include "my_jacobi_2d_kernel.hu"
__global__ void kernel0(int *A, int *B)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    if (t0 >= 1 && t1 >= 1)
      B[t0 * 4 + t1] = ((((A[t0 * 4 + t1] + A[t0 * 4 + (t1 - 1)]) + A[t0 * 4 + (t1 + 1)]) + A[(t0 + 1) * 4 + t1]) + A[(t0 - 1) * 4 + t1]);
}
__global__ void kernel1(int *A, int *B)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    if (t0 >= 1 && t1 >= 1)
      A[t0 * 4 + t1] = ((((B[t0 * 4 + t1] + B[t0 * 4 + (t1 - 1)]) + B[t0 * 4 + (t1 + 1)]) + B[(t0 + 1) * 4 + t1]) + B[(t0 - 1) * 4 + t1]);
}
