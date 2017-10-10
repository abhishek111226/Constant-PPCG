#include <assert.h>
#include <stdio.h>
#define HOSTCODE true 
#include "my_jacobi_2d_kernel.hu"
//#include<stdio.h>
int A[4][4];
int B[4][4];
int main()
{	

	int t, i, j,tm,tj;
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

	    int *dev_A;
	    int *dev_B;
	    
	    cudaCheckReturn(cudaMalloc((void **) &dev_A, (4) * (4) * sizeof(int)));
	    cudaCheckReturn(cudaMalloc((void **) &dev_B, (4) * (4) * sizeof(int)));
	    
	    
	    
	    cudaCheckReturn(cudaMemcpy(dev_A, A, (4) * (4) * sizeof(int), cudaMemcpyHostToDevice));
	    cudaCheckReturn(cudaMemcpy(dev_B, B, (4) * (4) * sizeof(int), cudaMemcpyHostToDevice));
	    {
	      dim3 k0_dimBlock(3, 3);
	      dim3 k0_dimGrid(1, 1);
	      kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_A, dev_B);
	      cudaCheckKernel();
	    }
	    
	    
	    {
	      dim3 k1_dimBlock(3, 3);
	      dim3 k1_dimGrid(1, 1);
	      kernel1 <<<k1_dimGrid, k1_dimBlock>>> (dev_A, dev_B);
	      cudaCheckKernel();
	    }
	    
	    
	    cudaCheckReturn(cudaMemcpy(A, dev_A, (4) * (4) * sizeof(int), cudaMemcpyDeviceToHost));
	    cudaCheckReturn(cudaMemcpy(B, dev_B, (4) * (4) * sizeof(int), cudaMemcpyDeviceToHost));
	    
	    
	    cudaCheckReturn(cudaFree(dev_A));
	    cudaCheckReturn(cudaFree(dev_B));
	  }

  return 0;
}
