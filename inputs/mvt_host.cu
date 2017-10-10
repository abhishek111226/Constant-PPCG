#include <assert.h>
#include <stdio.h>
#define HOSTCODE true 
#include "mvt_kernel.hu"
#define _PB_N 100
int x1[100];
int x2[100];
int y_1[100];
int y_2[100];
int A[100][100];
int n = 100;
void init_array()
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      x1[i] = (int) (i % n) ;
      x2[i] = (int) ((i + 1) % n) ;
      y_1[i] = (int) ((i + 3) % n) ;
      y_2[i] = (int) ((i + 4) % n);
      for (j = 0; j < n; j++)
	A[i][j] = (int) (i*j % n);
    }
}

int main()
{
  int i,j;
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
    int *dev_x1;
    int *dev_x2;
    
    cudaCheckReturn(cudaMalloc((void **) &dev_A, (100) * (100) * sizeof(int)));
    cudaCheckReturn(cudaMalloc((void **) &dev_x1, (100) * sizeof(int)));
    cudaCheckReturn(cudaMalloc((void **) &dev_x2, (100) * sizeof(int)));
    
    
    init_const_scop_0(y_1,y_2);
    
    cudaCheckReturn(cudaMemcpy(dev_A, A, (100) * (100) * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckReturn(cudaMemcpy(dev_x1, x1, (100) * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckReturn(cudaMemcpy(dev_x2, x2, (100) * sizeof(int), cudaMemcpyHostToDevice));
    {
      dim3 k0_dimBlock(32);
      dim3 k0_dimGrid(4);
      kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_A, dev_x1);
      cudaCheckKernel();
    }
    
    
    {
      dim3 k1_dimBlock(32);
      dim3 k1_dimGrid(4);
      kernel1 <<<k1_dimGrid, k1_dimBlock>>> (dev_A, dev_x2);
      cudaCheckKernel();
    }
    
    
    cudaCheckReturn(cudaMemcpy(x1, dev_x1, (100) * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheckReturn(cudaMemcpy(x2, dev_x2, (100) * sizeof(int), cudaMemcpyDeviceToHost));
    
    
    cudaCheckReturn(cudaFree(dev_A));
    cudaCheckReturn(cudaFree(dev_x1));
    cudaCheckReturn(cudaFree(dev_x2));
  }
}

/*enum RWbar 
{
	write,	0
	read,	1
	invalid,2 
	error,  3 
	none,   4
	read_inside_loop 5
}; */
