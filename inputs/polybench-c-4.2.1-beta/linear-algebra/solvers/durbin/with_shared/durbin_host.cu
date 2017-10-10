#include <stdio.h>
#define HOSTCODE true 
#include "durbin_kernel.hu"
/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* durbin.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "durbin.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_1D(r,N,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      r[i] = (n+1-i);
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(y,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("y");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, y[i]);
  }
  POLYBENCH_DUMP_END("y");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_durbin(int n,
		   DATA_TYPE POLYBENCH_1D(r,N,n),
		   DATA_TYPE POLYBENCH_1D(y,N,n))
{
 DATA_TYPE z[N];
 DATA_TYPE alpha;
 DATA_TYPE beta;
 DATA_TYPE sum;

 int i,k;

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

   float *dev_alpha;
   float *dev_beta;
   float *dev_r;
   float *dev_sum;
   float *dev_y;
   float *dev_z;
   
   cudaCheckReturn(cudaMalloc((void **) &dev_alpha, sizeof(float)));
   cudaCheckReturn(cudaMalloc((void **) &dev_beta, sizeof(float)));
   cudaCheckReturn(cudaMalloc((void **) &dev_r, (120) * sizeof(float)));
   cudaCheckReturn(cudaMalloc((void **) &dev_sum, sizeof(float)));
   cudaCheckReturn(cudaMalloc((void **) &dev_y, (120) * sizeof(float)));
   cudaCheckReturn(cudaMalloc((void **) &dev_z, (119) * sizeof(float)));
   
   
   beta = 1.F;
   alpha = (-r[0]);
   y[0] = (-r[0]);
   cudaCheckReturn(cudaMemcpy(dev_alpha, &alpha, sizeof(float), cudaMemcpyHostToDevice));
   cudaCheckReturn(cudaMemcpy(dev_beta, &beta, sizeof(float), cudaMemcpyHostToDevice));
   cudaCheckReturn(cudaMemcpy(dev_r, r, (120) * sizeof(float), cudaMemcpyHostToDevice));
   cudaCheckReturn(cudaMemcpy(dev_y, y, (120) * sizeof(float), cudaMemcpyHostToDevice));
   for (int c0 = 471; c0 <= 56639; c0 += 1) {
     if ((c0 - 117) % 472 == 0)
       {
         dim3 k0_dimBlock;
         dim3 k0_dimGrid;
         kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_r, dev_sum, dev_y, c0);
         cudaCheckKernel();
       }
       
       
     if (c0 <= 56049 && (c0 - 119) % 470 == 0)
       {
         dim3 k1_dimBlock;
         dim3 k1_dimGrid;
         kernel1 <<<k1_dimGrid, k1_dimBlock>>> (dev_alpha, dev_beta, c0);
         cudaCheckKernel();
       }
       
       
     if (c0 >= 943 && (c0 - 1) % 471 == 0) {
       {
         dim3 k2_dimBlock;
         dim3 k2_dimGrid;
         kernel2 <<<k2_dimGrid, k2_dimBlock>>> (dev_alpha, dev_y, c0);
         cudaCheckKernel();
       }
       
       
     } else if ((c0 - 235) % 472 == 0) {
       {
         dim3 k3_dimBlock;
         dim3 k3_dimGrid;
         kernel3 <<<k3_dimGrid, k3_dimBlock>>> (dev_alpha, dev_beta, dev_r, dev_sum, c0);
         cudaCheckKernel();
       }
       
       
     } else if (c0 <= 56049 && c0 % 471 == 0)
       {
         dim3 k4_dimBlock;
         dim3 k4_dimGrid;
         kernel4 <<<k4_dimGrid, k4_dimBlock>>> (dev_sum, c0);
         cudaCheckKernel();
       }
       
       
     if (c0 >= 943 && (c0 - 471) % 472 == 0) {
       {
         dim3 k5_dimBlock(32);
         dim3 k5_dimGrid(4);
         kernel5 <<<k5_dimGrid, k5_dimBlock>>> (dev_y, dev_z, c0);
         cudaCheckKernel();
       }
       
       
     } else if ((c0 - 353) % 472 == 0)
       {
         dim3 k6_dimBlock(32);
         dim3 k6_dimGrid(4);
         kernel6 <<<k6_dimGrid, k6_dimBlock>>> (dev_alpha, dev_y, dev_z, c0);
         cudaCheckKernel();
       }
       
       
   }
   cudaCheckReturn(cudaMemcpy(y, dev_y, (120) * sizeof(float), cudaMemcpyDeviceToHost));
   
   
   cudaCheckReturn(cudaFree(dev_alpha));
   cudaCheckReturn(cudaFree(dev_beta));
   cudaCheckReturn(cudaFree(dev_r));
   cudaCheckReturn(cudaFree(dev_sum));
   cudaCheckReturn(cudaFree(dev_y));
   cudaCheckReturn(cudaFree(dev_z));
 }

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_1D_ARRAY_DECL(r, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(r));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_durbin (n,
		 POLYBENCH_ARRAY(r),
		 POLYBENCH_ARRAY(y));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(y)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(r);
  POLYBENCH_FREE_ARRAY(y);

  return 0;
}
