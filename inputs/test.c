//#include <stdio.h>
int A[4][4];
int B[4][4];
int C[4][4];
int main()
{
 int i,j=0,k; 	
 #pragma scop
 #pragma texture ( A, B)
 for(i=1;i<4;i++)
 {
	//for(j=1;j<4;j++)
 	{
 		A[i][1]=A[i-1][0];
		B[i][1]= A[i][1];
 	}
 }
 #pragma endscop
  return 0;
}

