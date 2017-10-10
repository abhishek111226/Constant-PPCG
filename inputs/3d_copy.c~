//#include <stdio.h>
int A[3][3][3];
int B[3][3][3];
int main()
{
 int i,j,k; 	
 #pragma scop
 for(i=0;i<3;i++)
 {
 	for(j=0;j<3;j++)
 	{
		 	for(k=0;k<3;k++)
		 	{
				B[i][j][k]=A[i][j][k];
			}
	}
 }
 #pragma endscop
  return 0;
}

