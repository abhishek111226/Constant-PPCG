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
  #pragma scop	
  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      x1[i] = x1[i] + A[i][j] * y_1[j];
  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      x2[i] = x2[i] + A[j][i] * y_2[j];
  #pragma endscop
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
