#include "../../common/common.hpp"
#include "../../common/timer.hpp"
#include <cassert>
#include <cstdio>

extern "C" void j3d7pt(float*, int, int, int, float*);

extern "C" void j3d7pt_gold(float*, int, int, int, float*);

int main(int argc, char** argv) {
  int height, width_y, width_x;
  if (argc == 4) {
    height = atoi(argv[1]);
    width_y = atoi(argv[2]);
    width_x = atoi(argv[3]);
  }
  else {
    height = 13;
    width_y = 2016;
    width_x = 2016;
  }

  float (*input)[width_y][width_x] = (float (*)[width_y][width_x])
    getRandom3DArray<float>(height, width_y, width_x);
  float (*output)[width_y][width_x] = (float (*)[width_y][width_x])
    getZero3DArray<float>(height, width_y, width_x);
  float (*output_gold)[width_y][width_x] = (float (*)[width_y][width_x])
    getZero3DArray<float>(height, width_y, width_x);

  j3d7pt((float*)input, height, width_y, width_x, (float*)output);

  j3d7pt_gold((float*)input, height, width_y, width_x, (float*)output_gold);

#ifdef PRINT_OUTPUT
  printf("Output :\n");
  print3DArray<float>
    (width_y, width_x, (float*)output, 8, height-8, 8, width_y-8, 8,
     width_x-8);
  printf("\nOutput Gold:\n");
  print3DArray<float>
    (width_y, width_x, (float*)output_gold, 8, height-8, 8, width_y-8 , 8,
     width_x-8);
#endif

  double error =
    checkError3D<float>
    (width_y, width_x, (float*)output, (float*) output_gold, 8, height-8, 8,
     width_y-8, 8, width_x-8);
  printf("[Test] RMS Error : %e\n",error);
  if (error > TOLERANCE)
    return -1;

  double time = 0.0;
  for (int i = 0; i < 0; i++) {
    memset((void*)output, 0, sizeof(float)*height*width_y*width_x);
    startTimer();
    j3d7pt((float*)input, height, width_y, width_x, (float*)output);
    stopTimer();
    time += getElapsedTime();
  }
  printf
    ("[Test] Elapsed Time (average of %d runs) : %e (ms)\n", NUMRUNS,
     time/NUMRUNS);

  delete[] input;
  delete[] output;
  delete[] output_gold;
}
