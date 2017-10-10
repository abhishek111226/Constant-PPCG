#include "../../common/common.hpp"
#include "../../common/timer.hpp"
#include <cassert>
#include <cstdio>

extern "C" void j3d13pt(float*, int, int, int, float*);

extern "C" void j3d13pt_gold(float*, int, int, int, float*);

int main(int argc, char** argv) {
  int height, width_y, width_x;
  if (argc == 4) {
    height = atoi(argv[1]);
    width_y = atoi(argv[2]);
    width_x = atoi(argv[3]);
  }
  else {
    height = 60;
    width_y = 1536;
    width_x = 1536;
  }

  float (*input)[width_y][width_x] = (float (*)[width_y][width_x])
    getRandom3DArray<float>(height, width_y, width_x);
  float (*output)[width_y][width_x] = (float (*)[width_y][width_x])
    getZero3DArray<float>(height, width_y, width_x);
  float (*output_gold)[width_y][width_x] = (float (*)[width_y][width_x])
    getZero3DArray<float>(height, width_y, width_x);

  j3d13pt((float*)input, height, width_y, width_x, (float*)output);

  j3d13pt_gold((float*)input, height, width_y, width_x, (float*)output_gold);

#ifdef PRINT_OUTPUT
  printf("Output :\n");
  print3DArray<float>
    (width_y, width_x, (float*)output, 6, height-6, 6, width_y-6, 6,
     width_x-6);
  printf("\nOutput Gold:\n");
  print3DArray<float>
    (width_y, width_x, (float*)output_gold, 6, height-6, 6, width_y-6, 6,
     width_x-6);
#endif

  double error =
    checkError3D<float>
    (width_y, width_x, (float*)output, (float*) output_gold, 6, height-6, 6,
     width_y-6, 6, width_x-6);
  printf("[Test] RMS Error : %e\n",error);
  if (error > TOLERANCE)
    return -1;

  double time = 0.0;
  for (int i = 0; i < NUMRUNS; i++) {
    memset((void*)output, 0, sizeof(float)*height*width_y*width_x);
    startTimer();
    j3d13pt((float*)input, height, width_y, width_x, (float*)output);
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
