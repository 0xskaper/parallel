#include <stdio.h>
__global__ void fromGPU(void) { printf("From GPU!"); }

int main(void) {
  printf("from CPU!");
  fromGPU<<<1, 10>>>();
  cudaDeviceReset();
  return 0;
}
