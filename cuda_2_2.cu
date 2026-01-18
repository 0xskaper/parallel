#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char **argv) {
  int N;
  dim3 block(1024);
  dim3 grid((N + block.x - 1) / block.x);
  printf("Block Size -> %d || Grid Size -> %d", block.x, grid.x);
  block.x = 512;
  grid.x = ((N + block.x - 1) / block.x);
  printf("Block Size -> %d || Grid Size -> %d", block.x, grid.x);
  block.x = 256;
  grid.x = ((N + block.x - 1) / block.x);
  printf("Block Size -> %d || Grid Size -> %d", block.x, grid.x);
  block.x = 128;
  grid.x = ((N + block.x - 1) / block.x);
  printf("Block Size -> %d || Grid Size -> %d", block.x, grid.x);

  cudaDeviceReset();
  return 0;
}
