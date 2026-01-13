#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

__global__ void checkIndex(void) {
  printf("THREAD IDX: (%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z);
  printf("BLOCK IDX: (%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z);
  printf("BLOCK DIMENTION: (%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
  printf("GRID DIMENTION: (%d, %d, %d)\n", gridDim.z, gridDim.y, gridDim.z);
}

int main(int argc, char **argv) {
  int N = 6;
  dim3 block(3);
  dim3 grid((N + block.x - 1) / block.x);

  printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
  printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);

  checkIndex<<<grid, block>>>();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    return -1;
  }
  // cudaDeviceReset();
  cudaDeviceSynchronize();

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
    return -1;
  }
  return 0;
}
