#include <cuda_runtime.h>
#include <stdio.h>

__global__ void wrapKernel() {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  ib += ia + tid;
}

int main(int agc, char **argv) {
  int dev = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  int maxWrapsPerSM = deviceProp.maxThreadsPerMultiProcessor / 32;
  printf("\n Max Wraps -> %d", maxWrapsPerSM);
  return 0;
}
