#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <sys/time.h>

double cpuSecond() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

__global__ void warmup() {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  ib += ia + tid;
}

__global__ void mathKernel_I(float *c) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float a, b;
  a = b = 0.0f;

  if (tid % 2 == 0) {

    a = 100.0f;
  } else {
    b = 200.0f;
  }
  c[tid] = a + b;
}

__global__ void mathKernel_II(float *c) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float a, b;
  a = b = 0.0f;
  if ((tid / 32) % 2 == 0) {
    a = 100.0f;
  } else {
    b = 200.0f;
  }

  c[tid] = a + b;
}

__global__ void mathKernel_III(float *c) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float a = 0.0f, b = 0.0f;
  bool pred = (tid % 2 == 0);
  // Both paths execute via predication — no warp divergence
  if (pred)
    a = 100.0f;
  if (!pred)
    b = 200.0f;
  c[tid] = a + b;
}
int main(int argc, char **argv) {
  int dev = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  printf("%s using Device %d: %s\n", argv[0], dev, deviceProp.name);

  int size = 1 << 16;
  int blockSize = 32;
  if (argc > 1)
    size = atoi(argv[1]);
  if (argc > 2)
    blockSize = atoi(argv[2]);
  printf("Data Size -> %d", size);

  dim3 block(blockSize);
  dim3 grid((size + block.x - 1) / block.x, 1);
  printf("Execution Configure (block %d Grid %d)\n", block.x, grid.x);

  float *d_C;
  size_t nBytes = size * sizeof(float);
  cudaMalloc((float **)&d_C, nBytes);

  double iStart, iElapsed;
  cudaDeviceSynchronize();
  iStart = cpuSecond();
  warmup<<<grid, block>>>();
  cudaDeviceSynchronize();
  iElapsed = cpuSecond() - iStart;
  printf("\nWarmup<<<%d, %d>>> -> Elapsed -> %3.2f Sec \n", grid.x, block.x,
         iElapsed);

  iStart = cpuSecond();
  mathKernel_I<<<grid, block>>>(d_C);
  cudaDeviceSynchronize();
  iElapsed = cpuSecond() - iStart;
  printf("mathKernel<<<%d, %d>>> -> Elapsed -> %3.2f Sec\n", grid.x, block.x,
         iElapsed);

  iStart = cpuSecond();
  mathKernel_II<<<grid, block>>>(d_C);
  cudaDeviceSynchronize();
  iElapsed = cpuSecond() - iStart;
  printf("mathKernel<<<%d, %d>>> -> Elapsed -> %3.2f Sec\n", grid.x, block.x,
         iElapsed);

  iStart = cpuSecond();
  mathKernel_III<<<grid, block>>>(d_C);
  cudaDeviceSynchronize();
  iElapsed = cpuSecond() - iStart;
  printf("mathKernel<<<%d, %d>>> -> Elapsed -> %3.2f Sec\n", grid.x, block.x,
         iElapsed);

  cudaFree(d_C);
  cudaDeviceReset();
  return EXIT_SUCCESS;
}
