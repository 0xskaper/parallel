#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

#define CHECK(call)                                                            \
  {                                                                            \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      printf("Error -> %s | %d\n", __FILE__, __LINE__);                        \
      printf("Code -> %d | Reason -> %s\n", error, cudaGetErrorString(error)); \
      exit(1);                                                                 \
    }                                                                          \
  }

int sumArray(int *input, int N) {
  int sum = 0;
  for (int i = 0; i < N; i++) {
    sum += input[i];
  }

  return sum;
}

__global__ void reduceNeigbored(int *input, int *blockSum, unsigned int N) {
  unsigned int tid = threadIdx.x;
  unsigned int globalIdx = blockIdx.x * blockDim.x + tid;
  int *blockData = input + blockIdx.x * blockDim.x;
  if (globalIdx >= N)
    return;
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    if ((tid % (2 * stride)) == 0) {
      blockData[tid] += blockData[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0)
    blockSum[blockIdx.x] = blockData[0];
}

__global__ void reduceNeighborless(int *input, int *blockSum, unsigned int N) {
  unsigned int tid = threadIdx.x;
  unsigned int globalIdx = blockIdx.x * blockDim.x + tid;
  int *blockData = input + blockIdx.x * blockDim.x;
  if (globalIdx >= N)
    return;
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    int index = 2 * stride * tid;
    if (index < blockDim.x) {
      blockData[index] += blockData[index + stride];
    }
    __syncthreads();
  }
  if (tid == 0)
    blockSum[blockIdx.x] = blockData[0];
}

int main(int argc, char **argv) {
  int device = 0;
  CHECK(cudaSetDevice(device));
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);

  int power = 24;
  if (argc < 1)
    power = atoi(argv[1]);
  int n = 1 << power;
  int blockSize = 512;
  int gridSize = (n + blockSize - 1) / blockSize;

  printf("Array Size: %d | Grid: %d | Block: %d\n", n, gridSize, blockSize);
  size_t bytes = n * sizeof(int);

  int *h_input = (int *)malloc(bytes);
  int *h_output = (int *)malloc(gridSize * sizeof(int));
  for (int i = 0; i < n; i++)
    h_input[i] = (int)(rand() & 0xFF);

  int cpuSum = sumArray(h_input, n);

  int *d_input, *d_output;
  CHECK(cudaMalloc(&d_input, bytes));
  CHECK(cudaMalloc(&d_output, gridSize * sizeof(int)));
  CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));

  CHECK(cudaEventRecord(start));
  reduceNeigbored<<<gridSize, blockSize>>>(d_input, d_output, n);
  CHECK(cudaEventRecord(stop));
  CHECK(cudaEventSynchronize(stop));

  reduceNeighborless<<<gridSize, blockSize>>>(d_input, d_output, n);

  float ms = 0.0f;
  CHECK(cudaEventElapsedTime(&ms, start, stop));

  CHECK(cudaMemcpy(h_output, d_output, gridSize * sizeof(int),
                   cudaMemcpyDeviceToHost));
  int gpuSum = 0;
  for (int i = 0; i < gridSize; i++)
    gpuSum += h_output[i];

  printf("GPU SUM -> %d\n", cpuSum);
  printf("CPU SUM -> %d %s\n", gpuSum,
         gpuSum == cpuSum ? "(MATCH)" : "(MISMATCH)");
  printf("Kernel Time: %.3f ms\n", ms);

  CHECK(cudaFree(d_input));
  CHECK(cudaFree(d_output));
  free(h_input);
  free(h_output);
  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));

  return 0;
}
