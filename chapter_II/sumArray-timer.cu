#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

#define CHECK(call)                                                            \
  {                                                                            \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      printf("Error -> %s||%d\n", __FILE__, __LINE__);                         \
    }                                                                          \
    exit(1);                                                                   \
  }

void synthetic(float *ip, int N) {
  for (int i = 0; i < N; i++)
    ip[i] = i;
}

double cpuSecond() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.0e-6);
}

void checkResult(float *hostRef, float *gpuRef, const int N) {
  double epsilon = 1.0E-8;
  bool match = 1;
  for (int i = 0; i < N; i++) {
    if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
      match = 0;
      printf("Arrays do not match!\n");
      printf("Host %5.2f != GPU %5.2f @ Current %d\n", hostRef[i], gpuRef[i],
             i);
      break;
    }
  }
  if (match)
    printf("Arrays match.\n\n");
}

__global__ void sumArraysOnGPU(float *A, float *B, float *C) {
  int idx = threadIdx.x;
  C[idx] = A[idx] + B[idx];
}

void sumArraysOnCPU(float *A, float *B, float *C, const int N) {
  for (int i = 0; i < N; i++) {
    C[i] = A[i] + B[i];
  }
}

int main(int argc, char **argv) {
  int dev = 0;
  cudaSetDevice(dev);

  int nElem = 1024;
  printf("Vector size: %d\n", nElem);

  size_t nBytes = nElem * sizeof(float);

  float *h_A, *h_B, *hostRef, *gpuRef;
  h_A = (float *)malloc(nBytes);
  h_B = (float *)malloc(nBytes);
  hostRef = (float *)malloc(nBytes);
  gpuRef = (float *)malloc(nBytes);

  synthetic(h_A, nElem);
  synthetic(h_B, nElem);

  memset(hostRef, 0, nBytes);
  memset(gpuRef, 0, nBytes);

  float *d_A, *d_B, *d_C;

  cudaMalloc((float **)&d_A, nBytes);
  cudaMalloc((float **)&d_B, nBytes);
  cudaMalloc((float **)&d_C, nBytes);

  cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

  dim3 block(nElem);
  dim3 grid(nElem / block.x);

  printf("BLOCK.x -> %d\n", block.x);

  double iStart = cpuSecond();
  sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C);
  cudaDeviceSynchronize();
  double iElapsed = cpuSecond() - iStart;
  printf("Execution configuration <<<%d, %d>>> || TIME -> %f \n", grid.x,
         block.x, iElapsed);

  cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

  iStart = cpuSecond();
  sumArraysOnCPU(h_A, h_B, hostRef, nElem);
  iElapsed = cpuSecond() - iStart;

  printf("\n TIME -> %f \n", iElapsed);
  checkResult(hostRef, gpuRef, nElem);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  free(h_A);
  free(h_B);
  free(hostRef);
  free(gpuRef);

  return (0);
}
