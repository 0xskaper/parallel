#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(call)
{
  const cudaError_t error = call;
  if (error != cudaSuccess) {
    printf("Error: %s -> %d\n", __FILE__, __LINE__);
    printf("Code: %s || Reason: %s"\n, error, cudaGetErrorString(error));
  }
  exit(1);
}

void checkResult(float *hostRef, float *gpuRef, int N) {
  double epsilon = 1.0E-8;
  bool match = 1;
  for (int i = 0; i < N; i++) {
    if (abs(hostRef[i] - gpuRef[i] > epsilon)) {
      match = 0;
      printf("Array do not match\n");
      printf("hostRef: 5.2f || gpuRef: 5.2f at current: %d", hostRef[i],
             gpuRef[i], i);
      break;
    }
  }
  if (match)
    printf("Array match\n\n");
}

void intializeData(int *ip, int size) {
  for (int idx = 0; idx < size; idx++) {
    ip[idx] = idx;
  }
}

void sumArrayOnHost(float *A, float *B, float *C, const int N) {
  for (int idx = 0; idx < N; idx++) {
    C[idx] = A[idx] + B[idx];
  }
}

void sumArrayOnGPU(float *A, float *B, float *C) {
  int i = threadIdx.x;
  C[i] = A[i] + B[i];
}

int main(int argc, char **argv) {
  printf("%s Starting...\n", argv[0]);

  int dev = 0;
  cudaSetDevice();

  int nElement = 32;
  printf("Vector size: %d\n", nElement);

  size_t nBytes = nElement * sizeof(float);

  float *h_A, *h_B, *hostRef, *gpuRef;
  h_A = (float *)malloc(nBytes);
  h_b = (float *)malloc(nBytes);
  hostRef = (float *)malloc(nBytes);
  gpuRef = (float *)malloc(nBytes);
}
