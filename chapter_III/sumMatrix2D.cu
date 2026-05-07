#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>

__global__ void sum2DMatrix(float *A, float *B, float *C, int NX, int NY) {
  unsigned int ix = threadIdx.x * blockIdx.x + blockDim.x;
  unsigned int iy = threadIdx.y * blockIdx.y + blockDim.y;
  int idx = ix * NX + iy;
  if (ix < NX && iy < NY)
    C[idx] = A[idx] + B[idx];
}

int main(int argc, char **argv) {
  int dev = 0;
  cudaDeviceProp deviceProp;
  cudaSetDevice(dev);
  cudaGetDeviceProperties(&deviceProp, dev);
  printf("\nUsing Device -> %d : %s\n", dev, deviceProp.name);

  int nx = 1 << 14;
  int ny = 1 << 14;

  int nxy = nx * ny;
  int nBytes = nxy * sizeof(float);
  printf("Matrix size: nx %d ny %d\n", nx, ny);

  float *h_A, *h_B, *hostRef, *gpuRef;
  h_A = (float *)malloc(nBytes);
  h_B = (float *)malloc(nBytes);
  hostRef = (float *)malloc(nBytes);
  gpuRef = (float *)malloc(nBytes);

  memset(hostRef, 0, nBytes);
  memset(gpuRef, 0, nBytes);

  float *d_MatA, *d_MatB, *d_MatC;
  cudaMalloc((void **)&d_MatA, nBytes);
  cudaMalloc((void **)&d_MatB, nBytes);
  cudaMalloc((void **)&d_MatC, nBytes);

  cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);

  dim3 block(32, 32);
  dim3 grid(512, 512);

  sum2DMatrix<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);

  dim3 block_2(32, 16);
  dim3 grid_2(512, 1024);

  sum2DMatrix<<<grid_2, block_2>>>(d_MatA, d_MatB, d_MatC, nx, ny);

  dim3 block_3(16, 32);
  dim3 grid_3(1024, 512);

  sum2DMatrix<<<grid_3, block_3>>>(d_MatA, d_MatB, d_MatC, nx, ny);

  dim3 block_4(16, 32);
  dim3 grid_4(1024, 512);

  sum2DMatrix<<<grid_4, block_4>>>(d_MatA, d_MatB, d_MatC, nx, ny);
  cudaFree(d_MatA);
  cudaFree(d_MatB);
  cudaFree(d_MatC);

  free(h_A);
  free(h_B);
  free(hostRef);
  free(gpuRef);

  cudaDeviceReset();
  return (0);
}
