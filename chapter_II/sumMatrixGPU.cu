#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <sys/time.h>

#define CHECK(call)                                                            \
  {                                                                            \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      printf("Error -> %s | %d\n", __FILE__, __LINE__);                        \
      printf("Code -> %d || Reason: %s\n", error, cudaGetErrorString(error));  \
      exit(1);                                                                 \
    }                                                                          \
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

void synthetic(float *ip, int N) {
  for (int i = 0; i < N; i++) {
    ip[i] = i;
  }
}

double cpuSecond() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void sumMatrixOnCPU(float *A, float *B, float *C, const int nx, const int ny) {
  float *ia = A;
  float *ib = B;
  float *ic = C;
  for (int iy = 0; iy < ny; iy++) {
    for (int ix = 0; ix < nx; ix++) {
      ic[ix] = ia[ix] + ib[ix];
    }
    ia += nx;
    ib += nx;
    ic += nx;
  }
}

__global__ void sumMatrixOnGPU1D_1D(float *MatA, float *MatB, float *MatC,
                                    int nx, int ny) {
  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  if (ix < nx) {
    for (int iy = 0; iy < ny; iy++) {
      unsigned int idx = iy * nx + ix;
      MatC[idx] = MatA[idx] + MatB[idx];
    }
  }
}

__global__ void sumMatrixOnGPU(float *MatA, float *MatB, float *MatC, int nx,
                               int ny) {
  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int idx = iy * nx + ix;
  if (ix < nx && iy < ny)
    MatC[idx] = MatA[idx] + MatB[idx];
}

int main(int argc, char **argv) {
  printf("%s Starting...\n", argv[0]);
  int dev = 0;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  printf("\nUsing Device -> %d: %s\n", dev, deviceProp.name);
  CHECK(cudaSetDevice(dev));

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

  double iStart = cpuSecond();
  synthetic(h_A, nxy);
  synthetic(h_B, nxy);
  double iElapsed = cpuSecond() - iStart;

  memset(hostRef, 0, nBytes);
  memset(gpuRef, 0, nBytes);

  iStart = cpuSecond();
  sumMatrixOnCPU(h_A, h_B, hostRef, nx, ny);
  iElapsed = cpuSecond() - iStart;
  printf("CPU TIME ELAPSED -> %f Sec\n", iElapsed);

  float *d_MatA, *d_MatB, *d_MatC;
  cudaMalloc((void **)&d_MatA, nBytes);
  cudaMalloc((void **)&d_MatB, nBytes);
  cudaMalloc((void **)&d_MatC, nBytes);

  cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);

  // 2D_2D -> 32x32
  int dimx = 32;
  int dimy = 32;

  dim3 block(dimx, dimy);
  dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

  iStart = cpuSecond();
  sumMatrixOnGPU<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
  cudaDeviceSynchronize();
  iElapsed = cpuSecond() - iStart;
  printf("GPU 2D_2D <<<(%d, %d), (%d, %d)>>> ELAPSED -> %f Sec\n", grid.x,
         grid.y, block.x, block.y, iElapsed);
  cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);
  checkResult(hostRef, gpuRef, nxy);

  // 2D_2D -> 32x16
  dimx = 32;
  dimy = 32;

  dim3 block_1(dimx, dimy);
  dim3 grid_1((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

  iStart = cpuSecond();
  sumMatrixOnGPU<<<grid_1, block_1>>>(d_MatA, d_MatB, d_MatC, nx, ny);
  cudaDeviceSynchronize();
  iElapsed = cpuSecond() - iStart;
  printf("GPU 2D_2D <<<(%d, %d), (%d, %d)>>> ELAPSED -> %f Sec\n", grid_1.x,
         grid_1.y, block_1.x, block_1.y, iElapsed);
  cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);
  checkResult(hostRef, gpuRef, nxy);

  // 2D_2D -> 16x16
  dimx = 16;
  dimy = 16;

  dim3 block_2(dimx, dimy);
  dim3 grid_2((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

  iStart = cpuSecond();
  sumMatrixOnGPU<<<grid_2, block_2>>>(d_MatA, d_MatB, d_MatC, nx, ny);
  cudaDeviceSynchronize();
  iElapsed = cpuSecond() - iStart;
  printf("GPU 2D_2D <<<(%d, %d), (%d, %d)>>> ELAPSED -> %f Sec\n", grid_2.x,
         grid_2.y, block_2.x, block_2.y, iElapsed);
  cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);
  checkResult(hostRef, gpuRef, nxy);

  // 1D_1D
  dimx = 32;
  dimy = 1;

  dim3 block_3(dimx, dimy);
  dim3 grid_3((nx + block.x - 1) / block.x);

  iStart = cpuSecond();
  sumMatrixOnGPU1D_1D<<<grid_3, block_3>>>(d_MatA, d_MatB, d_MatC, nx, ny);
  cudaDeviceSynchronize();
  printf("GPU 1D_1D <<<(%d, %d), (%d, %d)>>> ELAPSED -> %f Sec\n", grid_3.x,
         grid_3.y, block_3.x, block_3.y, iElapsed);
  cudaMemcpy(gpuRef, hostRef, nBytes, cudaMemcpyDeviceToHost);
  checkResult(hostRef, gpuRef, nxy);

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
