#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(call)                                                            \
  {                                                                            \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      printf("\n Error -> %s | %d", __FILE__, __LINE__);                       \
      printf("Code -> %d | %s", error, cudaGetErrorString(error));             \
    }                                                                          \
  }

void initialInt(int *ip, int size) {
  for (int i = 0; i < size; i++) {
    ip[i] = i;
  }
}

void printMatrix(int *C, const int nx, const int ny) {
  int *ic = C;
  printf("\n MATRIX -> (%d, %d)\n", nx, ny);
  for (int iy = 0; iy < nx; iy++) {
    for (int ix = 0; ix < ny; ix++) {
      printf("%3d", ic[ix]);
    }
    ic += nx;
    printf("\n");
  }
  printf("\n");
}

__global__ void printThreadIDs(int *A, const int nx, const int ny) {
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  int iy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int idx = iy * nx + ix;

  printf("\nThread ID -> (%d, %d) | Block ID -> (%d, %d) | Coordinate -> (%d, "
         "%d) | Global Index -> %2d | VALUE -> %2d \n",
         threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx, A[idx]);
}

int main(int argc, char **argv) {
  printf("%s Starting...\n", argv[0]);
  int dev = 0;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&device, ))
}
