#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

// CUDA kernel 
__global__ void matrixMulKernel(float *A, float *B, float *C, int N) {
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float sum = 0.0;

    for (int tile = 0; tile < (N / TILE_WIDTH); ++tile) {
        // Loading tile method into mem
        tileA[threadIdx.y][threadIdx.x] = A[row * N + (tile * TILE_WIDTH + threadIdx.x)];
        tileB[threadIdx.y][threadIdx.x] = B[(tile * TILE_WIDTH + threadIdx.y) * N + col];
        __syncthreads();

        // Multiply the loaded tiles
        for (int i = 0; i < TILE_WIDTH; ++i) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

void matrixMul(float *h_A, float *h_B, float *h_C, int N) {
    int size = N * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH, 1);

    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    
    cudaEventRecord(start, 0);

    
    matrixMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

   
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("Time taken for matrix multiplication: %f ms\n", elapsedTime);

    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    int N = 1024;  // Matrix size NxN
    int size = N * N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize matrices A and B
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    matrixMul(h_A, h_B, h_C, N);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
