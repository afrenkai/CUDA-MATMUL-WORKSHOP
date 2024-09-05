import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

# Big Kernel Typescript Typescript  C = A \cdot B 
kernel_code = """
__global__ void matrixMulKernel(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
"""

# N x N matrix
N = 1024

# Block size for threads
BLOCK_SIZE = 16

# malloc for A, B, and C 
h_A = np.random.randn(N, N).astype(np.float32)
h_B = np.random.randn(N, N).astype(np.float32)
h_C = np.empty((N, N), dtype=np.float32)

#cuda malloc for A, B, and C
d_A = cuda.mem_alloc(h_A.nbytes)
d_B = cuda.mem_alloc(h_B.nbytes)
d_C = cuda.mem_alloc(h_C.nbytes)

# move A and B to CUDA device
cuda.memcpy_htod(d_A, h_A)
cuda.memcpy_htod(d_B, h_B)

# compile
mod = SourceModule(kernel_code)
matrixMulKernel = mod.get_function("matrixMulKernel")

# make dims
block = (BLOCK_SIZE, BLOCK_SIZE, 1)
grid = (N // BLOCK_SIZE, N // BLOCK_SIZE, 1)

# launch kernel
matrixMulKernel(d_A, d_B, d_C, np.int32(N), block=block, grid=grid)

# send from cuda to device
cuda.memcpy_dtoh(h_C, d_C)

# i, j = 0 of res mat
print("Result matrix C[0][0]:", h_C[0, 0])
cuda.Context.pop()
