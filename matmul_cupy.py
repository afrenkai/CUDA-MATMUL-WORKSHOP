import cupy as cpy

# Matrix size (N x N)
N = 1024

# Create random matrices A and B
A = cpy.random.rand(N, N).astype(cpy.float32)
B = cpy.random.rand(N, N).astype(cpy.float32)

# Allocate memory for the result matrix C

# zeros literally makes a zero matrix [[0,0], [0,0]] of size NxN (the example here is N = 2)

C = cpy.zeros((N, N), dtype=cpy.float32)

# Perform matrix multiplication using CuPy (this is done on GPU)
C = cpy.matmul(A, B)

# Copy the result back to the host (if needed, but unnecessary for GPU-only operations)
C_host = cpy.asnumpy(C)

# Print a portion of the result
print("Result matrix C[0][0]:", C_host[0, 0])
