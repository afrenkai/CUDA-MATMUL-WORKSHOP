import torch
import time

N = 1024

def matmul_cpu(N: int):
    A_cpu = torch.randn(N, N)
    B_cpu = torch.randn(N, N)
    start_time = time.time()
    
    C_cpu = torch.matmul(A_cpu, B_cpu)
    
    end_time = time.time()
    
    print(f"CPU result [0, 0]: {C_cpu[0, 0]}")
    print(f"Time taken on CPU: {end_time - start_time:.6f} seconds")

def matmul_gpu(N):
    A_gpu = torch.randn(N, N, device='cuda')
    B_gpu = torch.randn(N, N, device='cuda')
    start_time = time.time()
    C_gpu = torch.matmul(A_gpu, B_gpu)
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"GPU result [0, 0]: {C_gpu[0, 0]}")
    print(f"Time taken on GPU: {end_time - start_time:.6f} seconds")
matmul_cpu(N)
if torch.cuda.is_available():
    matmul_gpu(N)
else:
    print("GPU is not available")
