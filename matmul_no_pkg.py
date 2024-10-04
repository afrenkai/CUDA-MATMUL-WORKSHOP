import random
import time
from typing import List
Matrix = List[List[int]]

def is_matrix(matrix: Matrix) -> bool:
    if not matrix or not matrix[0]:
        return False

    row_len = len(matrix[0])
    for row in matrix:
        if len(row) != row_len:
            return False

    return True

def MatMul(A: Matrix, B: Matrix) -> List[List[int]]:
    if not is_matrix(A) or not is_matrix(B):
        raise ValueError("One or more of the matrices is not properly sized")

    m = len(A)    # Number of rows in A
    n = len(A[0]) # Number of columns in A and rows in B
    p = len(B[0]) # Number of columns in B

    # Initialize result matrix C with zeros
    C = []
    for x in range(m):
        row = []
        for j in range(p):
            row.append(0)
        C.append(row)

    # Perform matrix multiplication
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]

    return C

def run() -> None:
    # Generate two 1024 x 1024 random matrices
    size = 1024
    A = []
    B = []

    for i in range(size):
        row_A = []
        row_B = []
        for j in range(size):
            row_A.append(random.randint(0, 100))
            row_B.append(random.randint(0, 100))
        A.append(row_A)
        B.append(row_B)

    # Checking A and B
    if is_matrix(A):
        print("A is a valid matrix")
    else:
        print("A is not a valid matrix")

    if is_matrix(B):
        print("B is a valid matrix")
    else:
        print("B is not a valid matrix")

    # Measure time for matrix multiplication
    if is_matrix(A) and is_matrix(B):
        start_time = time.time()
        res = MatMul(A, B)
        end_time = time.time()
        print(f"Matrix multiplication took {end_time - start_time} seconds")
    else:
        print("One of the matrices is invalid, check the prior console output")


if __name__ == "__main__":
    run()
