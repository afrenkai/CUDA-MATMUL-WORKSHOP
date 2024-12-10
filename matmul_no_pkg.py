import random
import time
from typing import List

Matrix = List[List[float]]


# purpose of this is to A: show what matrices are literally a list of list of floats and B: that if any rows are misshaped we fail matrix structure check.
def is_matrix(matrix: Matrix) -> bool:
    if not matrix or not matrix[0]: # could potentially be a tensor, with several matrices
        return False

    row_len = len(matrix[0]) # len of row to determine what to check for each matrix in an O(n) operation
    for row in matrix:
        if len(row) != row_len: # check to determine we are viewing a valid matrix.
            return False
    # if all checks pass, return true
    return True

def MatMul(A: Matrix, B: Matrix) -> List[List[int]]:
    '''
    Args: 2 Matrices A and B. 
    Returns: Output Matrix C
    '''

    if not is_matrix(A) or not is_matrix(B):
        raise ValueError("One or more of the matrices is not properly sized") # check using method defined earlier

    m = len(A)    # Number of rows in A
    n = len(A[0]) # Number of columns in A and rows in B
    p = len(B[0]) # Number of columns in B

    # Initialize result matrix C with zeros
    C = []
    for x in range(m): # for each row...
        row = [] 
        for j in range(p): # for each col ...
            row.append(0) # add a 0 to the matrix
        C.append(row) # add the row to the final C matrix once it's filled with 0s

    # Perform matrix multiplication
    # standard O(n^3) matmul, not gonna do the sweaty 2.617 shit
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    '''
    example:

    A = [[1,2,3] [4,5,6]] 
    B = [[8,9,10], [11,12,13]]
    m should equal 2 (the row starting with 1 and the row starting with 4)
    n should equal 3
    p should equal 3 as well

    so outer loop would happen 2 times
    middle loop happens 3 times
    same for inner loop

    C[0][0] = A[0][0] * B[0][0] = 1 * 8
    etc.
    '''

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
