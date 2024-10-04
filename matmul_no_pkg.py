from typing import List
Matrix = List[List[int]]

def is_matrix(matrix: Matrix): bool:
    if not matrix or not matrix[0]:
        return False

    row_len = len(matrix[0])
    for row in matrix:
        if len(row) != row_len:
            return False

    return True


def MatMul(A: Matrix, B: Matrix) -> List[List[int]]:
    if not is_matrix(A) or not is_matrix(B):
        raise ValueError("One or More of the Matrices is not properly sized")
    m = len(A) # n rows in A
    n = len(A[0]) # n cols in A and rows in B
    p = len(B[0]) # n cols in B

    C = []
    for x in range (m):
        row = []
        for j in range(p):
            row.append(0)
        C.append(row)

    
    # C = [[0 for x in range (p)] for y in range (m)]

    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]

    return C

A = [[0, 2, 3, 9, 7, 3], [7, 0, 5, 6, 5, 4], [4, 2, 8, 7, 0, 0],[1, 4, 6, 8, 3, 7], [7, 4, 7, 5, 1, 8], [3, 8, 5, 8, 9, 4]]
B = [[0, 9, 3, 2, 6, 5], [5, 4, 9, 5, 1, 6]]
def run(A, B) -> None:
    # Checking A

    if is_matrix(A):
        print("A is a valid matrix")
    else:
        print("A is not a valid matrix")


    #Checking B


    if is_matrix(B):
        print('B is a valid Matrix')
    else:
        print('B is not a valid Matrix')


    if is_matrix(A) and is_matrix(B):
        res = MatMul(A,B)
        print(res)
    else:
        print("One of the matrices is invalid, check the prior console output")


if name = "__main__":
    run(A, B)
