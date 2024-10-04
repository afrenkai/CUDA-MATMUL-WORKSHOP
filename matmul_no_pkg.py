from typing import List
Matrix = List[List[int]]
def MatMul(A: Matrix, B: Matrix, m: int, n: int, p: int) -> List[List[int]]:
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
print(eval(A, Matrix))
