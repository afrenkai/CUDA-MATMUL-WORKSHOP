from Typing import List
def MatMul(m, n, p) -> List[List[int]]:
    A.rows = m
    A.cols = n
    B.rows = n
    B.cols = p
    C.rows = m
    C.cols = p

    for i in range(0, m-1):
        for j in range(0, p-1):
            for k in range(0, n-1):
                C[i][j] = C[i][j] + (A[i][k] * B[k][j])

    return C

