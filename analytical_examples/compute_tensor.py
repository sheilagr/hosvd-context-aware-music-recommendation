"""
Computes, entry by entry, the original symmetric tensor X used in
Chapter 2 (eq:cpd-example) as the running example for tensor rank versus
border rank:

    X = u o u o v  +  u o v o u  +  v o u o u

where "o" denotes the outer product (Chapter 2, Definition "Outer
Product"), and u = [1, 1]^T, v = [1, 2]^T are the two linearly
independent vectors chosen for this example.

This script does not use any external library: every entry is computed
directly from its definition, so the result can be checked by hand
against the values printed below and against eq:frontal_slices in the
thesis.
"""
u = [1, 1]
v = [1, 2]

def outer3(a, b, c):
    """Outer product of three vectors of length 2: a tensor in R^(2x2x2)."""
    X = [[[0.0 for _ in range(2)] for _ in range(2)] for _ in range(2)]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                X[i][j][k] = a[i] * b[j] * c[k]
    return X

def add3(A, B):
    """Entry-wise sum of two tensors in R^(2x2x2)."""
    C = [[[0.0 for _ in range(2)] for _ in range(2)] for _ in range(2)]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                C[i][j][k] = A[i][j][k] + B[i][j][k]
    return C

# The three rank-1 terms of eq:cpd-example
T1 = outer3(u, u, v)
T2 = outer3(u, v, u)
T3 = outer3(v, u, u)

X = add3(add3(T1, T2), T3)

print("Tensor X:")
for i in range(2):
    for j in range(2):
        for k in range(2):
            print(f"x_{i+1}{j+1}{k+1} = {X[i][j][k]}")

# The two frontal slices X_1, X_2 (fixing the third index k=1 and k=2)
# should match eq:frontal_slices in the thesis:
#   X_1 = [[3, 4], [4, 5]],  X_2 = [[4, 5], [5, 6]]
print("\nFrontal Slices:")
print("X_1:")
for i in range(2):
    print([X[i][j][0] for j in range(2)])
print("X_2:")
for i in range(2):
    print([X[i][j][1] for j in range(2)])
