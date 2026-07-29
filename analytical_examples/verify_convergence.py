"""
Numerically checks the convergence claim used in Chapter 2, Section
"Border Rank": the rank-2 tensor sequence X_n (defined analytically in
compute_analytical_slices.py) converges, entry by entry, to the fixed
frontal slices X_1, X_2 of eq:frontal_slices as n grows.

This script plugs in a single large value, n = 10000, and prints the
resulting frontal slices so they can be compared by eye against
eq:frontal_slices (X_1 = [[3, 4], [4, 5]], X_2 = [[4, 5], [5, 6]]): the
values below should already be very close to those exact numbers.
No external library is used; every operation is plain arithmetic on
nested Python lists.
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

def scale3(A, val):
    """Multiplies every entry of a tensor in R^(2x2x2) by a scalar."""
    C = [[[0.0 for _ in range(2)] for _ in range(2)] for _ in range(2)]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                C[i][j][k] = A[i][j][k] * val
    return C

# Large n, to approximate the n -> infinity limit
n = 10000.0
# z = u + v/n
u_plus = [u[0] + v[0]/n, u[1] + v[1]/n]

# X_n = n * z^o3  -  n * u^o3
T1 = scale3(outer3(u_plus, u_plus, u_plus), n)
T2 = scale3(outer3(u, u, u), -n)

X_approx = add3(T1, T2)

print("Approximated Tensor X for n = 10000:")
print("X_1:")
for i in range(2):
    print([round(X_approx[i][j][0], 4) for j in range(2)])
print("X_2:")
for i in range(2):
    print([round(X_approx[i][j][1], 4) for j in range(2)])

print("\nCompare against the exact eq:frontal_slices values from the thesis:")
print("X_1 = [[3, 4], [4, 5]],  X_2 = [[4, 5], [5, 6]]")
