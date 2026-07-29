"""
Computes, symbolically (using the sympy library for exact algebra, not
floating-point approximation), the frontal slices of the rank-2 tensor
sequence X_n used in Chapter 2, Section "Border Rank" to show that the
tensor X of eq:cpd-example (border rank 2) can be approximated
arbitrarily well by tensors of rank 2, even though X itself has tensor
rank 3.

The sequence is:
    X_n = n * (u + v/n)^o3  -  n * u^o3
where u = [1, 1]^T, v = [1, 2]^T, and w^o3 means w o w o w (the outer
product of w with itself three times). As n -> infinity, X_n converges
to X entry by entry (checked numerically in verify_convergence.py).
"""

import sympy as sp

n = sp.Symbol('n')
u = [1, 1]
v = [1, 2]
z = [1 + 1/n, 1 + 2/n]  # z = u + v/n

X_n = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(2)]

for i in range(2):
    for j in range(2):
        for k in range(2):
            # (X_n)_ijk = n * z_i * z_j * z_k  -  n * u_i * u_j * u_k
            X_n[i][j][k] = sp.simplify(n * z[i] * z[j] * z[k] - n * u[i] * u[j] * u[k])

print("Frontal slices of X_n (exact, as functions of n):")
print("(X_n)_1:")
for i in range(2):
    print([X_n[i][j][0] for j in range(2)])
print("(X_n)_2:")
for i in range(2):
    print([X_n[i][j][1] for j in range(2)])

print("\nAs n -> infinity, each entry above tends to the corresponding")
print("entry of the fixed frontal slices X_1, X_2 in eq:frontal_slices")
print("(numerically confirmed for a large finite n in verify_convergence.py).")
