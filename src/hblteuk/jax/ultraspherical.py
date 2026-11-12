# Ultraspherical (Olver-Townsend) equation rows for one subdomain of
#     P u'' + Q u' + U u = 0,   z = mid + h x,  x in [-1, 1],
# i.e. P u_xx + h Q u_x + h^2 U u = 0, imposed on the C^(2) coefficients of
# the residual instead of at nodes. Unknowns stay Chebyshev-T coefficients,
# so boundary rows, transfer and evaluation are the collocation solver's.
# Requires P, Q, U polynomial of degree <= 3 in z (true for the hyperboloidal
# Teukolsky ODE); rows are then exact and banded (lower 3, upper 7).
# Notes: dev/ULTRASPHERICAL.md; design:
# docs/superpowers/specs/2026-11-11-ultraspherical-radial-design.md
import functools

import jax
import jax.numpy as jnp
import numpy as np

DEG = 3                     # max polynomial degree of P, Q, U
X_FIT = np.cos(np.pi * (2 * np.arange(DEG + 1) + 1) / (2 * (DEG + 1)))
V_INV = np.linalg.inv(np.vander(X_FIT, DEG + 1, increasing=True))
SCHEMES = ("ultraspherical", "collocation")


def check_scheme(scheme):
    if scheme not in SCHEMES:
        raise ValueError("scheme must be 'ultraspherical' or 'collocation'")
    return scheme


def _x_mult(lam, N):
    # multiplication by x in the C^(lam) basis
    X = np.zeros((N, N))
    for k in range(N):
        if k + 1 < N:
            X[k + 1, k] = (k + 1) / (2 * (k + lam))
        if k >= 1:
            X[k - 1, k] = (k + 2 * lam - 1) / (2 * (k + lam))
    return X


@functools.lru_cache(maxsize=None)
def operators(n):
    """A, B, C, each (DEG+1, n-2, n): rows = sum_j P_j A_j + (hQ)_j B_j
    + (h^2 U)_j C_j for monomial coefficients P_j, Q_j, U_j in x."""
    N = n + 2 * DEG + 4     # padding keeps truncation away from kept rows
    k = np.arange(N)
    D1 = np.diag(k[1:].astype(float), 1)                   # T -> C^(1)
    D2 = np.diag(2.0 * k[2:], 2)                           # T -> C^(2)
    S0 = np.diag(np.r_[1.0, np.full(N - 1, 0.5)]) - np.diag(np.full(N - 2, 0.5), 2)
    S1 = np.diag(1.0 / (k + 1)) - np.diag(1.0 / (k[2:] + 1), 2)
    X2 = _x_mult(2, N)
    pw = [np.eye(N)]
    for _ in range(DEG):
        pw.append(X2 @ pw[-1])
    keep = (slice(0, n - 2), slice(0, n))
    A = np.stack([(p @ D2)[keep] for p in pw])
    B = np.stack([(p @ S1 @ D1)[keep] for p in pw])
    C = np.stack([(p @ S1 @ S0)[keep] for p in pw])
    return A, B, C


def monomials(values):
    """(..., DEG+1) values at X_FIT -> monomial coefficients in x."""
    return values @ V_INV.T


@functools.lru_cache(maxsize=None)
def _stacked(n):
    # (3 (DEG+1), (n-2) n): A, B, C rows flattened for one matrix product
    return np.concatenate([o.reshape(o.shape[0], -1) for o in operators(n)])


def equation_rows(pqu_fn, z0, z1, n):
    """(B, D, n-2, n) equation rows on subdomains [z0_d, z1_d] (either
    direction). pqu_fn(z (k,)) -> (P, Q, U) each (B, k), as for
    collocation.solve_fixed_multidomain_batch."""
    h = 0.5 * (z1 - z0)                                            # (D,)
    zf = 0.5 * (z1 + z0)[:, None] + h[:, None] * jnp.asarray(X_FIT)
    P, Q, U = (x.swapaxes(0, 1) for x in jax.vmap(pqu_fn)(zf))     # (B, D, 4)
    Vt = jnp.asarray(V_INV.T)
    h = h[None, :, None]
    cf = jnp.concatenate([P @ Vt, (h * Q) @ Vt, (h**2 * U) @ Vt], axis=-1)
    # one real GEMM per part: 3.5x faster than three complex einsums on H200
    # (dev/ULTRASPHERICAL.md)
    W = jnp.asarray(_stacked(n))
    B, D = cf.shape[:2]
    c2 = cf.reshape(B * D, -1)
    return ((c2.real @ W) + 1j * (c2.imag @ W)).reshape(B, D, n - 2, n)
