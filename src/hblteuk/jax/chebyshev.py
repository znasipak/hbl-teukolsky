# Chebyshev machinery for the batched solver: host-side collocation matrices
# (numpy, computed once) + device-side batched evaluation. See dev/PHASE2.md.
import functools

import jax
import jax.numpy as jnp
import numpy as np
import numpy.polynomial.chebyshev as ch


def cheby_D(n):
    # coefficient-space differentiation matrix (numpy twin of cheby.cheby_D_jit)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n, 2):
            D[i, j] = 2 * j if i > 0 else j
    return D


def collocation_matrices(n, chtype=1):
    """Host-side constants for an n-node solve: nodes, T0/T1/T2 at the nodes,
    boundary rows [value, d/dx] at x = -1 (bc_left) and x = +1 (bc_right)."""
    nodes = ch.chebpts1(n) if chtype == 1 else ch.chebpts2(n)
    D = cheby_D(n)
    T0 = ch.chebvander(nodes, n - 1)
    T1 = T0 @ D
    T2 = T1 @ D
    row_l = ch.chebvander(-1.0, n - 1)
    row_r = ch.chebvander(1.0, n - 1)
    bc_left = np.vstack([row_l, row_l @ D])
    bc_right = np.vstack([row_r, row_r @ D])
    return {"nodes": nodes, "D": D, "T0": T0, "T1": T1, "T2": T2,
            "bc_left": bc_left, "bc_right": bc_right}


@functools.partial(jax.jit, static_argnums=1)
def chebvander_jnp(x, n):
    # T_k(x) recurrence, columns k = 0..n-1
    cols = [jnp.ones_like(x), x]
    for _ in range(n - 2):
        cols.append(2 * x * cols[-1] - cols[-2])
    return jnp.stack(cols[:n], axis=-1)


def _domain_scaled_derivative(coeffs, subdomains, D, deriv):
    # d/dsigma in coefficient space: c -> c @ D.T, scaled per domain by dx/dz
    dxdz = 2.0 / (subdomains[1:] - subdomains[:-1])  # (Dm,)
    for _ in range(deriv):
        coeffs = (coeffs @ D.T) * dxdz[None, :, None]
    return coeffs


@functools.partial(jax.jit, static_argnums=(3, 4, 5))
def _eval_core(coeffs, subdomains, sigma, n, deriv, descending, D):
    # bin on an ascending copy, but evaluate with the ORIGINAL domain
    # orientation: flipping the list would negate x and break T_k parity
    coeffs = _domain_scaled_derivative(coeffs, subdomains, D, deriv)
    ndom = coeffs.shape[1]
    asc = subdomains[::-1] if descending else subdomains
    j = jnp.clip(jnp.searchsorted(asc, sigma, side="right") - 1, 0, ndom - 1)
    idx = (ndom - 1 - j) if descending else j
    z0 = subdomains[idx]
    z1 = subdomains[idx + 1]
    x = (2 * sigma - (z1 + z0)) / (z1 - z0)
    V = chebvander_jnp(x, n)                        # (Np, n)
    c = coeffs[:, idx, :]                           # (B, Np, n)
    return jnp.einsum("bpn,pn->bp", c, V)


def evaluate_multidomain(coeffs, subdomains, sigma, deriv=0, D=None):
    """Evaluate batched multi-domain Chebyshev series.

    coeffs (B, ndomains, n); subdomains (ndomains+1,) monotone either way;
    sigma (Np,). Returns (B, Np). Same semantics as the CPU
    MultiDomainChebyshev evaluation, batched over the leading axis.
    """
    coeffs = jnp.asarray(coeffs)
    subdomains_np = np.asarray(subdomains, dtype=np.float64)
    sigma = jnp.atleast_1d(jnp.asarray(sigma, dtype=jnp.float64))
    n = coeffs.shape[-1]
    if D is None:
        D = jnp.asarray(cheby_D(n))
    descending = bool(subdomains_np[0] > subdomains_np[-1])
    return _eval_core(coeffs, jnp.asarray(subdomains_np), sigma, n, deriv,
                      descending, D)
