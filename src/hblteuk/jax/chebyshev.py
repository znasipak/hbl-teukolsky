# Chebyshev machinery for the batched solver: host-side collocation matrices
# (numpy, computed once) + device-side batched evaluation. See dev/PHASE2.md.
import collections
import functools
from typing import NamedTuple

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


class EvalMatrix(NamedTuple):
    lo: int          # first touched domain (original orientation)
    hi: int          # one past the last touched domain
    E: jax.Array     # (3, (hi-lo)*n, Np): deriv 0..2 at the sigma points


_EVAL_CACHE = collections.OrderedDict()
_EVAL_CACHE_MAX = 64


def evaluation_matrix(subdomains, sigma, n):
    """Fixed-point evaluation operator: evaluate_multidomain applied to unit
    coefficient vectors over the touched domains, for deriv 0, 1, 2.
    Cached on (subdomains, sigma, n). See dev/JAX_PERF.md (Task 5)."""
    subs = np.ascontiguousarray(subdomains, dtype=np.float64)
    sig = np.ascontiguousarray(np.atleast_1d(sigma), dtype=np.float64)
    key = (subs.tobytes(), sig.tobytes(), int(n))
    em = _EVAL_CACHE.get(key)
    if em is not None:
        _EVAL_CACHE.move_to_end(key)
        return em
    # host binning, same rule as _eval_core
    ndom = len(subs) - 1
    descending = bool(subs[0] > subs[-1])
    asc = subs[::-1] if descending else subs
    j = np.clip(np.searchsorted(asc, sig, side="right") - 1, 0, ndom - 1)
    idx = (ndom - 1 - j) if descending else j
    lo, hi = int(idx.min()), int(idx.max()) + 1
    # unit vectors on the touched sub-list; points stay in the same bins
    k = (hi - lo) * n
    unit = np.eye(k).reshape(k, hi - lo, n)
    sub_slice = subs[lo:hi + 1]
    E = jnp.stack([evaluate_multidomain(unit, sub_slice, sig, deriv=d)
                   for d in (0, 1, 2)])
    em = EvalMatrix(lo, hi, E)
    _EVAL_CACHE[key] = em
    if len(_EVAL_CACHE) > _EVAL_CACHE_MAX:
        _EVAL_CACHE.popitem(last=False)
    return em


@functools.partial(jax.jit, static_argnums=(1, 2))
def _apply_eval(coeffs, lo, hi, E):
    c = coeffs[:, lo:hi].reshape(coeffs.shape[0], -1)
    return jnp.einsum("bk,dkp->dbp", c, E)


def evaluate_with_matrix(coeffs, em):
    """coeffs (B, ndomains, n) -> (3, B, Np): psi, dpsi, d2psi at em's sigma."""
    return _apply_eval(jnp.asarray(coeffs), em.lo, em.hi, em.E)
