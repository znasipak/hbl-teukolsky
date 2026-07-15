# Batched spin-weighted spheroidal harmonics via dense symmetric eigensolve.
# Mirrors the ARPACK path in hblteuk.swsh (spectral_sparse_matrix / generate_eigs);
# design rationale and validation notes: dev/PHASE1.md.
import functools

import jax
import jax.numpy as jnp
import numpy as np

from ..swsh import sphericalY_expansion


def _safe_div(num, den):
    # 0 where den == 0; used where a zero prefactor makes the term vanish anyway
    return jnp.where(den == 0, 0.0, num / jnp.where(den == 0, 1.0, den))


# spherical-spheroidal coupling coefficients, elementwise in l (cf. swsh.py k1*/k2*)
def _k2m2(s, l, m):
    t = (_safe_div(l - m - 1, l - 1) * _safe_div(l + m - 1, l - 1)
         * _safe_div(l - m, l) * _safe_div(l + m, l)
         * (l - s) / (2 * l - 1) * (l + s) / (2 * l - 1)
         * (l - s - 1) / (2 * l + 1) * _safe_div(l + s - 1, 2 * l - 3))
    return jnp.sqrt(jnp.abs(t)) * jnp.sign(t)


def _k2m1(s, l, m):
    t = (l - m) * (l + m) / (2 * l - 1) * (l - s) * (l + s) / (2 * l + 1)
    return -2 * m * s * _safe_div(jnp.sqrt(t), l * (l - 1) * (l + 1))


def _k2p0(s, l, m):
    t = (_safe_div(l * (l + 1) - 3 * m * m, (2 * l - 1) * l)
         * (l * (l + 1) - 3 * s * s) / ((2 * l + 3) * (l + 1)))
    return (1.0 + 2.0 * t) / 3.0


def _k2p1(s, l, m):
    t = ((l - m + 1) * (l + m + 1) / (2 * l + 1)
         * (l - s + 1) * (l + s + 1) / (2 * l + 3))
    return -2 * m * s * _safe_div(jnp.sqrt(t), l * (l + 1) * (l + 2))


def _k2p2(s, l, m):
    t = ((l - m + 1) / (l + 1) * (l + m + 1) / (l + 1)
         * (l - m + 2) / (l + 2) * (l + m + 2) / (l + 2)
         * (l - s + 2) / (2 * l + 3) * (l + s + 2) / (2 * l + 3)
         * (l - s + 1) / (2 * l + 1) * (l + s + 1) / (2 * l + 5))
    return jnp.sqrt(t)


def _k1m1(s, l, m):
    t = (l - m) * (l + m) / (2 * l - 1) * (l - s) * (l + s) / (2 * l + 1)
    return _safe_div(jnp.sqrt(t), l)


def _k1p0(s, l, m):
    return -_safe_div(m * s, l * (l + 1))


def _k1p1(s, l, m):
    t = ((l - m + 1) * (l + m + 1) / (2 * l + 3)
         * (l - s + 1) * (l + s + 1) / (2 * l + 1))
    return jnp.sqrt(t) / (l + 1)


def coupling_matrices(s, m, g, lmin, nmax):
    """Dense (B, nmax, nmax) spherical-spheroidal coupling matrices.

    Row/column j corresponds to spherical index l = lmin + j; same
    pentadiagonal bands as hblteuk.swsh.spectral_sparse_matrix.
    """
    m = m[:, None]
    g = g[:, None]
    l = lmin[:, None] + jnp.arange(nmax)[None, :]  # (B, nmax) float

    d0 = (-g * g * _k2p0(s, l, m) + 2 * s * g * _k1p0(s, l, m)
          + l * (l + 1) - s * (s + 1) - 2 * m * g + g * g)
    dp1 = -g * g * _k2p1(s, l, m) + 2 * s * g * _k1p1(s, l, m)  # M[j, j+1] at l_j
    dp2 = -g * g * _k2p2(s, l, m)                               # M[j, j+2] at l_j
    dm1 = -g * g * _k2m1(s, l, m) + 2 * s * g * _k1m1(s, l, m)  # M[j, j-1] at l_j
    dm2 = -g * g * _k2m2(s, l, m)                               # M[j, j-2] at l_j

    B = m.shape[0]
    M = jnp.zeros((B, nmax, nmax))
    j = jnp.arange(nmax)
    M = M.at[:, j, j].set(d0)
    M = M.at[:, j[:-1], j[:-1] + 1].set(dp1[:, :-1])
    M = M.at[:, j[:-2], j[:-2] + 2].set(dp2[:, :-2])
    M = M.at[:, j[1:], j[1:] - 1].set(dm1[:, 1:])
    M = M.at[:, j[2:], j[2:] - 2].set(dm2[:, 2:])
    return M


@functools.partial(jax.jit, static_argnums=(0, 4))
def _eigs_core(s, m, g, lmin, nmax, kval):
    M = coupling_matrices(s, m, g, lmin, nmax)
    w, v = jnp.linalg.eigh(M)  # ascending eigenvalues; matrix is symmetric
    idx = kval[:, None]
    lam = jnp.take_along_axis(w, idx, axis=1)[:, 0]
    vec = jnp.take_along_axis(v, idx[:, :, None], axis=2)[:, :, 0]
    # sign convention of SWSHSeriesBase.generate_eigs: dominant (l-lmin) entry > 0
    comp = jnp.take_along_axis(vec, idx, axis=1)[:, 0]
    vec = vec * jnp.where(comp < 0, -1.0, 1.0)[:, None]
    return lam, vec


def spheroidal_eigs_batched(s, l, m, g, nmax=None):
    """Eigenvalues and spherical-expansion coefficients for a batch of modes.

    Parameters: spin weight s (python int, common to batch); integer arrays
    l, m (B,); spheroidicities g (B,). Truncation nmax defaults to the
    largest per-mode value the CPU path would pick (kval + 22 + 2g).

    Returns (eigenvalues (B,), coeffs (B, nmax), lmin (B,)); coeffs[b, j] is
    the Y_{s, lmin_b + j, m_b} coefficient, CPU sign convention.
    """
    l = np.asarray(l, dtype=np.int64)
    m = np.asarray(m, dtype=np.int64)
    g = np.asarray(g, dtype=np.float64)
    lmin = np.maximum(abs(s), np.abs(m))
    kval = l - lmin
    if nmax is None:
        nmax = int(np.max(kval + np.round(20 + 2 * np.abs(g)) + 2))
    if np.any(kval >= nmax - 2):
        raise ValueError("nmax too small for requested (l, m) modes")

    lam, vec = _eigs_core(int(s), jnp.asarray(m, dtype=jnp.float64),
                          jnp.asarray(g), jnp.asarray(lmin, dtype=jnp.float64),
                          int(nmax), jnp.asarray(kval))
    return lam, vec, lmin


def sphericalY_basis_matrix(s, m, lmin, nterms, z):
    """CPU helper: (nterms, len(z)) matrix of Y_{s, lmin+j, m}(z) rows.

    Batched harmonic values follow as coeffs @ basis. Heavy special-function
    setup stays on host per GPU plan §2.2.
    """
    z = np.atleast_1d(z)
    out = np.empty((nterms, len(z)))
    for j in range(nterms):
        out[j] = sphericalY_expansion(s, lmin + j, m, z)
    return out
