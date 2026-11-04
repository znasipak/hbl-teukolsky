# Batched spin-weighted spheroidal harmonics via dense symmetric eigensolve.
# Mirrors the ARPACK path in hblteuk.swsh (spectral_sparse_matrix / generate_eigs);
# design rationale and validation notes: dev/PHASE1.md.
import functools

import jax
import jax.numpy as jnp
import numpy as np

from scipy.special import gammaln

from ..swsh import Asljm, dAsljm, sphericalY_expansion


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


# ---- scalar Ylm ladder (device) + spin-coupling tables (host) ---------------
# sYlm = sin^-|s| sum_j Asljm Y_jm  (dS: -sin^-(|s|+1) sum_j dAsljm Y_jm), so
# every harmonic and derivative is one contraction against the scalar Y_{L,|m|}
# ladder. Fixed-m recurrence, ascending L; no factorial overflow (CPU lpmv
# path dies near m ~ 150). dev/JAX_PERF.md, Task 2.

@jax.jit
def _ylm_core(mf, log_seed, sign, coeffs, th):
    z = jnp.cos(th)[None, :]
    x = jnp.sin(th)[None, :]
    mc = mf[:, None]
    y0 = sign[:, None] * jnp.exp(log_seed[:, None] + mc * jnp.log(x))
    y1 = z * jnp.sqrt(2 * mc + 3) * y0
    acc = coeffs[:, 0, None] * y0 + coeffs[:, 1, None] * y1

    def step(carry, xs):
        ym2, ym1, acc = carry
        t, c = xs
        L = mc + t
        a = jnp.sqrt((4 * L**2 - 1) / (L**2 - mc**2))
        b = jnp.sqrt(((L - 1)**2 - mc**2) / (4 * (L - 1)**2 - 1))
        y = a * (z * ym1 - b * ym2)
        return (ym1, y, acc + c[:, None] * y), None

    T = coeffs.shape[1]
    ts = jnp.arange(2, T, dtype=mf.dtype)
    (_, _, acc), _ = jax.lax.scan(step, (y0, y1, acc), (ts, coeffs.T[2:]))
    return acc


def ylm_contract(m, coeffs, th):
    """sum_t coeffs[b, t] * Ylm(|m_b| + t, |m_b|, cos th) -> (B, nth).

    m (B,) ints (host), coeffs (B, T) with T >= 2, th (nth,) in (0, pi).
    CPU Ylm normalization at |m| (Condon-Shortley phase); the m < 0 sign is
    carried by the coupling tables, not here.
    """
    ma = np.abs(np.asarray(m, dtype=np.int64))
    log_seed = (0.5 * gammaln(2 * ma + 1) - ma * np.log(2.0) - gammaln(ma + 1)
                + 0.5 * np.log((2 * ma + 1) / (4 * np.pi)))
    sign = np.where(ma % 2 == 0, 1.0, -1.0)
    coeffs = jnp.asarray(coeffs)
    if coeffs.shape[1] < 2:
        coeffs = jnp.pad(coeffs, ((0, 0), (0, 2 - coeffs.shape[1])))
    return _ylm_core(jnp.asarray(ma, dtype=jnp.float64), jnp.asarray(log_seed),
                     jnp.asarray(sign), coeffs, jnp.asarray(th, jnp.float64))


@functools.lru_cache(maxsize=4096)
def _coupling_table(s, m, lmin, nterms, deriv):
    # [j, t] = (d)Asljm(s, lmin+j, |m|+t, m); Ylm(L, m) = (-1)^|m| Ylm(L, |m|)
    T = nterms + 2 * abs(s) + 2
    A = dAsljm if deriv else Asljm
    sgn = (-1.0)**abs(m) if m < 0 else 1.0
    tab = np.array([[A(s, lmin + j, abs(m) + t, m) for t in range(T)]
                    for j in range(nterms)], dtype=np.float64)
    tab *= sgn
    tab.flags.writeable = False
    return tab


def ylm_coupling_coeffs(s, m, lmin, coeffs, deriv):
    """Host: spheroidal coefficients (B, nterms) -> Ylm-ladder coefficients
    (B, nterms + 2|s| + 2) for S (deriv 0) or the dS numerator (deriv 1)."""
    m = np.asarray(m, dtype=np.int64)
    lmin = np.asarray(lmin, dtype=np.int64)
    coeffs = np.asarray(coeffs)
    nterms = coeffs.shape[1]
    out = np.zeros((len(m), nterms + 2 * abs(s) + 2),
                   dtype=np.result_type(coeffs.dtype, np.float64))
    for mm, ll in set(zip(m.tolist(), lmin.tolist())):
        sel = (m == mm) & (lmin == ll)
        out[sel] = coeffs[sel] @ _coupling_table(int(s), mm, ll, nterms,
                                                 int(deriv))
    return out
