# Ultraspherical equation rows (jax/ultraspherical.py) on ODEs with known
# solutions. Design: docs/superpowers/specs/2026-11-11-ultraspherical-radial-design.md
import numpy as np
import numpy.polynomial.chebyshev as ch
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
from scipy.special import airy

from hblteuk.jax.chebyshev import collocation_matrices


def _closure(Pf, Qf, Uf):
    # pqu_fn over a batch of one: z (k,) -> (P, Q, U) each (1, k)
    def pqu(z):
        return tuple(jnp.broadcast_to(jnp.asarray(f(z), jnp.complex128),
                                      z.shape)[None, :] for f in (Pf, Qf, Uf))
    return pqu


def solve(Pf, Qf, Uf, n, bc, z0=-1.0, z1=1.0):
    # one domain; bc = (u, du/dz) at z0
    from hblteuk.jax.ultraspherical import equation_rows
    L = np.asarray(equation_rows(_closure(Pf, Qf, Uf), jnp.array([z0]),
                                 jnp.array([z1]), n))[0, 0]
    h = 0.5 * (z1 - z0)
    M = np.vstack([collocation_matrices(n)["bc_left"], L])
    rhs = np.zeros(n, dtype=complex)
    rhs[:2] = bc[0], h * bc[1]
    return np.linalg.solve(M, rhs)


def test_monomial_fit_exact():
    from hblteuk.jax.ultraspherical import X_FIT, monomials
    f = lambda x: 2 - 3j * x + 0.5 * x**2 + 4 * x**3
    np.testing.assert_allclose(monomials(np.array([f(x) for x in X_FIT])),
                               [2, -3j, 0.5, 4], atol=1e-14)


def test_operators_are_banded():
    from hblteuk.jax.ultraspherical import operators
    for M in operators(40):
        r, c = np.nonzero(np.abs(M).sum(axis=0) > 0)
        assert np.all(c - r <= 7) and np.all(r - c <= 3)


def test_harmonic_oscillator():
    c = solve(lambda z: 1, lambda z: 0, lambda z: 1, 32, (1.0, 0.0))
    x = np.linspace(-1, 1, 41)
    np.testing.assert_allclose(ch.chebval(x, c), np.cos(x + 1), atol=1e-14)


def test_airy_variable_coefficient():
    # u'' - z u = 0 on z in [-8, 2], marched from z = 2 down (h = -5)
    ai, aip, _, _ = airy(2.0)
    c = solve(lambda z: 1, lambda z: 0, lambda z: -z, 64, (ai, aip),
              z0=2.0, z1=-8.0)
    t = np.linspace(-1, 1, 51)
    np.testing.assert_allclose(ch.chebval(t, c).real, airy(-3 - 5 * t)[0],
                               atol=1e-13)


def test_cubic_coefficients_match_collocation():
    # all four slots used, against the nodal collocation solve
    P = lambda x: 3 + x + 0.2 * x**3
    Q = lambda x: 1 - 0.5j * x**2
    U = lambda x: -2 + 0.3 * x
    n = 48
    c_us = solve(P, Q, U, n, (1.0, 0.5))
    m = collocation_matrices(n)
    nd = m["nodes"]
    M = (np.array([P(x) for x in nd])[:, None] * m["T2"]
         + np.array([Q(x) for x in nd])[:, None] * m["T1"]
         + np.array([U(x) for x in nd])[:, None] * m["T0"])
    M[0], M[1] = m["bc_left"]
    rhs = np.zeros(n, complex)
    rhs[:2] = (1.0, 0.5)
    np.testing.assert_allclose(c_us, np.linalg.solve(M, rhs), atol=1e-13)


@pytest.mark.parametrize("s", [-2, 2])
def test_teuk_pqu_is_cubic(s):
    # equation_rows assumes P, Q, U are polynomials of degree <= 3 in sigma;
    # a fifth point off the fit nodes must agree with the cubic interpolant
    from hblteuk.jax.teukolsky import teuk_pqu
    from hblteuk.jax.ultraspherical import X_FIT, monomials
    rng = np.random.default_rng(s + 5)
    lam = jnp.asarray(rng.normal(size=3) + 1j * rng.normal(size=3))
    ma = jnp.asarray(rng.normal(size=3))
    om = jnp.asarray(rng.normal(size=3) + 0.1j * rng.normal(size=3))
    kappa, mid, h, x5 = 0.6, 0.4, 0.3, 0.37
    fit = teuk_pqu(jnp.asarray(mid + h * X_FIT), kappa, s, lam, ma, om)
    at5 = teuk_pqu(jnp.asarray([mid + h * x5]), kappa, s, lam, ma, om)
    for f, g in zip(fit, at5):
        c = monomials(np.asarray(f))
        interp = c @ (x5 ** np.arange(4))
        scale = np.max(np.abs(np.asarray(f)))
        assert np.max(np.abs(interp - np.asarray(g)[:, 0])) <= 1e-13 * scale


def test_equation_rows_shape():
    from hblteuk.jax.ultraspherical import equation_rows

    def pqu(z):
        w = jnp.arange(1.0, 4.0)[:, None]
        return (jnp.ones((3, z.shape[0])) + 0j, 0j * w * z, -w * z + 0j)

    z = jnp.linspace(-8.0, 2.0, 6)
    assert equation_rows(pqu, z[:-1], z[1:], 24).shape == (3, 5, 22, 24)


def test_equation_rows_equal_operator_sum():
    # rows are sum_j P_j A_j + (hQ)_j B_j + (h^2 U)_j C_j per domain, for
    # complex cubic coefficients; pins any reformulation of the contraction
    from hblteuk.jax.ultraspherical import (X_FIT, equation_rows, monomials,
                                            operators)
    rng = np.random.default_rng(7)
    cP, cQ, cU = (rng.normal(size=(2, 4)) + 1j * rng.normal(size=(2, 4))
                  for _ in range(3))

    def pqu(z):
        zp = z[None, :] ** jnp.arange(4)[:, None]            # (4, k)
        return tuple(jnp.asarray(c) @ zp for c in (cP, cQ, cU))

    n, z = 20, np.array([0.1, 0.5, 0.2])
    got = np.asarray(equation_rows(pqu, jnp.asarray(z[:-1]),
                                   jnp.asarray(z[1:]), n))
    A, B, C = operators(n)
    for d in range(2):
        h, mid = 0.5 * (z[d + 1] - z[d]), 0.5 * (z[d + 1] + z[d])
        x = mid + h * X_FIT
        Pc, Qc, Uc = (monomials(np.polynomial.polynomial.polyval(x, c.T))
                      for c in (cP, cQ, cU))
        want = (np.einsum("bj,jrc->brc", Pc, A)
                + np.einsum("bj,jrc->brc", h * Qc, B)
                + np.einsum("bj,jrc->brc", h**2 * Uc, C))
        np.testing.assert_allclose(got[:, d], want, rtol=0,
                                   atol=1e-13 * np.abs(want).max())
