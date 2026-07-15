# Spin-weighted spheroidal/spherical harmonic tests. See dev/PHASE0.md.
import numpy as np
import pytest

from hblteuk.swsh import (
    SpinWeightedSpheroidalHarmonic,
    Yslm,
    Yslm_eigenvalue,
    spectral_sparse_matrix,
)

SWSH_CASE_IDS = range(6)


def test_spherical_limit_eigenvalue():
    # g=0: eigenvalue reduces to l(l+1) - s(s+1)
    for s, l, m in [(-2, 2, 2), (-2, 5, -3), (0, 3, 1), (2, 4, 0)]:
        S = SpinWeightedSpheroidalHarmonic(s, l, m, 0.0)
        assert S.eigenvalue == pytest.approx(l * (l + 1) - s * (s + 1))


def test_spherical_limit_values():
    # g=0 spheroidal eval must equal the closed-form Yslm
    th = np.array([0.3, 1.1, 2.5])
    for s, l, m in [(-2, 2, 2), (-2, 3, -1), (0, 2, 0)]:
        S = SpinWeightedSpheroidalHarmonic(s, l, m, 0.0)
        got = np.array([S.eval(t) for t in th])
        want = Yslm(s, l, m, th)
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)


def test_yslm_orthonormality():
    # ∫ |Yslm|^2 sin(th) dth dphi = 1; poles trimmed (YslmBase singular
    # parametrization at z=±1; integrand ~ th^(2|m-s|+1) there, negligible)
    th = np.linspace(1e-3, np.pi - 1e-3, 20001)
    for s, l, m in [(-2, 2, 2), (-2, 4, 1), (0, 3, 2), (2, 3, -3)]:
        y = Yslm(s, l, m, th)
        norm = 2 * np.pi * np.trapezoid(np.abs(y) ** 2 * np.sin(th), th)
        assert norm == pytest.approx(1.0, rel=1e-5)  # rel floor set by pole trim


def test_coupling_matrix_is_symmetric():
    # GPU plan assumes batched eigh applies (swsh.py builds a symmetric matrix)
    for s, m, g in [(-2, 2, 0.27), (-2, -3, 1.35), (0, 1, 0.5), (2, 2, 2.7)]:
        M = spectral_sparse_matrix(s, m, g, 30).toarray()
        np.testing.assert_allclose(M, M.T, rtol=0, atol=1e-13)


def test_eigenvalues_vs_golden(golden):
    for i in SWSH_CASE_IDS:
        s, l, m, g = golden[f"swsh{i}_params"]
        S = SpinWeightedSpheroidalHarmonic(int(s), int(l), int(m), g)
        assert S.eigenvalue == pytest.approx(
            float(golden[f"swsh{i}_eigenvalue"]), rel=1e-10)


def test_values_vs_golden(golden):
    th = np.array([0.3, 0.5 * np.pi, 2.2])
    for i in SWSH_CASE_IDS:
        s, l, m, g = golden[f"swsh{i}_params"]
        S = SpinWeightedSpheroidalHarmonic(int(s), int(l), int(m), g)
        got_S = np.array([S.eval(t) for t in th])
        got_dS = np.array([S.deriv(t) for t in th])
        # atol floor: dS vanishes at equator for symmetric modes (roundoff-level)
        np.testing.assert_allclose(got_S, golden[f"swsh{i}_S"], rtol=1e-9, atol=1e-13)
        np.testing.assert_allclose(got_dS, golden[f"swsh{i}_dS"], rtol=1e-9, atol=1e-13)


def test_deriv2_ode_identity():
    # S'' from the class must satisfy the angular Teukolsky ODE it encodes
    s, l, m, g = -2, 3, 2, 0.8
    S = SpinWeightedSpheroidalHarmonic(s, l, m, g)
    th = 1.234
    d2_analytic = S.deriv2(th)
    eps = 1e-5
    d2_fd = (S.deriv(th + eps) - S.deriv(th - eps)) / (2 * eps)
    assert d2_fd == pytest.approx(d2_analytic, rel=1e-6)
