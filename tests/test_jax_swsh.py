# Batched JAX spheroidal harmonics vs CPU/ARPACK reference. See dev/PHASE1.md.
import numpy as np
import pytest

jax = pytest.importorskip("jax")

from hblteuk.jax import spheroidal_eigs_batched, sphericalY_basis_matrix
from hblteuk.swsh import SpinWeightedSpheroidalHarmonic


def cpu_reference(s, l, m, g):
    S = SpinWeightedSpheroidalHarmonic(s, l, m, g)
    return S.eigenvalue, np.asarray(S.coeffs)


def assert_coeffs_match(got, want, atol):
    # representations may differ in truncation length; tails are ~eps
    ncmp = min(len(got), len(want))
    np.testing.assert_allclose(got[:ncmp], want[:ncmp], atol=atol, rtol=0)
    for tail in (got[ncmp:], want[ncmp:]):
        assert np.all(np.abs(tail) < atol)


def test_x64_enabled():
    lam, vec, _ = spheroidal_eigs_batched(-2, [2], [2], [0.27])
    assert lam.dtype == np.float64 and vec.dtype == np.float64


def test_spherical_limit():
    l = np.arange(2, 9)
    lam, vec, lmin = spheroidal_eigs_batched(-2, l, np.full_like(l, 2), np.zeros(len(l)))
    np.testing.assert_allclose(lam, l * (l + 1) - (-2) * (-2 + 1), atol=1e-12)
    for b, lb in enumerate(l):
        want = np.zeros(vec.shape[1])
        want[lb - lmin[b]] = 1.0
        np.testing.assert_allclose(vec[b], want, atol=1e-12)


def test_vs_cpu_sweep():
    s = -2
    cases = [(l, m, g) for l in range(2, 13) for m in (-l, -2, 0, 1, l)
             for g in (0.27, 2.7) if abs(m) <= l]
    l_arr = np.array([c[0] for c in cases])
    m_arr = np.array([c[1] for c in cases])
    g_arr = np.array([c[2] for c in cases])
    lam, vec, lmin = spheroidal_eigs_batched(s, l_arr, m_arr, g_arr)

    for b, (l, m, g) in enumerate(cases):
        lam_cpu, coeffs_cpu = cpu_reference(s, l, m, g)
        assert lam[b] == pytest.approx(lam_cpu, rel=1e-10), (l, m, g)
        assert_coeffs_match(np.asarray(vec[b]), coeffs_cpu, atol=1e-10)


def test_vs_golden(golden):
    for i in range(6):
        s, l, m, g = golden[f"swsh{i}_params"]
        lam, vec, lmin = spheroidal_eigs_batched(
            int(s), [int(l)], [int(m)], [g])
        assert lam[0] == pytest.approx(float(golden[f"swsh{i}_eigenvalue"]),
                                       rel=1e-10)
        assert_coeffs_match(np.asarray(vec[0]), golden[f"swsh{i}_coeffs"],
                            atol=1e-10)


def test_eigenvector_convention():
    lam, vec, lmin = spheroidal_eigs_batched(
        -2, [2, 5, 9], [2, -3, 0], [0.27, 1.35, 2.7])
    norms = np.linalg.norm(vec, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-13)
    l_arr, lmin_arr = np.array([2, 5, 9]), lmin
    for b in range(3):
        assert vec[b, l_arr[b] - lmin_arr[b]] > 0


def test_harmonic_evaluation():
    # coeffs @ basis must reproduce CPU Sslm evaluation
    s, cases = -2, [(2, 2, 0.27), (5, -3, 1.35), (10, 7, 2.7)]
    th = np.array([0.3, 0.5 * np.pi, 2.2])
    z = np.cos(th)
    lam, vec, lmin = spheroidal_eigs_batched(
        s, [c[0] for c in cases], [c[1] for c in cases], [c[2] for c in cases])
    for b, (l, m, g) in enumerate(cases):
        basis = sphericalY_basis_matrix(s, m, int(lmin[b]), vec.shape[1], z)
        got = np.asarray(vec[b]) @ basis
        S = SpinWeightedSpheroidalHarmonic(s, l, m, g)
        want = np.array([S.eval(t) for t in th])
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)


def test_nmax_override_and_validation():
    lam1, _, _ = spheroidal_eigs_batched(-2, [2], [2], [0.27])
    lam2, _, _ = spheroidal_eigs_batched(-2, [2], [2], [0.27], nmax=60)
    assert lam1[0] == pytest.approx(float(lam2[0]), rel=1e-12)
    with pytest.raises(ValueError):
        spheroidal_eigs_batched(-2, [30], [2], [0.27], nmax=10)
