# Batched JAX source integration vs CPU reference. See dev/PHASE3.md.
import numpy as np
import pytest

jax = pytest.importorskip("jax")

from hblteuk import collocode
from hblteuk.collocode import generate_subdomain_list
from hblteuk.jax import (equatorial_swsh_values, integrate_source_batch,
                         orbit_samples, solve_homogeneous_batch,
                         spheroidal_eigs_batched)
from hblteuk.mode import TeukolskyPointParticleModeGenerator
from hblteuk.swsh import SpinWeightedSpheroidalHarmonic
from hblteuk.teuk import sigma_r
from .conftest import ORBIT, SOLVER_KWARGS

A = ORBIT[0]
KAPPA = np.sqrt(1 - A**2)
MODES = [(2, 2, n) for n in range(-3, 4)] + [(3, 1, 1), (10, 7, 0), (4, -3, 1)]


@pytest.fixture(scope="module")
def table(orbit):
    l = np.array([c[0] for c in MODES])
    m = np.array([c[1] for c in MODES])
    n = np.array([c[2] for c in MODES])
    om = np.array([orbit.mode_frequency(mm, 0, nn) for _, mm, nn in MODES])
    lam, coeffs, lmin = spheroidal_eigs_batched(-2, l, m, A * om)
    return l, m, n, om, np.asarray(lam), np.asarray(coeffs), lmin


@pytest.fixture(scope="module")
def setup(orbit, table):
    l, m, n, om, lam, coeffs, lmin = table
    smin = sigma_r(orbit.radial_roots[0], KAPPA)
    smax = sigma_r(orbit.radial_roots[1], KAPPA)
    subs_up = generate_subdomain_list([0, smax], 150, "arcsinh7")
    subs_in = generate_subdomain_list([1, smin], 150, "arcsinh7")
    hom = solve_homogeneous_batch(-2, KAPPA, lam, m * A, om, subs_up, subs_in)
    samples = orbit_samples(orbit, nsamples=2**10)
    S, dS, d2S = equatorial_swsh_values(-2, m, lmin, coeffs, lam, A * om)
    return samples, hom, subs_up, subs_in, (S, dS, d2S)


def run(setup, table, method, nsamples=None):
    samples, hom, subs_up, subs_in, (S, dS, d2S) = setup
    l, m, n, om, lam, _, _ = table
    return integrate_source_batch(samples, lam, m, n, om, S, dS, d2S,
                                  hom, subs_up, subs_in, method=method)


def test_equatorial_swsh_values(table):
    l, m, n, om, lam, coeffs, lmin = table
    S, dS, d2S = equatorial_swsh_values(-2, m, lmin, coeffs, lam, A * om)
    th = 0.5 * np.pi
    for b, (ll, mm, _) in enumerate(MODES):
        ref = SpinWeightedSpheroidalHarmonic(-2, ll, mm, A * om[b])
        assert S[b] == pytest.approx(ref.eval(th), rel=1e-10)
        assert dS[b] == pytest.approx(ref.deriv(th), rel=1e-10, abs=1e-12)
        assert d2S[b] == pytest.approx(ref.deriv2(th), rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("method", ["fft", "quad"])
def test_vs_cpu(orbit, setup, table, method):
    res = run(setup, table, method)
    amps = np.asarray(res.amplitudes)
    solver = collocode.CollocationODEMultiDomainFixedStepSolver(n=32)
    gen = TeukolskyPointParticleModeGenerator(orbit, solver,
                                              dict(SOLVER_KWARGS))
    for b, (l, m, n) in enumerate(MODES):
        mode = gen(-2, l, m, 0, n)
        cpu = np.array([mode.amplitudes["In"], mode.amplitudes["Up"]])
        np.testing.assert_allclose(amps[b], cpu, rtol=1e-7,
                                   err_msg=f"{(l, m, n)} {method}")


def test_vs_golden(setup, table, golden):
    res = run(setup, table, "fft")
    amps = np.asarray(res.amplitudes)
    for j, n in enumerate(golden["ecc_n"]):
        b = MODES.index((2, 2, int(n)))
        np.testing.assert_allclose(amps[b], golden["ecc_amplitudes"][j],
                                   rtol=1e-7, err_msg=f"n={n}")


def test_fft_equals_quad(setup, table):
    a1 = np.asarray(run(setup, table, "fft").amplitudes)
    a2 = np.asarray(run(setup, table, "quad").amplitudes)
    np.testing.assert_allclose(a1, a2, rtol=1e-12)


def test_error_estimates(orbit, setup, table):
    # fine-grid estimates small; coarse-grid actual error bounded by estimate
    res_fine = run(setup, table, "fft")
    assert float(np.max(np.asarray(res_fine.errors))) < 1e-8

    samples_c = orbit_samples(orbit, nsamples=2**6)
    _, hom, subs_up, subs_in, (S, dS, d2S) = setup
    l, m, n, om, lam, _, _ = table
    res_c = integrate_source_batch(samples_c, lam, m, n, om, S, dS, d2S,
                                   hom, subs_up, subs_in, method="fft")
    actual = np.abs(np.asarray(res_c.amplitudes)
                    - np.asarray(res_fine.amplitudes)) \
        / np.abs(np.asarray(res_fine.amplitudes))
    est = np.asarray(res_c.errors)
    assert np.max(est) > np.max(np.asarray(res_fine.errors))  # coarser -> larger
    # estimate must not dangerously underestimate the true error
    # (overestimating a nearly-converged mode is safe and does occur)
    sel = actual > 1e-12
    ratio = est[sel] / actual[sel]
    assert np.all(ratio > 1e-2)


def test_conjugate_symmetry(orbit, setup, table):
    # A_{l,-m,-n} = (-1)^l conj(A_{l,m,n}) for equatorial orbits
    samples, hom, subs_up, subs_in, _ = setup
    l, m, n, om, lam, coeffs, lmin = table
    lam2, coeffs2, lmin2 = spheroidal_eigs_batched(-2, l, -m, -A * om)
    lam2 = np.asarray(lam2)
    hom2 = solve_homogeneous_batch(-2, KAPPA, lam2, -m * A, -om,
                                   subs_up, subs_in)
    S2, dS2, d2S2 = equatorial_swsh_values(-2, -m, lmin2,
                                           np.asarray(coeffs2), lam2, -A * om)
    r1 = np.asarray(run(setup, table, "fft").amplitudes)
    r2 = np.asarray(integrate_source_batch(
        samples, lam2, -m, -n, -om, S2, dS2, d2S2, hom2, subs_up, subs_in,
        method="fft").amplitudes)
    for b, (ll, _, _) in enumerate(MODES):
        np.testing.assert_allclose(r2[b], (-1.)**ll * np.conj(r1[b]),
                                   rtol=1e-7, err_msg=str(MODES[b]))
