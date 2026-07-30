# Batched spherical-inclined (1D q_theta) source integration vs the
# committed pybhpt snapshot. See dev/PHASE5B.md.
import pathlib

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from hblteuk.collocode import generate_subdomain_list
from hblteuk.geo import MiniGeo
from hblteuk.jax import (flux_modes, solve_homogeneous_batch, solve_mode_table,
                         solve_spherical_mode_table, spheroidal_eigs_batched)
from hblteuk.jax.source import (integrate_source_spherical_batch,
                                polar_samples, polar_swsh_values)
from hblteuk.swsh import SpinWeightedSpheroidalHarmonic
from hblteuk.teuk import sigma_r

REF_PATH = pathlib.Path(__file__).parent / "golden" / "pybhpt_reference_v2.npz"
COLS = dict(l=0, m=1, k=2, omega=3, lam=4, zin_re=5, zin_im=6,
            zup_re=7, zup_im=8, edot_i=9, edot_h=10, ldot_i=11, ldot_h=12)


@pytest.fixture(scope="module")
def ref():
    return np.load(REF_PATH)


@pytest.fixture(scope="module")
def sph_orbit(ref):
    return MiniGeo(*ref["sph_params"])


def test_polar_samples(sph_orbit):
    s = polar_samples(sph_orbit, nsamples=2**6)
    assert s.qth.shape == s.thp.shape == s.uthp.shape == (2**6,)
    assert s.uthp[0] == 0.0 and s.uthp[2**5] == 0.0   # turning points
    assert np.all(s.uthp[1:2**5] > 0) and np.all(s.uthp[2**5 + 1:] < 0)
    thmin = np.arccos(np.sqrt(sph_orbit.polar_roots[0]))
    assert np.all((s.thp >= thmin - 1e-12) & (s.thp <= np.pi - thmin + 1e-12))
    assert s.rp == pytest.approx(sph_orbit.radial_roots[0])


def test_polar_swsh_values(ref, sph_orbit):
    table = ref["sph_table"]
    l = table[:, COLS["l"]].astype(int)
    m = table[:, COLS["m"]].astype(int)
    om = table[:, COLS["omega"]]
    a = sph_orbit.a
    lam, coeffs, lmin = spheroidal_eigs_batched(-2, l, m, a * om)
    th = polar_samples(sph_orbit, nsamples=2**5).thp
    S, dS, d2S = polar_swsh_values(-2, m, lmin, np.asarray(coeffs),
                                   np.asarray(lam), a * om, th)
    # SpinWeightedSpheroidalHarmonic.eval/deriv/deriv2 only accept scalar
    # theta (Sslm assigns per-point into a preallocated array internally),
    # so compare pointwise rather than passing the whole theta grid.
    for b in range(len(l)):
        cpu = SpinWeightedSpheroidalHarmonic(-2, int(l[b]), int(m[b]),
                                             a * om[b])
        S_cpu = np.array([cpu.eval(t) for t in th])
        dS_cpu = np.array([cpu.deriv(t) for t in th])
        d2S_cpu = np.array([cpu.deriv2(t) for t in th])
        np.testing.assert_allclose(np.asarray(S)[b], S_cpu,
                                   rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(np.asarray(dS)[b], dS_cpu,
                                   rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(np.asarray(d2S)[b], d2S_cpu,
                                   rtol=1e-10, atol=1e-12)


@pytest.fixture(scope="module")
def sph_setup(ref, sph_orbit):
    table = ref["sph_table"]
    l = table[:, COLS["l"]].astype(int)
    m = table[:, COLS["m"]].astype(int)
    k = table[:, COLS["k"]].astype(int)
    om = np.array([sph_orbit.mode_frequency(int(mm), int(kk), 0)
                   for mm, kk in zip(m, k)])
    a = sph_orbit.a
    kappa = np.sqrt(1 - a**2)
    lam, coeffs, lmin = spheroidal_eigs_batched(-2, l, m, a * om)
    lam = np.asarray(lam)
    coeffs = np.asarray(coeffs)
    sp = sigma_r(sph_orbit.radial_roots[0], kappa)
    subs_up = generate_subdomain_list([0, sp], 150, "arcsinh7")
    subs_in = generate_subdomain_list([1, sp], 150, "arcsinh7")
    hom = solve_homogeneous_batch(-2, kappa, lam, m * a, om, subs_up, subs_in)
    samples = polar_samples(sph_orbit, nsamples=2**8)
    return samples, (l, m, k, om, lam), (coeffs, lmin, a), hom, subs_up, subs_in


def run_sph(sph_setup, method, samples=None):
    # S/dS/d2S depend on the theta grid, so they must be recomputed for
    # whichever samples are in use (found while implementing this task:
    # reusing swsh values precomputed on a different-density grid breaks
    # with a shape mismatch -- see dev/PHASE5B.md).
    smp, (l, m, k, om, lam), (coeffs, lmin, a), hom, su, si = sph_setup
    if samples is not None:
        smp = samples
    S, dS, d2S = polar_swsh_values(-2, m, lmin, coeffs, lam, a * om, smp.thp)
    return integrate_source_spherical_batch(smp, lam, m, k, om, S, dS, d2S,
                                            hom, su, si, method=method)


@pytest.mark.parametrize("method", ["fft", "quad"])
def test_amplitudes_vs_pybhpt(ref, sph_setup, method):
    # THE gate: batched sphinc Z == pybhpt Z, magnitude AND phase
    table = ref["sph_table"]
    smp, (l, m, k, om, lam), _, _, _, _ = sph_setup
    np.testing.assert_allclose(om, table[:, COLS["omega"]], rtol=1e-12)
    np.testing.assert_allclose(lam, table[:, COLS["lam"]], rtol=1e-10)
    amps = np.asarray(run_sph(sph_setup, method).amplitudes)
    zin = table[:, COLS["zin_re"]] + 1j * table[:, COLS["zin_im"]]
    zup = table[:, COLS["zup_re"]] + 1j * table[:, COLS["zup_im"]]
    np.testing.assert_allclose(amps[:, 0], zin, rtol=1e-7)
    np.testing.assert_allclose(amps[:, 1], zup, rtol=1e-7)


def test_fft_equals_quad(sph_setup):
    a1 = np.asarray(run_sph(sph_setup, "fft").amplitudes)
    a2 = np.asarray(run_sph(sph_setup, "quad").amplitudes)
    np.testing.assert_allclose(a1, a2, rtol=1e-12)


def test_error_estimates(sph_orbit, sph_setup):
    res_fine = run_sph(sph_setup, "fft")
    assert float(np.max(np.asarray(res_fine.errors))) < 1e-8
    coarse = polar_samples(sph_orbit, nsamples=2**4)
    res_c = run_sph(sph_setup, "fft", samples=coarse)
    actual = np.abs(np.asarray(res_c.amplitudes)
                    - np.asarray(res_fine.amplitudes)) \
        / np.abs(np.asarray(res_fine.amplitudes))
    est = np.asarray(res_c.errors)
    sel = actual > 1e-12
    # never a dangerous underestimate (overestimation is fine by design)
    assert np.all(est[sel] / actual[sel] > 1e-2)


@pytest.fixture(scope="module")
def sol(ref, sph_orbit):
    t = ref["sph_table"]
    return solve_spherical_mode_table(sph_orbit, -2,
                                      t[:, COLS["l"]].astype(int),
                                      t[:, COLS["m"]].astype(int),
                                      t[:, COLS["k"]].astype(int))


def test_pipeline_wrapper_matches_gate(ref, sol):
    table = ref["sph_table"]
    zin = table[:, COLS["zin_re"]] + 1j * table[:, COLS["zin_im"]]
    zup = table[:, COLS["zup_re"]] + 1j * table[:, COLS["zup_im"]]
    amps = np.asarray(sol.amplitudes)
    np.testing.assert_allclose(amps[:, 0], zin, rtol=1e-7)
    np.testing.assert_allclose(amps[:, 1], zup, rtol=1e-7)


def test_fluxes_vs_pybhpt(ref, sph_orbit, sol):
    table = ref["sph_table"]
    fx = flux_modes(sol.amplitudes, sol.m, sol.omega, sol.eigenvalues,
                    sph_orbit.a)
    Edot, Ldot = np.asarray(fx.Edot), np.asarray(fx.Ldot)
    np.testing.assert_allclose(Edot[:, 0], table[:, COLS["edot_i"]], rtol=1e-7)
    np.testing.assert_allclose(Edot[:, 1], table[:, COLS["edot_h"]], rtol=1e-7)
    np.testing.assert_allclose(Ldot[:, 0], table[:, COLS["ldot_i"]], rtol=1e-7)
    np.testing.assert_allclose(Ldot[:, 1], table[:, COLS["ldot_h"]], rtol=1e-7)


def test_equatorial_degeneration():
    # x = 1: the polar machinery must reproduce the equatorial path exactly
    orbit = MiniGeo(0.9, 10.0, 0.0, 1.0)
    l = np.array([2, 3, 4])
    m = np.array([2, 2, 3])
    sph = solve_spherical_mode_table(orbit, -2, l, m, np.zeros_like(l))
    eq = solve_mode_table(orbit, -2, l, m, np.zeros_like(l))
    np.testing.assert_allclose(np.asarray(sph.amplitudes),
                               np.asarray(eq.amplitudes), rtol=1e-9)


def test_conjugate_symmetry(ref, sph_orbit, sol):
    # Z_{l,-m,-k} = (-1)^(l+k) conj(Z_{l,m,k}) from an independent solve
    t = ref["sph_table"]
    l = t[:, COLS["l"]].astype(int)
    m = t[:, COLS["m"]].astype(int)
    k = t[:, COLS["k"]].astype(int)
    sol2 = solve_spherical_mode_table(sph_orbit, -2, l, -m, -k)
    a1 = np.asarray(sol.amplitudes)
    a2 = np.asarray(sol2.amplitudes)
    for b in range(len(l)):
        np.testing.assert_allclose(
            a2[b], (-1.0)**(l[b] + k[b]) * np.conj(a1[b]), rtol=1e-7,
            err_msg=f"(l,m,k)=({l[b]},{m[b]},{k[b]})")


def test_static_mode_rejected(sph_orbit):
    with pytest.raises(ValueError, match="static"):
        solve_spherical_mode_table(sph_orbit, -2, np.array([2]),
                                   np.array([0]), np.array([0]))
