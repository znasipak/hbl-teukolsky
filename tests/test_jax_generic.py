# Batched generic-orbit (2D torus) source integration vs the committed
# pybhpt snapshot. See dev/PHASE5C.md.
import pathlib

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from hblteuk.collocode import generate_subdomain_list
from hblteuk.geo import MiniGeo
from hblteuk.jax import (flux_modes, solve_generic_mode_table,
                         solve_homogeneous_batch, solve_mode_table,
                         solve_spherical_mode_table, spheroidal_eigs_batched)
from hblteuk.jax.source import (generic_samples,
                                integrate_source_generic_batch,
                                polar_swsh_values)
from hblteuk.teuk import sigma_r

REF_PATH = pathlib.Path(__file__).parent / "golden" / "pybhpt_reference_v2.npz"
COLS = dict(l=0, m=1, k=2, n=3, omega=4, lam=5, zin_re=6, zin_im=7,
            zup_re=8, zup_im=9, edot_i=10, edot_h=11, ldot_i=12, ldot_h=13,
            qdot_i=14, qdot_h=15)


@pytest.fixture(scope="module")
def ref():
    return np.load(REF_PATH)


@pytest.fixture(scope="module")
def gen_orbit(ref):
    return MiniGeo(*ref["gen_params"])


def test_generic_samples(gen_orbit):
    s = generic_samples(gen_orbit, nsamples=2**6, ntheta=2**5)
    assert s.rp.shape == s.urp.shape == (2**6,)
    assert s.thp.shape == s.uthp.shape == (2**5,)
    assert s.urp[0] == 0.0 and s.urp[2**5] == 0.0
    assert s.uthp[0] == 0.0 and s.uthp[2**4] == 0.0
    assert np.all(s.urp[1:2**5] > 0)      # rmin -> rmax on (0, pi)
    assert np.all(s.uthp[1:2**4] > 0)     # prograde: theta_min -> theta_max
    with pytest.raises(ValueError):
        generic_samples(MiniGeo(0.9, 10.0, 0.0, 0.5))   # e = 0
    with pytest.raises(ValueError):
        generic_samples(MiniGeo(0.9, 10.0, 0.2, 1.0))   # |x| = 1


@pytest.fixture(scope="module")
def gen_setup(ref, gen_orbit):
    t = ref["gen_table"]
    l = t[:, COLS["l"]].astype(int)
    m = t[:, COLS["m"]].astype(int)
    k = t[:, COLS["k"]].astype(int)
    n = t[:, COLS["n"]].astype(int)
    om = np.array([gen_orbit.mode_frequency(int(mm), int(kk), int(nn))
                   for mm, kk, nn in zip(m, k, n)])
    a = gen_orbit.a
    kappa = np.sqrt(1 - a**2)
    lam, coeffs, lmin = spheroidal_eigs_batched(-2, l, m, a * om)
    lam = np.asarray(lam)
    coeffs = np.asarray(coeffs)
    smin = sigma_r(gen_orbit.radial_roots[0], kappa)
    smax = sigma_r(gen_orbit.radial_roots[1], kappa)
    subs_up = generate_subdomain_list([0, smax], 150, "arcsinh7")
    subs_in = generate_subdomain_list([1, smin], 150, "arcsinh7")
    hom = solve_homogeneous_batch(-2, kappa, lam, m * a, om, subs_up, subs_in)
    samples = generic_samples(gen_orbit, nsamples=2**9, ntheta=2**7)
    return samples, (l, m, k, n, om, lam), (coeffs, lmin, a), hom, subs_up, subs_in


def run_gen(gen_setup, method, samples=None):
    smp, (l, m, k, n, om, lam), (coeffs, lmin, a), hom, su, si = gen_setup
    if samples is not None:
        smp = samples
    S, dS, d2S = polar_swsh_values(-2, m, lmin, coeffs, lam, a * om, smp.thp)
    return integrate_source_generic_batch(smp, lam, m, k, n, om, S, dS, d2S,
                                          hom, su, si, method=method)


@pytest.mark.parametrize("method", ["fft", "fft2"])
def test_amplitudes_vs_pybhpt(ref, gen_setup, method):
    # THE gate: batched generic Z == pybhpt Z, magnitude AND phase
    t = ref["gen_table"]
    smp, (l, m, k, n, om, lam), _, _, _, _ = gen_setup
    np.testing.assert_allclose(om, t[:, COLS["omega"]], rtol=1e-12)
    np.testing.assert_allclose(lam, t[:, COLS["lam"]], rtol=1e-10)
    amps = np.asarray(run_gen(gen_setup, method).amplitudes)
    zin = t[:, COLS["zin_re"]] + 1j * t[:, COLS["zin_im"]]
    zup = t[:, COLS["zup_re"]] + 1j * t[:, COLS["zup_im"]]
    np.testing.assert_allclose(amps[:, 0], zin, rtol=1e-7)
    np.testing.assert_allclose(amps[:, 1], zup, rtol=1e-7)


def test_contraction_equals_ifft2(gen_setup):
    a1 = np.asarray(run_gen(gen_setup, "fft").amplitudes)
    a2 = np.asarray(run_gen(gen_setup, "fft2").amplitudes)
    np.testing.assert_allclose(a1, a2, rtol=1e-12)


def test_error_estimates(gen_orbit, gen_setup):
    res_fine = run_gen(gen_setup, "fft")
    assert float(np.max(np.asarray(res_fine.errors))) < 1e-6
    coarse = generic_samples(gen_orbit, nsamples=2**6, ntheta=2**4)
    res_c = run_gen(gen_setup, "fft", samples=coarse)
    actual = np.abs(np.asarray(res_c.amplitudes)
                    - np.asarray(res_fine.amplitudes)) \
        / np.abs(np.asarray(res_fine.amplitudes))
    est = np.asarray(res_c.errors)
    sel = actual > 1e-12
    assert np.all(est[sel] / actual[sel] > 1e-2)   # never underestimate


@pytest.fixture(scope="module")
def sol(ref, gen_orbit):
    t = ref["gen_table"]
    return solve_generic_mode_table(gen_orbit, -2,
                                    t[:, COLS["l"]].astype(int),
                                    t[:, COLS["m"]].astype(int),
                                    t[:, COLS["k"]].astype(int),
                                    t[:, COLS["n"]].astype(int),
                                    nsamples=2**9, ntheta=2**7)


def test_pipeline_wrapper_matches_gate(ref, sol):
    t = ref["gen_table"]
    zin = t[:, COLS["zin_re"]] + 1j * t[:, COLS["zin_im"]]
    zup = t[:, COLS["zup_re"]] + 1j * t[:, COLS["zup_im"]]
    amps = np.asarray(sol.amplitudes)
    np.testing.assert_allclose(amps[:, 0], zin, rtol=1e-7)
    np.testing.assert_allclose(amps[:, 1], zup, rtol=1e-7)


def test_fluxes_and_carter_vs_pybhpt(ref, gen_orbit, sol):
    t = ref["gen_table"]
    fx = flux_modes(sol.amplitudes, sol.m, sol.omega, sol.eigenvalues,
                    gen_orbit.a, k=sol.k, n=sol.n,
                    carter_frequencies=gen_orbit.carter_frequencies())
    Edot, Ldot, Qdot = (np.asarray(fx.Edot), np.asarray(fx.Ldot),
                        np.asarray(fx.Qdot))
    for arr, ci, ch in [(Edot, "edot_i", "edot_h"),
                        (Ldot, "ldot_i", "ldot_h"),
                        (Qdot, "qdot_i", "qdot_h")]:
        np.testing.assert_allclose(arr[:, 0], t[:, COLS[ci]], rtol=1e-7)
        np.testing.assert_allclose(arr[:, 1], t[:, COLS[ch]], rtol=1e-7)


def test_flux_modes_backward_compatible(sol, gen_orbit):
    fx = flux_modes(sol.amplitudes, sol.m, sol.omega, sol.eigenvalues,
                    gen_orbit.a)
    assert fx.Qdot is None


def test_equatorial_degeneration():
    # x -> 1 exact limit unavailable (generic_samples rejects |x| = 1), so
    # near-equatorial continuity: x = 1 equatorial vs x = 0.999999 generic
    # at k = 0 (smooth in x)
    l = np.array([2, 3]); m = np.array([2, 2])
    eq = solve_mode_table(MiniGeo(0.9, 10.0, 0.2, 1.0), -2, l, m,
                          np.array([1, 0]))
    gen = solve_generic_mode_table(MiniGeo(0.9, 10.0, 0.2, 0.999999), -2,
                                   l, m, np.zeros(2, int), np.array([1, 0]),
                                   nsamples=2**9, ntheta=2**4)
    np.testing.assert_allclose(np.asarray(gen.amplitudes),
                               np.asarray(eq.amplitudes), rtol=1e-4)


def test_spherical_degeneration():
    # e -> 0 continuity against Increment B at n = 0
    l = np.array([2, 3]); m = np.array([2, 2]); k = np.array([1, -1])
    sph = solve_spherical_mode_table(MiniGeo(0.9, 10.0, 0.0, 0.5), -2, l, m, k)
    gen = solve_generic_mode_table(MiniGeo(0.9, 10.0, 1e-6, 0.5), -2,
                                   l, m, k, np.zeros(2, int),
                                   nsamples=2**6, ntheta=2**8)
    np.testing.assert_allclose(np.asarray(gen.amplitudes),
                               np.asarray(sph.amplitudes), rtol=1e-4)


def test_conjugate_symmetry(ref, gen_orbit, sol):
    # Z_{l,-m,-k,-n} = (-1)^(l+k) conj(Z_{l,m,k,n}) -- NO (-1)^n factor in
    # this code's conventions (qr = 0 at periapsis, tr/phir odd), matching
    # Phase 3's (-1)^l equatorial result; verified empirically across all
    # 12 snapshot modes at machine precision (dev/PHASE5C.md)
    t = ref["gen_table"]
    l = t[:, COLS["l"]].astype(int)
    m = t[:, COLS["m"]].astype(int)
    k = t[:, COLS["k"]].astype(int)
    n = t[:, COLS["n"]].astype(int)
    sol2 = solve_generic_mode_table(gen_orbit, -2, l, -m, -k, -n,
                                    nsamples=2**9, ntheta=2**7)
    a1 = np.asarray(sol.amplitudes)
    a2 = np.asarray(sol2.amplitudes)
    for b in range(len(l)):
        np.testing.assert_allclose(
            a2[b], (-1.0)**(l[b] + k[b]) * np.conj(a1[b]), rtol=1e-7,
            err_msg=f"(l,m,k,n)=({l[b]},{m[b]},{k[b]},{n[b]})")


def test_static_mode_rejected(gen_orbit):
    with pytest.raises(ValueError, match="static"):
        solve_generic_mode_table(gen_orbit, -2, np.array([2]),
                                 np.array([0]), np.array([0]), np.array([0]))
