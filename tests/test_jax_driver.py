# Unified solver dispatch, canonical-half flux summation, and the adaptive
# total-flux driver. See dev/PHASE5D_DESIGN.md and dev/PHASE5D.md.
import numpy as np
import pytest

jax = pytest.importorskip("jax")

from hblteuk.geo import MiniGeo
from hblteuk.jax import (solve_generic_mode_table, solve_mode_table,
                        solve_modes, solve_spherical_mode_table)

A = 0.9


@pytest.fixture(scope="module")
def orb_ecc_eq():
    return MiniGeo(A, 10.0, 0.2, 1.0)


@pytest.fixture(scope="module")
def orb_sph():
    return MiniGeo(A, 10.0, 0.0, 0.5)


@pytest.fixture(scope="module")
def orb_gen():
    return MiniGeo(A, 8.0, 0.3, 0.6)


def test_dispatch_equatorial(orb_ecc_eq):
    l = np.array([2, 3]); m = np.array([2, 2]); n = np.array([1, -1])
    got = solve_modes(orb_ecc_eq, -2, l, m, np.zeros(2, int), n,
                      nsamples=2**8, ndomains=40)
    ref = solve_mode_table(orb_ecc_eq, -2, l, m, n,
                           nsamples=2**8, ndomains=40)
    np.testing.assert_array_equal(np.asarray(got.amplitudes),
                                  np.asarray(ref.amplitudes))
    np.testing.assert_array_equal(got.k, np.zeros(2, int))
    np.testing.assert_array_equal(got.n, n)


def test_dispatch_spherical(orb_sph):
    l = np.array([2, 3]); m = np.array([2, 2]); k = np.array([1, -1])
    got = solve_modes(orb_sph, -2, l, m, k, np.zeros(2, int), ntheta=2**7)
    ref = solve_spherical_mode_table(orb_sph, -2, l, m, k, nsamples=2**7)
    np.testing.assert_array_equal(np.asarray(got.amplitudes),
                                  np.asarray(ref.amplitudes))
    np.testing.assert_array_equal(got.n, np.zeros(2, int))


def test_dispatch_generic(orb_gen):
    l = np.array([2]); m = np.array([2]); k = np.array([1]); n = np.array([1])
    got = solve_modes(orb_gen, -2, l, m, k, n, nsamples=2**8, ntheta=2**6)
    ref = solve_generic_mode_table(orb_gen, -2, l, m, k, n,
                                   nsamples=2**8, ntheta=2**6)
    np.testing.assert_array_equal(np.asarray(got.amplitudes),
                                  np.asarray(ref.amplitudes))


def test_dispatch_degenerate_axis_raises(orb_ecc_eq, orb_sph):
    two = np.array([2]); one = np.array([1])
    with pytest.raises(ValueError, match="k"):
        solve_modes(orb_ecc_eq, -2, two, two, one, one)
    with pytest.raises(ValueError, match="n"):
        solve_modes(orb_sph, -2, two, two, one, one)


from hblteuk.jax import sum_fluxes


@pytest.fixture(scope="module")
def gen_half(orb_gen):
    # canonical half-window: m > 0 rows only, small (l, k, n) box --
    # every axis needs spread so edge_report covers l, k, and n
    modes = [(ll, mm, kk, nn) for ll in (2, 3) for mm in (1, 2)
             for kk in (-1, 0, 1) for nn in (-1, 0, 1)]
    arr = np.array(modes)
    return solve_modes(orb_gen, -2, arr[:, 0], arr[:, 1], arr[:, 2],
                       arr[:, 3], nsamples=2**8, ntheta=2**6)


def test_sum_fluxes_doubling_matches_full_plane(orb_gen, gen_half):
    got = sum_fluxes(gen_half, orb_gen)
    # full plane: canonical rows plus their (-m, -k, -n) partners, no doubling
    l, m, k, n = gen_half.l, gen_half.m, gen_half.k, gen_half.n
    full = solve_modes(orb_gen, -2, np.concatenate([l, l]),
                       np.concatenate([m, -m]), np.concatenate([k, -k]),
                       np.concatenate([n, -n]), nsamples=2**8, ntheta=2**6)
    from hblteuk.jax import flux_modes
    fx = flux_modes(full.amplitudes, full.m, full.omega, full.eigenvalues,
                    orb_gen.a, k=full.k, n=full.n,
                    carter_frequencies=orb_gen.carter_frequencies())
    for tot, F in [(got.Edot, fx.Edot), (got.Ldot, fx.Ldot),
                   (got.Qdot, fx.Qdot)]:
        np.testing.assert_allclose(tot, np.asarray(F).sum(axis=0), rtol=1e-7)


def test_sum_fluxes_noncanonical_raises(orb_gen, gen_half):
    l, m, k, n = gen_half.l, gen_half.m, gen_half.k, gen_half.n
    bad = solve_modes(orb_gen, -2, l, -m, -k, -n, nsamples=2**8, ntheta=2**6)
    with pytest.raises(ValueError, match="canonical"):
        sum_fluxes(bad, orb_gen)


def test_sum_fluxes_reports(orb_gen, gen_half):
    got = sum_fluxes(gen_half, orb_gen, tol=1e-1)
    assert set(got.edge_report) == {"l", "k", "n"}
    for frac in got.edge_report.values():
        assert frac.shape == (3,) and np.all(frac >= 0)
    assert set(got.converged) == {"Edot", "Ldot", "Qdot"}
    assert np.all(np.asarray(got.error_bound) >= 0)
    assert len(got.table["l"]) == len(gen_half.l)


def test_sum_fluxes_equatorial_qdot_zero(orb_ecc_eq):
    l = np.array([2, 2]); m = np.array([2, 2])
    n = np.array([0, 1]); k = np.zeros(2, int)
    sol = solve_modes(orb_ecc_eq, -2, l, m, k, n, nsamples=2**8, ndomains=40)
    got = sum_fluxes(sol, orb_ecc_eq)
    np.testing.assert_allclose(got.Qdot, 0.0, atol=1e-14)
    assert "k" not in got.edge_report      # degenerate axis: no k spread


from hblteuk.jax import total_fluxes


def brute_total(orbit, lmax, k_half, n_half, **kw):
    modes = [(ll, mm, kk, nn) for ll in range(2, lmax + 1)
             for mm in range(0, ll + 1)
             for kk in range(-k_half, k_half + 1)
             for nn in range(-n_half, n_half + 1)
             if (mm > 0) or (mm == 0 and kk > 0)
             or (mm == 0 and kk == 0 and nn > 0)]
    arr = np.array(modes)
    sol = solve_modes(orbit, -2, arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3],
                      return_radial=False, **kw)
    return sum_fluxes(sol, orbit)


def test_total_fluxes_circular_quadrupole():
    orbit = MiniGeo(0.0, 500.0, 0.0, 1.0)
    got = total_fluxes(orbit, tol=1e-4)
    newt = 32.0 / 5.0 * orbit.frequencies[2] ** (10.0 / 3.0)
    assert got.Edot[0] == pytest.approx(newt, rel=2e-2)   # 1PN-sized residual
    assert all(got.converged.values())
    assert len(got.rounds) >= 1


def test_total_fluxes_vs_brute_force_ecc(orb_ecc_eq):
    got = total_fluxes(orb_ecc_eq, tol=1e-5)
    ref = brute_total(orb_ecc_eq, lmax=got.table["l"].max() + 2, k_half=0,
                      n_half=int(got.table["n"].max()) + 4, nsamples=2**10)
    np.testing.assert_allclose(got.Edot, ref.Edot, rtol=5e-5)
    np.testing.assert_allclose(got.Ldot, ref.Ldot, rtol=5e-5)


def test_total_fluxes_vs_brute_force_sph(orb_sph):
    got = total_fluxes(orb_sph, tol=1e-5)
    ref = brute_total(orb_sph, lmax=got.table["l"].max() + 2,
                      k_half=int(got.table["k"].max()) + 3, n_half=0,
                      ntheta=2**8)
    np.testing.assert_allclose(got.Edot, ref.Edot, rtol=5e-5)
    np.testing.assert_allclose(got.Qdot, ref.Qdot, rtol=5e-4)


def test_total_fluxes_generic_smoke(orb_gen):
    # coarser tol keeps runtime bounded; full validation via the notebook
    got = total_fluxes(orb_gen, tol=1e-4)
    assert all(got.converged.values())
    assert got.Edot[0] > 0
    assert np.any(got.table["k"] != 0) and np.any(got.table["n"] != 0)


def test_total_fluxes_cap_warns(orb_ecc_eq):
    with pytest.warns(UserWarning, match="lmax_cap"):
        got = total_fluxes(orb_ecc_eq, tol=1e-12, lmax_cap=4)
    assert not all(got.converged.values())
