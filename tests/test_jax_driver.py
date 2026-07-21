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


from hblteuk.jax.driver import (_peak_window_modes, _k_peaks, _n_peak_mag,
                                _k_offset, _n_offset)


def test_peak_centers_formulas():
    # k peaks where |m+k| = l; |n| peaks at round(m / (1-e)^1.5)
    assert _k_peaks(5, 2) == (3, -7)                # l-m, -(l+m)
    assert _n_peak_mag(0.0, 5) == 5                 # (1-0)^1.5 = 1
    assert _n_peak_mag(0.3, 4) == round(4 / 0.7 ** 1.5)
    assert _k_offset(5, 2, 3) == 0                  # exactly on l-m peak
    assert _n_offset(0.0, 5, -5) == 0               # on the -n peak


def test_peak_window_canonical_and_centered(orb_gen):
    modes = _peak_window_modes(orb_gen, lmax=6, dk=2, dn=2)
    assert len(modes) == len(set(modes))            # deduplicated
    for (l, m, k, n) in modes:
        # canonical half only
        assert (m > 0) or (m == 0 and k > 0) or (m == 0 and k == 0 and n > 0)
        assert 2 <= l <= 6 and 0 <= m <= l          # natural (l, m)
        assert _k_offset(l, m, k) <= 2              # inside k band
        assert _n_offset(orb_gen.e, m, n) <= 2      # inside n band


def test_peak_window_degenerate_collapse():
    from hblteuk.geo import MiniGeo
    eq = MiniGeo(0.9, 10.0, 0.2, 1.0)               # equatorial: no k
    modes = _peak_window_modes(eq, lmax=6, dk=2, dn=2)
    assert all(k == 0 for (l, m, k, n) in modes)
    circ = MiniGeo(0.9, 10.0, 0.0, 0.5)             # circular: no n
    modes = _peak_window_modes(circ, lmax=6, dk=2, dn=2)
    assert all(n == 0 for (l, m, k, n) in modes)


def test_sum_fluxes_peak_edges(orb_gen):
    modes = _peak_window_modes(orb_gen, lmax=5, dk=3, dn=3)
    arr = np.array(modes)
    sol = solve_modes(orb_gen, -2, arr[:, 0], arr[:, 1], arr[:, 2],
                      arr[:, 3], nsamples=2**8, ntheta=2**6)
    got = sum_fluxes(sol, orb_gen, tol=1e-1, peak_edges=True)
    assert set(got.edge_report) <= {"l", "dk", "dn"}
    assert {"dk", "dn"} <= set(got.edge_report)      # both bands have spread
    for frac in got.edge_report.values():
        assert frac.shape == (3,) and np.all(frac >= 0)
    # box edge_report (default) is unaffected
    box = sum_fluxes(sol, orb_gen, tol=1e-1)
    assert "l" in box.edge_report


def test_total_fluxes_peak_matches_box_ecc(orb_ecc_eq):
    box = total_fluxes(orb_ecc_eq, tol=1e-5, strategy="box")
    peak = total_fluxes(orb_ecc_eq, tol=1e-5, strategy="peak")
    np.testing.assert_allclose(peak.Edot, box.Edot, rtol=5e-5)
    np.testing.assert_allclose(peak.Ldot, box.Ldot, rtol=5e-5)


def test_total_fluxes_peak_matches_box_sph(orb_sph):
    box = total_fluxes(orb_sph, tol=1e-5, strategy="box")
    peak = total_fluxes(orb_sph, tol=1e-5, strategy="peak")
    np.testing.assert_allclose(peak.Edot, box.Edot, rtol=5e-5)
    np.testing.assert_allclose(peak.Qdot, box.Qdot, rtol=5e-4)


def test_total_fluxes_peak_matches_box_generic(orb_gen):
    # correctness on the 2D case (coarse tol to bound runtime)
    box = total_fluxes(orb_gen, tol=1e-4, strategy="box")
    peak = total_fluxes(orb_gen, tol=1e-4, strategy="peak")
    np.testing.assert_allclose(peak.Edot, box.Edot, rtol=5e-4)
    np.testing.assert_allclose(peak.Qdot, box.Qdot, rtol=5e-3)


def test_peak_selection_prunes(orb_gen):
    # the payoff: at matched extents the peak selector picks far fewer modes
    # than the full box rectangle (ridge pruning; measured ~0.5x). Cheap,
    # deterministic -- the adaptive per-orbit mode count is shown in the
    # notebook, where the savings grow with eccentricity/inclination.
    from hblteuk.jax.driver import _window_modes
    box = _window_modes(10, 8, 10)
    peak = _peak_window_modes(orb_gen, lmax=10, dk=3, dn=3)
    assert len(peak) < 0.7 * len(box)


def test_total_fluxes_peak_cap_warns(orb_ecc_eq):
    with pytest.warns(UserWarning, match="lmax_cap"):
        got = total_fluxes(orb_ecc_eq, tol=1e-12, lmax_cap=4, strategy="peak")
    assert not all(got.converged.values())


def test_total_fluxes_bad_strategy(orb_ecc_eq):
    with pytest.raises(ValueError, match="strategy"):
        total_fluxes(orb_ecc_eq, strategy="nope")


from hblteuk.jax import flux_efficiency


def test_flux_efficiency_basic(orb_ecc_eq):
    got = total_fluxes(orb_ecc_eq, tol=1e-5, strategy="box")
    eff = flux_efficiency(got, tol=1e-5)
    N = len(got.table["l"])
    assert eff["n_solved"] == N
    for ch, (ci, ch_) in eff["n_required"].items():
        assert 0 <= ci <= N and 0 <= ch_ <= N   # zero channels (eq Qdot) -> 0
    assert eff["n_required"]["Edot"][0] > 0     # nonzero channel needs modes
    assert 0 < eff["efficiency"] <= 1.0
    assert set(eff["per_channel_efficiency"]) == {"Edot", "Ldot", "Qdot"}


def test_flux_efficiency_required_set_reproduces_total(orb_ecc_eq):
    got = total_fluxes(orb_ecc_eq, tol=1e-5, strategy="box")
    eff = flux_efficiency(got, tol=1e-5)
    contrib = np.abs(got.table["Edot"][:, 0])
    order = np.argsort(contrib)[::-1]
    ni = eff["n_required"]["Edot"][0]
    kept = got.table["Edot"][order[:ni], 0].sum()
    assert abs(kept - got.Edot[0]) <= 1e-5 * abs(got.Edot[0])
