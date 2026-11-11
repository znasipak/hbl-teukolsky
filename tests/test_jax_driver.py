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


@pytest.mark.parametrize("strategy", ["box", "peak"])
def test_bucketing_matches_unpadded(orb_ecc_eq, strategy):
    # bucket=True pads every round's batch to the chunk size (the GPU
    # compile-amortization path); results must match the unpadded run to the
    # batch-reproducibility floor, with identical mode count and round count.
    off = total_fluxes(orb_ecc_eq, tol=1e-5, strategy=strategy, bucket=False)
    on = total_fluxes(orb_ecc_eq, tol=1e-5, strategy=strategy, bucket=True)
    for ch in ("Edot", "Ldot", "Qdot"):
        np.testing.assert_allclose(getattr(on, ch), getattr(off, ch),
                                   rtol=1e-7, atol=1e-30)
    assert len(on.table["l"]) == len(off.table["l"])
    assert len(on.rounds) == len(off.rounds)


def test_bucketing_auto_off_on_cpu(orb_ecc_eq):
    # "auto" must not pad on CPU (padding only helps GPU compile amortization
    # and wastes CPU compute); assert it equals the explicit bucket=False path
    from hblteuk.jax.driver import _want_bucket
    assert _want_bucket("auto") is (jax.default_backend() == "gpu")
    assert _want_bucket(False) is False and _want_bucket(True) is True


# --- Phase 5F: gradient (decay-predicted) mode selection ---
from hblteuk.jax.driver import _decay_step


def test_decay_step_geometric_predicts_and_stops():
    # exact geometric decay r=0.1: S_j = 100 * 0.1**j, edge=1.0
    S = np.array([100.0, 10.0, 1.0])
    # budget above the edge -> tail 0.111 and edge 1.0 both under 2.0 -> done
    done, step = _decay_step({"Edot": S}, {"Edot": 2.0})
    assert done and step == 0
    # tighter budget 1e-3: tail 0.111 needs 3 more shells (1e-2 tail after 2,
    # 1.1e-3 -> still >1e-3; 3rd shell brings tail to 1.1e-4 < 1e-3)
    done, step = _decay_step({"Edot": S}, {"Edot": 1e-3})
    assert not done and step == 3


def test_decay_step_not_decaying_forces_growth():
    S = np.array([1.0, 1.0, 2.0])             # r=2 (increasing) -> unresolved
    done, step = _decay_step({"Edot": S}, {"Edot": 1e-6})
    assert not done and step == 1


def test_decay_step_cap_and_multichannel_binds_max():
    Sfast = np.array([1.0, 1e-3])             # r=1e-3, converges fast
    Sslow = np.array([1.0, 0.9])              # r=0.9, long tail
    done, step = _decay_step({"Edot": Sfast, "Ldot": Sslow},
                             {"Edot": 1e-9, "Ldot": 1e-9}, cap=3)
    assert not done and step == 3             # slow channel binds, capped at 3


from hblteuk.jax.driver import _m_offset, _gradient_window_modes


def test_m_offset():
    assert _m_offset(10, 8, 10) == 2
    assert _m_offset(10, 10, 10) == 0


def test_gradient_window_bands_m(orb_gen):
    # c_m = top (m=l); dm=1 keeps only m in {l-1, l}; canonical half only
    lmax = 8
    c_m = {l: l for l in range(2, lmax + 1)}
    modes = _gradient_window_modes(orb_gen, lmax, c_m, dm=1, dk=1, dn=1)
    ls = sorted({l for l, m, k, n in modes})
    assert ls == list(range(2, lmax + 1))
    for l, m, k, n in modes:
        assert l - 1 <= m <= l                     # m banded near center=l
        assert (m > 0) or (m == 0 and k > 0) or (m == 0 and k == 0 and n > 0)


def test_gradient_window_low_l_covers_all_m(orb_gen):
    # at l=2 with center=2, dm=5 -> band covers all m in [0,2]
    c_m = {2: 2}
    modes = _gradient_window_modes(orb_gen, 2, c_m, dm=5, dk=0, dn=0)
    ms = sorted({m for l, m, k, n in modes if l == 2})
    assert ms == [0, 1, 2]


from hblteuk.jax.driver import _m_center_seed, _recenter_m


def test_m_center_seed(orb_gen):
    c_m = _m_center_seed(orb_gen, 6)
    assert c_m == {2: 2, 3: 3, 4: 4, 5: 5, 6: 6}


def test_recenter_interior_argmax_moves_center():
    # l=10, current center 10, dm=3 (band 7..10). Peak weight at m=8 (interior)
    l_arr = np.full(4, 10)
    m_arr = np.array([7, 8, 9, 10])
    w_arr = np.array([1.0, 9.0, 2.0, 1.0])       # argmax at m=8
    new_c_m, grow = _recenter_m(l_arr, m_arr, w_arr, {10: 10}, dm=3)
    assert new_c_m[10] == 8 and grow is False


def test_recenter_boundary_argmax_grows_not_moves():
    # peak sits on the lower band edge (m=7) -> band too narrow, grow
    l_arr = np.full(4, 10)
    m_arr = np.array([7, 8, 9, 10])
    w_arr = np.array([9.0, 1.0, 1.0, 1.0])       # argmax at m=7 (band edge)
    new_c_m, grow = _recenter_m(l_arr, m_arr, w_arr, {10: 10}, dm=3)
    assert new_c_m[10] == 10 and grow is True


def test_total_fluxes_gradient_matches_box_ecc(orb_ecc_eq):
    box = total_fluxes(orb_ecc_eq, tol=1e-5, strategy="box")
    grad = total_fluxes(orb_ecc_eq, tol=1e-5, strategy="gradient")
    np.testing.assert_allclose(grad.Edot, box.Edot, rtol=5e-5)
    np.testing.assert_allclose(grad.Ldot, box.Ldot, rtol=5e-5)


def test_total_fluxes_gradient_matches_box_generic(orb_gen):
    box = total_fluxes(orb_gen, tol=1e-4, strategy="box")
    grad = total_fluxes(orb_gen, tol=1e-4, strategy="gradient")
    np.testing.assert_allclose(grad.Edot, box.Edot, rtol=5e-4)
    np.testing.assert_allclose(grad.Qdot, box.Qdot, rtol=5e-3)


def test_total_fluxes_gradient_solves_fewer_than_peak(orb_gen):
    # the payoff: gradient's non-uniform hull solves strictly fewer modes
    # than peak at matched tol, while staying within tolerance (checked above)
    peak = total_fluxes(orb_gen, tol=1e-4, strategy="peak")
    grad = total_fluxes(orb_gen, tol=1e-4, strategy="gradient")
    assert len(grad.table["l"]) < len(peak.table["l"])


def test_total_fluxes_gradient_degenerate(orb_ecc_eq, orb_sph):
    # equatorial: no k modes solved; spherical: no n modes solved
    g_eq = total_fluxes(orb_ecc_eq, tol=1e-4, strategy="gradient")
    assert set(np.unique(g_eq.table["k"])) == {0}
    g_sph = total_fluxes(orb_sph, tol=1e-4, strategy="gradient")
    assert set(np.unique(g_sph.table["n"])) == {0}


def test_decay_step_safety_gate_blocks_early_stop():
    # ratio looks finished (edge tiny vs prev -> predicted tail tiny) BUT the
    # edge shell itself still exceeds budget -> must NOT stop: the model-free
    # max(tail, S_edge) gate fires.
    S = np.array([1000.0, 0.5])              # r=5e-4, tail~2.5e-4, edge=0.5
    done, step = _decay_step({"Edot": S}, {"Edot": 0.1})
    assert not done and step >= 1


def test_total_fluxes_gradient_bad_strategy_message(orb_ecc_eq):
    with pytest.raises(ValueError, match="gradient"):
        total_fluxes(orb_ecc_eq, strategy="nope")


# --- retrograde orbits and driver termination guards ---

@pytest.mark.parametrize("orbit_args", [(A, 10.0, 0.2, -1.0),    # equatorial
                                        (A, 12.0, 0.0, -0.5)])   # spherical
def test_total_fluxes_retrograde_converges(orbit_args):
    # x < 0 makes Ldot (and for equatorial, Qdot) negative. The edge and
    # budget checks are magnitude-based (np.abs) throughout, so sign must not
    # affect convergence -- these orbits must terminate normally.
    got = total_fluxes(MiniGeo(*orbit_args), tol=1e-4, strategy="box")
    assert all(got.converged.values())
    assert got.Ldot[0] < 0                      # retrograde: negative Ldot
    assert got.Edot[0] > 0                       # energy flux stays positive


@pytest.mark.parametrize("strategy", ["box", "peak", "gradient"])
def test_driver_stops_when_k_axis_stalls(strategy):
    # Regression: only lmax had a cap, so if l converged while k did not the
    # loop grew k without bound and never returned. Force k to stall with l
    # converging, and require a return plus a warning naming the stuck axis.
    with pytest.warns(UserWarning, match="stalled on"):
        got = total_fluxes(MiniGeo(A, 12.0, 0.0, -0.5), tol=1e-4,
                           strategy=strategy, k_cap=4)
    assert not all(got.converged.values())
    assert len(got.rounds) < 20                  # bounded, not runaway


@pytest.mark.parametrize("strategy", ["box", "peak", "gradient"])
def test_driver_stops_at_max_modes(strategy):
    # k and n windows can both double per round, so the per-axis caps still
    # permit impractically large runs; max_modes is the guard that fires first.
    with pytest.warns(UserWarning, match="max_modes"):
        got = total_fluxes(MiniGeo(A, 12.0, 0.0, -0.5), tol=1e-12,
                           strategy=strategy, max_modes=500)
    assert not all(got.converged.values())


def test_sum_fluxes_rejects_non_finite_modes():
    # one NaN mode must not pass as a converged total
    from types import SimpleNamespace
    from hblteuk.geo import MiniGeo
    from hblteuk.jax import solve_modes, sum_fluxes

    orbit = MiniGeo(0.9, 10.0, 0.2, 0.7)
    l, m, k, n = (np.array(c) for c in ((2, 2), (2, 1), (0, 0), (0, 1)))
    sol = solve_modes(orbit, -2, l, m, k, n, nsamples=2**6, ntheta=2**5,
                      ndomains=40, return_radial=False)
    amp = np.array(sol.amplitudes)
    amp[1] = np.nan
    bad = SimpleNamespace(l=sol.l, m=sol.m, k=sol.k, n=sol.n, omega=sol.omega,
                          eigenvalues=sol.eigenvalues, amplitudes=amp,
                          errors=np.array(sol.errors))
    with pytest.raises(FloatingPointError, match=r"\(2, 1, 0, 1\)"):
        sum_fluxes(bad, orbit, tol=1e-6)


class _Stop(Exception):
    pass


@pytest.mark.parametrize("given,expected", [(None, 32), (64, 64)])
def test_total_fluxes_passes_ndomains_to_solver(monkeypatch, given, expected):
    # default 32 radial subdomains (dev/JAX_PERF.md: <= 5e-12 from 150 on 14
    # orbits at tol 1e-6); the chunk model and the solver must see the same
    from hblteuk.geo import MiniGeo
    from hblteuk.jax import driver, total_fluxes

    seen = {}

    def fake(*a, **k):
        seen.update(k)
        raise _Stop

    monkeypatch.setattr(driver, "solve_modes", fake)
    kw = {} if given is None else dict(ndomains=given)
    for strategy in ("box", "peak", "gradient"):
        seen.clear()
        with pytest.raises(_Stop):
            total_fluxes(MiniGeo(0.9, 10.0, 0.2, 0.7), tol=1e-2,
                         strategy=strategy, **kw)
        assert seen["ndomains"] == expected, strategy


def test_bucketed_solve_pads_to_ladder(monkeypatch):
    # small rounds pad to the next bucket rung, not to the whole chunk
    from hblteuk.geo import MiniGeo
    from hblteuk.jax import driver
    from hblteuk.jax.pipeline import _bucket, _chunk_size

    seen = {}

    def fake(orbit, s, l, m, k, n, **kw):
        seen["B"] = len(l)
        raise _Stop

    monkeypatch.setattr(driver, "solve_modes", fake)
    arr = np.array([(2, 2, 0, 1)] * 1038)
    with pytest.raises(_Stop):
        driver._bucketed_solve(MiniGeo(0.9, 10.0, 0.2, 0.7), -2, arr, 512, 128,
                               32.0, pad=True, ndomains=32)
    cap = _chunk_size(32.0, 512, 32, 32, True)
    assert cap > 2000 and seen["B"] == _bucket(1038, cap) == 1280


_SCHEME_TABLE = [(2, 2, 1, 1), (2, 1, -1, -1), (3, 3, 0, 2), (4, 2, 2, 0),
                 (5, 4, -1, 3), (3, -2, 1, 1)]


@pytest.mark.parametrize("kind", ["equatorial", "spherical", "generic",
                                  "retrograde"])
def test_scheme_default_and_agreement(kind, orb_ecc_eq, orb_sph, orb_gen):
    # every solver path: the default is ultraspherical, and at 150 domains it
    # agrees with collocation on resolved modes
    orbit = {"equatorial": orb_ecc_eq, "spherical": orb_sph,
             "generic": orb_gen,
             "retrograde": MiniGeo(A, 10.0, 0.3, -0.6)}[kind]
    l, m, k, n = (np.array(c) for c in zip(*_SCHEME_TABLE))
    if kind == "equatorial":
        k = 0 * k
    if kind == "spherical":
        n = 0 * n
    kw = dict(nsamples=2**8, ntheta=2**6)
    sol = {sc: solve_modes(orbit, -2, l, m, k, n, **kw, **(
        {} if sc is None else dict(scheme=sc)))
        for sc in (None, "ultraspherical", "collocation")}
    np.testing.assert_array_equal(np.asarray(sol[None].amplitudes),
                                  np.asarray(sol["ultraspherical"].amplitudes))
    a_us = np.asarray(sol["ultraspherical"].amplitudes)
    a_col = np.asarray(sol["collocation"].amplitudes)
    ok = np.asarray(sol["collocation"].errors) < 1e-8
    assert ok.sum() >= ok.size // 2, kind
    rel = np.abs(a_us - a_col) / np.abs(a_col)
    assert np.all(rel[ok] < 1e-10), (kind, rel[ok].max())


@pytest.mark.parametrize("given,expected", [(None, "ultraspherical"),
                                            ("collocation", "collocation")])
def test_total_fluxes_passes_scheme_to_solver(monkeypatch, given, expected):
    from hblteuk.jax import driver, total_fluxes

    seen = {}

    def fake(*a, **k):
        seen.update(k)
        raise _Stop

    monkeypatch.setattr(driver, "solve_modes", fake)
    kw = {} if given is None else dict(scheme=given)
    for strategy in ("box", "peak", "gradient"):
        seen.clear()
        with pytest.raises(_Stop):
            total_fluxes(MiniGeo(0.9, 10.0, 0.2, 0.7), tol=1e-2,
                         strategy=strategy, **kw)
        assert seen["scheme"] == expected, strategy
