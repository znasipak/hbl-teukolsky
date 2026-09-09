# Old-API compatibility layer tests (src/hblteuk/legacy.py). The wrappers must
# reproduce the current teuk primitives exactly -- same boundary-condition
# normalization, same TS stabilization -- so every value test is against the
# modern HyperboloidalTeukolskySolver rather than stored data.
import numpy as np
import pytest

from hblteuk import collocode, teuk
from hblteuk.legacy import (
    HyperboloidalTeukolsky,
    HyperboloidalTeukolskySolution,
    RadialTeukolsky,
    RadialTeukolskySolution,
)
from .conftest import ORBIT, SOLVER_KWARGS

A = ORBIT[0]
KAPPA = np.sqrt(1 - A**2)
HORIZON = 1 + KAPPA
LM = (2, 2)

# (a, s, l, m) cases for the cross-check against the modern solver. l=2 is the
# cheap sanity case; the a=0.99, l=16 cases are where the unstabilized march has
# visibly drifted (Up at s=-2: 8e-9 vs 5e-11 stabilized; In at s=+2: 6e-8 vs
# 1e-12), so they are what actually pin use_ts_transform's default.
SOLVER_CASES = [(A, -2, 2, 2), (A, 2, 2, 2), (0.99, -2, 16, 16), (0.99, 2, 16, 16)]

RADIAL_FLAGS = {"slicing": "time", "scaled": False, "compactification": False}


@pytest.fixture(scope="module")
def omega(orbit):
    return orbit.mode_frequency(LM[1], 0, 0)


def mode_frequency(a, m):
    from hblteuk.geo import MiniGeo
    return MiniGeo(a, *ORBIT[1:]).mode_frequency(m, 0, 0)


def solve_legacy(cls, s, omega, a=A, lm=LM, **kwargs):
    l, m = lm
    obj = cls(a, s, l, m, omega)
    obj.solve(**kwargs)
    return obj


def solve_modern(legacy, s, omega, a=A, lm=LM):
    # same sigma domains the legacy cutoffs picked, so the two solves are
    # comparable point by point
    l, m = lm
    solver = teuk.HyperboloidalTeukolskySolver(
        domains=[[0, max(legacy._psi["Up"].domain)],
                 [min(legacy._psi["In"].domain), 1]],
        solver=collocode.CollocationODEMultiDomainFixedStepSolver(n=32),
        **SOLVER_KWARGS)
    return solver(a, s, l, m, omega)


def interior_sigma(psi, n=25, pad=0.02):
    dom = psi.domain
    return np.linspace(min(dom) + pad, max(dom) - pad, n)


@pytest.fixture(scope="module")
def legacy_hbl(omega):
    return solve_legacy(HyperboloidalTeukolsky, -2, omega)


@pytest.mark.parametrize("a,s,l,m", SOLVER_CASES)
def test_matches_modern_solver(a, s, l, m):
    # both stabilized branches: s=-2 stabilizes Up, s=+2 stabilizes In
    w = mode_frequency(a, m)
    legacy = solve_legacy(HyperboloidalTeukolsky, s, w, a=a, lm=(l, m))
    hbl = solve_modern(legacy, s, w, a=a, lm=(l, m))
    assert legacy.eigenvalue == pytest.approx(hbl.eigenvalue, rel=1e-12)
    for bc in ("Up", "In"):
        sig = interior_sigma(legacy._psi[bc])
        for d in (0, 1):
            got = legacy.solution[bc](sig, deriv=d)
            want = hbl.psi[bc](sig, deriv=d)
            np.testing.assert_allclose(
                got, want, rtol=1e-9, atol=1e-9 * np.max(np.abs(want)),
                err_msg=f"a={a} s={s} l={l} {bc} deriv={d}")


def test_ts_transform_flag(omega, legacy_hbl):
    # use_ts_transform=False marches s=-2 Up in its unstable direction; at l=2
    # that march has not yet drifted, so the two paths still agree. The l=16
    # case in SOLVER_CASES is what tests that the stabilized path is the
    # accurate one (see dev/ACCURACY_DEBUG_PLAN.md).
    plain = solve_legacy(HyperboloidalTeukolsky, -2, omega, use_ts_transform=False)
    for bc in ("Up", "In"):
        sig = interior_sigma(legacy_hbl._psi[bc])
        got = plain.solution[bc](sig)
        want = legacy_hbl.solution[bc](sig)
        np.testing.assert_allclose(
            got, want, rtol=1e-7, atol=1e-8 * np.max(np.abs(want)), err_msg=bc)


def test_radial_matches_hyperboloidal(omega):
    # RadialTeukolsky must apply the time-slicing, unscaled, uncompactified
    # conversion to the same psi the hyperboloidal wrapper returns
    rad = solve_legacy(RadialTeukolsky, -2, omega)
    hbl = solve_modern(rad, -2, omega)
    l, m = LM
    for bc in ("Up", "In"):
        sig = interior_sigma(rad._psi[bc])
        r = teuk.r_sigma(sig, KAPPA)
        ref = teuk.HyperboloidalTeukolskySolution(
            A, -2, l, m, omega, hbl.eigenvalue, hbl.psi[bc])
        sol = rad.solution[bc]
        for d, got in [(0, sol(r)), (0, sol.sol(r)), (1, sol(r, deriv=1)),
                       (1, sol.deriv(r))]:
            want = ref(r, deriv=d, **RADIAL_FLAGS)
            np.testing.assert_allclose(
                got, want, rtol=1e-9, atol=1e-9 * np.max(np.abs(want)),
                err_msg=f"{bc} deriv={d}")
        # linspace must evaluate the same solution it returns abscissae for
        r_ls, R_ls = sol.linspace(25.0, n=9)
        np.testing.assert_array_equal(R_ls, sol(r_ls))


def test_hyperboloidal_accessors_agree(legacy_hbl):
    # sol/deriv/__call__/linspace are thin aliases; none may drift in deriv order
    sol = legacy_hbl.solution["Up"]
    sig = interior_sigma(legacy_hbl._psi["Up"])
    np.testing.assert_array_equal(sol.sol(sig), sol(sig))
    np.testing.assert_array_equal(sol.deriv(sig), sol(sig, deriv=1))
    assert not np.allclose(sol.sol(sig), sol.deriv(sig))
    sig_ls, psi_ls = sol.linspace(n=9)
    np.testing.assert_array_equal(psi_ls, sol(sig_ls))
    assert sig_ls[0] == pytest.approx(sol.domain[0])
    assert sig_ls[-1] == pytest.approx(sol.domain[1])


def test_cross_conversion_shares_psi(omega, legacy_hbl):
    # get_teukolsky / get_hyperboloidal must rewrap the solved psi, not resolve
    rad = legacy_hbl.get_teukolsky("Up")
    assert isinstance(rad, RadialTeukolskySolution)
    back = RadialTeukolsky(A, -2, *LM, omega)
    back._psi = legacy_hbl._psi
    hyp = back.get_hyperboloidal("Up")
    assert isinstance(hyp, HyperboloidalTeukolskySolution)
    sig = interior_sigma(legacy_hbl._psi["Up"])
    np.testing.assert_array_equal(hyp(sig), legacy_hbl.solution["Up"](sig))
    r = teuk.r_sigma(sig, KAPPA)
    np.testing.assert_array_equal(rad(r), rad._sol(r, deriv=0, **RADIAL_FLAGS))
    assert set(legacy_hbl.get_teukolsky().keys()) == {"In", "Up"}


def test_solve_accepts_single_bc(omega):
    hbl = solve_legacy(HyperboloidalTeukolsky, -2, omega, bc="Up")
    assert hbl.solution["Up"] is not None
    assert hbl.solution["In"] is None
    assert hbl("Up") is hbl.solution["Up"]
    assert hbl() is hbl.solution


def test_eigenvalue_override(omega):
    hbl = HyperboloidalTeukolsky(A, -2, *LM, omega, eigenvalue=3.14)
    assert hbl.eigenvalue == 3.14
    assert hbl.kappa == pytest.approx(KAPPA)
    assert hbl.horizon == pytest.approx(HORIZON)


def test_cutoff_maps_radii_to_sigma(omega):
    # scalar cutoff clamps both branches at the same r; a pair is [rmin, rmax]
    scalar = solve_legacy(HyperboloidalTeukolsky, -2, omega, cutoff=20.0)
    assert max(scalar._psi["Up"].domain) == pytest.approx(teuk.sigma_r(20.0, KAPPA))
    assert min(scalar._psi["In"].domain) == pytest.approx(teuk.sigma_r(20.0, KAPPA))
    pair = solve_legacy(HyperboloidalTeukolsky, -2, omega, cutoff=[3.0, 50.0])
    assert max(pair._psi["Up"].domain) == pytest.approx(teuk.sigma_r(3.0, KAPPA))
    assert min(pair._psi["In"].domain) == pytest.approx(teuk.sigma_r(50.0, KAPPA))


def test_radial_domains_and_linspace(omega):
    rad = solve_legacy(RadialTeukolsky, -2, omega)
    # Up reaches sigma=0 (spatial infinity); In is anchored at the horizon
    assert rad.domain["Up"][1] == np.inf
    assert rad.domain["In"][0] == pytest.approx(HORIZON)
    r, R = rad.solution["Up"].linspace(25.0, n=7)
    assert r[0] == pytest.approx(teuk.r_sigma(max(rad._psi["Up"].domain), KAPPA))
    assert r[-1] == 25.0
    assert r.shape == R.shape == (7,)
    r_in = rad.solution["In"].linspace_var(3.0, n=7)
    assert r_in[0] == 3.0
    assert r_in[-1] == pytest.approx(teuk.r_sigma(min(rad._psi["In"].domain), KAPPA))


def test_repr_latex(omega, legacy_hbl):
    rad = solve_legacy(RadialTeukolsky, -2, omega, bc="Up")
    for obj in (legacy_hbl, legacy_hbl.solution["Up"], rad, rad.solution["Up"]):
        assert isinstance(obj._repr_latex_(), str)
