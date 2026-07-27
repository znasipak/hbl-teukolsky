# Teukolsky-Starobinsky branch stabilization. The marches are unstable on
# exactly one branch per spin (s=-2 Up, s=+2 In); solving at the opposite spin
# and mapping back removes a ~1e-8 error floor. See dev/ACCURACY_DEBUG_PLAN.md.
import numpy as np
import pytest

from hblteuk import collocode
from hblteuk.collocode import generate_subdomain_list
from hblteuk.geo import MiniGeo
from hblteuk.swsh import swsh_eigenvalue
from hblteuk.teuk import (HyperboloidalTeukolskySolver, sigma_r, teuk_sys,
                          ts_in_minus_to_plus_2, ts_up_plus_to_minus_2)
from .conftest import ORBIT

A = ORBIT[0]
KAPPA = np.sqrt(1 - A**2)
L, M, N = 2, 2, 3


def _geom():
    orbit = MiniGeo(*ORBIT)
    smin = sigma_r(orbit.radial_roots[0], KAPPA)
    smax = sigma_r(orbit.radial_roots[1], KAPPA)
    return orbit, smin, smax


def _solve(ndom, s, w, lam, **kw):
    _, smin, smax = _geom()
    su = generate_subdomain_list([0, smax], ndom, "arcsinh7")
    si = generate_subdomain_list([1, smin], ndom, "arcsinh7")
    solver = collocode.CollocationODEFixedMultiDomainFixedStepSolver(n=32)
    ts = HyperboloidalTeukolskySolver(
        domains=[[0, su[-1]], [si[-1], 1]], solver=solver,
        subdomains={"Up": su, "In": si})
    return ts(A, s, L, M, w, eigenvalue=lam, **kw), su, si


def _eigs():
    orbit, _, _ = _geom()
    w = orbit.mode_frequency(M, 0, N)
    lam_m2 = swsh_eigenvalue(-2, L, M, A * w)
    return w, lam_m2, lam_m2 - 4.0, lam_m2 + 2.0     # w, lam_-2, lam_+2, lam_CH


def _residual(psi, sig, s, lam, w):
    P, Q, U = teuk_sys(sig, KAPPA, s, lam, M * A, w)
    p, dp, d2p = psi(sig), psi(sig, deriv=1), psi(sig, deriv=2)
    resid = np.abs(P * d2p + Q * dp + U * p)
    scale = np.abs(P * d2p) + np.abs(Q * dp) + np.abs(U * p)
    return np.max(resid / scale)


def test_lambda_ch_is_spin_invariant():
    w, lam_m2, lam_p2, lam_ch = _eigs()
    # lambda_CH = lambda_s + s(s+1); the exact relation that lets us skip a
    # second spheroidal eigensolve when switching spin
    assert lam_m2 + 2.0 == pytest.approx(
        swsh_eigenvalue(2, L, M, A * w) + 6.0, rel=1e-12)
    assert lam_p2 == pytest.approx(swsh_eigenvalue(2, L, M, A * w), rel=1e-12)


def test_ts_transform_matches_pointwise_identity():
    # exact correctness check: the projected spectral representation must
    # reproduce the pointwise TS identity f_TS*Psi + g_TS*Psi' it is built from
    from hblteuk.teuk import (
        hbl_teukolsky_starobinsky_system_minus_to_plus_2,
        hbl_teukolsky_starobinsky_system_plus_to_minus_2)
    w, lam_m2, lam_p2, lam_ch = _eigs()
    _, smin, smax = _geom()
    hp, _, _ = _solve(150, 2, w, lam_p2)
    hm, _, si = _solve(150, -2, w, lam_m2)

    sig = np.linspace(0.02 * smax, 0.98 * smax, 40)
    f, g = hbl_teukolsky_starobinsky_system_plus_to_minus_2(
        sig, KAPPA, lam_ch, M * A, w)
    want = f * hp.psi["Up"](sig) + g * hp.psi["Up"](sig, deriv=1)
    got = ts_up_plus_to_minus_2(hp.psi["Up"], KAPPA, lam_ch, M * A, w)(sig)
    np.testing.assert_allclose(got, want, rtol=1e-13)

    sigI = np.linspace(si[-1] + 0.01, 0.99, 40)
    f, g = hbl_teukolsky_starobinsky_system_minus_to_plus_2(
        sigI, KAPPA, lam_ch, M * A, w)
    want = f * hm.psi["In"](sigI) + g * hm.psi["In"](sigI, deriv=1)
    got = ts_in_minus_to_plus_2(hm.psi["In"], KAPPA, lam_ch, M * A, w)(sigI)
    np.testing.assert_allclose(got, want, rtol=1e-13)


def test_ts_transform_ode_residual_is_bounded():
    # The transformed coefficients are a *projection*, so unlike a collocation
    # solve the ODE is not enforced at the nodes and the spectrally-computed
    # psi'' carries amplified error (~1e-8; adding basis functions does not
    # help -- it is differentiation amplification, not truncation). This does
    # not affect production: the source integrator builds psi'' from the ODE
    # itself (source.py:104, the spectral deriv=2 call there is commented out),
    # so only psi and psi' matter -- see the refinement tests below.
    w, lam_m2, lam_p2, lam_ch = _eigs()
    _, _, smax = _geom()
    hp, _, _ = _solve(150, 2, w, lam_p2)
    psi_m2 = ts_up_plus_to_minus_2(hp.psi["Up"], KAPPA, lam_ch, M * A, w)
    sig = np.linspace(0.02 * smax, 0.98 * smax, 40)
    assert _residual(psi_m2, sig, -2, lam_m2, w) < 1e-6


def _drift(route):
    a, b = route(150), route(300)
    return np.max(np.abs(b - a) / np.abs(a))


def test_ts_up_route_is_refinement_stable():
    # the decisive property: the TS route converges under refinement while the
    # direct s=-2 Up march does not (it is roundoff-path dependent)
    w, lam_m2, lam_p2, lam_ch = _eigs()
    _, _, smax = _geom()
    sig = np.linspace(0.02 * smax, 0.98 * smax, 25)

    def ts_route(ndom):
        hp, _, _ = _solve(ndom, 2, w, lam_p2)
        p = ts_up_plus_to_minus_2(hp.psi["Up"], KAPPA, lam_ch, M * A, w)(sig)
        return p / p[0]

    def direct(ndom):
        hm, _, _ = _solve(ndom, -2, w, lam_m2)
        d = hm.psi["Up"](sig)
        return d / d[0]

    ts_drift, dir_drift = _drift(ts_route), _drift(direct)
    assert ts_drift < 1e-12                    # measured ~1.9e-15
    assert dir_drift > 100 * ts_drift          # measured ~2.7e-8


def test_ts_in_route_is_refinement_stable():
    # symmetric statement for s=+2, where In is the unstable branch
    w, lam_m2, lam_p2, lam_ch = _eigs()
    _, smin, _ = _geom()
    sig = np.linspace(smin + 0.02, 0.99, 25)

    def ts_route(ndom):
        hm, _, _ = _solve(ndom, -2, w, lam_m2)
        p = ts_in_minus_to_plus_2(hm.psi["In"], KAPPA, lam_ch, M * A, w)(sig)
        return p / p[0]

    def direct(ndom):
        hp, _, _ = _solve(ndom, 2, w, lam_p2)
        d = hp.psi["In"](sig)
        return d / d[0]

    ts_drift, dir_drift = _drift(ts_route), _drift(direct)
    assert ts_drift < 1e-12
    assert dir_drift > 100 * ts_drift


def test_stabilize_flag_matches_manual_transform():
    w, lam_m2, lam_p2, lam_ch = _eigs()
    _, smin, smax = _geom()

    hp, _, _ = _solve(150, 2, w, lam_p2)
    manual_up = ts_up_plus_to_minus_2(hp.psi["Up"], KAPPA, lam_ch, M * A, w)
    got_m2, _, _ = _solve(150, -2, w, lam_m2, stabilize=True)
    sig = np.linspace(0.02 * smax, 0.98 * smax, 25)
    np.testing.assert_allclose(got_m2.psi["Up"](sig), manual_up(sig),
                               rtol=1e-13)

    hm, _, si = _solve(150, -2, w, lam_m2)
    manual_in = ts_in_minus_to_plus_2(hm.psi["In"], KAPPA, lam_ch, M * A, w)
    got_p2, _, _ = _solve(150, 2, w, lam_p2, stabilize=True)
    sigI = np.linspace(si[-1] + 0.01, 0.99, 25)
    np.testing.assert_allclose(got_p2.psi["In"](sigI), manual_in(sigI),
                               rtol=1e-13)


def test_stabilize_leaves_the_stable_branch_untouched():
    # s=-2 In and s=+2 Up are already stable; the flag must not touch them
    w, lam_m2, lam_p2, lam_ch = _eigs()
    _, smin, smax = _geom()
    off_m2, _, si = _solve(150, -2, w, lam_m2)
    on_m2, _, _ = _solve(150, -2, w, lam_m2, stabilize=True)
    sigI = np.linspace(si[-1] + 0.01, 0.99, 25)
    np.testing.assert_array_equal(on_m2.psi["In"](sigI),
                                  off_m2.psi["In"](sigI))

    off_p2, _, _ = _solve(150, 2, w, lam_p2)
    on_p2, _, _ = _solve(150, 2, w, lam_p2, stabilize=True)
    sig = np.linspace(0.02 * smax, 0.98 * smax, 25)
    np.testing.assert_array_equal(on_p2.psi["Up"](sig), off_p2.psi["Up"](sig))


def test_stabilize_is_off_by_default():
    # default must remain the unstabilized path, bit for bit
    w, lam_m2, lam_p2, lam_ch = _eigs()
    _, _, smax = _geom()
    sig = np.linspace(0.02 * smax, 0.98 * smax, 25)
    a, _, _ = _solve(150, -2, w, lam_m2)
    b, _, _ = _solve(150, -2, w, lam_m2, stabilize=False)
    np.testing.assert_array_equal(a.psi["Up"](sig), b.psi["Up"](sig))


def test_stabilize_noop_for_other_spins():
    # s=0 has no TS partner here; the flag must fall through untouched
    w, lam_m2, lam_p2, lam_ch = _eigs()
    _, _, smax = _geom()
    lam0 = swsh_eigenvalue(0, L, M, A * w)
    sig = np.linspace(0.02 * smax, 0.98 * smax, 25)
    off, _, _ = _solve(150, 0, w, lam0)
    on, _, _ = _solve(150, 0, w, lam0, stabilize=True)
    np.testing.assert_array_equal(on.psi["Up"](sig), off.psi["Up"](sig))


@pytest.mark.parametrize("branch", ["Up", "In"])
def test_ts_first_derivative_is_refinement_stable(branch):
    # psi' is the other quantity the source integrator consumes (it builds
    # psi'' from the ODE). Spectral differentiation of a projected
    # representation costs a few orders vs psi itself, but still lands far
    # below the ~1e-8 floor of the direct unstable march.
    w, lam_m2, lam_p2, lam_ch = _eigs()
    _, smin, smax = _geom()
    if branch == "Up":
        sig = np.linspace(0.02 * smax, 0.98 * smax, 25)

        def ts_route(ndom):
            hp, _, _ = _solve(ndom, 2, w, lam_p2)
            p = ts_up_plus_to_minus_2(hp.psi["Up"], KAPPA, lam_ch,
                                      M * A, w)(sig, deriv=1)
            return p / p[0]

        def direct(ndom):
            d = _solve(ndom, -2, w, lam_m2)[0].psi["Up"](sig, deriv=1)
            return d / d[0]
    else:
        sig = np.linspace(smin + 0.02, 0.99, 25)

        def ts_route(ndom):
            hm, _, _ = _solve(ndom, -2, w, lam_m2)
            p = ts_in_minus_to_plus_2(hm.psi["In"], KAPPA, lam_ch,
                                      M * A, w)(sig, deriv=1)
            return p / p[0]

        def direct(ndom):
            d = _solve(ndom, 2, w, lam_p2)[0].psi["In"](sig, deriv=1)
            return d / d[0]

    ts_drift, dir_drift = _drift(ts_route), _drift(direct)
    assert ts_drift < 1e-9                  # measured ~2e-11
    assert dir_drift > 100 * ts_drift
