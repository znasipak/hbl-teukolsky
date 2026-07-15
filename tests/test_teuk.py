# Homogeneous radial Teukolsky solution tests. See dev/PHASE0.md.
import numpy as np
import pytest

from hblteuk import collocode
from hblteuk.teuk import (
    HyperboloidalTeukolskySolver,
    q_hbl,
    sigma_r,
    teuk_sys,
    u_hbl,
)
from .conftest import ORBIT, SOLVER_KWARGS

HOM_CASE_IDS = range(4)


@pytest.fixture(scope="module")
def teuk_solver(orbit):
    a = ORBIT[0]
    kappa = np.sqrt(1 - a**2)
    rmin, rmax = orbit.radial_roots[1], orbit.radial_roots[0]
    smin, smax = sigma_r(rmax, kappa), sigma_r(rmin, kappa)
    solver = collocode.CollocationODEMultiDomainFixedStepSolver(n=32)
    return HyperboloidalTeukolskySolver(
        domains=[[0, smax], [smin, 1]], solver=solver, **SOLVER_KWARGS)


def solve_case(orbit, teuk_solver, s, l, m, n):
    omega = orbit.mode_frequency(m, 0, n)
    return teuk_solver(ORBIT[0], s, l, m, omega), omega


def test_ode_residual(orbit, teuk_solver):
    # P psi'' + Q psi' + U psi = 0 at interior points, both branches
    a = ORBIT[0]
    kappa = np.sqrt(1 - a**2)
    hbl, omega = solve_case(orbit, teuk_solver, -2, 2, 2, 0)
    for bc in ("Up", "In"):
        dom = hbl.psi[bc].domains
        sig = np.linspace(min(dom) + 0.01, max(dom) - 0.01, 30)
        P, Q, U = teuk_sys(sig, kappa, -2, hbl.eigenvalue, 2 * a, omega)
        psi = hbl.psi[bc](sig)
        dpsi = hbl.psi[bc](sig, deriv=1)
        d2psi = hbl.psi[bc](sig, deriv=2)
        resid = P * d2psi + Q * dpsi + U * psi
        scale = np.abs(P * d2psi) + np.abs(Q * dpsi) + np.abs(U * psi)
        assert np.max(np.abs(resid) / scale) < 1e-8


def test_boundary_conditions(orbit, teuk_solver):
    # regular-singular BC: psi'(sigma_b) = -U/Q * psi(sigma_b)
    a = ORBIT[0]
    kappa = np.sqrt(1 - a**2)
    hbl, omega = solve_case(orbit, teuk_solver, -2, 3, 1, 0)
    lam = hbl.eigenvalue
    for bc, sig_b in [("Up", 0.0), ("In", 1.0)]:
        ratio = -u_hbl(sig_b, kappa, -2, lam, a, omega) / q_hbl(sig_b, kappa, -2, a, omega)
        psi = hbl.psi[bc](np.array([sig_b]))[0]
        dpsi = hbl.psi[bc](np.array([sig_b]), deriv=1)[0]
        assert dpsi / psi == pytest.approx(ratio, rel=1e-9)


def test_homogeneous_vs_golden(orbit, teuk_solver, golden):
    for i in HOM_CASE_IDS:
        s, l, m, n, omega_g = golden[f"hom{i}_params"]
        hbl, omega = solve_case(orbit, teuk_solver, int(s), int(l), int(m), int(n))
        assert omega == pytest.approx(omega_g, rel=1e-13)
        assert hbl.eigenvalue == pytest.approx(
            float(golden[f"hom{i}_eigenvalue"]), rel=1e-10)
        for bc, tag in [("Up", "up"), ("In", "in")]:
            sig = golden[f"hom{i}_sig_{tag}"]
            for d in (0, 1, 2):
                got = hbl.psi[bc](sig, deriv=d)
                np.testing.assert_allclose(
                    got, golden[f"hom{i}_psi_{tag}_d{d}"], rtol=1e-10,
                    err_msg=f"case {i} {bc} deriv={d}")


def test_fixed_vs_dynamic_solver(orbit, teuk_solver):
    # frozen-subdomain solver (grid/GPU path) must match adaptive solver
    a = ORBIT[0]
    hbl_dyn, omega = solve_case(orbit, teuk_solver, -2, 2, 2, 0)
    fixed = collocode.CollocationODEFixedMultiDomainFixedStepSolver(n=32)
    kw = dict(SOLVER_KWARGS)
    kw["subdomains"] = {"Up": hbl_dyn.psi["Up"].domains,
                        "In": hbl_dyn.psi["In"].domains}
    solver2 = HyperboloidalTeukolskySolver(
        domains=[[0, max(hbl_dyn.psi["Up"].domains)],
                 [min(hbl_dyn.psi["In"].domains), 1]],
        solver=fixed, **kw)
    hbl_fix = solver2(a, -2, 2, 2, omega)
    for bc in ("Up", "In"):
        dom = hbl_dyn.psi[bc].domains
        sig = np.linspace(min(dom) + 0.02, max(dom) - 0.02, 17)
        np.testing.assert_allclose(
            hbl_fix.psi[bc](sig), hbl_dyn.psi[bc](sig), rtol=1e-10)
