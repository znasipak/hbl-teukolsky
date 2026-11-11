# Batched JAX radial Teukolsky solver vs CPU reference. See dev/PHASE2.md.
import numpy as np
import pytest

jax = pytest.importorskip("jax")

from hblteuk import collocode
from hblteuk.cheby import MultiDomainChebyshev
from hblteuk.collocode import generate_subdomain_list
from hblteuk.jax import evaluate_multidomain, solve_homogeneous_batch
from hblteuk.jax.teukolsky import teuk_pqu
from hblteuk.swsh import swsh_eigenvalue
from hblteuk.teuk import HyperboloidalTeukolskySolver, sigma_r
from .conftest import ORBIT

A = ORBIT[0]
KAPPA = np.sqrt(1 - A**2)
NDOM, NNODE = 150, 32
MODES = [(2, 2, 0), (3, 1, 0), (10, 7, 0), (2, 2, 3), (4, -3, 1)]  # (l, m, n)


@pytest.fixture(scope="module")
def subs(orbit):
    smin = sigma_r(orbit.radial_roots[0], KAPPA)
    smax = sigma_r(orbit.radial_roots[1], KAPPA)
    return (generate_subdomain_list([0, smax], NDOM, "arcsinh7"),
            generate_subdomain_list([1, smin], NDOM, "arcsinh7"))


@pytest.fixture(scope="module")
def batch(orbit, subs):
    om = np.array([orbit.mode_frequency(m, 0, n) for _, m, n in MODES])
    lam = np.array([swsh_eigenvalue(-2, l, m, A * w)
                    for (l, m, _), w in zip(MODES, om)])
    ma = np.array([m * A for _, m, _ in MODES])
    res = solve_homogeneous_batch(-2, KAPPA, lam, ma, om, subs[0], subs[1],
                                  n=NNODE)
    return res, lam, ma, om


@pytest.fixture(scope="module")
def cpu_solutions(orbit, subs, batch):
    _, lam, ma, om = batch
    solver = collocode.CollocationODEFixedMultiDomainFixedStepSolver(n=NNODE)
    ts = HyperboloidalTeukolskySolver(
        domains=[[0, subs[0][-1]], [subs[1][-1], 1]], solver=solver,
        subdomains={"Up": subs[0], "In": subs[1]})
    return [ts(A, -2, l, m, om[b], eigenvalue=lam[b])
            for b, (l, m, _) in enumerate(MODES)]


def sample_points(subs):
    return (np.linspace(0.02 * subs[0][-1], 0.98 * subs[0][-1], 25),
            np.linspace(subs[1][-1] + 0.01, 0.99, 25))


def test_vs_cpu_fixed_solver(subs, batch, cpu_solutions):
    res = batch[0]
    sig_up, sig_in = sample_points(subs)
    for tag, coeffs, sub, sig in [("Up", res.coeffs_up, subs[0], sig_up),
                                  ("In", res.coeffs_in, subs[1], sig_in)]:
        for d in (0, 1, 2):
            got = np.asarray(evaluate_multidomain(coeffs, sub, sig, deriv=d))
            for b, hbl in enumerate(cpu_solutions):
                want = hbl.psi[tag](sig, deriv=d)
                # Re-tightened after TS branch stabilization. The residual
                # JAX-vs-CPU gap is localized near sigma=0 (2.5e-9), where
                # arcsinh7 puts the narrowest subdomains; over the orbital
                # range, which is what feeds amplitudes, it is 1.1e-12.
                np.testing.assert_allclose(got[b], want, rtol=1e-8,
                                           err_msg=f"{MODES[b]} {tag} d{d}")


def test_ode_residual(subs, batch):
    res, lam, ma, om = batch
    sig_up, sig_in = sample_points(subs)
    import jax.numpy as jnp
    for coeffs, sub, sig in [(res.coeffs_up, subs[0], sig_up),
                             (res.coeffs_in, subs[1], sig_in)]:
        psi = np.asarray(evaluate_multidomain(coeffs, sub, sig))
        dpsi = np.asarray(evaluate_multidomain(coeffs, sub, sig, deriv=1))
        d2psi = np.asarray(evaluate_multidomain(coeffs, sub, sig, deriv=2))
        P, Q, U = teuk_pqu(jnp.asarray(sig), KAPPA, -2,
                           jnp.asarray(lam, dtype=jnp.complex128),
                           jnp.asarray(ma),
                           jnp.asarray(om, dtype=jnp.complex128))
        resid = np.abs(np.asarray(P) * d2psi + np.asarray(Q) * dpsi
                       + np.asarray(U) * psi)
        scale = (np.abs(np.asarray(P) * d2psi) + np.abs(np.asarray(Q) * dpsi)
                 + np.abs(np.asarray(U) * psi))
        assert np.max(resid / scale) < 1e-8


def test_boundary_conditions(subs, batch):
    res, lam, ma, om = batch
    import jax.numpy as jnp
    for coeffs, sub, sig_b in [(res.coeffs_up, subs[0], 0.0),
                               (res.coeffs_in, subs[1], 1.0)]:
        psi = np.asarray(evaluate_multidomain(coeffs, sub, np.array([sig_b])))
        dpsi = np.asarray(evaluate_multidomain(coeffs, sub,
                                               np.array([sig_b]), deriv=1))
        _, Q, U = teuk_pqu(jnp.array([sig_b]), KAPPA, -2,
                           jnp.asarray(lam, dtype=jnp.complex128),
                           jnp.asarray(ma),
                           jnp.asarray(om, dtype=jnp.complex128))
        ratio = -np.asarray(U)[:, 0] / np.asarray(Q)[:, 0]
        np.testing.assert_allclose(dpsi[:, 0] / psi[:, 0], ratio, rtol=1e-9)


def test_solver_residuals_small(batch):
    res = batch[0]
    assert float(res.errors_up.max()) < 1e-9
    assert float(res.errors_in.max()) < 1e-9


def test_vs_golden(orbit, subs, golden):
    # golden was produced by the adaptive solver; fixed-domain equivalence
    # is established in test_teuk.py::test_fixed_vs_dynamic_solver
    params = [golden[f"hom{i}_params"] for i in range(4)]
    lam_g = np.array([float(golden[f"hom{i}_eigenvalue"]) for i in range(4)])
    om = np.array([p[4] for p in params])
    ma = np.array([p[2] * A for p in params])
    res = solve_homogeneous_batch(-2, KAPPA, lam_g, ma, om, subs[0], subs[1],
                                  n=NNODE)
    for i in range(4):
        for tag, coeffs, sub in [("up", res.coeffs_up, subs[0]),
                                 ("in", res.coeffs_in, subs[1])]:
            sig = golden[f"hom{i}_sig_{tag}"]
            for d in (0, 1, 2):
                got = np.asarray(evaluate_multidomain(coeffs, sub, sig,
                                                      deriv=d))[i]
                want = golden[f"hom{i}_psi_{tag}_d{d}"]
                # golden used per-mode adaptive subdomains; the march's
                # accuracy is absolute relative to the domain peak (psi''
                # spans ~10 decades on In), so scale atol to the peak
                np.testing.assert_allclose(
                    got, want, rtol=1e-7, atol=1e-8 * np.max(np.abs(want)),
                    err_msg=f"golden case {i} {tag} d{d}")


def test_eval_matches_cpu_container(subs, batch):
    # both orientations, all derivative orders, off-node points
    res = batch[0]
    rng = np.random.default_rng(7)
    for coeffs, sub in [(res.coeffs_up, subs[0]), (res.coeffs_in, subs[1])]:
        lo, hi = min(sub[0], sub[-1]), max(sub[0], sub[-1])
        sig = rng.uniform(lo + 1e-4, hi - 1e-4, 40)
        cpu = MultiDomainChebyshev(np.asarray(coeffs[2]), sub)
        for d in (0, 1, 2):
            got = np.asarray(evaluate_multidomain(coeffs, sub, sig,
                                                  deriv=d))[2]
            np.testing.assert_allclose(got, cpu(sig, deriv=d), rtol=1e-11)


def test_static_mode_rejected(subs):
    with pytest.raises(ValueError, match="static"):
        solve_homogeneous_batch(-2, KAPPA, [4.0], [0.0], [0.0],
                                subs[0], subs[1])


def _eval_sigma_sets(orbit, sub):
    # orbit samples, sigma exactly on three interior boundaries, one domain
    smin = sigma_r(orbit.radial_roots[0], KAPPA)
    smax = sigma_r(orbit.radial_roots[1], KAPPA)
    asc = np.sort(sub)
    mid = len(asc) // 2
    return {"orbit": np.linspace(smin, smax, 37),
            "boundaries": np.array([asc[1], asc[mid], asc[-2],
                                    0.5 * (asc[mid] + asc[mid + 1])]),
            "one_domain": np.linspace(asc[mid] + 1e-3 * (asc[mid + 1] - asc[mid]),
                                      asc[mid + 1] - 1e-3 * (asc[mid + 1] - asc[mid]),
                                      9)}


def test_evaluation_matrix_matches_multidomain(orbit, subs, batch):
    from hblteuk.jax.chebyshev import evaluate_with_matrix, evaluation_matrix
    res = batch[0]
    for coeffs, sub in [(res.coeffs_up, subs[0]), (res.coeffs_in, subs[1])]:
        for tag, sig in _eval_sigma_sets(orbit, sub).items():
            em = evaluation_matrix(sub, sig, NNODE)
            got = np.asarray(evaluate_with_matrix(coeffs, em))
            assert got.shape == (3, len(MODES), len(sig))
            for d in (0, 1, 2):
                ref = np.asarray(evaluate_multidomain(coeffs, sub, sig,
                                                      deriv=d))
                err = np.max(np.abs(got[d] - ref)) / np.max(np.abs(ref))
                assert err < 1e-13, f"{tag} d{d}: {err:.2e}"
            if tag == "one_domain":
                assert em.hi - em.lo == 1


def test_evaluation_matrix_cached(orbit, subs):
    from hblteuk.jax.chebyshev import evaluation_matrix
    sig = _eval_sigma_sets(orbit, subs[1])["orbit"]
    assert (evaluation_matrix(subs[1], sig, NNODE)
            is evaluation_matrix(np.array(subs[1]), sig.copy(), NNODE))


@pytest.mark.parametrize("s", [-2, 2])
def test_parallel_march_matches_reference(orbit, subs, s):
    # Strict where the discrete problem is well conditioned: the stable
    # branch (In at s=-2, Up at s=+2) of resolved modes. Unstable branches sit
    # at a ~1e-9 rounding floor (the reference disagrees with itself there
    # when only the batch size changes), and (30, 25, 0) is under-resolved in
    # one wide, 1e6-growth subdomain (n=32 off by 1e-5 vs n=64 for BOTH
    # methods); those get a loose gross-error bound. dev/JAX_PERF.md, Task 6.
    import jax.numpy as jnp
    from hblteuk.jax.collocation import (
        _solve_fixed_multidomain_batch_reference, solve_fixed_multidomain_batch)
    from hblteuk.jax.teukolsky import (boundary_conditions_in,
                                       boundary_conditions_up)
    modes = MODES + [(30, 25, 0), (40, -40, 2)]
    om = jnp.array([orbit.mode_frequency(m, 0, n) for _, m, n in modes])
    lam = jnp.array([swsh_eigenvalue(s, l, m, A * w)
                     for (l, m, _), w in zip(modes, np.asarray(om))])
    ma = jnp.array([m * A for _, m, _ in modes], dtype=jnp.float64)

    def pqu(z):
        return teuk_pqu(z, KAPPA, s, lam, ma, om)

    stable = "in" if s == -2 else "up"
    for fam, bc, sub in [("up", boundary_conditions_up, subs[0]),
                         ("in", boundary_conditions_in, subs[1])]:
        y0 = bc(KAPPA, s, lam, ma, om)
        c_new, e_new = solve_fixed_multidomain_batch(pqu, y0, sub, n=NNODE)
        c_ref, e_ref = _solve_fixed_multidomain_batch_reference(
            pqu, y0, sub, n=NNODE)
        c_new, c_ref = np.asarray(c_new), np.asarray(c_ref)
        for b, mode in enumerate(modes):
            tol = 1e-11 if fam == stable and mode != (30, 25, 0) else 1e-6
            err = (np.max(np.abs(c_new[b] - c_ref[b]))
                   / np.max(np.abs(c_ref[b])))
            assert err < tol, f"s={s} {fam} {mode}: {err:.2e}"
        # residual estimates agree above the rounding floor (to 0.3% in the
        # under-resolved domain)
        np.testing.assert_allclose(np.asarray(e_new), np.asarray(e_ref),
                                   rtol=1e-2, atol=1e-11)


@pytest.mark.parametrize("s", [-2, 2])
def test_ultraspherical_march_matches_collocation(orbit, subs, s):
    # Resolved stable-branch modes: same answer as collocation at 150 x 32 to
    # the test_parallel_march_matches_reference bound. Elsewhere (unstable
    # branches; (30, 25, 0) and (40, -40, 2), which collocation itself gets
    # wrong by 1e-5 / 2e-7 at n = 32) judge both against n = 64 collocation:
    # ultraspherical no worse than 10x collocation. Both schemes run back to
    # back so a shared trace would show up here.
    import jax.numpy as jnp
    from hblteuk.jax.collocation import solve_fixed_multidomain_batch
    from hblteuk.jax.teukolsky import (boundary_conditions_in,
                                       boundary_conditions_up)
    modes = MODES + [(30, 25, 0), (40, -40, 2)]
    hard = {(30, 25, 0), (40, -40, 2)}
    om = jnp.array([orbit.mode_frequency(m, 0, n) for _, m, n in modes])
    lam = jnp.array([swsh_eigenvalue(s, l, m, A * w)
                     for (l, m, _), w in zip(modes, np.asarray(om))])
    ma = jnp.array([m * A for _, m, _ in modes], dtype=jnp.float64)

    def pqu(z):
        return teuk_pqu(z, KAPPA, s, lam, ma, om)

    stable = "in" if s == -2 else "up"
    pts = sample_points(subs)
    for fam, bc, sub, sig in [("up", boundary_conditions_up, subs[0], pts[0]),
                              ("in", boundary_conditions_in, subs[1], pts[1])]:
        y0 = bc(KAPPA, s, lam, ma, om)
        c_col, _ = solve_fixed_multidomain_batch(pqu, y0, sub, n=NNODE)
        c_us, e_us = solve_fixed_multidomain_batch(pqu, y0, sub, n=NNODE,
                                                   scheme="ultraspherical")
        c_ref, _ = solve_fixed_multidomain_batch(pqu, y0, sub, n=64)
        v_col, v_us, v_ref = (np.asarray(evaluate_multidomain(c, sub, sig))
                              for c in (c_col, c_us, c_ref))
        v_col, v_us, v_ref = (v / v[:, :1] for v in (v_col, v_us, v_ref))
        c_col, c_us, e_us = (np.asarray(x) for x in (c_col, c_us, e_us))
        assert np.all(np.isfinite(e_us))
        for b, mode in enumerate(modes):
            if fam == stable and mode not in hard:
                err = (np.max(np.abs(c_us[b] - c_col[b]))
                       / np.max(np.abs(c_col[b])))
                assert err < 1e-11, f"s={s} {fam} {mode}: {err:.2e}"
                assert e_us[b].max() < 1e-8, f"s={s} {fam} {mode}"
            else:
                e_u, e_c = (np.max(np.abs(v[b] - v_ref[b]) / np.abs(v_ref[b]))
                            for v in (v_us, v_col))
                assert e_u <= max(10 * e_c, 1e-12), \
                    f"s={s} {fam} {mode}: us {e_u:.2e} col {e_c:.2e}"


def test_ultraspherical_fewer_domains(orbit, subs, batch):
    # at 8 domains ultraspherical is never worse than collocation and holds
    # 1e-9 on the low-l modes, against 150 collocation domains (stabilized).
    # (10, 7, 0) In needs more domains on this orbit: 5.7e-5 here, 0.16 for
    # collocation; the driver default is set from flux totals (Task 4).
    _, lam, ma, om = batch
    smax, smin = subs[0][-1], subs[1][-1]
    su = generate_subdomain_list([0, smax], 8, "arcsinh7")
    si = generate_subdomain_list([1, smin], 8, "arcsinh7")
    ref = solve_homogeneous_batch(-2, KAPPA, lam, ma, om, subs[0], subs[1],
                                  n=NNODE, scheme="collocation")
    got = {sc: solve_homogeneous_batch(-2, KAPPA, lam, ma, om, su, si,
                                       n=NNODE, scheme=sc)
           for sc in ("ultraspherical", "collocation")}
    low_l = np.array([l <= 4 for l, _, _ in MODES])
    for k, (sub8, sig) in enumerate(zip((su, si), sample_points(subs))):
        b = np.asarray(evaluate_multidomain(ref[k], subs[k], sig))
        b = b / b[:, :1]
        err = {}
        for sc, h in got.items():
            a = np.asarray(evaluate_multidomain(h[k], sub8, sig))
            err[sc] = np.max(np.abs(a / a[:, :1] - b) / np.abs(b), axis=1)
        fam = ("up", "in")[k]
        assert np.all(err["ultraspherical"]
                      <= np.maximum(err["collocation"], 1e-12)), fam
        assert np.all(err["ultraspherical"][low_l] < 1e-9), fam


def test_scheme_rejected(subs, batch):
    _, lam, ma, om = batch
    with pytest.raises(ValueError, match="scheme must be"):
        solve_homogeneous_batch(-2, KAPPA, lam, ma, om, subs[0], subs[1],
                                n=NNODE, scheme="tau")
