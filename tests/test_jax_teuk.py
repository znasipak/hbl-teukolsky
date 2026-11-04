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
