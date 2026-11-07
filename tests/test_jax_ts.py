# Batched Teukolsky-Starobinsky stabilization vs the CPU path.
# See dev/ACCURACY_DEBUG_PLAN.md for why the transform is built as an ODE
# solve with TS-supplied Dirichlet endpoints rather than as a projection.
import numpy as np
import pytest

jax = pytest.importorskip("jax")

from hblteuk import collocode
from hblteuk.collocode import generate_subdomain_list
from hblteuk.geo import MiniGeo
from hblteuk.jax import evaluate_multidomain, solve_homogeneous_batch
from hblteuk.jax.starobinsky import (flipped_spin, ts_solve_batch,
                                     unstable_branch)
from hblteuk.swsh import swsh_eigenvalue
from hblteuk.teuk import (HyperboloidalTeukolskySolver, sigma_r,
                          hbl_teukolsky_starobinsky_system_plus_to_minus_2)
from .conftest import ORBIT

A = ORBIT[0]
KAPPA = np.sqrt(1 - A**2)
NDOM, NNODE = 150, 32
MODES = [(2, 2, 0), (3, 1, 0), (2, 2, 3), (4, -3, 1)]


@pytest.fixture(scope="module")
def geom():
    orbit = MiniGeo(*ORBIT)
    smin = sigma_r(orbit.radial_roots[0], KAPPA)
    smax = sigma_r(orbit.radial_roots[1], KAPPA)
    su = generate_subdomain_list([0, smax], NDOM, "arcsinh7")
    si = generate_subdomain_list([1, smin], NDOM, "arcsinh7")
    return orbit, su, si, smin, smax


def test_branch_spec():
    assert unstable_branch(-2) == "up" and flipped_spin(-2) == 2
    assert unstable_branch(2) == "in" and flipped_spin(2) == -2
    assert unstable_branch(0) is None and flipped_spin(0) is None


def test_fg_expressions_evaluate_on_jax_arrays():
    # the CPU f_TS/g_TS are pure arithmetic, so they are reused as-is in the
    # JAX path; this pins that they agree on jnp input (x64 is on via
    # `import hblteuk.jax`)
    import jax.numpy as jnp
    sig = np.linspace(1e-3, 0.11, 17)
    args = (0.43589, 5.243658341220, 1.8, 0.4548)
    f_np, g_np = hbl_teukolsky_starobinsky_system_plus_to_minus_2(sig, *args)
    f_jx, g_jx = hbl_teukolsky_starobinsky_system_plus_to_minus_2(
        jnp.asarray(sig), *args)
    np.testing.assert_allclose(np.asarray(f_jx), f_np, rtol=1e-13)
    np.testing.assert_allclose(np.asarray(g_jx), g_np, rtol=1e-13)


def test_ts_solve_batch_matches_cpu_stabilized(geom):
    # JAX batched stabilization must reproduce the CPU stabilize=True solution
    orbit, su, si, smin, smax = geom
    om = np.array([orbit.mode_frequency(m, 0, n) for _, m, n in MODES])
    lam = np.array([swsh_eigenvalue(-2, l, m, A * w)
                    for (l, m, _), w in zip(MODES, om)])
    ma = np.array([m * A for _, m, _ in MODES])

    # stable s=+2 Up batch, then TS to s=-2
    flip = solve_homogeneous_batch(2, KAPPA, lam - 4.0, ma, om, su, si,
                                   n=NNODE)
    got = ts_solve_batch(flip.coeffs_up, su, KAPPA, -2, lam, ma, om, n=NNODE)

    solver = collocode.CollocationODEFixedMultiDomainFixedStepSolver(n=NNODE)
    ts = HyperboloidalTeukolskySolver(
        domains=[[0, su[-1]], [si[-1], 1]], solver=solver,
        subdomains={"Up": su, "In": si})
    sig = np.linspace(0.02 * smax, 0.98 * smax, 25)
    ev = np.asarray(evaluate_multidomain(got, su, sig))
    for b, (l, m, n) in enumerate(MODES):
        cpu = ts(A, -2, l, m, om[b], eigenvalue=lam[b], stabilize=True)
        want = cpu.psi["Up"](sig)
        np.testing.assert_allclose(ev[b], want, rtol=1e-9,
                                   err_msg=f"{MODES[b]}")


def test_ts_solve_batch_is_refinement_stable(geom):
    # the payoff, in the batched path: the TS route converges under
    # refinement while the direct s=-2 Up march does not
    orbit, _, _, smin, smax = geom
    l, m, n = 2, 2, 3
    w = orbit.mode_frequency(m, 0, n)
    lam = np.array([swsh_eigenvalue(-2, l, m, A * w)])
    ma = np.array([m * A])
    om = np.array([w])
    sig = np.linspace(0.02 * smax, 0.98 * smax, 25)

    def routes(nd):
        su = generate_subdomain_list([0, smax], nd, "arcsinh7")
        si = generate_subdomain_list([1, smin], nd, "arcsinh7")
        flip = solve_homogeneous_batch(2, KAPPA, lam - 4.0, ma, om, su, si,
                                       n=NNODE)
        c = ts_solve_batch(flip.coeffs_up, su, KAPPA, -2, lam, ma, om,
                           n=NNODE)
        direct = solve_homogeneous_batch(-2, KAPPA, lam, ma, om, su, si,
                                         n=NNODE, stabilize=False)
        t = np.asarray(evaluate_multidomain(c, su, sig))[0]
        d = np.asarray(evaluate_multidomain(direct.coeffs_up, su, sig))[0]
        return t / t[0], d / d[0]

    t150, d150 = routes(150)
    t300, d300 = routes(300)
    ts_drift = np.max(np.abs(t300 - t150) / np.abs(t150))
    dir_drift = np.max(np.abs(d300 - d150) / np.abs(d150))
    assert ts_drift < 1e-11
    assert dir_drift > 100 * ts_drift


def test_ts_solve_batch_rejects_unsupported_spin(geom):
    orbit, su, si, _, _ = geom
    om = np.array([0.3])
    with pytest.raises(ValueError, match="TS stabilization"):
        ts_solve_batch(np.zeros((1, len(su) - 1, NNODE), complex), su, KAPPA,
                       0, np.array([4.0]), np.array([0.0]), om, n=NNODE)


def test_solve_homogeneous_batch_stabilize_flag(geom):
    # the flag must reproduce the manual TS route, leave the stable branch
    # alone, and default to off
    orbit, su, si, smin, smax = geom
    om = np.array([orbit.mode_frequency(m, 0, n) for _, m, n in MODES])
    lam = np.array([swsh_eigenvalue(-2, l, m, A * w)
                    for (l, m, _), w in zip(MODES, om)])
    ma = np.array([m * A for _, m, _ in MODES])

    off = solve_homogeneous_batch(-2, KAPPA, lam, ma, om, su, si, n=NNODE,
                                  stabilize=False)
    on = solve_homogeneous_batch(-2, KAPPA, lam, ma, om, su, si, n=NNODE,
                                 stabilize=True)
    flip = solve_homogeneous_batch(2, KAPPA, lam - 4.0, ma, om, su, si,
                                   n=NNODE)
    manual = ts_solve_batch(flip.coeffs_up, su, KAPPA, -2, lam, ma, om,
                            n=NNODE)

    np.testing.assert_allclose(np.asarray(on.coeffs_up),
                               np.asarray(manual), rtol=1e-12)
    # In is already stable at s=-2 and must be untouched
    np.testing.assert_array_equal(np.asarray(on.coeffs_in),
                                  np.asarray(off.coeffs_in))
    # on by default
    dflt = solve_homogeneous_batch(-2, KAPPA, lam, ma, om, su, si, n=NNODE)
    np.testing.assert_array_equal(np.asarray(dflt.coeffs_up),
                                  np.asarray(on.coeffs_up))


def test_solve_homogeneous_batch_stabilize_noop_other_spin(geom):
    # s=0 has no TS partner here; the flag must fall through untouched
    orbit, su, si, _, _ = geom
    om = np.array([orbit.mode_frequency(2, 0, 0)])
    lam = np.array([swsh_eigenvalue(0, 2, 2, A * om[0])])
    ma = np.array([2 * A])
    off = solve_homogeneous_batch(0, KAPPA, lam, ma, om, su, si, n=NNODE,
                                  stabilize=False)
    on = solve_homogeneous_batch(0, KAPPA, lam, ma, om, su, si, n=NNODE,
                                 stabilize=True)
    np.testing.assert_array_equal(np.asarray(on.coeffs_up),
                                  np.asarray(off.coeffs_up))


def test_stabilized_amplitudes_tighten_the_pybhpt_gate():
    # The strongest available check: pybhpt is an independent implementation.
    # The ~3.7e-9 agreement recorded as "the gate" in dev/PHASE5A.md was our
    # error, not an irreducible floor -- stabilizing collapses it to ~1e-13.
    import pathlib
    from hblteuk.jax import solve_mode_table
    ref = np.load(pathlib.Path(__file__).parent / "golden"
                  / "pybhpt_reference_v2.npz")
    C = dict(l=0, m=1, n=2, zin_re=5, zin_im=6, zup_re=7, zup_im=8)
    for tag in ("ecc", "circ"):
        t = ref[f"{tag}_table"]
        orbit = MiniGeo(*ref[f"{tag}_params"])
        l = t[:, C["l"]].astype(int)
        m = t[:, C["m"]].astype(int)
        n = t[:, C["n"]].astype(int)
        zup = t[:, C["zup_re"]] + 1j * t[:, C["zup_im"]]

        off = np.asarray(solve_mode_table(orbit, -2, l, m, n,
                                          stabilize=False).amplitudes)
        on = np.asarray(solve_mode_table(orbit, -2, l, m, n,
                                         stabilize=True).amplitudes)
        d_off = np.max(np.abs(off[:, 1] - zup) / np.abs(zup))
        d_on = np.max(np.abs(on[:, 1] - zup) / np.abs(zup))
        assert d_on < 1e-11, f"{tag}: Z^Up vs pybhpt {d_on:.2e}"
        assert d_on < d_off / 100, f"{tag}: {d_on:.2e} vs {d_off:.2e}"


@pytest.mark.parametrize("orbit_kind", ["eccentric", "spherical"])
def test_total_fluxes_stabilize_threads_through(orbit_kind):
    # the flag must reach the flux driver and leave the converged answer
    # unchanged well within tol (it corrects a ~1e-8 level error)
    from hblteuk.jax import total_fluxes
    orb = (MiniGeo(0.9, 10.0, 0.2, 1.0) if orbit_kind == "eccentric"
           else MiniGeo(0.9, 10.0, 0.0, 0.5))
    off = total_fluxes(orb, tol=1e-5, strategy="box", stabilize=False)
    on = total_fluxes(orb, tol=1e-5, strategy="box", stabilize=True)
    for ch in ("Edot", "Ldot", "Qdot"):
        np.testing.assert_allclose(getattr(on, ch), getattr(off, ch),
                                   rtol=1e-5, atol=1e-30)
    assert len(on.table["l"]) == len(off.table["l"])


def _mode_params(orbit, s):
    om = np.array([orbit.mode_frequency(m, 0, n) for _, m, n in MODES])
    lam = np.array([swsh_eigenvalue(s, l, m, A * w)
                    for (l, m, _), w in zip(MODES, om)])
    ma = np.array([m * A for _, m, _ in MODES])
    return lam, ma, om


def test_stabilized_marches_each_branch_once(geom, monkeypatch):
    # one march per family: the unstable branch at the flipped spin, the
    # stable branch at s -- not both branches twice
    from hblteuk.jax import teukolsky
    orbit, su, si, _, _ = geom
    lam, ma, om = _mode_params(orbit, -2)
    calls = []
    real = teukolsky.solve_fixed_multidomain_batch

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(teukolsky, "solve_fixed_multidomain_batch", counting)
    with jax.disable_jit():
        solve_homogeneous_batch(-2, KAPPA, lam[:1], ma[:1], om[:1], su, si,
                                n=NNODE, stabilize=True)
    assert len(calls) == 2


@pytest.mark.parametrize("s", [-2, 2])
def test_stabilized_equals_core_reference(geom, s):
    # the pre-optimization route: both branches at both spins, keep one each
    from hblteuk.jax.teukolsky import _solve_core
    orbit, su, si, _, _ = geom
    lam, ma, om = _mode_params(orbit, s)
    s_flip = flipped_spin(s)
    lam_flip = lam + (s * (s + 1) - s_flip * (s_flip + 1))
    jnp = jax.numpy
    subs = (jnp.asarray(su), jnp.asarray(si))
    args = (float(KAPPA),)
    flip = _solve_core(s_flip, *args, jnp.asarray(lam_flip, dtype=complex),
                       jnp.asarray(ma), jnp.asarray(om, dtype=complex),
                       subs, NNODE)
    plain = _solve_core(s, *args, jnp.asarray(lam, dtype=complex),
                        jnp.asarray(ma), jnp.asarray(om, dtype=complex),
                        subs, NNODE)
    new = solve_homogeneous_batch(s, KAPPA, lam, ma, om, su, si, n=NNODE,
                                  stabilize=True)
    if unstable_branch(s) == "up":
        ref = (ts_solve_batch(flip.coeffs_up, su, KAPPA, s, lam, ma, om,
                              n=NNODE), plain.coeffs_in,
               flip.errors_up, plain.errors_in)
    else:
        ref = (plain.coeffs_up, ts_solve_batch(flip.coeffs_in, si, KAPPA, s,
                                               lam, ma, om, n=NNODE),
               plain.errors_up, flip.errors_in)
    for got, want in zip(new, ref):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want),
                                   rtol=1e-13, atol=0)


def test_ts_solve_batch_is_jitted_and_matches_eager(geom):
    # one compiled program: eager dispatch costs ~22 ms per chunk on H200
    # (dev/JAX_PERF.md), and compiling must not move the result
    from hblteuk.jax.starobinsky import _ts_solve_core

    orbit, su, si, _, _ = geom
    om = np.array([orbit.mode_frequency(m, 0, n) for _, m, n in MODES])
    lam = np.array([swsh_eigenvalue(-2, l, m, A * w)
                    for (l, m, _), w in zip(MODES, om)])
    ma = np.array([m * A for _, m, _ in MODES])
    flip = solve_homogeneous_batch(2, KAPPA, lam - 4.0, ma, om, su, si,
                                   n=NNODE)

    got = ts_solve_batch(flip.coeffs_up, su, KAPPA, -2, lam, ma, om, n=NNODE)
    size = _ts_solve_core._cache_size()
    assert size >= 1
    ts_solve_batch(flip.coeffs_up, su, KAPPA, -2, lam, ma, om, n=NNODE)
    assert _ts_solve_core._cache_size() == size          # reused, not recompiled

    with jax.disable_jit():
        ref = ts_solve_batch(flip.coeffs_up, su, KAPPA, -2, lam, ma, om,
                             n=NNODE)
    got, ref = np.asarray(got), np.asarray(ref)
    assert np.max(np.abs(got - ref)) / np.max(np.abs(ref)) < 1e-12


@pytest.mark.parametrize("target,branch", [(-2, "up"), (2, "in")])
def test_ts_renormalization_is_exact_at_low_frequency(target, branch):
    # the boundary value of the mapped solution is below roundoff at small
    # omega (Up: 16 omega^4). Dividing by a measured value gave noise, or NaN
    # when it rounded to zero on H200, so the factor must be the closed form.
    from hblteuk.jax.starobinsky import _SPEC
    from hblteuk.jax.teukolsky import _bc_ratio

    orbit = MiniGeo(0.9, 8.0, 0.3, 0.7)
    a, kappa = orbit.a, np.sqrt(1 - orbit.a**2)
    su = generate_subdomain_list([0, sigma_r(orbit.radial_roots[1], kappa)],
                                 NDOM, "arcsinh7")
    si = generate_subdomain_list([1, sigma_r(orbit.radial_roots[0], kappa)],
                                 NDOM, "arcsinh7")
    modes = [(13, 11, -1, -15), (14, 11, -1, -15), (2, 2, 0, 0), (4, 3, 1, 2)]
    om = np.array([orbit.mode_frequency(m, k, n) for _, m, k, n in modes])
    lam = np.array([swsh_eigenvalue(target, l, m, a * w)
                    for (l, m, _, _), w in zip(modes, om)])
    ma = np.array([m * a for _, m, _, _ in modes])

    source = flipped_spin(target)
    flip = solve_homogeneous_batch(
        source, kappa, lam + target * (target + 1) - source * (source + 1),
        ma, om, su, si, n=NNODE, stabilize=False)
    coeffs, subs, sb = ((flip.coeffs_up, su, 0.0) if branch == "up"
                        else (flip.coeffs_in, si, 1.0))
    raw = np.asarray(ts_solve_batch(coeffs, subs, kappa, target, lam, ma, om,
                                    n=NNODE, renormalize=False))
    out = np.asarray(ts_solve_batch(coeffs, subs, kappa, target, lam, ma, om,
                                    n=NNODE))

    lam_c = lam + 0j
    lam_ch = lam_c + target * (target + 1)
    sig = np.array([[sb]])
    f, g = _SPEC[target][2](sig, kappa, lam_ch[:, None], ma[:, None],
                            om[:, None] + 0j)
    bc = np.asarray(_bc_ratio(sb, kappa, source,
                              jax.numpy.asarray(lam_ch - source * (source + 1)),
                              jax.numpy.asarray(ma),
                              jax.numpy.asarray(om + 0j)))
    ratio = np.ravel(f) + np.ravel(g) * bc
    if branch == "up":
        np.testing.assert_allclose(ratio, 16 * om**4, rtol=1e-12)
    np.testing.assert_allclose(out[:, 7, 3] * ratio, raw[:, 7, 3], rtol=1e-12)
