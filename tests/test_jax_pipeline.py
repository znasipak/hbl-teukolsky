# End-to-end pipeline, projection, and mode-summation tests. See dev/PHASE4.md.
import numpy as np
import pytest

jax = pytest.importorskip("jax")

from hblteuk import collocode
from hblteuk.geo import MiniGeo
from hblteuk.mode import TeukolskyPointParticleModeGridGenerator
from hblteuk.swsh import SpinWeightedSpheroidalHarmonic
from hblteuk.jax import (assemble_field, evaluate_multidomain,
                         project_to_spherical, solve_circular_grid,
                         solve_mode_table, swsh_theta_values,
                         waveform_timeseries)

LRANGE = [2, 5]


@pytest.fixture(scope="module")
def circ_orbit():
    return MiniGeo(0.9, 10.0, 0.0, 1.0)


@pytest.fixture(scope="module")
def cpu_gen(circ_orbit):
    solver = collocode.CollocationODEFixedMultiDomainFixedStepSolver(n=32)
    gen = TeukolskyPointParticleModeGridGenerator(
        circ_orbit, solver=solver,
        solver_kwargs={"subdomains": 150, "tol": 1e-13, "spacing": "arcsinh7"})
    gen.optimize(-2, LRANGE[1])
    return gen


@pytest.fixture(scope="module")
def grids(circ_orbit, cpu_gen):
    cpu = cpu_gen(-2, LRANGE)
    jx = solve_circular_grid(circ_orbit, -2, LRANGE, cpu_gen)
    return cpu, jx


def test_grid_amplitudes_eigenvalues(grids):
    cpu, jx = grids
    for key in ("In", "Up"):
        np.testing.assert_allclose(jx.amplitudes[key], cpu.amplitudes[key],
                                   rtol=1e-7)
    np.testing.assert_allclose(jx.eigenvalues, cpu.eigenvalues, rtol=1e-10)
    np.testing.assert_array_equal(jx.modes, cpu.modes)
    np.testing.assert_allclose(jx.frequencies, cpu.frequencies, rtol=1e-14)


def test_grid_coupling_and_harmonics(grids):
    cpu, jx = grids
    w = min(jx.coupling.shape[1], cpu.coupling.shape[1])
    np.testing.assert_allclose(jx.coupling[:, :w], cpu.coupling[:, :w],
                               atol=1e-12)
    for arr in (jx.coupling[:, w:], cpu.coupling[:, w:]):
        assert arr.size == 0 or np.max(np.abs(arr)) < 1e-12
    z = np.cos(np.linspace(0.3, 2.8, 7))
    np.testing.assert_allclose(jx.Sslm(z), cpu.Sslm(z), atol=1e-12)


def test_grid_radial_solutions(grids):
    cpu, jx = grids
    sig = np.linspace(0.25, 0.95, 9)
    for key in ("In", "Up"):
        got = jx.Rslm[key](sig)
        want = cpu.Rslm[key](sig)
        scale = np.max(np.abs(want))
        np.testing.assert_allclose(got, want, atol=1e-7 * scale)


def test_projection_vs_cpu_cast(cpu_gen, grids):
    cpu, jx = grids
    lmax = LRANGE[1]
    cpu_Y = cpu_gen.cast_to_spherical(cpu, lmax)

    # rebuild batched inputs from the grid: amplitude-scaled Up coeffs
    coeffs = np.moveaxis(np.asarray(cpu.Rslm["Up"].coeffs), -1, 0)  # (B, D, n)
    l_arr, m_arr = cpu.modes[:, 0], cpu.modes[:, 1]
    lmin_arr = np.maximum(2, np.abs(m_arr))
    proj, out_l, out_m = project_to_spherical(cpu.coupling, coeffs,
                                              m_arr, lmin_arr, l_arr)
    sig = np.linspace(0.25, 0.95, 6)
    got = np.asarray(evaluate_multidomain(proj, cpu.domains["Up"], sig))
    want = cpu_Y.Rslm["Up"](sig)              # (nsig, nmodes)
    scale = np.max(np.abs(want))
    for i, (L, M) in enumerate(zip(out_l, out_m)):
        k = cpu_Y.mode_index(int(L), int(M), 0, 0)
        np.testing.assert_allclose(got[i], want[:, k], atol=1e-10 * scale,
                                   err_msg=f"(L,m)=({L},{M})")


def test_swsh_theta_values(circ_orbit, grids):
    cpu, jx = grids
    th = np.linspace(0.3, 2.8, 6)
    # spot-check a few spheroidal modes against the per-mode CPU class
    for (l, m) in [(2, 2), (3, -1), (5, 4)]:
        i = cpu.mode_index(l, m, 0, 0)
        gamma = circ_orbit.a * cpu.frequencies[i]
        lmin = max(2, abs(m))
        vals = np.asarray(swsh_theta_values(
            -2, np.array([m]), np.array([lmin]),
            cpu.coupling[i][None, :], th))[0]
        ref = SpinWeightedSpheroidalHarmonic(-2, l, m, gamma)
        want = np.array([ref.eval(t) for t in th])
        np.testing.assert_allclose(vals, want, rtol=1e-9, atol=1e-12)


def test_assemble_field_vs_direct(grids):
    cpu, jx = grids
    B = 6
    rng = np.random.default_rng(3)
    amps = rng.normal(size=B) + 1j * rng.normal(size=B)
    radial = rng.normal(size=(B, 4)) + 1j * rng.normal(size=(B, 4))
    angular = rng.normal(size=(B, 3))
    m_arr = np.array([2, 2, -1, 3, 1, -2])
    omega = rng.normal(size=B)
    phi = np.array([0.0, 1.3])
    t = 2.7
    got = np.asarray(assemble_field(amps, radial, angular, m_arr, omega,
                                    phi, t=t))
    want = np.zeros((4, 3, 2), dtype=complex)
    for b in range(B):
        for si in range(4):
            for hi in range(3):
                for pi in range(2):
                    want[si, hi, pi] += (amps[b] * radial[b, si]
                                         * angular[b, hi]
                                         * np.exp(1j * (m_arr[b] * phi[pi]
                                                        - omega[b] * t)))
    np.testing.assert_allclose(got, want, rtol=1e-13)


def test_waveform_timeseries_vs_field(grids):
    cpu, jx = grids
    B = 5
    rng = np.random.default_rng(4)
    amps = rng.normal(size=B) + 1j * rng.normal(size=B)
    r0 = rng.normal(size=B) + 1j * rng.normal(size=B)
    angular = rng.normal(size=(B, 2))
    m_arr = np.array([1, 2, 3, -2, 2])
    omega = rng.normal(size=B)
    phi = np.array([0.7])
    t_arr = np.linspace(0, 10, 5)
    got = np.asarray(waveform_timeseries(amps, r0, angular, m_arr, omega,
                                         phi, t_arr))
    for k, t in enumerate(t_arr):
        want = np.asarray(assemble_field(amps, r0[:, None], angular, m_arr,
                                         omega, phi, t=t))[0]
        np.testing.assert_allclose(got[:, :, k], want, rtol=1e-12)


def test_mode_table_rejects_static(circ_orbit):
    with pytest.raises(ValueError, match="static"):
        solve_mode_table(circ_orbit, -2, [2, 2], [2, 0], [0, 0])
