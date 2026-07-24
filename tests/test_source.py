# Source integration and mode-grid regression tests. See dev/PHASE0.md.
import numpy as np
import pytest

from hblteuk import collocode
from hblteuk.mode import (
    PointParticleModeGrid,
    TeukolskyPointParticleModeGenerator,
    TeukolskyPointParticleModeGridGenerator,
)
from .conftest import SOLVER_KWARGS


@pytest.fixture(scope="module")
def mode_gen(orbit):
    solver = collocode.CollocationODEMultiDomainFixedStepSolver(n=32)
    return TeukolskyPointParticleModeGenerator(orbit, solver, SOLVER_KWARGS)


def test_eccentric_amplitudes_vs_golden(mode_gen, golden):
    for j, n in enumerate(golden["ecc_n"]):
        mode = mode_gen(-2, 2, 2, 0, int(n))
        got = np.array([mode.amplitudes["In"], mode.amplitudes["Up"]])
        # cross-platform LAPACK reproducibility floor ~1e-8 (golden was
        # generated on a different BLAS build)
        np.testing.assert_allclose(
            got, golden["ecc_amplitudes"][j], rtol=1e-7,
            err_msg=f"n={n}")


def test_amplitude_precision_estimates(mode_gen, golden):
    # error estimates should stay small for well-converged low-n modes
    for j, n in enumerate(golden["ecc_n"]):
        prec = golden["ecc_precisions"][j]
        assert np.all(prec < 1e-6), f"n={n}: {prec}"


def test_grid_vs_golden(orbit, golden):
    lmax = int(golden["grid_lmax"])
    solver = collocode.CollocationODEFixedMultiDomainFixedStepSolver(n=32)
    gen = TeukolskyPointParticleModeGridGenerator(
        orbit, solver=solver, solver_kwargs=dict(SOLVER_KWARGS))
    gen.optimize(-2, lmax)
    psi4 = gen(-2, [2, lmax])

    np.testing.assert_array_equal(psi4.modes, golden["grid_modes"])
    np.testing.assert_allclose(
        psi4.frequencies, golden["grid_frequencies"], rtol=1e-13)
    np.testing.assert_allclose(
        psi4.eigenvalues, golden["grid_eigenvalues"], rtol=1e-10)
    # amplitudes carry the cross-platform LAPACK floor (~1e-8); frequencies
    # and eigenvalues above are deterministic and stay tight
    np.testing.assert_allclose(
        psi4.amplitudes["In"], golden["grid_amp_in"], rtol=1e-7)
    np.testing.assert_allclose(
        psi4.amplitudes["Up"], golden["grid_amp_up"], rtol=1e-7)


def test_grid_conjugate_symmetry(golden):
    # m<0 amplitudes stored as (-1)^l conj(m>0): verify from golden arrays
    modes = golden["grid_modes"]
    amp = golden["grid_amp_up"]
    lmax = int(golden["grid_lmax"])
    grid = PointParticleModeGrid.__new__(PointParticleModeGrid)
    # reconstruct index map without a source: use stored modes array directly
    index = {tuple(mode): i for i, mode in enumerate(map(tuple, modes))}
    for (l, m, k, n), i in index.items():
        if m <= 0:
            continue
        j = index[(l, -m, k, n)]
        assert amp[j] == pytest.approx((-1) ** l * np.conj(amp[i]), rel=1e-12)


def test_mode_index_roundtrip(orbit):
    grid = PointParticleModeGrid(-2, [2, 6], [0, 0], [0, 0], orbit)
    seen = set()
    for i, (l, m, k, n) in enumerate(grid.modes):
        assert grid.mode_index(l, m, k, n) == i
        seen.add(i)
    assert seen == set(range(grid.mode_num))
