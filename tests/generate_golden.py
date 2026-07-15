# Generate golden regression data for Phase 0 (see dev/PHASE0.md).
# Usage: python tests/generate_golden.py
# Writes tests/golden/golden_v1.npz. Regenerate ONLY on intentional physics
# changes; commit the result.
import pathlib

import numpy as np

from hblteuk import collocode
from hblteuk.geo import MiniGeo
from hblteuk.mode import (
    TeukolskyPointParticleModeGenerator,
    TeukolskyPointParticleModeGridGenerator,
)
from hblteuk.swsh import SpinWeightedSpheroidalHarmonic
from hblteuk.teuk import HyperboloidalTeukolskySolver, sigma_r

OUT = pathlib.Path(__file__).parent / "golden" / "golden_v1.npz"

ORBIT = (0.9, 10.0, 0.2, 1.0)  # a, p, e, x
SOLVER_KWARGS = {"subdomains": 150, "tol": 1e-13, "spacing": "arcsinh7"}

SWSH_CASES = [(-2, 2, 2, 0.0), (-2, 2, 2, 0.27), (-2, 5, -3, 1.35),
              (-2, 10, 7, 2.7), (0, 3, 1, 0.5), (2, 2, 2, 0.27)]

HOM_CASES = [(-2, 2, 2, 0), (-2, 3, 1, 0), (-2, 10, 7, 0), (-2, 2, 2, 3)]

ECC_N = range(-3, 4)  # amplitudes for (s,l,m,n)=(-2,2,2,n)
GRID_LMAX = 5


def swsh_block(data):
    for i, (s, l, m, g) in enumerate(SWSH_CASES):
        S = SpinWeightedSpheroidalHarmonic(s, l, m, g)
        th = np.array([0.3, 0.5 * np.pi, 2.2])
        data[f"swsh{i}_params"] = np.array([s, l, m, g])
        data[f"swsh{i}_eigenvalue"] = S.eigenvalue
        data[f"swsh{i}_coeffs"] = S.coeffs
        data[f"swsh{i}_S"] = np.array([S.eval(t) for t in th])
        data[f"swsh{i}_dS"] = np.array([S.deriv(t) for t in th])


def homogeneous_block(data, orbit):
    a = ORBIT[0]
    kappa = np.sqrt(1 - a**2)
    rmin, rmax = orbit.radial_roots[1], orbit.radial_roots[0]
    smin, smax = sigma_r(rmax, kappa), sigma_r(rmin, kappa)
    solver = collocode.CollocationODEMultiDomainFixedStepSolver(n=32)
    teuk_solver = HyperboloidalTeukolskySolver(
        domains=[[0, smax], [smin, 1]], solver=solver, **SOLVER_KWARGS)

    for i, (s, l, m, n) in enumerate(HOM_CASES):
        omega = orbit.mode_frequency(m, 0, n)
        hbl = teuk_solver(a, s, l, m, omega)
        # evaluate on interior points of each domain: representation-independent
        sig_up = np.linspace(0.05 * smax, 0.95 * smax, 11)
        sig_in = np.linspace(smin + 0.05 * (1 - smin), 1 - 0.05 * (1 - smin), 11)
        data[f"hom{i}_params"] = np.array([s, l, m, n, omega])
        data[f"hom{i}_eigenvalue"] = hbl.eigenvalue
        data[f"hom{i}_sig_up"] = sig_up
        data[f"hom{i}_sig_in"] = sig_in
        for d in (0, 1, 2):
            data[f"hom{i}_psi_up_d{d}"] = hbl.psi["Up"](sig_up, deriv=d)
            data[f"hom{i}_psi_in_d{d}"] = hbl.psi["In"](sig_in, deriv=d)


def eccentric_block(data, orbit):
    solver = collocode.CollocationODEMultiDomainFixedStepSolver(n=32)
    gen = TeukolskyPointParticleModeGenerator(orbit, solver, SOLVER_KWARGS)
    amps, precs = [], []
    for n in ECC_N:
        mode = gen(-2, 2, 2, 0, n)
        amps.append([mode.amplitudes["In"], mode.amplitudes["Up"]])
        precs.append([mode.precisions["In"], mode.precisions["Up"]])
    data["ecc_n"] = np.array(list(ECC_N))
    data["ecc_amplitudes"] = np.array(amps)
    data["ecc_precisions"] = np.array(precs)


def grid_block(data, orbit):
    solver = collocode.CollocationODEFixedMultiDomainFixedStepSolver(n=32)
    gen = TeukolskyPointParticleModeGridGenerator(
        orbit, solver=solver, solver_kwargs=SOLVER_KWARGS)
    gen.optimize(-2, GRID_LMAX)
    psi4 = gen(-2, [2, GRID_LMAX])
    data["grid_lmax"] = np.array(GRID_LMAX)
    data["grid_modes"] = psi4.modes
    data["grid_frequencies"] = psi4.frequencies
    data["grid_eigenvalues"] = psi4.eigenvalues
    data["grid_amp_in"] = psi4.amplitudes["In"]
    data["grid_amp_up"] = psi4.amplitudes["Up"]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    orbit = MiniGeo(*ORBIT)
    data = {"orbit_params": np.array(ORBIT)}
    swsh_block(data)
    homogeneous_block(data, orbit)
    eccentric_block(data, orbit)
    grid_block(data, orbit)
    np.savez_compressed(OUT, **data)
    size = OUT.stat().st_size / 1024
    print(f"wrote {OUT} ({size:.0f} kB, {len(data)} arrays)")


if __name__ == "__main__":
    main()
