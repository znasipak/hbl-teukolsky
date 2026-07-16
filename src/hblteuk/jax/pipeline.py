# End-to-end batched pipeline: mode table -> eigenvalues, homogeneous
# solutions, amplitudes; plus a circular-grid wrapper that reproduces the
# CPU TeukolskyPointParticleModeGridGenerator output. See dev/PHASE4.md.
from typing import NamedTuple

import jax
import numpy as np

from ..collocode import generate_subdomain_list
from ..teuk import sigma_r
from ..utils import chop_matrix
from .source import equatorial_swsh_values, integrate_source_batch, orbit_samples
from .swsh import spheroidal_eigs_batched
from .teukolsky import solve_homogeneous_batch


class ModeTableSolution(NamedTuple):
    l: np.ndarray            # (B,)
    m: np.ndarray
    n: np.ndarray
    omega: np.ndarray
    eigenvalues: np.ndarray
    coupling: np.ndarray     # (B, nterms) spherical-expansion coefficients
    lmin: np.ndarray
    hom: object              # HomogeneousBatch
    subdomains_up: np.ndarray
    subdomains_in: np.ndarray
    amplitudes: jax.Array    # (B, 2) [In, Up]
    errors: jax.Array        # (B, 2)


def solve_mode_table(orbit, s, l, m, n, subdomains=None, ndomains=150,
                     nnodes=32, spacing="arcsinh7", nsamples=2**10,
                     method="fft"):
    """Solve a radiative (s = -2, equatorial) mode table end to end.

    l, m, n: (B,) integer arrays; every mode must be radiative (omega != 0).
    subdomains: optional (subs_up, subs_in) pair (e.g. from the CPU grid
    generator's optimize()); defaults to `spacing`-spaced structures.
    """
    l = np.asarray(l, dtype=np.int64)
    m = np.asarray(m, dtype=np.int64)
    n = np.asarray(n, dtype=np.int64)
    a = orbit.a
    kappa = float(np.sqrt(1 - a**2))
    omega = np.array([orbit.mode_frequency(int(mm), 0, int(nn))
                      for mm, nn in zip(m, n)])
    if np.any(omega == 0):
        raise ValueError("static (omega = 0) modes must go through the CPU "
                         "path; drop them from the table")

    if subdomains is None:
        smin = sigma_r(orbit.radial_roots[0], kappa)
        smax = sigma_r(orbit.radial_roots[1], kappa)
        subs_up = generate_subdomain_list([0, smax], ndomains, spacing)
        subs_in = generate_subdomain_list([1, smin], ndomains, spacing)
    else:
        subs_up, subs_in = subdomains

    lam, coupling, lmin = spheroidal_eigs_batched(s, l, m, a * omega)
    lam = np.asarray(lam)
    hom = solve_homogeneous_batch(s, kappa, lam, m * a, omega,
                                  subs_up, subs_in, n=nnodes)
    samples = orbit_samples(orbit, nsamples=nsamples)
    S, dS, d2S = equatorial_swsh_values(s, m, lmin, np.asarray(coupling),
                                        lam, a * omega)
    src = integrate_source_batch(samples, lam, m, n, omega, S, dS, d2S,
                                 hom, subs_up, subs_in, method=method)
    return ModeTableSolution(l, m, n, omega, lam, np.asarray(coupling), lmin,
                             hom, np.asarray(subs_up), np.asarray(subs_in),
                             src.amplitudes, src.errors)


def _negative_m_coupling(coupling, l, lmin):
    # CPU convention (mode.py solve_teukolsky_mode_data): m -> -m flips the
    # parity of alternating coefficients, plus a global sign for even l+lmin
    out = coupling.copy()
    out[::2] *= -1
    if np.abs(l + lmin) % 2 == 0:
        out *= -1
    return out


def solve_circular_grid(orbit, s, lrange, cpu_grid_generator):
    """Batched replacement for TeukolskyPointParticleModeGridGenerator.solve
    on a circular orbit (n = k = 0 grid).

    cpu_grid_generator: an optimized CPU generator; supplies the frozen
    subdomain structures and solves the static m = 0 column (omega = 0),
    which stays on the CPU path by design. Returns a PointParticleModeGrid
    wired identically to the CPU version (amplitude-scaled radial grids,
    chopped coupling, MultiModeSpinWeightedSpherical).
    """
    from ..cheby import MultiGridMultiDomainChebyshev
    from ..mode import PointParticleModeGrid
    from ..swsh import MultiModeSpinWeightedSpherical

    psi4 = PointParticleModeGrid(s, lrange, [0, 0], [0, 0], orbit)
    lmax = psi4.lmax
    subs = cpu_grid_generator.solver_kwargs.get("subdomains")
    if not isinstance(subs, dict):
        raise ValueError("run cpu_grid_generator.optimize(s, lmax) first so "
                         "the grid shares one frozen subdomain structure")
    subs_up, subs_in = np.asarray(subs["Up"]), np.asarray(subs["In"])

    # radiative modes: m > 0 solved batched; m < 0 filled by conjugation
    modes_pos = [(l, m) for l in psi4.larray for m in range(1, l + 1)]
    l_arr = np.array([c[0] for c in modes_pos])
    m_arr = np.array([c[1] for c in modes_pos])
    sol = solve_mode_table(orbit, s, l_arr, m_arr, np.zeros_like(l_arr),
                           subdomains=(subs_up, subs_in))
    amps = np.asarray(sol.amplitudes)

    nnodes = sol.hom.coeffs_up.shape[-1]
    ndom_up, ndom_in = len(subs_up) - 1, len(subs_in) - 1
    Rslm = {"In": np.empty((psi4.mode_num, ndom_in, nnodes), np.complex128),
            "Up": np.empty((psi4.mode_num, ndom_up, nnodes), np.complex128)}
    amplitudes = {"In": np.empty(psi4.mode_num, np.complex128),
                  "Up": np.empty(psi4.mode_num, np.complex128)}
    SslmGrid = np.zeros((psi4.mode_num, lmax + 100))
    eigenvalues = np.empty(psi4.mode_num)

    cu = np.asarray(sol.hom.coeffs_up)
    ci = np.asarray(sol.hom.coeffs_in)
    for b, (l, m) in enumerate(modes_pos):
        lmin = max(abs(s), abs(m))
        coup = sol.coupling[b][:lmax + 100 - lmin]
        i = psi4.mode_index(l, m, 0, 0)
        Rslm["Up"][i], Rslm["In"][i] = cu[b], ci[b]
        amplitudes["In"][i], amplitudes["Up"][i] = amps[b]
        SslmGrid[i][:len(coup)] = coup
        eigenvalues[i] = sol.eigenvalues[b]

        j = psi4.mode_index(l, -m, 0, 0)
        Rslm["Up"][j], Rslm["In"][j] = np.conj(cu[b]), np.conj(ci[b])
        amplitudes["In"][j] = (-1)**l * np.conj(amps[b, 0])
        amplitudes["Up"][j] = (-1)**l * np.conj(amps[b, 1])
        lmin_c = max(2, abs(m))
        SslmGrid[j][:len(coup)] = _negative_m_coupling(coup, l, lmin_c)
        eigenvalues[j] = sol.eigenvalues[b]

    # static m = 0 column: CPU path (different ODE and scaling conventions)
    for l in psi4.larray:
        mode = cpu_grid_generator.generate_mode(
            s, int(l), 0, 0, 0, cpu_grid_generator.teuk_solver,
            cpu_grid_generator.integrate_kwargs)
        i = psi4.mode_index(int(l), 0, 0, 0)
        Rslm["In"][i] = mode.Rslm.psi["In"].coeffs
        Rslm["Up"][i] = mode.Rslm.psi["Up"].coeffs
        amplitudes["In"][i] = mode.amplitudes["In"]
        amplitudes["Up"][i] = mode.amplitudes["Up"]
        coup = mode.Sslm.coefficients
        SslmGrid[i][:len(coup)] = coup
        eigenvalues[i] = mode.eigenvalue

    # final wiring mirrors the CPU solve(): amplitude-scaled radial grids,
    # chopped coupling, per-m spherical-expansion container
    psi4.Rslm = {
        key: MultiGridMultiDomainChebyshev(
            np.ascontiguousarray(amplitudes[key][:, None, None] * Rslm[key]),
            subs_k)
        for key, subs_k in [("In", subs_in), ("Up", subs_up)]}
    psi4.coupling = chop_matrix(SslmGrid, buffer=lmax - 2, tol=1e-50)
    coeffs_arr = psi4.group_by_m_modes(psi4.coupling)
    gamma_arr = psi4.group_by_m_modes(orbit.a * psi4.frequencies)
    psi4.Sslm = MultiModeSpinWeightedSpherical(s, psi4.m_mode_arr, gamma_arr,
                                               coeffs_arr, tol=None)
    psi4.eigenvalues = eigenvalues
    psi4.amplitudes = amplitudes
    psi4.domains["In"], psi4.domains["Up"] = subs_in, subs_up
    return psi4
