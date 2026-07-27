# End-to-end batched pipeline: mode table -> eigenvalues, homogeneous
# solutions, amplitudes; plus a circular-grid wrapper that reproduces the
# CPU TeukolskyPointParticleModeGridGenerator output. See dev/PHASE4.md.
from typing import NamedTuple

import jax
import numpy as np

from ..collocode import generate_subdomain_list
from ..teuk import sigma_r
from ..utils import chop_matrix
from .source import (equatorial_swsh_values, generic_samples,
                     integrate_source_batch, integrate_source_generic_batch,
                     integrate_source_spherical_batch, orbit_samples,
                     polar_samples, polar_swsh_values)
from .swsh import spheroidal_eigs_batched
from .teukolsky import HomogeneousBatch, solve_homogeneous_batch

# per-mode device working-set model (complex128). Constants calibrated
# 2026-07-16 via dev/calibrate_chunk_model.py (measured K_SRC=38.3,
# K_RAD=5.3; shipped at x1.5 headroom -- dev/MEMORY_CHUNKING_DESIGN.md,
# Results): _K_SRC counts live (B, nsamples) arrays in the source stage,
# _K_RAD live (B, ndomains, nnodes) arrays in the radial stage.
_K_SRC = 58
_K_RAD = 8


def _per_mode_bytes(nsamples, ndomains, nnodes):
    return (_K_SRC * nsamples + _K_RAD * ndomains * nnodes) * 16


def _chunk_size(max_memory_gb, nsamples, ndomains, nnodes):
    """Modes per chunk under the budget; None disables chunking."""
    if max_memory_gb is None:
        return None
    return max(1, int(max_memory_gb * 2**30
                      // _per_mode_bytes(nsamples, ndomains, nnodes)))


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
                     method="fft", max_memory_gb=8.0, return_radial=True,
                     stabilize=False):
    """Solve a radiative (s = -2, equatorial) mode table end to end.

    l, m, n: (B,) integer arrays; every mode must be radiative (omega != 0).
    subdomains: optional (subs_up, subs_in) pair (e.g. from the CPU grid
    generator's optimize()); defaults to `spacing`-spaced structures.
    max_memory_gb bounds the device working set by solving the table in
    equal-shape mode chunks -- mathematically identical results to the
    unchunked path (rtol ~1e-8; JAX's batched linear algebra is not
    bit-reproducible across batch sizes, dev/MEMORY_CHUNKING_DESIGN.md);
    None disables chunking.
    return_radial=False drops the homogeneous-solution coefficients
    (hom=None) to bound output memory in flux-only workflows.
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

    # table-global spheroidal truncation (same formula as the
    # spheroidal_eigs_batched default): pinned so chunking cannot change
    # per-mode results or coupling widths across chunk boundaries
    g = a * omega
    lmin_all = np.maximum(abs(s), np.abs(m))
    nmax = int(np.max((l - lmin_all) + np.round(20 + 2 * np.abs(g)) + 2))

    samples = orbit_samples(orbit, nsamples=nsamples)

    def solve_chunk(idx):
        lam, coupling, lmin = spheroidal_eigs_batched(s, l[idx], m[idx],
                                                      g[idx], nmax=nmax)
        lam = np.asarray(lam)
        hom = solve_homogeneous_batch(s, kappa, lam, m[idx] * a, omega[idx],
                                      subs_up, subs_in, n=nnodes,
                                      stabilize=stabilize)
        S, dS, d2S = equatorial_swsh_values(s, m[idx], lmin,
                                            np.asarray(coupling), lam, g[idx])
        src = integrate_source_batch(samples, lam, m[idx], n[idx],
                                     omega[idx], S, dS, d2S,
                                     hom, subs_up, subs_in, method=method)
        return lam, np.asarray(coupling), lmin, hom, src

    B = len(l)
    csize = _chunk_size(max_memory_gb, nsamples, len(subs_up) - 1, nnodes)
    if csize is None or csize >= B:
        lam, coupling, lmin, hom, src = solve_chunk(np.arange(B))
        amplitudes, errors = src.amplitudes, src.errors
        if not return_radial:
            hom = None
    else:
        lams, coups, lmins, homs, amps, errs = [], [], [], [], [], []
        for start in range(0, B, csize):
            idx = np.arange(start, min(start + csize, B))
            keep = len(idx)
            if keep < csize:  # pad so every chunk reuses one compiled trace
                idx = np.concatenate([idx, np.full(csize - keep, idx[-1])])
            lam_c, coup_c, lmin_c, hom_c, src_c = solve_chunk(idx)
            lams.append(lam_c[:keep])
            coups.append(coup_c[:keep])
            lmins.append(lmin_c[:keep])
            amps.append(np.asarray(src_c.amplitudes)[:keep])
            errs.append(np.asarray(src_c.errors)[:keep])
            if return_radial:
                homs.append(HomogeneousBatch(*(np.asarray(x)[:keep]
                                               for x in hom_c)))
        lam = np.concatenate(lams)
        coupling = np.concatenate(coups)
        lmin = np.concatenate(lmins)
        amplitudes = np.concatenate(amps)
        errors = np.concatenate(errs)
        hom = (HomogeneousBatch(*(np.concatenate([h[i] for h in homs])
                                  for i in range(len(HomogeneousBatch._fields))))
               if return_radial else None)

    return ModeTableSolution(l, m, n, omega, lam, np.asarray(coupling), lmin,
                             hom, np.asarray(subs_up), np.asarray(subs_in),
                             amplitudes, errors)


def _negative_m_coupling(coupling, l, lmin):
    # CPU convention (mode.py solve_teukolsky_mode_data): m -> -m flips the
    # parity of alternating coefficients, plus a global sign for even l+lmin
    out = coupling.copy()
    out[::2] *= -1
    if np.abs(l + lmin) % 2 == 0:
        out *= -1
    return out


class SphericalModeSolution(NamedTuple):
    l: np.ndarray            # (B,)
    m: np.ndarray
    k: np.ndarray
    omega: np.ndarray
    eigenvalues: np.ndarray
    coupling: np.ndarray     # (B, nterms) spherical-expansion coefficients
    lmin: np.ndarray
    hom: object              # HomogeneousBatch
    subdomains_up: np.ndarray
    subdomains_in: np.ndarray
    amplitudes: jax.Array    # (B, 2) [In, Up]
    errors: jax.Array        # (B, 2)


def solve_spherical_mode_table(orbit, s, l, m, k, subdomains=None,
                               ndomains=150, nnodes=32, spacing="arcsinh7",
                               nsamples=2**8, method="fft", stabilize=False):
    """Solve a radiative (s = -2) spherical-inclined mode table end to end.

    orbit: MiniGeo with e = 0 (any |x| <= 1). l, m, k: (B,) integers;
    omega = m*Omega_phi + k*Omega_theta must be nonzero for every mode.
    The radial stage collapses to the single orbital radius; the source is
    the k-th Fourier coefficient over the polar period (dev/PHASE5B.md).
    """
    if orbit.e != 0:
        raise ValueError("solve_spherical_mode_table requires e = 0; "
                         "eccentric-equatorial tables use solve_mode_table")
    l = np.asarray(l, dtype=np.int64)
    m = np.asarray(m, dtype=np.int64)
    k = np.asarray(k, dtype=np.int64)
    a = orbit.a
    kappa = float(np.sqrt(1 - a**2))
    omega = np.array([orbit.mode_frequency(int(mm), int(kk), 0)
                      for mm, kk in zip(m, k)])
    if np.any(omega == 0):
        raise ValueError("static (omega = 0) modes must go through the CPU "
                         "path; drop them from the table")

    sp = sigma_r(float(orbit.radial_roots[0]), kappa)
    if subdomains is None:
        subs_up = generate_subdomain_list([0, sp], ndomains, spacing)
        subs_in = generate_subdomain_list([1, sp], ndomains, spacing)
    else:
        subs_up, subs_in = subdomains

    lam, coupling, lmin = spheroidal_eigs_batched(s, l, m, a * omega)
    lam = np.asarray(lam)
    hom = solve_homogeneous_batch(s, kappa, lam, m * a, omega,
                                  subs_up, subs_in, n=nnodes,
                                  stabilize=stabilize)
    samples = polar_samples(orbit, nsamples=nsamples)
    S, dS, d2S = polar_swsh_values(s, m, lmin, np.asarray(coupling), lam,
                                   a * omega, samples.thp)
    src = integrate_source_spherical_batch(samples, lam, m, k, omega,
                                           S, dS, d2S, hom, subs_up, subs_in,
                                           method=method)
    return SphericalModeSolution(l, m, k, omega, lam, np.asarray(coupling),
                                 lmin, hom, np.asarray(subs_up),
                                 np.asarray(subs_in),
                                 src.amplitudes, src.errors)


def solve_circular_grid(orbit, s, lrange, cpu_grid_generator,
                        max_memory_gb=8.0):
    """Batched replacement for TeukolskyPointParticleModeGridGenerator.solve
    on a circular orbit (n = k = 0 grid).

    cpu_grid_generator: an optimized CPU generator; supplies the frozen
    subdomain structures and solves the static m = 0 column (omega = 0),
    which stays on the CPU path by design. Returns a PointParticleModeGrid
    wired identically to the CPU version (amplitude-scaled radial grids,
    chopped coupling, MultiModeSpinWeightedSpherical). max_memory_gb is
    forwarded to solve_mode_table; the grid always retains the radial
    coefficients (return_radial is not exposed here).
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
                           subdomains=(subs_up, subs_in),
                           max_memory_gb=max_memory_gb)
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


class GenericModeSolution(NamedTuple):
    l: np.ndarray            # (B,)
    m: np.ndarray
    k: np.ndarray
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


def solve_generic_mode_table(orbit, s, l, m, k, n, subdomains=None,
                             ndomains=150, nnodes=32, spacing="arcsinh7",
                             nsamples=2**10, ntheta=2**8, method="fft",
                             max_memory_gb=8.0, return_radial=True,
                             stabilize=False):
    """Solve a radiative (s = -2) generic-orbit mode table end to end.

    orbit: MiniGeo with e != 0 and |x| < 1. l, m, k, n: (B,) integers;
    omega = m*Omega_phi + k*Omega_theta + n*Omega_r must be nonzero.
    2D torus source contracted over q_theta (dev/PHASE5C.md); working set
    O(B*nq) so the chunking cost model of solve_mode_table applies with
    nsamples = nq (ntheta does not enter). max_memory_gb / return_radial
    as in solve_mode_table.
    """
    l = np.asarray(l, dtype=np.int64)
    m = np.asarray(m, dtype=np.int64)
    k = np.asarray(k, dtype=np.int64)
    n = np.asarray(n, dtype=np.int64)
    a = orbit.a
    kappa = float(np.sqrt(1 - a**2))
    omega = np.array([orbit.mode_frequency(int(mm), int(kk), int(nn))
                      for mm, kk, nn in zip(m, k, n)])
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

    # table-global spheroidal truncation, pinned across chunks (see
    # solve_mode_table)
    g = a * omega
    lmin_all = np.maximum(abs(s), np.abs(m))
    nmax = int(np.max((l - lmin_all) + np.round(20 + 2 * np.abs(g)) + 2))

    samples = generic_samples(orbit, nsamples=nsamples, ntheta=ntheta)

    def solve_chunk(idx):
        lam, coupling, lmin = spheroidal_eigs_batched(s, l[idx], m[idx],
                                                      g[idx], nmax=nmax)
        lam = np.asarray(lam)
        hom = solve_homogeneous_batch(s, kappa, lam, m[idx] * a, omega[idx],
                                      subs_up, subs_in, n=nnodes,
                                      stabilize=stabilize)
        S, dS, d2S = polar_swsh_values(s, m[idx], lmin,
                                       np.asarray(coupling), lam, g[idx],
                                       samples.thp)
        src = integrate_source_generic_batch(samples, lam, m[idx], k[idx],
                                             n[idx], omega[idx], S, dS, d2S,
                                             hom, subs_up, subs_in,
                                             method=method)
        return lam, np.asarray(coupling), lmin, hom, src

    B = len(l)
    csize = _chunk_size(max_memory_gb, nsamples, len(subs_up) - 1, nnodes)
    if csize is None or csize >= B:
        lam, coupling, lmin, hom, src = solve_chunk(np.arange(B))
        amplitudes, errors = src.amplitudes, src.errors
        if not return_radial:
            hom = None
    else:
        lams, coups, lmins, homs, amps, errs = [], [], [], [], [], []
        for start in range(0, B, csize):
            idx = np.arange(start, min(start + csize, B))
            keep = len(idx)
            if keep < csize:  # pad so every chunk reuses one compiled trace
                idx = np.concatenate([idx, np.full(csize - keep, idx[-1])])
            lam_c, coup_c, lmin_c, hom_c, src_c = solve_chunk(idx)
            lams.append(lam_c[:keep])
            coups.append(coup_c[:keep])
            lmins.append(lmin_c[:keep])
            amps.append(np.asarray(src_c.amplitudes)[:keep])
            errs.append(np.asarray(src_c.errors)[:keep])
            if return_radial:
                homs.append(HomogeneousBatch(*(np.asarray(x)[:keep]
                                               for x in hom_c)))
        lam = np.concatenate(lams)
        coupling = np.concatenate(coups)
        lmin = np.concatenate(lmins)
        amplitudes = np.concatenate(amps)
        errors = np.concatenate(errs)
        hom = (HomogeneousBatch(*(np.concatenate([h[i] for h in homs])
                                  for i in range(len(HomogeneousBatch._fields))))
               if return_radial else None)

    return GenericModeSolution(l, m, k, n, omega, lam, np.asarray(coupling),
                               lmin, hom, np.asarray(subs_up),
                               np.asarray(subs_in), amplitudes, errors)
