# Unified solver dispatch, canonical-half flux summation, and the staged
# adaptive total-flux driver. Layered strictly on the per-geometry table
# solvers and flux_modes -- no new physics. Design: dev/PHASE5D_DESIGN.md;
# writeup: dev/PHASE5D.md.
import warnings
from typing import NamedTuple

import numpy as np

from .fluxes import flux_modes
from .pipeline import (GenericModeSolution, solve_generic_mode_table,
                       solve_mode_table, solve_spherical_mode_table)


def _as_generic(sol, k, n):
    # repack a 1D-path solution into the GenericModeSolution shape
    return GenericModeSolution(sol.l, sol.m, np.asarray(k, np.int64),
                               np.asarray(n, np.int64), sol.omega,
                               sol.eigenvalues, sol.coupling, sol.lmin,
                               sol.hom, sol.subdomains_up, sol.subdomains_in,
                               sol.amplitudes, sol.errors)


def solve_modes(orbit, s, l, m, k, n, **solver_kwargs):
    """Dispatch a (l, m, k, n) mode table to the cheapest valid solver.

    equatorial (|x| = 1): requires k == 0, radial 1D path (uses `nsamples`);
    spherical (e = 0, |x| < 1): requires n == 0, polar 1D path (uses
    `ntheta` as its sampling density); generic: 2D path (uses both).
    Returns a GenericModeSolution regardless of path (missing index
    zero-filled). Static (omega = 0) rows raise, as in every solver.
    """
    l = np.asarray(l, np.int64)
    m = np.asarray(m, np.int64)
    k = np.asarray(k, np.int64)
    n = np.asarray(n, np.int64)
    equatorial = abs(orbit.x) == 1
    spherical = orbit.e == 0
    if equatorial and np.any(k != 0):
        raise ValueError("equatorial orbit has no k modes; nonzero k at "
                         f"rows {np.nonzero(k != 0)[0].tolist()}")
    if spherical and not equatorial and np.any(n != 0):
        raise ValueError("spherical orbit has no n modes; nonzero n at "
                         f"rows {np.nonzero(n != 0)[0].tolist()}")
    if equatorial:
        solver_kwargs.pop("ntheta", None)
        sol = solve_mode_table(orbit, s, l, m, n, **solver_kwargs)
        return _as_generic(sol, np.zeros_like(l), n)
    if spherical:
        ntheta = solver_kwargs.pop("ntheta", None)
        solver_kwargs.pop("nsamples", None)
        if ntheta is not None:
            solver_kwargs["nsamples"] = ntheta
        # solve_spherical_mode_table predates chunking (no n-axis -> smallest
        # tables); it has no max_memory_gb / return_radial parameters
        solver_kwargs.pop("max_memory_gb", None)
        solver_kwargs.pop("return_radial", None)
        sol = solve_spherical_mode_table(orbit, s, l, m, k, **solver_kwargs)
        return _as_generic(sol, k, np.zeros_like(l))
    return solve_generic_mode_table(orbit, s, l, m, k, n, **solver_kwargs)


class FluxSummary(NamedTuple):
    Edot: np.ndarray          # (2,) [infinity, horizon], doubled totals
    Ldot: np.ndarray
    Qdot: np.ndarray
    error_bound: np.ndarray   # (3, 2) channels (E, L, Q) x [inf, hor]
    edge_report: dict         # axis -> (3,) per-channel edge fraction
    converged: object         # None, or dict channel -> bool
    table: dict               # per-mode indices + fluxes actually summed
    rounds: tuple = ()        # driver-only: (window, n_new, totals) log


_CHANNELS = ("Edot", "Ldot", "Qdot")


def sum_fluxes(sol, orbit, tol=None):
    """Doubled flux totals + truncation diagnostics for a canonical-half
    mode table (m > 0, or m = 0 & k > 0, or m = 0 & k = 0 & n > 0; each
    row counts twice via the (m,k,n) -> (-m,-k,-n) conjugation symmetry).

    Returns FluxSummary; edge_report[axis] is the per-channel fraction of
    |total| carried by the window-boundary cells of that axis (only axes
    with spread appear); error_bound[channel] = [inf, hor] absolute bounds
    from the per-mode amplitude error estimates (flux err ~ 2x amp err).
    """
    l, m, k, n = (np.asarray(x) for x in (sol.l, sol.m, sol.k, sol.n))
    canonical = (m > 0) | ((m == 0) & (k > 0)) | ((m == 0) & (k == 0) & (n > 0))
    if not np.all(canonical):
        raise ValueError("rows outside the canonical half-space at "
                         f"{np.nonzero(~canonical)[0].tolist()[:10]}")

    fx = flux_modes(sol.amplitudes, sol.m, sol.omega, sol.eigenvalues,
                    orbit.a, k=sol.k, n=sol.n,
                    carter_frequencies=orbit.carter_frequencies())
    F = {"Edot": 2 * np.asarray(fx.Edot), "Ldot": 2 * np.asarray(fx.Ldot),
         "Qdot": 2 * np.asarray(fx.Qdot)}
    totals = {c: F[c].sum(axis=0) for c in _CHANNELS}

    err = np.asarray(sol.errors)          # (B, 2) [In, Up] amplitude rel err
    error_bound = np.array(
        [[np.sum(2 * err[:, 1] * np.abs(F[c][:, 0])),      # infinity <- Up
          np.sum(2 * err[:, 0] * np.abs(F[c][:, 1]))]      # horizon <- In
         for c in _CHANNELS])

    edge_report = {}
    for axis, v in (("l", l), ("k", k), ("n", n)):
        if v.max() == v.min():
            continue                      # degenerate axis: no window edge
        # l is summed from |s| upward and only truncated at the top, so its
        # edge is the highest-l slab alone; k, n are centered on 0 and
        # truncated at both ends, so both extremes count
        if axis == "l":
            edge = v == v.max()
        else:
            edge = (v == v.max()) | (v == v.min())
        edge_report[axis] = np.array(
            [np.abs(F[c][edge]).sum() / max(np.abs(F[c]).sum(), 1e-300)
             for c in _CHANNELS])

    converged = None
    if tol is not None:
        converged = {c: bool(all(edge_report[a][i] < tol
                                 for a in edge_report))
                     for i, c in enumerate(_CHANNELS)}

    table = dict(l=l, m=m, k=k, n=n, omega=np.asarray(sol.omega),
                 Edot=F["Edot"], Ldot=F["Ldot"], Qdot=F["Qdot"], errors=err)
    return FluxSummary(totals["Edot"], totals["Ldot"], totals["Qdot"],
                       error_bound, edge_report, converged, table)


def _next_pow2(x):
    return 1 << max(0, int(np.ceil(np.log2(max(x, 1)))))


def _window_modes(lmax, k_half, n_half):
    # canonical half of the rectangle l in [2, lmax], m in [0, l]
    return [(l, m, k, n) for l in range(2, lmax + 1)
            for m in range(0, l + 1)
            for k in range(-k_half, k_half + 1)
            for n in range(-n_half, n_half + 1)
            if (m > 0) or (m == 0 and k > 0)
            or (m == 0 and k == 0 and n > 0)]


class _Accum:
    # accumulated mode set across rounds; the minimal solution-shaped view
    # sum_fluxes needs (l/m/k/n/omega/eigenvalues/amplitudes/errors)
    def __init__(self):
        self.l = np.empty(0, np.int64); self.m = np.empty(0, np.int64)
        self.k = np.empty(0, np.int64); self.n = np.empty(0, np.int64)
        self.omega = np.empty(0); self.eigenvalues = np.empty(0)
        self.amplitudes = np.empty((0, 2), np.complex128)
        self.errors = np.empty((0, 2))
        self.solved = set()

    def merge(self, sol):
        self.l = np.concatenate([self.l, sol.l])
        self.m = np.concatenate([self.m, sol.m])
        self.k = np.concatenate([self.k, sol.k])
        self.n = np.concatenate([self.n, sol.n])
        self.omega = np.concatenate([self.omega, sol.omega])
        self.eigenvalues = np.concatenate([self.eigenvalues, sol.eigenvalues])
        self.amplitudes = np.concatenate(
            [self.amplitudes, np.asarray(sol.amplitudes)])
        self.errors = np.concatenate([self.errors, np.asarray(sol.errors)])
        self.solved.update(zip(sol.l.tolist(), sol.m.tolist(),
                               sol.k.tolist(), sol.n.tolist()))


def total_fluxes(orbit, s=-2, tol=1e-6, lmax_cap=30, max_memory_gb=8.0,
                 verbose=False):
    """Adaptive staged-round flux driver: grows a rectangular canonical
    (l, m, k, n) window in batched rounds until every axis' edge
    contribution < tol x running total per channel (l axis: last two
    slabs, 2-consecutive), or lmax_cap (warns, converged=False).
    """
    equatorial = abs(orbit.x) == 1
    circular = orbit.e == 0
    k_half = 0 if equatorial else max(2, int(np.ceil(4 * (1 - abs(orbit.x)))) + 2)
    n_half = 0 if circular else max(3, int(np.ceil(10 * orbit.e / (1 - orbit.e))))
    lmax = 4
    acc = _Accum()
    rounds = []

    while True:
        new = [md for md in _window_modes(lmax, k_half, n_half)
               if md not in acc.solved]
        if new:
            arr = np.array(new)
            sol = solve_modes(
                orbit, s, arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3],
                nsamples=max(2**9, _next_pow2(4 * n_half)),
                ntheta=max(2**7, _next_pow2(4 * k_half)),
                max_memory_gb=max_memory_gb, return_radial=False)
            acc.merge(sol)
        elif not rounds:
            raise RuntimeError("empty initial window")
        summary = sum_fluxes(acc, orbit, tol=tol)
        rounds.append(dict(window=(lmax, k_half, n_half), n_new=len(new),
                           totals={c: getattr(summary, c) for c in _CHANNELS}))
        if verbose:
            print(f"round {len(rounds)}: window l<={lmax}, |k|<={k_half}, "
                  f"|n|<={n_half}, +{len(new)} modes, "
                  f"Edot_inf={summary.Edot[0]:.6e}")

        # l stop: last TWO l-slabs each below tol x total, every channel
        # (summary.table[c] are the doubled per-mode fluxes, shape (B, 2);
        # magnitudes over both flux sides combined in the criterion)
        def slab_frac(lv):
            sel = acc.l == lv
            return max(np.abs(summary.table[c][sel]).sum()
                       / max(np.abs(summary.table[c]).sum(), 1e-300)
                       for c in _CHANNELS)
        l_done = (slab_frac(lmax) < tol) and (slab_frac(lmax - 1) < tol)
        k_done = "k" not in summary.edge_report or \
            all(f < tol for f in summary.edge_report["k"])
        n_done = "n" not in summary.edge_report or \
            all(f < tol for f in summary.edge_report["n"])

        if l_done and k_done and n_done:
            return summary._replace(rounds=tuple(rounds))
        if not l_done and lmax >= lmax_cap:
            warnings.warn(f"total_fluxes hit lmax_cap={lmax_cap} before "
                          "converging; results are truncated", UserWarning)
            conv = {c: False for c in _CHANNELS}
            return summary._replace(converged=conv, rounds=tuple(rounds))
        if not l_done:
            lmax = min(lmax + 2, lmax_cap)
        if not k_done:
            k_half *= 2
        if not n_done:
            n_half *= 2
