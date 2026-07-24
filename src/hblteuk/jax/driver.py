# Unified solver dispatch, canonical-half flux summation, and the staged
# adaptive total-flux driver. Layered strictly on the per-geometry table
# solvers and flux_modes -- no new physics. Design: dev/PHASE5D_DESIGN.md;
# writeup: dev/PHASE5D.md.
import math
import warnings
from typing import NamedTuple

import numpy as np

from .fluxes import flux_modes
from .pipeline import (GenericModeSolution, _chunk_size,
                       solve_generic_mode_table, solve_mode_table,
                       solve_spherical_mode_table)


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


def sum_fluxes(sol, orbit, tol=None, peak_edges=False):
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
    if peak_edges:
        # peak strategy knobs: the l extent (top slab, like box), and the
        # k/n band offsets from their (l,m)-peaks (dk, dn). Each edge is the
        # outermost shell of that knob.
        koff = np.array([_k_offset(ll, mm, kk)
                         for ll, mm, kk in zip(l, m, k)])
        noff = np.array([_n_offset(orbit.e, mm, nn)
                         for mm, nn in zip(m, n)])
        axes = (("l", l), ("dk", koff), ("dn", noff))
    else:
        # box: l is summed from |s| upward and only truncated at the top, so
        # its edge is the highest-l slab alone; k, n are centered on 0 and
        # truncated at both ends, so both extremes count
        axes = (("l", l), ("k", k), ("n", n))
    for axis, v in axes:
        if v.max() == v.min():
            continue                      # degenerate/zero-spread axis
        if peak_edges or axis == "l":
            edge = v == v.max()           # outermost shell
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


def _decay_step(shell_sums, budget, cap=4):
    """Decay-model step for one growth direction. shell_sums maps each
    channel to its per-offset-level flux-magnitude sums S_0..S_edge; budget
    maps each channel to its absolute allowance B_c. For each channel:
    estimate the geometric decay r = S_edge/S_{edge-1}, the tail beyond the
    edge = S_edge*r/(1-r), and the shells needed to drop that tail under B_c.
    The direction is done when max(tail, S_edge) < B_c for every channel (the
    S_edge term is the model-free safety gate). Returns (done, step) with the
    step = min(max over channels of the predicted shells, cap), >=1 when not
    done. (First pass uses the last-two-shell ratio; a log-linear fit over the
    outer 2-3 shells is a later robustness refinement.)
    """
    done_all = True
    step = 0
    for c, S in shell_sums.items():
        S = np.asarray(S, float)
        B = budget[c]
        edge = S[-1] if S.size else 0.0
        if S.size < 2 or edge <= 0.0:
            if edge < B:
                continue
            done_all = False
            step = max(step, 1)
            continue
        prev = S[-2]
        r = edge / prev if prev > 0 else np.inf
        if not (0.0 < r < 1.0):            # not decaying -> unresolved
            done_all = False
            step = max(step, 1)
            continue
        tail = edge * r / (1.0 - r)
        if max(tail, edge) < B:
            continue
        done_all = False
        need = math.log(B * (1.0 - r) / edge) / math.log(r) - 1.0
        step = max(step, int(math.ceil(need)) if need > 0 else 1)
    return done_all, min(step, cap)


# Fields sum_fluxes/_Accum.merge actually read off a solution; a bucketed
# solve returns this trimmed view so the padding rows never leak downstream.
class _SolView(NamedTuple):
    l: np.ndarray
    m: np.ndarray
    k: np.ndarray
    n: np.ndarray
    omega: np.ndarray
    eigenvalues: np.ndarray
    amplitudes: np.ndarray
    errors: np.ndarray


def _want_bucket(bucket):
    # "auto" (default): pad only on GPU, where XLA recompilation per distinct
    # batch shape dominates small adaptive rounds (A100: ~10 s/round compile,
    # trivial compute). On CPU compile is cheap and padding only wastes
    # compute, so skip. True/False force the choice (True is for tests, to
    # exercise the padded path on CPU).
    if bucket in (True, False):
        return bucket
    import jax
    return jax.default_backend() == "gpu"


def _bucketed_solve(orbit, s, arr, nsamples, ntheta, max_memory_gb,
                    pad, ndomains=150, nnodes=32):
    """solve_modes with the batch padded to the pipeline chunk size so every
    adaptive round reuses one compiled trace. Padding rows are duplicates of
    real modes (kept valid) and are sliced off before return. csize is the
    same bucket the chunked path already pads to (pipeline.py), so a padded
    small round and a large chunked round share the identical XLA trace ->
    one compile for the whole driver run instead of one per round.
    """
    B = len(arr)
    kw = dict(nsamples=nsamples, ntheta=ntheta,
              max_memory_gb=max_memory_gb, return_radial=False)
    if pad:
        csize = _chunk_size(max_memory_gb, nsamples, ndomains, nnodes)
        if csize is not None and B < csize:
            arr = arr[np.arange(csize) % B]      # first B rows == arr (csize>=B)
    sol = solve_modes(orbit, s, arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3],
                      **kw)
    return _SolView(sol.l[:B], sol.m[:B], sol.k[:B], sol.n[:B],
                    sol.omega[:B], sol.eigenvalues[:B],
                    np.asarray(sol.amplitudes)[:B],
                    np.asarray(sol.errors)[:B])


def _window_modes(lmax, k_half, n_half):
    # canonical half of the rectangle l in [2, lmax], m in [0, l]
    return [(l, m, k, n) for l in range(2, lmax + 1)
            for m in range(0, l + 1)
            for k in range(-k_half, k_half + 1)
            for n in range(-n_half, n_half + 1)
            if (m > 0) or (m == 0 and k > 0)
            or (m == 0 and k == 0 and n > 0)]


# Spectral-peak model, read at FIXED (l, m): for a given (l, m) the flux
# concentrates at the polar mode k where |m+k| = l (k = l-m or k = -(l+m))
# and at the radial modes |n| ~ round(m / (1-e)^1.5) (from the empirical
# ridge m = round(|n|(1-e)^1.5)). At high eccentricity (1-e)^1.5 is small so
# the peak |n| departs sharply from 0 -- exactly where box wastes the most
# modes. The formula only needs to land NEAR the peak: a symmetric band of
# half-width dn around +-n_peak tests both sides so the adaptive check can
# see the fall-off (or keep growing toward the true peak). n = 0 is NOT
# forced in -- at high e it may not matter. Iterating (l, m) naturally
# (m in [0, l], as box does) and banding only k, n around these
# (l,m)-centers prunes the two expensive axes. dev/PHASE5E.md.

def _k_peaks(l, m):
    return (l - m, -(l + m))


def _n_peak_mag(e, m):
    return int(round(m / (1 - e) ** 1.5)) if e < 1 else 0


def _k_offset(l, m, k):
    # distance to the nearer k-peak center at this (l, m)
    a, b = _k_peaks(l, m)
    return min(abs(k - a), abs(k - b))


def _n_offset(e, m, n):
    # distance to the nearer +-n-peak center for this m
    p = _n_peak_mag(e, m)
    return min(abs(n - p), abs(n + p))


def _m_offset(l, m, c_m):
    # distance to this l's m-ridge center (gradient strategy)
    return abs(m - c_m)


def _gradient_window_modes(orbit, lmax, c_m, dm, dk, dn):
    """Canonical-half mode list banded around the ridge in all four
    directions: m in [c_m[l]-dm, c_m[l]+dm] (clipped to [0, l]), k banded
    (half-width dk) around _k_peaks(l, m), n banded (half-width dn) around
    +-_n_peak_mag(e, m). Degenerate collapse: equatorial -> k=0 only;
    circular -> n=0 only.
    """
    e = orbit.e
    equatorial = abs(orbit.x) == 1
    circular = e == 0
    out = set()
    for l in range(2, lmax + 1):
        center = c_m.get(l, l)
        mlo = max(0, center - dm)
        mhi = min(l, center + dm)
        for m in range(mlo, mhi + 1):
            k_centers = (0,) if equatorial else _k_peaks(l, m)
            p = _n_peak_mag(e, m)
            n_centers = (0,) if circular else (p, -p)
            for kc in k_centers:
                krange = (0,) if equatorial else range(kc - dk, kc + dk + 1)
                for k in krange:
                    for nc in n_centers:
                        nrange = ((0,) if circular
                                  else range(nc - dn, nc + dn + 1))
                        for n in nrange:
                            if ((m > 0) or (m == 0 and k > 0)
                                    or (m == 0 and k == 0 and n > 0)):
                                out.add((l, m, k, n))
    return sorted(out)


def _m_center_seed(orbit, lmax):
    # prograde l=m dominance; data-driven recenter corrects per orbit
    return {l: l for l in range(2, lmax + 1)}


def _recenter_m(l_arr, m_arr, w_arr, c_m, dm):
    """Per l, move the m-center to the argmax of accumulated weight w_arr,
    but only when that argmax is interior to the current band [c-dm, c+dm].
    A boundary argmax means the band has not captured the peak -> keep the
    center and flag grow=True so the M direction expands before we trust it.
    """
    l_arr = np.asarray(l_arr)
    m_arr = np.asarray(m_arr)
    w_arr = np.asarray(w_arr, float)
    new_c_m = dict(c_m)
    grow = False
    for l in np.unique(l_arr):
        l = int(l)
        sel = l_arr == l
        if not sel.any():
            continue
        ms, ws = m_arr[sel], w_arr[sel]
        peak_m = int(ms[int(np.argmax(ws))])
        c = c_m.get(l, l)
        lo, hi = c - dm, c + dm
        if lo < peak_m < hi:
            new_c_m[l] = peak_m
        elif peak_m <= lo or peak_m >= hi:
            grow = True
    return new_c_m, grow


def _peak_window_modes(orbit, lmax, dk, dn):
    # canonical (l, m, k, n) with l in [2, lmax], m in [0, l], and k, n
    # banded (half-widths dk, dn) around their (l, m)-peaks. Degenerate
    # orbits collapse the banded axis: equatorial (|x| = 1) -> k = 0 only;
    # circular (e = 0) -> n = 0 only.
    e = orbit.e
    equatorial = abs(orbit.x) == 1
    circular = e == 0
    out = set()
    for l in range(2, lmax + 1):
        for m in range(0, l + 1):
            k_centers = (0,) if equatorial else _k_peaks(l, m)
            p = _n_peak_mag(e, m)
            n_centers = (0,) if circular else (p, -p)
            for kc in k_centers:
                krange = (0,) if equatorial else range(kc - dk, kc + dk + 1)
                for k in krange:
                    for nc in n_centers:
                        nrange = (0,) if circular \
                            else range(nc - dn, nc + dn + 1)
                        for n in nrange:
                            if (m > 0) or (m == 0 and k > 0) \
                                    or (m == 0 and k == 0 and n > 0):
                                out.add((l, m, k, n))
    return sorted(out)


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
                 strategy="box", verbose=False, bucket="auto"):
    """Adaptive staged-round flux driver. strategy="box" grows a full
    rectangular canonical (l, m, k, n) window; strategy="peak" grows four
    knobs (k_ext, n_ext, dm, dl) around the spectral ridges l ~ |m+k| and
    m ~ round(|n|(1-e)^1.5), solving far fewer modes on high-e generic
    orbits (dev/PHASE5E.md). Both stop when every axis'/knob's edge
    contribution < tol x total per channel, or lmax_cap (warns,
    converged=False).

    bucket="auto" pads each round's batch to the pipeline chunk size on GPU
    (one shared XLA trace instead of a recompile per round; big win on small
    adaptive runs where compile dominates) and is off on CPU; True/False
    force it. Fluxes are bit-identical to the unpadded path either way.
    """
    if strategy not in ("box", "peak"):
        raise ValueError(f"strategy must be 'box' or 'peak', got {strategy!r}")
    pad = _want_bucket(bucket)
    if strategy == "peak":
        return _total_fluxes_peak(orbit, s, tol, lmax_cap, max_memory_gb,
                                  verbose, pad)
    return _total_fluxes_box(orbit, s, tol, lmax_cap, max_memory_gb, verbose,
                             pad)


def _total_fluxes_box(orbit, s, tol, lmax_cap, max_memory_gb, verbose,
                      pad=False):
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
            sol = _bucketed_solve(
                orbit, s, arr,
                nsamples=max(2**9, _next_pow2(4 * n_half)),
                ntheta=max(2**7, _next_pow2(4 * k_half)),
                max_memory_gb=max_memory_gb, pad=pad)
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


def _total_fluxes_peak(orbit, s, tol, lmax_cap, max_memory_gb, verbose,
                       pad=False):
    # iterate (l, m) naturally to lmax; band k, n around their (l, m)-peaks
    # with half-widths dk, dn. Three adaptive knobs. l uses the box two-slab
    # rule; dk/dn grow while their outermost offset shell contributes > tol.
    lmax = 4
    dk = 2
    dn = 4
    acc = _Accum()
    rounds = []

    while True:
        new = [md for md in _peak_window_modes(orbit, lmax, dk, dn)
               if md not in acc.solved]
        if new:
            arr = np.array(new)
            sol = _bucketed_solve(
                orbit, s, arr,
                nsamples=max(2**9, _next_pow2(4 * dn)),
                ntheta=max(2**7, _next_pow2(4 * dk)),
                max_memory_gb=max_memory_gb, pad=pad)
            acc.merge(sol)
        elif not rounds:
            raise RuntimeError("empty initial peak window")
        summary = sum_fluxes(acc, orbit, tol=tol, peak_edges=True)
        rounds.append(dict(window=(lmax, dk, dn), n_new=len(new),
                           totals={c: getattr(summary, c) for c in _CHANNELS}))
        if verbose:
            print(f"round {len(rounds)}: lmax={lmax}, dk={dk}, dn={dn}, "
                  f"+{len(new)} modes ({len(acc.l)} total), "
                  f"Edot_inf={summary.Edot[0]:.6e}")

        # per-knob outer-two-shell stop: a knob is done when its two
        # outermost shells each carry < tol x total for every channel.
        koff = np.array([_k_offset(ll, mm, kk)
                         for ll, mm, kk in zip(acc.l, acc.m, acc.k)])
        noff = np.array([_n_offset(orbit.e, mm, nn)
                         for mm, nn in zip(acc.m, acc.n)])
        shells = dict(l=acc.l, dk=koff, dn=noff)

        def shell_frac(vals, level):
            sel = vals == level
            if not sel.any():
                return 0.0
            return max(np.abs(summary.table[c][sel]).sum()
                       / max(np.abs(summary.table[c]).sum(), 1e-300)
                       for c in _CHANNELS)

        def knob_status(name, top):
            # (done, step): a knob is done when its two outermost shells each
            # carry < tol x total. When not done, the step scales with how
            # far the boundary/volume ratio exceeds tol -- less-converged
            # directions (more boundary power) expand more, so the box grows
            # non-uniformly toward whichever edges still hold flux. The step
            # is bounded to [1, 2] shells: dk and dn multiply, so larger
            # simultaneous steps blow the mode count up geometrically (a tiny
            # initial box is all-boundary, edge fraction ~ 1, and would
            # otherwise trigger a max jump on every axis at once).
            v = shells[name]
            if v.max() == v.min():       # degenerate/collapsed knob
                return True, 0
            f, f2 = shell_frac(v, top), shell_frac(v, top - 1)
            if f < tol and f2 < tol:
                return True, 0
            step = 2 if max(f, f2) > 10 * tol else 1
            return False, step

        l_done, l_step = knob_status("l", lmax)
        dk_done, dk_step = knob_status("dk", dk)
        dn_done, dn_step = knob_status("dn", dn)

        if l_done and dk_done and dn_done:
            return summary._replace(rounds=tuple(rounds))
        if not l_done and lmax >= lmax_cap:
            warnings.warn(f"total_fluxes hit lmax_cap={lmax_cap} before "
                          "converging; results are truncated", UserWarning)
            conv = {c: False for c in _CHANNELS}
            return summary._replace(converged=conv, rounds=tuple(rounds))
        # grow only the un-converged knobs, each by its own proportional step
        if not l_done:
            lmax = min(lmax + l_step, lmax_cap)
        if not dk_done:
            dk += dk_step
        if not dn_done:
            dn += dn_step


def flux_efficiency(summary, tol):
    """How many solved modes are actually required to reach `tol`, per
    channel. Magnitude-tail definition: for each channel/side, sort modes
    by |contribution| descending and count the smallest prefix whose
    omitted tail (sum of remaining magnitudes) < tol x |total|. The tail
    magnitude bounds the truncated-sum error, so this is a sound
    "modes that matter" count even with L/Q sign cancellation.
    efficiency = binding required count / n_solved (1 = no waste).
    """
    N = len(summary.table["l"])
    n_required = {}
    for c in _CHANNELS:
        counts = []
        for side in (0, 1):
            total = abs(getattr(summary, c)[side])
            if total == 0:
                counts.append(0)
                continue
            contrib = np.sort(np.abs(summary.table[c][:, side]))[::-1]
            # tail[i] = sum of contrib[i:]; find smallest k with tail[k] < thr
            tail = np.concatenate([np.cumsum(contrib[::-1])[::-1], [0.0]])
            k = int(np.searchsorted(-tail, -tol * total))
            counts.append(min(k, N))
        n_required[c] = (counts[0], counts[1])
    per_channel = {c: max(n_required[c]) / max(N, 1) for c in _CHANNELS}
    efficiency = max(per_channel.values()) if N else 0.0
    return dict(n_solved=N, n_required=n_required,
                efficiency=efficiency, per_channel_efficiency=per_channel)
