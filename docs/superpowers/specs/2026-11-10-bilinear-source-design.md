# Bilinear-form generic source integral: design

Date: 2026-11-10. Branch: `kerr-jax`. Prototype and measurements:
`dev/bilinear_source2.py`, `dev/BILINEAR_SOURCE.md`.

## Intent

The generic-orbit source integral (`integrate_source_generic_batch`, scan over
q_theta in `_integrate_generic_core`) is now the largest stage of a generic
flux run: 0.78 s of 1.71 s on H200 for the reference case
(a, p, e, x) = (0.9, 8, 0.3, 0.7), tol 1e-6. The goal behind the whole effort is
to bring that case toward the colleague's 0.5 s (A30).

A factored prototype recasts the 2D torus integral as matrix products and runs
at 1.0-1.2 us/mode on H200 against 12-22 us/mode, agreeing with the current
integrator to roundoff. This work puts it into the library as the default for
generic orbits, with per-mode error estimates that keep `error_bound`
trustworthy.

## How the method works

At torus point (q_r_i, q_theta_j), the integrand for mode b and radial family
(Up or In) is `sum_c A_c(i, j) rho_{c,b}(i) phase_r(i) w_b(j)`, with A_c from
`_source_coefficients`. Each term of A_c factors as

    K_t(i, j)  x  R_t(r_i; m, omega)  x  P_t(theta_j; m, omega, S, dS, d2S)

orbit-only kernel, radial mode factor, polar mode factor. Kt depends only on r;
L1, L2, L2p only on theta, so nothing is multiplied out. 12 terms, 4 distinct
polar factors (L1L2S, "ann", L2S, S), 6 radial factors. The term table is in
`dev/bilinear_source2.py` (`TERMS`, `kernels`, `radial_factors`,
`polar_factors`) and is checked pointwise against `_source_coefficients`.

Per batch: X_t = K_t @ (P_t w)^T, a matrix product shared by both families;
then contract with u_t = R_t rho_c phase_r over q_r. Normalization, Wronskian
and rescale are the library's.

## Goals

1. Same physics: amplitudes match `method="fft"` to roundoff (prototype:
   <= 7e-12 on flux-carrying modes; 2.5e-9 on large random batches, which a CPU
   check traced to per-mode noise at the reference level, shared by all
   methods). Existing pybhpt reference tests pass with the new default.
2. Trustworthy error bounds: per-mode errors stay conservative, so
   `error_bound` compared against `tol` means what it does today. Values may
   differ from the current estimates but must never be smaller in a way that
   understates error.
3. Drop-in: `solve_generic_mode_table` and `total_fluxes` pick it up with no
   change to how they are called. Return type stays `SourceAmplitudes`
   (amplitudes (B, 2) [In, Up], errors (B, 2)).
4. Faster in practice: reference case on H200 drops from 1.71 s toward the
   projected ~1.0 s.

## Non-goals

- Spherical-inclined and equatorial integrators stay as they are.
- Half-torus symmetry, kernel compression, ultraspherical radial solve.
- Removing the current scan path.

## Anti-goals (failure even if it runs)

- Error estimates that understate error relative to today's.
- Amplitude changes beyond roundoff on flux-carrying modes.
- Device memory growth that breaks chunking (the model must still bound it).

## Constraints

- FP64/complex128 throughout; jitted; works with chunked and padded batches.
- Same orbit requirements as `generic_samples` (e != 0, |x| < 1).
- Consistency across orbits (user): interface and methodology should be shaped
  so the spherical and equatorial paths can adopt the same approach later.
  Concretely: a `method` argument with the same name and meaning, and the
  factor tables (`radial_factors`, `polar_factors`, kernel terms) written so a
  fixed-r or fixed-theta caller can reuse them.

## Decisions

User's:
- Generic orbits only for now; consistent interface and methodology across
  orbit types where possible.
- Error estimates feed `error_bound`, which is compared against tol to trust a
  result. They must stay trustworthy, not necessarily identical.

Mine (flagged for review):
- Placement: `integrate_source_generic_batch(..., method="bilinear")` becomes
  the default. `"fft"` (current scan) and `"fft2"` (small-batch diagnostic)
  remain selectable. `solve_generic_mode_table`'s `method` default changes to
  match.
- q_r tail and q_theta check keep their exact definitions. Instead of
  contracting straight to the bin-n coefficient, form each family's
  theta-integrated radial integrand J_b(q_r_i) = sum_t u_t(b, i) X_t(b, i)
  (u without the bin weight), then FFT over q_r as today. That gives c (bin n)
  and the tail estimate unchanged. For the q_theta check, split the polar
  weights by column parity into even and odd halves, form J_even and J_odd the
  same way (no extra flops: the columns are split, not duplicated), and keep
  `err_th = 0.5 |1 - c_odd / c_even|` with the library's half-grid scaling.
- Cancellation floor: today `canc = 1e-15 / |c / (absJ / (2 nth))|`, with
  absJ from mean |J| per theta slice, which is not bilinear. Replace absJ with
  the bound sum_t |u_t| (|K_t| @ |v_t|), computed with real matrix products.
  It is >= the true mean |J| by the triangle inequality, so the floor is
  conservative. Cost: about +25% (real vs complex products). As today, one
  absJ sums both families (bound for Up plus bound for In) and serves both.
- Memory model: the new working set is K (12 nq nth, shared), V (12 B nth),
  X_even and X_odd (2 x 12 B nq), J (2 families x 3, B nq). `_K_SRC` is shared
  by every orbit type, so leave it unchanged unless GPU calibration
  (`dev/calibrate_chunk_model.py`) shows the generic path needs more. If it
  needs less, do not lower the shared constant.

## Tests

- Factored terms reproduce `_source_coefficients` pointwise, several (m, omega)
  including m=14 and omega ~ 1e-4 (port of `dev/test_bilinear_source2.py`).
- `method="bilinear"` vs `method="fft"` on a small generic table: amplitudes to
  ~1e-10 relative on flux-carrying modes; q_r tail and q_theta estimates agree
  closely (same J up to roundoff); cancellation floor new >= old per mode.
- Existing generic, driver and chunking suites pass with the new default.
- H200: `cluster/bench_target.py` before and after; spot-check fluxes on the
  survey grid against the previous run.

## Left to the builder

- Module layout (private functions in `source.py` or a new module), naming.
- Whether to cache kernels per samples object (building them is ~0.6 ms on
  H200, so building per call is acceptable).
- Exact tolerances in tests, within the goals above.
