# Bilinear-Form Generic Source Integral Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the factored bilinear-form source integral the default generic-orbit source path, with per-mode error estimates that never understate today's.

**Architecture:** A new module holds the orbit-only kernel table and the radial and polar mode-factor tables. A jitted core forms each family's theta-integrated radial integrand from 12 shared matrix products, then reuses today's FFT bin extraction, q_r tail and q_theta even/odd estimates unchanged; only the cancellation floor switches to a triangle-inequality bound. `integrate_source_generic_batch` and `solve_generic_mode_table` default to it; `"fft"` and `"fft2"` stay.

**Tech Stack:** Python, JAX (x64), NumPy, pytest. Env: `~/miniforge3/envs/hblteuk-env/bin/python`.

**Spec:** `docs/superpowers/specs/2026-11-10-bilinear-source-design.md`. Prototype: `dev/bilinear_source2.py`, measurements `dev/BILINEAR_SOURCE.md`.

## Global Constraints

- FP64/complex128 throughout; jitted; works with chunked and padded batches.
- Same orbit requirements as `generic_samples` (e != 0, |x| < 1).
- Return type stays `SourceAmplitudes` (amplitudes (B, 2) [In, Up], errors (B, 2)).
- `method` values: `"bilinear"` (new default), `"fft"`, `"fft2"`.
- Error estimates: q_r tail and q_theta check keep their exact current definitions; cancellation floor uses absJ_bound = sum_t |u_t| (|K_t| @ |v_t|), summed over both families, never smaller than today's absJ.
- `_K_SRC` is shared by all orbit types: never lower it; raise only if GPU calibration of the generic path needs it.
- Spherical and equatorial integrators unchanged.
- Commits only when the user says so; code + tests in one commit per task.
- Concise comments; long explanations go in `dev/*.md`.

## Review Focus

1. Padded/chunked batches with duplicated rows: results equal the unchunked solve (covered by the existing `test_generic_chunked_matches_unchunked`, which runs the new default after Task 3).
2. Near-zero |omega| with large m, e.g. (13, 11, -1, -15) on (0.9, 8, 0.3, 0.7): finite amplitudes matching `"fft"` (Task 2 test).
3. Modes whose amplitude cancels to roundoff: errors must stay finite and >= today's (Task 2 test over all modes, no flux filter).
4. Negative n and n = 0 / k = 0 modes: the FFT bin `mod(n, nq)` must match `"fft"` (Task 2 table includes them).
5. Non-power-of-two even grids (nsamples=96, ntheta=40): same agreement with `"fft"` (Task 2 test).

---

### Task 1: Factor tables

**Files:**
- Create: `src/hblteuk/jax/source_bilinear.py`
- Test: `tests/test_jax_bilinear.py`

**Interfaces:**
- Produces:
  - `TERMS`: tuple of 12 `(c: int, radial: str, polar: str)`, exactly the prototype's order and names.
  - `kernels(a, En, Lz, rp, urp, th, uthp) -> jax.Array` (12, nq, nth) complex; inputs 1D (nq,) and (nth,).
  - `radial_factors(a, m, om, rp) -> dict[str, jax.Array]` keys `"1", "-iKt/D", "Kt/D", "quad", "iKt", "iKt/D"`, broadcasting m, om (B, 1) with rp (1, nq).
  - `polar_factors(a, m, om, th, S, dS, d2S) -> dict[str, jax.Array]` keys `"L1L2S", "ann", "L2S", "S"`, broadcasting (B, 1) with (1, nth).

- [ ] **Step 1: Write the failing test** `test_terms_reproduce_source_coefficients`: port of `dev/test_bilinear_source2.py`. Orbit `MiniGeo(0.9, 8.0, 0.3, 0.7)`, `generic_samples(nsamples=64, ntheta=32)`, (m, omega) in ((14.0, 1.2), (-3.0, 0.017), (0.0, 1.3e-4)), random complex S, dS, d2S (seed 3). Rebuild A0, A1, A2 as `sum_t K[t] * R[radial] * P[polar]` grouped by `c`; assert `max|got - want| <= 1e-13 * max|want|` for each c against `_source_coefficients`.
- [ ] **Step 2: Run it, expect ImportError**: `~/miniforge3/envs/hblteuk-env/bin/python -m pytest tests/test_jax_bilinear.py -q`
- [ ] **Step 3: Implement the three functions and `TERMS`** in `source_bilinear.py`, moving the prototype's `_orbit`, `kernels`, `radial_factors`, `polar_factors`. Import `SQRT2` from `.source`.
- [ ] **Step 4: Run it, expect PASS.**
- [ ] **Step 5: Commit** (on user go-ahead): `git add src/hblteuk/jax/source_bilinear.py tests/test_jax_bilinear.py`.

### Task 2: Jitted core with error estimates

**Files:**
- Modify: `src/hblteuk/jax/source_bilinear.py`, `src/hblteuk/jax/source.py` (extract the argument assembly)
- Test: `tests/test_jax_bilinear.py`

**Interfaces:**
- Consumes: Task 1 tables.
- Produces:
  - `source._generic_args(samples, lam, m, k, n, omega, S, dS, d2S, hom, subdomains_up, subdomains_in) -> tuple`: the `args` tuple `integrate_source_generic_batch` builds today (psi via `_psi_at`, radial/polar arrays, dtype casts), extracted unchanged so every method gets identical inputs.
  - `_integrate_generic_bilinear(s, kappa, a, En, Lz, gamma_mino, radial_arrs, polar_arrs, lam, m, k, n, omega, psi_arrs) -> SourceAmplitudes`, jitted with `static_argnums=(0,)`. Same arguments and argument meaning as `source._integrate_generic_core`.

Algorithm (what the signature does not determine):
1. Normalize psi per family by `norm_up`, `norm_in` as `_integrate_generic_core` does.
2. `rho_c` for c = 0, 1, 2 from Zs, dZs, d2Zs, gs, dgs (the B0/B1/B2 algebra of `_integrate_generic_core`).
3. Polar weights `pw = exp(i(omega tth - m phith + k qth))` (no 1/nth), split into even columns (`idx % 2 == 0`) and odd columns.
4. For half h in (even, odd): `X_h[t] = K[t][:, cols_h] @ (P[polar_t] * pw)[:, cols_h]^T`, shape (T, B, nq).
5. Per family f and half h: `J_{f,h}(b, i) = sum_t R[radial_t] rho_{c_t}(b, i) * phase_r(b, i) * X_h[t, b, i]`, phase_r = `exp(i(omega tr - m phir))`.
6. Replicate the library from here: `J = (J_even + J_odd) / nth`; `spec = ifft(J)`; c = bin `mod(n, nq)`; `tail_r` and `err_r` as today; halves `2 J_even / nth`, `2 J_odd / nth` through the same ifft bin for `err_th`.
7. absJ bound: `sum_f sum_t sum_i |R rho|(b, i) * (|K[t]| @ |P_t|^T)(i, b) / nq` (|phase_r| = |pw| = 1), the bound on the library's `absJ` (sum over theta slices of the mean over q_r of |J_up| + |J_in|). Then `canc = 1e-15 / |c / (absJ / (2 nth))|`, `err = max(err_r, err_th, canc)`.
8. Wronskian, rescale, `const_w`, amps exactly as `_integrate_generic_core`.

- [ ] **Step 1: Write the failing tests** in `tests/test_jax_bilinear.py`. Inputs: `solve_homogeneous_batch` at 32 domains and `polar_swsh_values`, as in `tests/test_jax_generic.py::gen_setup`, then `args = _generic_args(...)`; compare `_integrate_generic_bilinear(*args)` with `_integrate_generic_core(*args)`:
  - `test_bilinear_matches_fft_amplitudes`: on `cluster/bench_modes_target.npz` modes (13, 11, -1, -15), (14, 11, -1, -15), (2, 2, 0, 0), (2, 2, 0, 1), (4, 3, 1, -2), (8, 6, -5, -2), (12, 7, 5, 10), (3, 0, 1, 0), orbit (0.9, 8, 0.3, 0.7), 32 domains, `generic_samples(512, 128)`: amplitudes of `_integrate_generic_bilinear` vs `_integrate_generic_core`, `rtol=1e-10`, and all finite.
  - `test_bilinear_error_estimates_never_smaller`: same inputs; `errors_new >= errors_old * (1 - 1e-6)` for every mode and family; all finite.
  - `test_bilinear_non_pow2_grid`: same modes with `generic_samples(96, 40)`: amplitudes agree `rtol=1e-10`.
- [ ] **Step 2: Run them, expect failure** (function missing).
- [ ] **Step 3: Extract `_generic_args`** (no behaviour change; `integrate_source_generic_batch` calls it) **and implement `_integrate_generic_bilinear`** per the algorithm above.
- [ ] **Step 4: Run Task 1 and 2 tests, expect PASS.**
- [ ] **Step 5: Commit** (on user go-ahead).

### Task 3: Default method

**Files:**
- Modify: `src/hblteuk/jax/source.py` (`integrate_source_generic_batch`), `src/hblteuk/jax/pipeline.py:467` (`solve_generic_mode_table` default)
- Test: `tests/test_jax_generic.py`

**Interfaces:**
- Consumes: `_integrate_generic_bilinear` (Task 2), with kernels built per call from `samples` via `kernels(...)`.
- Produces: `integrate_source_generic_batch(..., method="bilinear")` accepting `"bilinear" | "fft" | "fft2"`; `ValueError("method must be 'bilinear', 'fft' or 'fft2'")` otherwise. `solve_generic_mode_table(..., method="bilinear")`. Docstrings name all three.

- [ ] **Step 1: Write the failing tests** in `tests/test_jax_generic.py`: extend `test_amplitudes_vs_pybhpt` parametrization to `["bilinear", "fft", "fft2"]`; add `test_bilinear_equals_fft(gen_setup)` asserting amplitudes `rtol=1e-12`, errors `new >= old * (1 - 1e-6)`; add `test_generic_default_is_bilinear` checking `inspect.signature` defaults of both functions are `"bilinear"`; add an unknown-method `ValueError` check.
- [ ] **Step 2: Run, expect failure.**
- [ ] **Step 3: Wire it:** in `integrate_source_generic_batch`, for `"bilinear"` call `_integrate_generic_bilinear(*_generic_args(...))`; the core builds K from the orbit arrays it already receives; change both defaults.
- [ ] **Step 4: Run** `tests/test_jax_generic.py tests/test_jax_bilinear.py tests/test_jax_chunking.py tests/test_jax_driver.py`, expect PASS; then the full suite in the background (`pytest tests/ -q`, ~15 min), expect all pass.
- [ ] **Step 5: Record** CPU timing (`dev/bilinear_compare.py` or equivalent) in `dev/BILINEAR_SOURCE.md`; **commit** (on user go-ahead) and push.

### Task 4: GPU validation and memory model

**Files:**
- Modify: `dev/calibrate_chunk_model.py` and `cluster/calibrate_chunk_model.py` (add a generic-path config), `dev/JAX_PERF.md`, `dev/BILINEAR_SOURCE.md`
- Possibly modify: `src/hblteuk/jax/pipeline.py` `_K_SRC` (raise only)

**Interfaces:**
- Consumes: pushed commit from Task 3.

- [ ] **Step 1: Add a generic config** to the calibration script: `solve_generic_mode_table` on the reference orbit, B = 800, nsamples 512, ntheta 128, 32 domains; print measured peak device memory against `_per_mode_bytes`.
- [ ] **Step 2: H200 run** (user): `MARCH=0 CALIBRATE=1` benchmark job plus `cluster/slurm_bench_target.sbatch`, tag = commit.
- [ ] **Step 3: Check:** measured/modelled <= 1 for the generic config; if above 1, raise `_K_SRC` just enough, with a test in `tests/test_jax_chunking.py` pinning the new value. Reference case `Edot_inf` matches 1.8209785850e-04 to 1e-9; record before/after per-stage table (baseline 1.71 s at 6ea0ba2).
- [ ] **Step 4: Commit** any `_K_SRC` change (on user go-ahead).
