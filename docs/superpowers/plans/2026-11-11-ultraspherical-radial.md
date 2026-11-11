# Ultraspherical Radial Solve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ultraspherical (Olver-Townsend) equation rows in the JAX radial march and TS map, selectable by `scheme`, default everywhere, so `total_fluxes` can run with fewer radial subdomains.

**Architecture:** A new module `src/hblteuk/jax/ultraspherical.py` holds the cached operator tables and builds the n-2 equation rows per subdomain from P, Q, U sampled at 4 points. `collocation.solve_fixed_multidomain_batch` and `starobinsky._ts_solve_core` swap their nodal rows for these when `scheme="ultraspherical"`; unknowns, boundary rows, 2x2 transfer, residual estimate and evaluation are unchanged. `scheme` is threaded from `solve_homogeneous_batch` up to `total_fluxes`; an H200 scan then fixes the driver's domain count.

**Tech Stack:** JAX (FP64), NumPy, pytest. Python: `~/miniforge3/envs/hblteuk-env/bin/python`.

**Spec:** `docs/superpowers/specs/2026-11-11-ultraspherical-radial-design.md`. Prototype: `dev/ultraspherical.py`, `dev/us_solver.py`, `dev/test_ultraspherical.py`, findings `dev/ULTRASPHERICAL.md`.

## Global Constraints

- Accuracy bar: flux totals within 1e-9 relative of collocation at 150 domains on every survey orbit.
- FP64/complex128; jitted; batching, chunking and padding behaviour unchanged.
- Same unknowns (Chebyshev-T coefficients), same boundary rows, 2x2 transfer and evaluation.
- One argument name everywhere: `scheme`, values exactly `"ultraspherical"` or `"collocation"`; anything else raises `ValueError("scheme must be 'ultraspherical' or 'collocation'")`.
- `errors` stay the relative nodal residual at the collocation nodes (`||res|| / ||c||`), whichever scheme built the rows.
- Chunk memory model (`pipeline._per_mode_bytes`) unchanged.
- Pipeline functions keep `ndomains=150`; only `total_fluxes` gets the scanned default.
- Not in scope: banded solver, s=+2 horizon route, source/angular/driver logic changes.
- Commit only on the user's go-ahead. Concise code comments; long notes in `dev/*.md`. `dev/` and `cluster/` are gitignored.

## Review Focus

1. Descending subdomains (In branch, sigma from 1 down): h < 0 in every row. Covered by Task 2's In-branch comparison; Task 1 test uses a negative h.
2. Static-arg recompiles: `scheme` must be static in `_solve_branch` and `_ts_solve_core`, and the two schemes must not share a cached trace. Task 2 test calls both schemes back to back in one process.
3. Coefficients stop being cubic (someone edits `teuk_pqu`): rows silently wrong. Task 1 guard test.
4. Unstable branch (s=-2 Up marched directly, `stabilize=False`): ultraspherical must not be worse than collocation's ~1e-9 floor by orders of magnitude. Task 2 test with the existing loose 1e-6 bound.
5. Retrograde / spherical / equatorial paths share the solve: Task 3 runs the three pipeline entry points with both schemes.

---

### Task 1: Operator tables and equation rows

**Files:**
- Create: `src/hblteuk/jax/ultraspherical.py`
- Test: `tests/test_jax_ultraspherical.py`

**Interfaces:**
- Produces:
  - `DEG = 3`; `X_FIT: np.ndarray (4,)` (Chebyshev points of the first kind, as prototype); `V_INV: np.ndarray (4, 4)`.
  - `operators(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]`, each `(4, n-2, n)`, `functools.lru_cache`d. Port of `dev/ultraspherical.operators` unchanged (padding `N = n + 2*DEG + 4`).
  - `monomials(values: (..., 4)) -> (..., 4)`.
  - `equation_rows(pqu_fn, z0: (D,), z1: (D,), n: int) -> jax.Array (B, D, n-2, n)`: samples `pqu_fn` at `mid + h*X_FIT` per domain (`h = (z1 - z0)/2`, may be negative), fits monomials, contracts with `operators(n)` as `system_rows` does. `pqu_fn(z (k,)) -> (P, Q, U)` each `(B, k)`, same contract as collocation.

- [ ] **Step 1: Write failing tests** in `tests/test_jax_ultraspherical.py`, ported from `dev/test_ultraspherical.py` against the library module: `test_monomial_fit_exact`, `test_operators_are_banded` (lower 3, upper 7 at n=40), `test_harmonic_oscillator` (cos, atol 1e-14), `test_airy_variable_coefficient` (atol 1e-13), `test_cubic_coefficients_match_collocation` (atol 1e-13). The `solve` helper builds rows with `equation_rows` on one domain instead of `system_rows`; Airy runs on `z0=2.0, z1=-8.0` (descending, h = -5) with BCs at z=2.
  Add:

```python
@pytest.mark.parametrize("s", [-2, 2])
def test_teuk_pqu_is_cubic(s):
    # equation_rows assumes P, Q, U are polynomials of degree <= 3 in sigma;
    # a fifth point off the fit nodes must agree with the cubic interpolant
    from hblteuk.jax.teukolsky import teuk_pqu
    from hblteuk.jax.ultraspherical import X_FIT, monomials
    rng = np.random.default_rng(s + 5)
    lam = jnp.asarray(rng.normal(size=3) + 1j * rng.normal(size=3))
    ma = jnp.asarray(rng.normal(size=3))
    om = jnp.asarray(rng.normal(size=3) + 0.1j * rng.normal(size=3))
    kappa, mid, h, x5 = 0.6, 0.4, 0.3, 0.37
    fit = teuk_pqu(jnp.asarray(mid + h * X_FIT), kappa, s, lam, ma, om)
    at5 = teuk_pqu(jnp.asarray([mid + h * x5]), kappa, s, lam, ma, om)
    for f, g in zip(fit, at5):
        c = monomials(np.asarray(f))
        interp = c @ (x5 ** np.arange(4))
        scale = np.max(np.abs(np.asarray(f)))
        assert np.max(np.abs(interp - np.asarray(g)[:, 0])) <= 1e-13 * scale
```

  and `test_equation_rows_shape`: B=3 Airy-style closure over 5 domains returns `(3, 5, n-2, n)`.

- [ ] **Step 2: Run, expect FAIL** (`ModuleNotFoundError: hblteuk.jax.ultraspherical`)
  `~/miniforge3/envs/hblteuk-env/bin/python -m pytest tests/test_jax_ultraspherical.py -v`

- [ ] **Step 3: Implement `src/hblteuk/jax/ultraspherical.py`** per Interfaces. Header comment: method in two lines, pointer to the spec and `dev/ULTRASPHERICAL.md`.

- [ ] **Step 4: Run, expect PASS** (same command).

- [ ] **Step 5: Commit** (on go-ahead): `git add src/hblteuk/jax/ultraspherical.py tests/test_jax_ultraspherical.py`, message "Add ultraspherical operator tables for the radial solve".

---

### Task 2: `scheme` in the march, TS map and homogeneous solve

**Files:**
- Modify: `src/hblteuk/jax/collocation.py` (`solve_fixed_multidomain_batch`)
- Modify: `src/hblteuk/jax/starobinsky.py` (`_ts_solve_core`, `ts_solve_batch`)
- Modify: `src/hblteuk/jax/teukolsky.py` (`_solve_branch`, `_solve_core`, `solve_homogeneous_batch`)
- Test: `tests/test_jax_teuk.py`, `tests/test_jax_ts.py`

**Interfaces:**
- Consumes: `ultraspherical.equation_rows(pqu_fn, z0, z1, n)`.
- Produces:
  - `solve_fixed_multidomain_batch(pqu_fn, y0, subdomains, n=32, chtype=1, scheme="collocation")`. Default stays collocation here: it is the generic-ODE solver and ultraspherical needs cubic coefficients.
  - `ts_solve_batch(..., n=32, chtype=1, renormalize=True, scheme="collocation")`; `_ts_solve_core` gains `scheme` as a static arg (static_argnums extended).
  - `solve_homogeneous_batch(..., n=32, stabilize=True, scheme="collocation")`; `_solve_branch` gains static `scheme`; passed to both marches and the TS map. (Default flips in Task 3.)
  - Validation in `solve_homogeneous_batch` and `ts_solve_batch` (the public entry points) with the Global Constraints message.

- [ ] **Step 1: Write failing tests.**
  In `tests/test_jax_teuk.py`:
  - `test_ultraspherical_march_matches_collocation[s=-2,2]`: same modes and closure as `test_parallel_march_matches_reference`; both families; `solve_fixed_multidomain_batch(..., scheme=...)` both ways, back to back. Coefficient relative error < 1e-11 on the stable branch for resolved modes, < 1e-6 otherwise (the existing bounds). Errors: finite, and on the stable branch for resolved modes `e_us.max() < 1e-8`; record the measured max in the ledger.
  - `test_ultraspherical_fewer_domains`: `solve_homogeneous_batch` (stabilized) on `MODES` at 8 domains with `scheme="ultraspherical"` vs 150 domains with `"collocation"`; values at `sample_points` (normalised by the first sample, as `test_ts_solve_batch_is_refinement_stable`) agree to 1e-9 relative on both branches.
  - `test_scheme_rejected`: `solve_homogeneous_batch(..., scheme="tau")` raises `ValueError` matching `"scheme must be"`.
  In `tests/test_jax_ts.py`:
  - `test_ts_ultraspherical_matches_collocation`: flipped-spin coefficients from `solve_homogeneous_batch(2, ...)` at 150 domains, `ts_solve_batch` both schemes, relative error < 1e-11.
  - `test_ts_scheme_rejected`: same `ValueError` from `ts_solve_batch`.

- [ ] **Step 2: Run, expect FAIL** (`TypeError: unexpected keyword argument 'scheme'`)
  `~/miniforge3/envs/hblteuk-env/bin/python -m pytest tests/test_jax_teuk.py tests/test_jax_ts.py -v -k "ultraspherical or scheme"`

- [ ] **Step 3: Implement.** In both solvers the matrix is `[2 boundary rows; n-2 equation rows]`; for ultraspherical build the equation rows with `equation_rows(pqu_fn, z0, z1, n)` and concatenate under the existing boundary rows (left value+derivative for the march, `bcL`/`bcR` Dirichlet for TS), as `dev/us_solver.py` does. Keep P, Q, U at the collocation nodes for the residual in both schemes. `_ts_solve_core` builds its `pqu_fn` from `teuk_pqu(..., s_target, ...)`.

- [ ] **Step 4: Run, expect PASS**, then the full radial suites:
  `~/miniforge3/envs/hblteuk-env/bin/python -m pytest tests/test_jax_teuk.py tests/test_jax_ts.py tests/test_jax_ultraspherical.py -q`

- [ ] **Step 5: Commit** (on go-ahead): message "Add a scheme option to the radial march and TS map".

---

### Task 3: Thread `scheme` to the drivers and make ultraspherical the default

**Files:**
- Modify: `src/hblteuk/jax/teukolsky.py`, `src/hblteuk/jax/starobinsky.py` (defaults)
- Modify: `src/hblteuk/jax/pipeline.py` (`solve_mode_table`, `solve_spherical_mode_table`, `solve_generic_mode_table`)
- Modify: `src/hblteuk/jax/driver.py` (`_bucketed_solve`, `total_fluxes`, `_total_fluxes_box`/`_peak`/`_gradient`)
- Test: `tests/test_jax_pipeline.py`, `tests/test_jax_driver.py`

**Interfaces:**
- Consumes: Task 2 signatures.
- Produces:
  - `scheme="ultraspherical"` default on `solve_homogeneous_batch`, `ts_solve_batch`, the three pipeline mode-table functions and `total_fluxes`; `solve_fixed_multidomain_batch` keeps `"collocation"`.
  - Pipeline functions pass `scheme=scheme` to `solve_homogeneous_batch`; `solve_modes` forwards it via `**solver_kwargs` (no change); `_bucketed_solve(..., scheme="ultraspherical")` puts it in `kw`; `total_fluxes(..., scheme="ultraspherical")` passes it to each strategy helper, which passes it to `_bucketed_solve`. Docstrings: one line each.

- [ ] **Step 1: Write failing tests.**
  `tests/test_jax_pipeline.py::test_scheme_default_and_agreement[equatorial, spherical, generic]`: a 6-8 mode table per orbit type (reuse existing fixtures' orbits), via `solve_modes` at 150 domains: default amplitudes equal `scheme="ultraspherical"` exactly, and agree with `scheme="collocation"` to 1e-10 relative on modes with error < 1e-8.
  `tests/test_jax_driver.py::test_total_fluxes_passes_scheme_to_solver`: copy of `test_total_fluxes_passes_ndomains_to_solver`, parametrized `(None, "ultraspherical"), ("collocation", "collocation")`, asserting `seen["scheme"]` for all three strategies.

- [ ] **Step 2: Run, expect FAIL** (`KeyError: 'scheme'` / `TypeError`).

- [ ] **Step 3: Implement** per Interfaces.

- [ ] **Step 4: Run, expect PASS**, then the whole suite (pybhpt references included):
  `~/miniforge3/envs/hblteuk-env/bin/python -m pytest tests -q`
  Expected: all pass. A failing vs-CPU or pybhpt comparison is reported to the user with its numbers, not fixed by loosening a tolerance.

- [ ] **Step 5: Commit** (on go-ahead): message "Make the ultraspherical radial scheme the default".

---

### Task 4: H200 domain scan and the `total_fluxes` default

**Files:**
- Modify: `cluster/bench_resolution.py` (gitignored), `cluster/slurm_resolution.sbatch` (gitignored)
- Modify: `src/hblteuk/jax/driver.py` (`ndomains` default and docstring in `total_fluxes` and the three helpers)
- Modify: `dev/ULTRASPHERICAL.md`, `dev/JAX_PERF.md` (results)
- Test: `tests/test_jax_driver.py`

**Interfaces:**
- Consumes: `total_fluxes(..., ndomains, scheme)`, `solve_modes(..., ndomains, scheme)`, `sum_fluxes`.
- Produces: `total_fluxes(..., ndomains=ND)` with ND the chosen count.

- [ ] **Step 1: Extend `cluster/bench_resolution.py`.** Drop the `domains()` patch (the driver now takes `ndomains`). Configurations are `(scheme, nd)`: reference `("collocation", 150)`, candidates `("ultraspherical", 32/16/8/4)` and `("collocation", 32)` (current default, for timing). Part 1 (reference case, timed): as now, per configuration, plus `rel vs A30`. Part 2 (survey `SUBSET`): run the reference `total_fluxes` once per orbit; for each candidate re-solve exactly the reference's mode table (`fx.table` l, m, k, n) with `solve_modes(..., ndomains=nd, scheme=scheme)` (chunked by `_chunk_size` as `_bucketed_solve` does) and `sum_fluxes`; report max relative difference over the 6 channel/side totals. This is the like-for-like comparison, including orbits that hit `max_modes`. Also report each candidate's own `total_fluxes` mode count and status. CLI: `--configs` as `scheme:nd` strings. Local smoke run with `--limit 1 --reps 1 --tol 1e-2` on CPU before handing off.

- [ ] **Step 2: User syncs and runs** `sbatch cluster/slurm_resolution.sbatch` on the H200; user syncs `resolution-<tag>.txt` back.

- [ ] **Step 3: Choose ND**: smallest candidate with every survey orbit at <= 1e-9 in Part 2. Record the table in `dev/ULTRASPHERICAL.md` and the timing line in `dev/JAX_PERF.md`. If no candidate below 32 passes, stop and report to the user.

- [ ] **Step 4: Write failing test**: update `test_total_fluxes_passes_ndomains_to_solver` so the default case expects ND; add `test_ultraspherical_default_matches_reference`: 40 modes of the reference orbit (0.9, 8, 0.3, 0.7), l <= 6, |k| <= 3, |n| <= 6, via `solve_modes` at `(ultraspherical, ND)` and `(collocation, 150)`, `sum_fluxes` totals within 1e-9 relative on all 6 channel/side entries.

- [ ] **Step 5: Run, expect FAIL** on the default assertion; set `ndomains=ND` defaults and docstring ("ND is within <measured> of 150-domain collocation on 14 orbits; dev/ULTRASPHERICAL.md"); run `tests/test_jax_driver.py` then `tests -q`, expect PASS.

- [ ] **Step 6: Commit** (on go-ahead): message "Default total_fluxes to ND radial subdomains with the ultraspherical scheme".
