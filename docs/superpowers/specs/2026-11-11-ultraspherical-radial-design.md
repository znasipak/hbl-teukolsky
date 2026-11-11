# Ultraspherical radial solve: design

Date: 2026-11-11. Branch: `kerr-jax`. Prototype and measurements:
`dev/ultraspherical.py`, `dev/us_solver.py`, `dev/ULTRASPHERICAL.md`.

## Intent

After the bilinear source (5abb41d), the radial solve is the largest stage of a
generic flux run: 0.66 s of 1.06 s on H200 for the reference case
(a, p, e, x) = (0.9, 8, 0.3, 0.7), tol 1e-6 (target ~0.5 s). Each mode does
three dense LU solves (flipped-spin march, stable march, TS map) per domain, at
32 domains x 32 nodes, so cost scales with D n^3.

The ultraspherical (Olver-Townsend) method writes each domain's ODE rows in
coefficient space. The hyperboloidal Teukolsky coefficients are polynomials in
sigma (P cubic, Q quadratic, U linear), so the rows are exact and banded. On
CPU it reached the accuracy floor with far fewer domains than collocation: at
8 x 32, worst flux-carrying mode 8.5e-11 against collocation's 2.1e-7. Fewer
domains with the same dense LU is the cheap win; a banded solver is not needed
for it.

## Goals

1. Ultraspherical rows available in the radial march and the TS map, with
   collocation still selectable.
2. A lower domain count for `total_fluxes`, chosen from H200 data, that keeps
   flux totals within 1e-9 relative (user: "1e-9 is fine for now") of
   collocation at 150 domains, the best-resolved reference, on every survey
   orbit (`cluster/bench_resolution.py` subset). Orbits that hit `max_modes`
   are compared on the same truncated mode set. (Reference choice: mine.)
3. Faster: reference-case radial from 0.66 s toward ~0.17 s (estimate for 8
   domains, 4x fewer LUs), measured on H200.

## Non-goals

- Batched banded solver (only pays off for few, large domains, e.g. 2 x 64).
- Removing the TS map via the s=+2 horizon route (user: separate, later).
- Source, angular or driver changes.

## Anti-goals

- Flux totals moving by more than 1e-9 relative on any survey orbit.
- Breaking the eccentric or spherical paths, which share the radial solve.
- A polynomial-degree assumption that silently breaks if the ODE changes.

## Constraints

- FP64; jitted; same batching, chunking and padding behaviour.
- Same unknowns (Chebyshev-T coefficients) and the same boundary rows, 2x2
  transfer and evaluation, so everything downstream is unchanged.
- Consistency across orbit types (user, earlier): one `scheme` argument with the
  same meaning wherever the radial solve is reachable.

## Decisions

User's:
- Accuracy bar 1e-9 relative on flux totals.
- s=+2 horizon route stays separate.

Mine (flagged):
- Interface: `scheme="ultraspherical" | "collocation"` on
  `solve_homogeneous_batch`, passed through `solve_mode_table`,
  `solve_spherical_mode_table`, `solve_generic_mode_table`, `solve_modes` and
  `total_fluxes`. Default `"ultraspherical"` everywhere once validated: at the
  pipeline default of 150 domains it matches collocation to 2e-13 (CPU), so
  only the driver's domain default changes behaviour materially.
- Domain default in `total_fluxes` set from the H200 scan (candidates 16, 8, 4
  at 32 nodes), the smallest that holds 1e-9 on every survey orbit. Pipeline
  functions keep `ndomains=150`.
- Coefficient polynomials: P, Q, U sampled at 4 points per domain and fitted
  exactly to a cubic in x (as the prototype). A test checks a 5th point
  against `teuk_pqu`, so a future higher-degree term fails loudly.
- TS map uses the same rows with its two Dirichlet rows, as the prototype.
- Residual error estimate stays the nodal residual at collocation nodes, so
  `errors` keep their meaning.
- Chunk memory model unchanged: matrices keep their (n, n) shape.

## Tests

- Operators: banded (lower 3, upper 7), cos and Airy solutions, cubic
  coefficients vs collocation (port of `dev/test_ultraspherical.py`).
- Cubic-fit guard on `teuk_pqu` at both spins.
- March and TS map: ultraspherical vs collocation coefficients at 150 x 32 on
  the stable branch, to the existing parallel-march tolerances.
- Existing pybhpt reference suites pass with the new default.
- Fewer domains: on reference-case modes, ultraspherical at the chosen domain
  count vs collocation at 150 domains, flux totals within 1e-9.
- H200: `cluster/bench_resolution.py` with `scheme` over domain counts (survey
  accuracy and reference-case time) before fixing the default.

## Left to the builder

- Where the operator tables live (`chebyshev.py` or a new module) and caching.
- Exact test tolerances within the goals above.
