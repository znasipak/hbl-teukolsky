# Batched homogeneous radial Teukolsky solutions on hyperboloidal slices.
# Ports teuk.teuk_sys / boundary conditions / HyperboloidalTeukolskySolver's
# radiative branch to a batched fixed-domain solve. See dev/PHASE2.md.
import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .collocation import solve_fixed_multidomain_batch
from .ultraspherical import check_scheme


def teuk_pqu(sigma, kappa, s, lam, ma, omega):
    """Radial ODE coefficients P, Q, U; sigma (n,), mode params (B,) -> (B, n).
    Same expressions as teuk.teuk_sys."""
    sig = sigma[None, :]
    lam = lam[:, None]
    ma = ma[:, None]
    om = omega[:, None]
    Kt = -((1 + s) * kappa + 1j * ma - 2j * (1 + kappa) * om) / kappa
    P = (sig**2 * (1 - sig)) + 0j * om
    Q = (-4j * kappa * om + 2 * sig * (1 + s - 2j * om * (1 - kappa))
         + sig**2 * (Kt + 4j * om - 2))
    U = (-2j * om * (2 * kappa * Kt + (1 + 2 * s) * (1 + kappa))
         + (1 - 4j * om) * Kt * sig - lam)
    return P, Q, U


def _bc_ratio(sigma_b, kappa, s, lam, ma, omega):
    # regular-singular boundary condition: dpsi/psi = -U/Q at sigma_b in {0, 1}
    _, Q, U = teuk_pqu(jnp.array([sigma_b]), kappa, s, lam, ma, omega)
    return -U[:, 0] / Q[:, 0]


def boundary_conditions_up(kappa, s, lam, ma, omega):
    y0 = jnp.ones_like(omega, dtype=jnp.complex128)
    return y0, _bc_ratio(0.0, kappa, s, lam, ma, omega) * y0


def boundary_conditions_in(kappa, s, lam, ma, omega):
    # horizon-side scale matches teuk.py (HyperboloidalTeukolskySolver.__call__)
    horizon = 1 + kappa
    y0 = 2 * kappa * jnp.exp(-1j * (2 * omega - 0.5 * ma / horizon)
                             * (1 + kappa + 2 * jnp.log(kappa)))
    return y0, _bc_ratio(1.0, kappa, s, lam, ma, omega) * y0


class HomogeneousBatch(NamedTuple):
    coeffs_up: jax.Array   # (B, D_up, n) complex
    coeffs_in: jax.Array   # (B, D_in, n) complex
    errors_up: jax.Array   # (B, D_up) per-domain solve residuals
    errors_in: jax.Array   # (B, D_in)


@functools.partial(jax.jit, static_argnums=(0, 1, 6, 7, 8))
def _solve_branch(s, kappa, lam, ma, omega, subs, n, branch,
                  scheme="ultraspherical"):
    # one family's march: "up" from sigma = 0, "in" from sigma = 1
    def pqu(z):
        return teuk_pqu(z, kappa, s, lam, ma, omega)

    bc = boundary_conditions_up if branch == "up" else boundary_conditions_in
    return solve_fixed_multidomain_batch(pqu, bc(kappa, s, lam, ma, omega),
                                         subs, n=n, scheme=scheme)


def _solve_core(s, kappa, lam, ma, omega, subs, n, scheme="ultraspherical"):
    subs_up, subs_in = subs
    cu, eu = _solve_branch(s, kappa, lam, ma, omega, subs_up, n, "up", scheme)
    ci, ei = _solve_branch(s, kappa, lam, ma, omega, subs_in, n, "in", scheme)
    return HomogeneousBatch(cu, ci, eu, ei)


def solve_homogeneous_batch(s, kappa, lam, ma, omega, subdomains_up,
                            subdomains_in, n=32, stabilize=True,
                            scheme="ultraspherical"):
    """Batched Up and In homogeneous solutions on shared subdomain structures.

    s: spin weight (python int); kappa = sqrt(1 - a^2) (python float);
    lam, ma, omega: (B,) eigenvalues, m*a, mode frequencies. subdomains_up
    marches from sigma = 0 (ascending), subdomains_in from sigma = 1
    (descending) — the layouts the grid generator's optimize() produces.

    Radiative modes only: the omega = 0 static branch uses a different ODE
    and stays on the CPU path (GPU plan Phase 2 scope).

    stabilize=True replaces the branch whose march is unstable at this spin
    (Up for s=-2, In for s=+2) by a solve at the opposite spin mapped back
    with Teukolsky-Starobinsky — removing a ~1e-8 error floor. The stable
    branch is untouched, as are other spins. Note the returned residuals for
    the stabilized branch belong to the opposite-spin march that actually ran.
    On by default since 2026-07-27; pass stabilize=False for the legacy path.

    scheme: "ultraspherical" or "collocation" ODE rows for every march and
    the TS map (collocation.solve_fixed_multidomain_batch).

    Returns HomogeneousBatch; evaluate with chebyshev.evaluate_multidomain.
    """
    check_scheme(scheme)
    lam = jnp.asarray(lam, dtype=jnp.complex128)
    ma = jnp.asarray(ma, dtype=jnp.float64)
    omega = jnp.asarray(omega, dtype=jnp.complex128)
    if np.any(np.abs(np.asarray(omega)) == 0):
        raise ValueError("static (omega = 0) modes are not supported by the "
                         "batched solver; use the CPU path")
    subs_up = np.asarray(subdomains_up, dtype=np.float64)
    subs_in = np.asarray(subdomains_in, dtype=np.float64)
    if not (abs(subs_up[0]) < 1e-10 and np.all(np.diff(subs_up) > 0)):
        raise ValueError("subdomains_up must ascend from sigma = 0")
    if not (abs(subs_in[0] - 1) < 1e-10 and np.all(np.diff(subs_in) < 0)):
        raise ValueError("subdomains_in must descend from sigma = 1")
    subs_up[0], subs_in[0] = 0.0, 1.0  # snap BC endpoints exactly

    subs = (jnp.asarray(subs_up), jnp.asarray(subs_in))
    from .starobinsky import flipped_spin, ts_solve_batch, unstable_branch
    branch = unstable_branch(int(s)) if stabilize else None
    if branch is None:
        return _solve_core(int(s), float(kappa), lam, ma, omega, subs, int(n),
                           scheme)

    # solve at the opposite spin, where this branch's march is stable, and map
    # back. lambda_CH = lambda_s + s(s+1) is spin invariant, so the flipped
    # eigenvalue is exact algebra -- no second spheroidal eigensolve.
    s_flip = flipped_spin(int(s))
    lam_flip = lam + (s*(s + 1) - s_flip*(s_flip + 1))
    # one march per family: the unstable branch at s_flip, the stable at s
    stable = "in" if branch == "up" else "up"
    sub = {"up": subs[0], "in": subs[1]}
    cf, ef = _solve_branch(int(s_flip), float(kappa), lam_flip, ma, omega,
                           sub[branch], int(n), branch, scheme)
    cs, es = _solve_branch(int(s), float(kappa), lam, ma, omega,
                           sub[stable], int(n), stable, scheme)
    subs_np = subs_up if branch == "up" else subs_in
    ct = ts_solve_batch(cf, subs_np, float(kappa), int(s), lam, ma, omega,
                        n=int(n), scheme=scheme)
    if branch == "up":
        return HomogeneousBatch(ct, cs, ef, es)
    return HomogeneousBatch(cs, ct, es, ef)
