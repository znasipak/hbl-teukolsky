# Batched homogeneous radial Teukolsky solutions on hyperboloidal slices.
# Ports teuk.teuk_sys / boundary conditions / HyperboloidalTeukolskySolver's
# radiative branch to a batched fixed-domain solve. See dev/PHASE2.md.
import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .collocation import solve_fixed_multidomain_batch


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


@functools.partial(jax.jit, static_argnums=(0, 1, 6))
def _solve_core(s, kappa, lam, ma, omega, subs, n):
    subs_up, subs_in = subs

    def pqu(z):
        return teuk_pqu(z, kappa, s, lam, ma, omega)

    cu, eu = solve_fixed_multidomain_batch(
        pqu, boundary_conditions_up(kappa, s, lam, ma, omega), subs_up, n=n)
    ci, ei = solve_fixed_multidomain_batch(
        pqu, boundary_conditions_in(kappa, s, lam, ma, omega), subs_in, n=n)
    return HomogeneousBatch(cu, ci, eu, ei)


def solve_homogeneous_batch(s, kappa, lam, ma, omega, subdomains_up,
                            subdomains_in, n=32):
    """Batched Up and In homogeneous solutions on shared subdomain structures.

    s: spin weight (python int); kappa = sqrt(1 - a^2) (python float);
    lam, ma, omega: (B,) eigenvalues, m*a, mode frequencies. subdomains_up
    marches from sigma = 0 (ascending), subdomains_in from sigma = 1
    (descending) — the layouts the grid generator's optimize() produces.

    Radiative modes only: the omega = 0 static branch uses a different ODE
    and stays on the CPU path (GPU plan Phase 2 scope).

    Returns HomogeneousBatch; evaluate with chebyshev.evaluate_multidomain.
    """
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

    return _solve_core(int(s), float(kappa), lam, ma, omega,
                       (jnp.asarray(subs_up), jnp.asarray(subs_in)), int(n))
