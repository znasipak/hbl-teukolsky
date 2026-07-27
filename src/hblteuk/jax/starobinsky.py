# Batched Teukolsky-Starobinsky branch stabilization.
#
# The radial march is unstable on exactly one branch per spin (s=-2 Up, s=+2
# In): there the target solution is subdominant, so roundoff seeding the
# growing solution amplifies ~7 orders over a long march. Solve that branch at
# the opposite spin, where it is dominant, and map it back with the TS relation
#
#     Psi_target = f_TS Psi_src + g_TS Psi_src'
#
# built as an ODE solution rather than a fit: per subdomain, assemble the
# target-spin collocation system and replace two rows with Dirichlet conditions
# valued from the TS relation. Fitting instead would force psi' to come from
# differentiating an interpolant, amplifying roundoff by ~2 n^2 / h and making
# the error high-frequency, which the oscillatory source integral passes
# straight through. See dev/ACCURACY_DEBUG_PLAN.md and hblteuk.teuk.
#
# The f_TS/g_TS expressions are reused directly from the CPU module: they are
# pure arithmetic in (sigma, kappa, lam, ma, omega), so they evaluate on JAX
# arrays unchanged (verified to 1e-13 against the numpy path).
import jax.numpy as jnp

from ..teuk import (hbl_teukolsky_starobinsky_system_minus_to_plus_2,
                    hbl_teukolsky_starobinsky_system_plus_to_minus_2)
from .chebyshev import collocation_matrices
from .teukolsky import teuk_pqu

# which branch each spin must borrow, and from where
_SPEC = {-2: ("up", 2, hbl_teukolsky_starobinsky_system_plus_to_minus_2),
         2: ("in", -2, hbl_teukolsky_starobinsky_system_minus_to_plus_2)}


def unstable_branch(s):
    """'up' for s=-2, 'in' for s=+2; None if this spin needs no stabilization."""
    return _SPEC[s][0] if s in _SPEC else None


def flipped_spin(s):
    """The spin whose march is stable for the branch that s solves badly."""
    return _SPEC[s][1] if s in _SPEC else None


def ts_solve_batch(coeffs_src, subdomains, kappa, s_target, lam, ma, omega,
                   n=32, chtype=1, renormalize=True):
    """Map a batched opposite-spin solution to s_target via Teukolsky-Starobinsky.

    coeffs_src: (B, D, n) Chebyshev coefficients of the *stable* solve, on the
    branch that s_target marches badly. subdomains: (D+1,). lam: (B,) the
    ordinary eigenvalue of s_target (lambda_CH follows by spin invariance).
    Returns (B, D, n) coefficients of the s_target solution.

    Every subdomain is independent here -- unlike the marching solver there is
    no scan, so this batches flat.
    """
    if s_target not in _SPEC:
        raise ValueError(f"no TS stabilization defined for s={s_target}")
    transform_sys = _SPEC[s_target][2]

    mats = collocation_matrices(n, chtype)
    T0 = jnp.asarray(mats["T0"])
    T1 = jnp.asarray(mats["T1"])
    T2 = jnp.asarray(mats["T2"])
    nodes = jnp.asarray(mats["nodes"])
    bcL, dbcL = jnp.asarray(mats["bc_left"])       # value, d/dx rows at x=-1
    bcR, dbcR = jnp.asarray(mats["bc_right"])      # value, d/dx rows at x=+1

    coeffs_src = jnp.asarray(coeffs_src)
    sub = jnp.asarray(subdomains, dtype=jnp.float64)
    lam = jnp.asarray(lam)
    ma = jnp.asarray(ma)
    omega = jnp.asarray(omega)
    lam_ch = lam + s_target*(s_target + 1)         # spin-invariant eigenvalue

    z0, z1 = sub[:-1], sub[1:]                     # (D,)
    dzdx = 0.5*(z1 - z0)                           # (D,)
    # physical nodes per subdomain, (D, n)
    z = 0.5*(z1 - z0)[:, None]*nodes[None, :] + 0.5*(z1 + z0)[:, None]

    # target-spin ODE coefficients on those nodes: (B, D, n)
    Dm = z.shape[0]
    P, Q, U = teuk_pqu(z.reshape(-1), kappa, s_target, lam, ma, omega)
    P = P.reshape(-1, Dm, n)
    Q = Q.reshape(-1, Dm, n)
    U = U.reshape(-1, Dm, n)

    # source solution and derivative at both subdomain endpoints: (B, D)
    psiL = jnp.einsum("i,bdi->bd", bcL, coeffs_src)
    psiR = jnp.einsum("i,bdi->bd", bcR, coeffs_src)
    dpsiL = jnp.einsum("i,bdi->bd", dbcL, coeffs_src)/dzdx[None, :]
    dpsiR = jnp.einsum("i,bdi->bd", dbcR, coeffs_src)/dzdx[None, :]

    # f_TS, g_TS at the endpoints, broadcast over the batch: (B, D)
    def fg(zz):
        return transform_sys(zz[None, :], kappa, lam_ch[:, None],
                             ma[:, None], omega[:, None])

    fL, gL = fg(z0)
    fR, gR = fg(z1)
    tsL = fL*psiL + gL*dpsiL
    tsR = fR*psiR + gR*dpsiR

    # collocation system, two rows replaced by the TS Dirichlet conditions
    M = (P[..., None]*T2[None, None]
         + (dzdx[None, :, None]*Q)[..., None]*T1[None, None]
         + (dzdx[None, :, None]**2*U)[..., None]*T0[None, None])
    M = M.at[:, :, 0, :].set(bcL)
    M = M.at[:, :, 1, :].set(bcR)
    rhs = jnp.zeros(M.shape[:3], dtype=M.dtype)
    rhs = rhs.at[:, :, 0].set(tsL)
    rhs = rhs.at[:, :, 1].set(tsR)

    out = jnp.linalg.solve(M, rhs[..., None])[..., 0]

    if renormalize:
        # the transform carries the Starobinsky constant, which does not cancel
        # out of the amplitudes (Z^Up is built from Psi_In over the Wronskian,
        # so it scales as 1/c). Restore the source solve's boundary value.
        src_b = jnp.einsum("i,bi->b", bcL, coeffs_src[:, 0])
        new_b = jnp.einsum("i,bi->b", bcL, out[:, 0])
        out = out*(src_b/new_b)[:, None, None]
    return out
