# Batched fixed-structure multi-domain spectral collocation: all subdomain
# systems solved in parallel for two unit boundary data, then a lax.scan over
# the 2x2 edge-value transfer only (dev/JAX_PERF.md, Task 6). Mirrors the CPU
# CollocationODEFixedMultiDomainFixedStepSolver + CoefficientGenerator
# (collocode.py); equivalence argument and design notes: dev/PHASE2.md.
import functools

import jax
import jax.numpy as jnp

from .chebyshev import collocation_matrices


def _step(pqu_fn, mats, carry, xs):
    psi, dpsi = carry                      # (B,) each: value and d/dsigma
    z_nodes, dzdx = xs                     # (n,), scalar

    P, Q, U = pqu_fn(z_nodes)              # (B, n) each
    M = (P[:, :, None] * mats["T2"]
         + (dzdx * Q)[:, :, None] * mats["T1"]
         + (dzdx**2 * U)[:, :, None] * mats["T0"])   # (B, n, n)
    M_ode = M
    M = M.at[:, 0, :].set(mats["bc_left"][0])
    M = M.at[:, 1, :].set(mats["bc_left"][1])

    # x-frame boundary data; norm guards overflow across long marches
    y0 = psi
    y1 = dpsi * dzdx
    norm = jnp.where(jnp.abs(y0) > 1, y0, 1.0 + 0j)
    rhs = jnp.zeros(M.shape[:2], dtype=M.dtype)
    rhs = rhs.at[:, 0].set(y0 / norm)
    rhs = rhs.at[:, 1].set(y1 / norm)

    c = jnp.linalg.solve(M, rhs[:, :, None])[:, :, 0]  # (B, n)
    # relative residual: absolute one scales with |c|, which can legitimately
    # grow ~1e50 across long marches (high-l In solutions)
    err = (jnp.linalg.norm(M_ode @ c[:, :, None], axis=(1, 2))
           / jnp.linalg.norm(c, axis=1))
    c = c * norm[:, None]

    psi_end = c @ mats["bc_right"][0]
    dpsi_end = (c @ mats["bc_right"][1]) / dzdx        # back to d/dsigma
    return (psi_end, dpsi_end), (c, err)


# not jitted here: pqu_fn is a per-call closure over traced batch params;
# jit is applied once at the driver level (jax/teukolsky.py)
def _march(pqu_fn, mats, y0, z_nodes_all, dzdx_all):
    step = functools.partial(_step, pqu_fn, mats)
    _, (coeffs, errors) = jax.lax.scan(step, y0, (z_nodes_all, dzdx_all))
    return coeffs.swapaxes(0, 1), errors.swapaxes(0, 1)  # (B, D, n), (B, D)


def _setup(subdomains, n, chtype):
    subdomains = jnp.asarray(subdomains, dtype=jnp.float64)
    mats_np = collocation_matrices(n, chtype)
    mats = {k: jnp.asarray(mats_np[k]) for k in ("T0", "T1", "T2",
                                                 "bc_left", "bc_right")}
    nodes = jnp.asarray(mats_np["nodes"])
    z0 = subdomains[:-1]
    z1 = subdomains[1:]
    dzdx_all = 0.5 * (z1 - z0)                                   # (D,)
    z_nodes_all = 0.5 * ((z1 - z0)[:, None] * nodes[None, :]
                         + (z1 + z0)[:, None])                   # (D, n)
    return mats, z_nodes_all, dzdx_all


def _transfer(carry, xs):
    # emit this domain's start state, advance it to the domain's end
    psi, dpsi = carry                      # (B,) each, sigma frame
    R, dzdx = xs                           # (B, 2, 2), scalar
    y1 = dpsi * dzdx
    psi_end = R[:, 0, 0] * psi + R[:, 0, 1] * y1
    dpsi_end = R[:, 1, 0] * psi + R[:, 1, 1] * y1
    return (psi_end, dpsi_end), (psi, dpsi)


def solve_fixed_multidomain_batch(pqu_fn, y0, subdomains, n=32, chtype=1):
    """Solve P psi'' + Q psi' + U psi = 0 for a batch of modes on a shared
    fixed subdomain structure.

    pqu_fn(z_nodes (n,)) -> (P, Q, U) each (B, n), closed over batch params.
    y0: (psi, dpsi/dsigma) at subdomains[0], each (B,) complex.
    subdomains: (ndomains+1,) monotone (either direction); the march starts
    at subdomains[0] where y0 is imposed, exactly like the CPU solver.

    Returns (coeffs (B, ndomains, n) complex, errors (B, ndomains)); errors
    are relative per-domain residuals ||M_ode @ c|| / ||c|| (the CPU
    CoefficientGenerator metric, normalized by the solution magnitude).
    """
    mats, z_nodes_all, dzdx_all = _setup(subdomains, n, chtype)
    P, Q, U = (x.swapaxes(0, 1) for x in jax.vmap(pqu_fn)(z_nodes_all))
    dz = dzdx_all[None, :, None, None]                       # (1, D, 1, 1)
    M = (P[..., None] * mats["T2"] + dz * Q[..., None] * mats["T1"]
         + dz**2 * U[..., None] * mats["T0"])                # (B, D, n, n)
    M = M.at[:, :, 0, :].set(mats["bc_left"][0])
    M = M.at[:, :, 1, :].set(mats["bc_left"][1])

    # u (psi=1, y1=0) and v (psi=0, y1=1) per domain, all domains at once
    rhs = jnp.zeros(M.shape[:3] + (2,), dtype=M.dtype)
    rhs = rhs.at[:, :, 0, 0].set(1.0).at[:, :, 1, 1].set(1.0)
    X = jnp.linalg.solve(M, rhs)                             # (B, D, n, 2)
    del M

    # edge values [psi_end, dpsi_end/dsigma] of u and v: R (B, D, 2, 2)
    br = mats["bc_right"]                                    # (2, n)
    R = jnp.einsum("bdnj,in->bdij", X, br)
    R = R.at[:, :, 1, :].divide(dzdx_all[None, :, None])

    y0 = (jnp.asarray(y0[0], dtype=jnp.complex128),
          jnp.asarray(y0[1], dtype=jnp.complex128))
    _, (psi, dpsi) = jax.lax.scan(
        _transfer, y0, (R.swapaxes(0, 1), dzdx_all))         # (D, B) each
    psi, dpsi = psi.T, dpsi.T                                # (B, D)
    c = (psi[..., None] * X[..., 0]
         + (dpsi * dzdx_all[None, :])[..., None] * X[..., 1])  # (B, D, n)

    dzv = dzdx_all[None, :, None]
    res = (P * (c @ mats["T2"].T) + dzv * Q * (c @ mats["T1"].T)
           + dzv**2 * U * (c @ mats["T0"].T))
    err = jnp.linalg.norm(res, axis=-1) / jnp.linalg.norm(c, axis=-1)
    return c, err


def _solve_fixed_multidomain_batch_reference(pqu_fn, y0, subdomains, n=32,
                                             chtype=1):
    # sequential per-domain march (pre-Task 6); equivalence-test reference
    mats, z_nodes_all, dzdx_all = _setup(subdomains, n, chtype)

    y0 = (jnp.asarray(y0[0], dtype=jnp.complex128),
          jnp.asarray(y0[1], dtype=jnp.complex128))
    return _march(pqu_fn, mats, y0, z_nodes_all, dzdx_all)
