# Batched per-mode gravitational-wave fluxes from Teukolsky amplitudes
# (s = -2). Formulas follow pybhpt cpp/src/fluxes.cpp (flux_mode,
# horizonConstant, horizonAmplitude); conventions validated numerically
# against pybhpt in tests/test_jax_fluxes.py. See dev/PHASE5A.md.
from typing import NamedTuple

import jax
import jax.numpy as jnp

ZERO_FREQ_TOL = 1e-8


def teukolsky_starobinsky_abs_C_squared(m, a, omega, lam):
    """|C|^2 in the lam = spheroidal-eigenvalue convention shared by
    hblteuk.swsh and pybhpt (lam_CH = lam + 2 for s = -2)."""
    maw = m * a * omega
    aw2 = (a * omega) ** 2
    return ((lam + 2) ** 2 + 4 * maw - 4 * aw2) \
        * (lam**2 + 36 * maw - 36 * aw2) \
        + (2 * lam + 3) * (96 * aw2 - 48 * maw) \
        + 144 * omega**2 * (1 - a**2)


def horizon_constant(m, a, omega, lam):
    """Ratio Edot^H / (|Z_In|^2 / 4 pi w^2) (pybhpt horizonConstant)."""
    rplus = 1 + jnp.sqrt(1 - a**2)
    kappa_h = omega - m * a / (2 * rplus)
    eps = jnp.sqrt(1 - a**2) / (4 * rplus)
    C2 = teukolsky_starobinsky_abs_C_squared(m, a, omega, lam)
    return (256 * (2 * rplus) ** 5 * kappa_h
            * (kappa_h**2 + 4 * eps**2) * (kappa_h**2 + 16 * eps**2)
            * omega**3 / C2)


class FluxModes(NamedTuple):
    Edot: jax.Array   # (B, 2): [infinity, horizon]
    Ldot: jax.Array   # (B, 2)


def flux_modes(amplitudes, m, omega, eigenvalues, a,
               zero_freq_tol=ZERO_FREQ_TOL):
    """Per-mode energy and angular-momentum fluxes (s = -2).

    amplitudes (B, 2) in the pipeline's [In, Up] order; m, omega,
    eigenvalues (B,); a the spin. Fluxes are per individual (l, m, k, n)
    mode with no -m doubling (grids that store explicit -m modes just sum).
    Static (|omega| < zero_freq_tol) modes contribute zero.

    Edot^inf = |Z_Up|^2 / (4 pi w^2); Edot^hor = alpha |Z_In|^2 / (4 pi w^2);
    Ldot = (m / w) Edot. Carter flux is (2 M_mkn / w) Edot with the Carter
    frequency combination M_mkn — deferred until the generic-orbit
    increments provide those geodesic constants (identically ~0 on
    equatorial orbits).
    """
    amplitudes = jnp.asarray(amplitudes)
    m = jnp.asarray(m, dtype=jnp.float64)
    omega = jnp.asarray(omega, dtype=jnp.float64)
    lam = jnp.asarray(eigenvalues, dtype=jnp.float64)

    radiative = jnp.abs(omega) > zero_freq_tol
    w = jnp.where(radiative, omega, 1.0)  # guarded; masked to zero below

    pref = 1.0 / (4 * jnp.pi * w**2)
    e_inf = pref * jnp.abs(amplitudes[:, 1]) ** 2
    e_hor = horizon_constant(m, a, w, lam) * pref * jnp.abs(amplitudes[:, 0]) ** 2
    Edot = jnp.where(radiative[:, None], jnp.stack([e_inf, e_hor], axis=1), 0.0)
    Ldot = jnp.where(radiative[:, None], (m / w)[:, None] * Edot, 0.0)
    return FluxModes(Edot, Ldot)
