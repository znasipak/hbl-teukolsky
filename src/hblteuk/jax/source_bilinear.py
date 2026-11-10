# Generic-orbit source integral as bilinear forms. Each term of
# source._source_coefficients factors as
#     (orbit-only function of r and theta) x (radial mode factor) x (polar mode factor)
# because Kt depends on r only and L1, L2, L2p on theta only, so nothing is
# multiplied out. Per mode the integral is then sum_t u_t^T K_t v_t, a set of
# matrix products. Design and measurements: dev/BILINEAR_SOURCE.md,
# docs/superpowers/specs/2026-11-10-bilinear-source-design.md.
import jax.numpy as jnp

from .source import SQRT2

# (coefficient c in A_c, radial factor, polar factor), one per kernel
TERMS = ((0, "1", "L1L2S"), (0, "1", "ann"), (0, "1", "L2S"),
         (0, "-iKt/D", "L2S"), (0, "Kt/D", "S"), (0, "quad", "S"),
         (0, "iKt", "S"), (1, "1", "L2S"), (1, "1", "S"), (1, "iKt/D", "S"),
         (1, "1", "S"), (2, "1", "S"))


def _orbit(a, En, Lz, rp, urp, sth, cth, uthp):
    Delta = rp**2 - 2 * rp + a**2
    rho = -1.0 / (rp - 1j * a * cth)
    rhob = jnp.conj(rho)
    Sigma = 1.0 / (rho * rhob)
    Cr = (En * (rp**2 + a**2) - a * Lz + urp) / (2 * Sigma)
    Cth = rho * (1j * sth * (a * En - Lz / sth**2) + uthp) / SQRT2
    return Delta, rho, rhob, Cr * Cr, Cr * Cth, Cth * Cth


def kernels(a, En, Lz, rp, urp, th, uthp):
    """Orbit-only kernels K_t on the torus grid, (12, nq, nth), in TERMS order.
    rp, urp: (nq,); th, uthp: (nth,)."""
    rp, urp = rp[:, None], urp[:, None]
    th, uthp = th[None, :], uthp[None, :]
    sth = jnp.sin(th)
    D, r, rb, Cnn, Cnm, Cmm = _orbit(a, En, Lz, rp, urp, sth, jnp.cos(th),
                                     uthp)
    d1 = 1.0 / (SQRT2 * D)
    d2 = d1**2
    K = [-r**-3 / rb * d2 * Cnn,                         # Ann0: L1L2S
         -r**-2 / rb * d2 * Cnn,                         # Ann0: rest
         r**-3 * (r + rb) * d1 * Cnm,                    # Anmbar0
         r**-3 * d1 * Cnm,                               #   (-i Kt/D) L2S
         r**-3 * (r - rb) * a * sth * d1 * Cnm,          #   (Kt/D) S
         r**-3 * rb * d2 * Cmm,                          # Ambarmbar0
         r**-3 * rb * d2 * D * r * Cmm,                  #   i Kt rho Delta
         r**-3 * d1 * Cnm,                               # A1: Anmbar1 L2S
         r**-3 * d1 * 1j * (r - rb) * a * sth * Cnm,     #   Anmbar1 S
         0.5 * r**-3 * rb * Cmm,                         #   Ambarmbar1 Kt
         -0.5 * r**-3 * rb * r * Cmm,                    #   Ambarmbar1 rho
         -0.25 * r**-3 * rb * Cmm]                       # A2
    shape = (rp.shape[0], th.shape[1])
    return jnp.stack([jnp.broadcast_to(k, shape) for k in K])


def radial_factors(a, m, om, rp):
    """Radial mode factors by TERMS name; m, om broadcast against rp."""
    D = rp**2 - 2 * rp + a**2
    Kt = (rp**2 + a**2) * om - m * a
    return {"1": jnp.ones_like(Kt), "-iKt/D": -1j * Kt / D, "Kt/D": Kt / D,
            "quad": 0.5 * Kt**2 + 1j * Kt * (1 - rp) + 1j * rp * om * D,
            "iKt": 1j * Kt, "iKt/D": 1j * Kt / D}


def polar_factors(a, m, om, th, S, dS, d2S):
    """Polar mode factors by TERMS name: the L-operator algebra of
    _source_coefficients; m, om broadcast against th and S."""
    sth, cth = jnp.sin(th), jnp.cos(th)
    L1 = -m / sth + a * om * sth + cth / sth
    L2 = -m / sth + a * om * sth + 2 * cth / sth
    L2p = m * cth / sth**2 + a * om * cth - 2.0 / sth**2
    L2S = dS + L2 * S
    L1L2S = d2S + L1 * dS + L2p * S + L2 * dS + L1 * L2 * S
    ann = (3j * a * sth * L1 * S + 3j * a * cth * S + 2j * a * sth * dS
           - 1j * a * sth * L2 * S)
    return {"L1L2S": L1L2S, "ann": ann, "L2S": L2S, "S": S}
