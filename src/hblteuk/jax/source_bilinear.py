# Generic-orbit source integral as bilinear forms. Each term of
# source._source_coefficients factors as
#     (orbit-only function of r and theta) x (radial mode factor) x (polar mode factor)
# because Kt depends on r only and L1, L2, L2p on theta only, so nothing is
# multiplied out. Per mode the integral is then sum_t u_t^T K_t v_t, a set of
# matrix products. Design and measurements: dev/BILINEAR_SOURCE.md,
# docs/superpowers/specs/2026-11-10-bilinear-source-design.md.
import functools

import jax
import jax.numpy as jnp

from .source import (SQRT2, SourceAmplitudes, _Z_sigma, _Z_sigma_deriv,
                     _Z_sigma_deriv2)

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


@functools.partial(jax.jit, static_argnums=(0,))
def _integrate_generic_bilinear(s, kappa, a, En, Lz, gamma_mino, radial_arrs,
                                polar_arrs, lam, m, k, n, omega, psi_arrs):
    """Drop-in for source._integrate_generic_core (same arguments, same
    SourceAmplitudes). The theta sum is 12 matrix products K_t @ v_t, shared
    by Up and In; bin extraction, q_r tail and q_theta even/odd estimates are
    the library's. The cancellation floor bounds mean |J| by sum_t |u||K||v|,
    so it is never smaller than the library's."""
    rp, urp, tr, phir, qr = radial_arrs
    thp, uthp, tth, phith, qth = polar_arrs
    (psi_up, dpsi_up, d2psi_up, psi_in, dpsi_in, d2psi_in,
     S, dS, d2S) = psi_arrs
    nq, nth = rp.shape[0], thp.shape[0]

    norm_up = jnp.max(jnp.abs(jnp.stack([psi_up, dpsi_up, d2psi_up])),
                      axis=(0, 2))
    norm_in = jnp.max(jnp.abs(jnp.stack([psi_in, dpsi_in, d2psi_in])),
                      axis=(0, 2))
    psi_up, dpsi_up, d2psi_up = (x / norm_up[:, None]
                                 for x in (psi_up, dpsi_up, d2psi_up))
    psi_in, dpsi_in, d2psi_in = (x / norm_in[:, None]
                                 for x in (psi_in, dpsi_in, d2psi_in))

    ma = m * a
    om = omega[:, None]
    sig = 2 * kappa / (rp - (1 - kappa))
    sigB = sig[None, :]
    Zs = _Z_sigma(sigB, kappa, s, ma[:, None], om)
    dZs = _Z_sigma_deriv(sigB, kappa, s, ma[:, None], om)
    d2Zs = _Z_sigma_deriv2(sigB, kappa, s, ma[:, None], om)
    gs = -sig**2 / (2 * kappa)
    dgs = -sig / kappa
    phase_r = jnp.exp(1j * (om * tr[None, :] - m[:, None] * phir[None, :]))
    pw = jnp.exp(1j * (om * tth[None, :] - m[:, None] * phith[None, :]
                       + k[:, None] * qth[None, :]))

    K = kernels(a, En, Lz, rp, urp, thp, uthp)                 # (T, nq, nth)
    R = radial_factors(a, m[:, None], om, rp[None, :])
    P = polar_factors(a, m[:, None], om, thp[None, :], S, dS, d2S)
    V = jnp.stack([P[pn] for _, _, pn in TERMS])               # (T, B, nth)
    Vw = V * pw[None]
    # even and odd theta columns separately: the q_theta check, no extra flops
    X_e = jnp.einsum("tij,tbj->tbi", K[:, :, 0::2], Vw[:, :, 0::2])
    X_o = jnp.einsum("tij,tbj->tbi", K[:, :, 1::2], Vw[:, :, 1::2])
    A = jnp.einsum("tij,tbj->tbi", jnp.abs(K), jnp.abs(V))     # |J| bound

    halves, absJ = [], 0.0
    for p0, p1, p2 in ((psi_up, dpsi_up, d2psi_up),
                       (psi_in, dpsi_in, d2psi_in)):
        rho = (Zs * p0,
               gs * dZs * p0 + gs * Zs * p1,
               gs * (dgs * dZs + gs * d2Zs) * p0
               + gs * (dgs * Zs + 2 * gs * dZs) * p1 + gs * gs * Zs * p2)
        U = jnp.stack([R[rn] * rho[c] for c, rn, _ in TERMS])  # (T, B, nq)
        Uph = U * phase_r[None]
        halves += [jnp.sum(Uph * X_e, axis=0), jnp.sum(Uph * X_o, axis=0)]
        absJ = absJ + jnp.einsum("tbi,tbi->b", jnp.abs(U), A) / nq
    Je_up, Jo_up, Je_in, Jo_in = halves

    # from here as _integrate_generic_core
    J = jnp.stack([(Je_up + Jo_up) / nth, (Je_in + Jo_in) / nth])
    spec = jnp.fft.ifft(J, axis=-1)
    nn = (jnp.mod(n, nq))[None, :, None]
    c = jnp.take_along_axis(spec, nn, axis=-1)[:, :, 0]
    freq = jnp.fft.fftfreq(nq, d=1.0 / nq)
    tail_r = jnp.max(jnp.where(jnp.abs(freq)[None, None, :] >= nq // 4,
                               jnp.abs(spec), 0.0), axis=-1)
    err_r = tail_r / jnp.abs(c)

    hv = jnp.stack([2 * Je_up / nth, 2 * Jo_up / nth,
                    2 * Je_in / nth, 2 * Jo_in / nth])
    ch = jnp.take_along_axis(jnp.fft.ifft(hv, axis=-1),
                             (jnp.mod(n, nq))[None, :, None], axis=-1)[:, :, 0]
    err_th = jnp.stack([0.5 * jnp.abs(1 - ch[1] / ch[0]),
                        0.5 * jnp.abs(1 - ch[3] / ch[2])])

    canc = 1e-15 / jnp.abs(c / (absJ / (2 * nth))[None, :])
    err = jnp.maximum(jnp.maximum(err_r, err_th), canc)

    W = psi_in[:, 0] * dpsi_up[:, 0] - psi_up[:, 0] * dpsi_in[:, 0]
    sig0, rp0 = sig[0], rp[0]
    rescale = (_Z_sigma(sig0, kappa, s, ma, omega)**2
               * (rp0**2 - 2 * rp0 + a**2)**(-1)
               * (-2 * kappa / (rp0 - (1 - kappa))**2))
    const_w = jnp.stack([norm_in, norm_up]) * rescale[None, :] * W[None, :]
    amps = -8 * jnp.pi * c / const_w / gamma_mino
    return SourceAmplitudes(amps.T, err.T)
