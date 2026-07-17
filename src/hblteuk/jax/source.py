# Batched point-particle source integration (s = -2, equatorial eccentric).
# Ports source.gravitational_source_eccentric to a full-radial-period batch:
# the amplitude is the n-th Fourier coefficient of the phase-stripped
# integrand J(qr), extracted by uniform quadrature ("quad", CPU-equivalent)
# or FFT ("fft", with an aliasing error estimate from the spectrum tail).
# Derivation, conventions, and validation notes: dev/PHASE3.md.
import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..swsh import sphericalY_expansion, sphericalY_expansion_deriv
from .chebyshev import evaluate_multidomain

SQRT2 = float(np.sqrt(2.0))


# ---- host-side orbit sampling ------------------------------------------------

class OrbitSamples(NamedTuple):
    qr: np.ndarray      # (nq,) uniform on [0, 2 pi), qr = 0 at periapsis
    rp: np.ndarray      # (nq,) particle radius
    urp: np.ndarray     # (nq,) radial four-velocity (signed over the period)
    tr: np.ndarray      # (nq,) oscillatory t(qr) (odd)
    phir: np.ndarray    # (nq,) oscillatory phi(qr) (odd)
    constants: tuple    # (En, Lz, Q)
    gamma: float        # Mino time Gamma
    a: float
    kappa: float


def orbit_samples(orbit, nsamples=2**10):
    """Sample geodesic quantities on a full-period uniform qr grid (host)."""
    qr = 2 * np.pi * np.arange(nsamples) / nsamples
    rp = orbit.r(qr)
    urp = orbit.ur(qr)
    urp[0] = 0.0  # turning points: kill roundoff, as the CPU path does
    half = nsamples // 2
    if nsamples % 2 == 0:
        urp[half] = 0.0
    a = orbit.a
    return OrbitSamples(qr, rp, urp, orbit.tr(qr), orbit.phir(qr),
                        tuple(orbit.constants), float(orbit.mino_frequencies[0]),
                        a, float(np.sqrt(1 - a**2)))


# ---- host-side polar sampling (spherical-inclined orbits, e = 0) --------------

class PolarSamples(NamedTuple):
    qth: np.ndarray     # (nq,) uniform on [0, 2 pi), qth = 0 at theta_min
    thp: np.ndarray     # (nq,) particle polar angle
    uthp: np.ndarray    # (nq,) d(theta)/d(lambda), signed over the period
    tth: np.ndarray     # (nq,) oscillatory t(qth) (odd)
    phith: np.ndarray   # (nq,) oscillatory phi(qth) (odd)
    rp: float           # the (single) orbital radius
    constants: tuple    # (En, Lz, Q)
    gamma: float        # Mino time Gamma
    a: float
    kappa: float


def polar_samples(orbit, nsamples=2**8):
    """Sample polar geodesic quantities on a full-period uniform qth grid."""
    if orbit.e != 0:
        raise ValueError("polar_samples requires a spherical (e = 0) orbit")
    qth = 2 * np.pi * np.arange(nsamples) / nsamples
    thp = orbit.th(qth)
    uthp = orbit.uth(qth)
    uthp[0] = 0.0  # turning points: kill roundoff, as the radial path does
    half = nsamples // 2
    if nsamples % 2 == 0:
        uthp[half] = 0.0
    a = orbit.a
    return PolarSamples(qth, thp, uthp, orbit.tth(qth), orbit.phith(qth),
                        float(orbit.radial_roots[0]), tuple(orbit.constants),
                        float(orbit.mino_frequencies[0]),
                        a, float(np.sqrt(1 - a**2)))


def polar_swsh_values(s, m_arr, lmin_arr, coeffs, eigenvalues, gamma_arr, th):
    """S, dS/dtheta, d2S/dtheta2 on a shared theta grid -> (B, nth) each."""
    from .summation import swsh_theta_values
    S = swsh_theta_values(s, m_arr, lmin_arr, coeffs, th)
    dS = swsh_theta_values(s, m_arr, lmin_arr, coeffs, th, deriv=1)
    d2S = swsh_theta_values(s, m_arr, lmin_arr, coeffs, th, deriv=2,
                            eigenvalues=eigenvalues, gamma=gamma_arr)
    return S, dS, d2S


# ---- equatorial spheroidal-harmonic values (host basis, batched dot) ---------

def equatorial_swsh_values(s, m_arr, lmin_arr, coeffs, eigenvalues, gamma_arr):
    """S, dS/dtheta, d2S/dtheta2 at theta = pi/2 for a batch of modes.

    coeffs (B, nterms) from spheroidal_eigs_batched; the Y-basis rows are
    built on host per unique (m, lmin). d2S comes from the angular ODE
    (swsh.deriv2), whose cot(theta) term vanishes at the equator.
    """
    coeffs = np.asarray(coeffs)
    B, nterms = coeffs.shape
    S = np.empty(B)
    dS = np.empty(B)
    z = np.array([0.0])
    for key in {(int(m), int(lm)) for m, lm in zip(m_arr, lmin_arr)}:
        m, lmin = key
        sel = (m_arr == m) & (lmin_arr == lmin)
        basis = np.array([sphericalY_expansion(s, lmin + j, m, z)[0]
                          for j in range(nterms)])
        dbasis = np.array([sphericalY_expansion_deriv(s, lmin + j, m, z)[0]
                           for j in range(nterms)])
        S[sel] = coeffs[sel] @ basis
        dS[sel] = coeffs[sel] @ dbasis
    g_eq = (-gamma_arr**2 - (np.asarray(m_arr) + 0.0)**2 + s
            + 2 * np.asarray(m_arr) * gamma_arr + eigenvalues)
    d2S = -g_eq * S
    return S, dS, d2S


# ---- jnp ports of the hyperboloidal Z-scaling chain (teuk.py) -----------------

def _Z_sigma(sig, kappa, s, ma, om):
    Phi = jnp.log(1 - sig) / (2 * kappa)
    h = (2 * kappa / sig - 2 * jnp.log(sig)
         - (1 + kappa) / kappa * jnp.log(1 - sig) + (1 - kappa)
         + 2 * jnp.log(kappa))
    return (sig / (2 * kappa) * (4 * kappa**2 * (1 - sig) / sig**2)**(-s)
            * jnp.exp(1j * (ma * Phi + om * h)))


def _Z_sigma_deriv(sig, kappa, s, ma, om):
    return _Z_sigma(sig, kappa, s, ma, om) * (
        kappa * om * sig**(-2) * (-2j)
        + sig**(-1) / (-1 + sig) * (-1 + sig + s * (-2 + sig)
                                    + om * sig * (-3j) + om * (2j))
        + (1j) / (2 * kappa * (-1 + sig)) * (ma - 2 * om))


def _Z_sigma_deriv2(sig, kappa, s, ma, om):
    # verbatim port of teuk.Z_sigma_deriv2
    return _Z_sigma(sig, kappa, s, ma, om) * (
        kappa**(-2) * sig**(-4) * (-1 + sig)**(-2) * (-0.25) * (
            ma**2 * sig**4
            - 4 * kappa**2 * sig**2 * s**2 * (-2 + sig)**2
            - 2 * ma * sig**2 * (4 * om * kappa**2 * (-1 + sig)
                                 + 2 * om * sig**2
                                 + kappa * sig * (-4 * om + 6 * om * sig
                                                  + (-2j) + sig * (1j)))
            + 4 * om * (4 * om * kappa**4 * (-1 + sig)**2
                        + 4 * om * sig * kappa**3 * (2 - 5 * sig + 3 * sig**2)
                        + om * sig**4
                        + kappa * sig**3 * (-4 * om + (-2j)
                                            + sig * (6 * om + (1j)))
                        + kappa**2 * sig**2 * ((2j)
                                               - 2 * sig * (4 * om + (3j))
                                               + sig**2 * (9 * om + (3j))))
            + kappa * sig * s * (4j) * (
                -1 * sig**2 * (-2 + sig) * (ma - 2 * om)
                + 4 * om * kappa**2 * (2 - 3 * sig + sig**2)
                + kappa * sig * (8 * om + sig**2 * (6 * om + (1j))
                                 - 2 * sig * (8 * om + (1j)) + (2j)))))


# ---- the batched integrand and Fourier extraction -----------------------------

class SourceAmplitudes(NamedTuple):
    amplitudes: jax.Array   # (B, 2) complex — [:, 0] "In", [:, 1] "Up" (CPU order)
    errors: jax.Array       # (B, 2) relative error estimates


@functools.partial(jax.jit, static_argnums=(0, 12))
def _integrate_core(s, kappa, a, En, Lz, gamma_mino, orbit_arrs,
                    lam, m, n, omega, psi_arrs, method):
    rp, urp, tr, phir, qr = orbit_arrs
    (psi_up, dpsi_up, d2psi_up, psi_in, dpsi_in, d2psi_in,
     S, dS, d2S) = psi_arrs
    nq = rp.shape[0]

    ma = m * a
    om = omega[:, None]
    sig = 2 * kappa / (rp - (1 - kappa))            # sigma(rp), (nq,)
    sigB = sig[None, :]

    # per-mode normalization of each family (CPU norm trick)
    norm_up = jnp.max(jnp.abs(jnp.stack([psi_up, dpsi_up, d2psi_up])),
                      axis=(0, 2))
    norm_in = jnp.max(jnp.abs(jnp.stack([psi_in, dpsi_in, d2psi_in])),
                      axis=(0, 2))
    psi_up, dpsi_up, d2psi_up = (x / norm_up[:, None]
                                 for x in (psi_up, dpsi_up, d2psi_up))
    psi_in, dpsi_in, d2psi_in = (x / norm_in[:, None]
                                 for x in (psi_in, dpsi_in, d2psi_in))

    # equatorial Newman-Penrose projections (source.py, sth = 1, cth = 0)
    Delta = rp**2 - 2 * rp + a**2
    DeltaSqrt2 = SQRT2 * Delta
    rho = -1.0 / rp
    rhobar = rho
    Sigma = 1.0 / (rho * rhobar)
    Kt = (rp**2 + a**2)[None, :] * om - ma[:, None]

    L1 = (-m + a * omega)[:, None]      # -m/sth + a w sth + cth/sth at eq
    L2 = L1
    L2p = -2.0
    L2S = dS[:, None] + L2 * S[:, None]
    L1L2S = (d2S[:, None] + L1 * dS[:, None] + L2p * S[:, None]
             + L2 * dS[:, None] + L1 * L2 * S[:, None])

    Ann0 = (-rho**(-2) * rhobar**(-1) * DeltaSqrt2**(-2)
            * (rho**(-1) * L1L2S + 3j * a * L1 * S[:, None]
               + 2j * a * dS[:, None] - 1j * a * L2 * S[:, None]))
    Anmbar0 = (rho**(-3) / DeltaSqrt2
               * ((rho + rhobar - 1j * Kt / Delta) * L2S))
    Anmbar1 = -(rho**(-3) / DeltaSqrt2) * L2S
    Ambarmbar0 = (rho**(-3) * rhobar * DeltaSqrt2**(-2)
                  * (0.5 * Kt**2 + 1j * Kt * (1 - rp + Delta * rho)[None, :]
                     + 1j * rp[None, :] * om * Delta[None, :]) * S[:, None])
    Ambarmbar1 = -0.5 * rho**(-3) * rhobar * (1j * Kt / Delta - rho) * S[:, None]
    Ambarmbar2 = -0.25 * rho**(-3) * rhobar * S[:, None]

    Cr = (En * (rp**2 + a**2) - a * Lz + urp) / (2 * Sigma)   # (nq,)
    Cth = rho * (1j * (a * En - Lz)) / SQRT2                  # (nq,)
    Cnn, Cnmbar, Cmbarmbar = Cr * Cr, Cr * Cth, Cth * Cth

    A0 = Ann0 * Cnn + Anmbar0 * Cnmbar + Ambarmbar0 * Cmbarmbar
    A1 = -(Anmbar1 * Cnmbar + Ambarmbar1 * Cmbarmbar)
    A2 = Ambarmbar2 * Cmbarmbar

    Zs = _Z_sigma(sigB, kappa, s, ma[:, None], om)
    dZs = _Z_sigma_deriv(sigB, kappa, s, ma[:, None], om)
    d2Zs = _Z_sigma_deriv2(sigB, kappa, s, ma[:, None], om)
    gs = -sig**2 / (2 * kappa)
    dgs = -sig / kappa

    B0 = A0 * Zs + A1 * gs * dZs + A2 * gs * (dgs * dZs + gs * d2Zs)
    B1 = A1 * gs * Zs + A2 * gs * (dgs * Zs + 2 * gs * dZs)
    B2 = A2 * gs * gs * Zs

    phase = jnp.exp(1j * (om * tr[None, :] - m[:, None] * phir[None, :]))
    J_up = (B0 * psi_up + B1 * dpsi_up + B2 * d2psi_up) * phase
    J_in = (B0 * psi_in + B1 * dpsi_in + B2 * d2psi_in) * phase
    J = jnp.stack([J_up, J_in])                     # (2, B, nq)

    if method == "fft":
        # amplitude = (1/N) sum_k J_k e^{+ i n qr_k} = ifft(J)[n mod N]
        spec = jnp.fft.ifft(J, axis=-1)             # (2, B, nq)
        c = jnp.take_along_axis(
            spec, (jnp.mod(n, nq))[None, :, None], axis=-1)[:, :, 0]
        # aliasing estimate: largest tail coefficient (top half of the band)
        k = jnp.fft.fftfreq(nq, d=1.0 / nq)
        tail = jnp.max(jnp.where(jnp.abs(k)[None, None, :] >= nq // 4,
                                 jnp.abs(spec), 0.0), axis=-1)
        int_err = tail / jnp.abs(c)
    else:
        fourier = jnp.exp(1j * n[:, None] * qr[None, :])
        c_even = jnp.mean((J * fourier)[:, :, ::2], axis=-1)
        c_odd = jnp.mean((J * fourier)[:, :, 1::2], axis=-1)
        c = 0.5 * (c_even + c_odd)
        int_err = 0.5 * jnp.abs(1 - c_odd / c_even)

    # cancellation-error estimate (CPU convention)
    canc = 1e-15 / jnp.abs(c / jnp.mean(jnp.abs(J), axis=-1))
    err = jnp.maximum(int_err, canc)

    # Wronskian at qr = 0 (periapsis) and the CPU rescale there
    W = psi_in[:, 0] * dpsi_up[:, 0] - psi_up[:, 0] * dpsi_in[:, 0]
    sig0 = sig[0]
    rp0 = rp[0]
    rescale = (_Z_sigma(sig0, kappa, s, ma, omega)**2
               * (rp0**2 - 2 * rp0 + a**2)**(-1)
               * (-2 * kappa / (rp0 - (1 - kappa))**2))
    const_w = jnp.stack([norm_in, norm_up]) * rescale[None, :] * W[None, :]

    amps = -8 * jnp.pi * c / const_w / gamma_mino   # (2, B): [In, Up]
    return SourceAmplitudes(amps.T, err.T)


def integrate_source_batch(samples, lam, m, n, omega, S_eq, dS_eq, d2S_eq,
                           hom, subdomains_up, subdomains_in,
                           method="fft"):
    """Batched Teukolsky-mode amplitudes for an equatorial eccentric orbit.

    samples: OrbitSamples from orbit_samples(orbit). lam/m/n/omega: (B,)
    mode table (s = -2 radiative). S_eq/dS_eq/d2S_eq: equatorial harmonic
    values from equatorial_swsh_values. hom: HomogeneousBatch on the given
    subdomain structures. method: "fft" (spectral extraction + aliasing
    error estimate) or "quad" (staggered uniform rules, CPU-equivalent).

    Returns SourceAmplitudes with amplitudes (B, 2) ordered [In, Up] and
    relative error estimates (B, 2).
    """
    if method not in ("fft", "quad"):
        raise ValueError("method must be 'fft' or 'quad'")
    s = -2
    sig = 2 * samples.kappa / (samples.rp - (1 - samples.kappa))

    psi = []
    for coeffs, subs in [(hom.coeffs_up, subdomains_up),
                         (hom.coeffs_in, subdomains_in)]:
        for d in (0, 1, 2):
            psi.append(evaluate_multidomain(coeffs, subs, sig, deriv=d))

    En, Lz, _ = samples.constants
    orbit_arrs = tuple(jnp.asarray(x) for x in
                       (samples.rp, samples.urp, samples.tr, samples.phir,
                        samples.qr))
    psi_arrs = tuple(psi) + (jnp.asarray(S_eq), jnp.asarray(dS_eq),
                             jnp.asarray(d2S_eq))
    return _integrate_core(
        s, samples.kappa, samples.a, En, Lz, samples.gamma, orbit_arrs,
        jnp.asarray(lam, dtype=jnp.complex128),
        jnp.asarray(m, dtype=jnp.float64),
        jnp.asarray(n, dtype=jnp.int64),
        jnp.asarray(omega, dtype=jnp.complex128),
        psi_arrs, method)


# ---- spherical-inclined integrand: 1D q_theta Fourier extraction --------------

@functools.partial(jax.jit, static_argnums=(0, 13))
def _integrate_spherical_core(s, kappa, a, En, Lz, gamma_mino, rp, polar_arrs,
                              lam, m, k, omega, psi_arrs, method):
    thp, uthp, tth, phith, qth = polar_arrs
    (psi_up, dpsi_up, d2psi_up, psi_in, dpsi_in, d2psi_in,
     S, dS, d2S) = psi_arrs            # psi_* (B,) at sigma(rp); S* (B, nq)
    nq = thp.shape[0]

    ma = m * a
    om = omega[:, None]
    sig = 2 * kappa / (rp - (1 - kappa))            # scalar sigma(rp)

    # per-mode normalization of each family (CPU norm trick)
    norm_up = jnp.max(jnp.abs(jnp.stack([psi_up, dpsi_up, d2psi_up])), axis=0)
    norm_in = jnp.max(jnp.abs(jnp.stack([psi_in, dpsi_in, d2psi_in])), axis=0)
    psi_up, dpsi_up, d2psi_up = (x / norm_up for x in
                                 (psi_up, dpsi_up, d2psi_up))
    psi_in, dpsi_in, d2psi_in = (x / norm_in for x in
                                 (psi_in, dpsi_in, d2psi_in))

    # general-theta Newman-Penrose projections (source.py full-theta forms)
    sth = jnp.sin(thp)
    cth = jnp.cos(thp)
    Delta = rp**2 - 2 * rp + a**2
    DeltaSqrt2 = SQRT2 * Delta
    rho = -1.0 / (rp - 1j * a * cth)                # (nq,)
    rhobar = jnp.conj(rho)
    Sigma = 1.0 / (rho * rhobar)
    Kt = (rp**2 + a**2) * om - ma[:, None]          # (B, 1)

    L1 = -m[:, None] / sth + a * om * sth + cth / sth          # (B, nq)
    L2 = -m[:, None] / sth + a * om * sth + 2 * cth / sth
    L2p = m[:, None] * cth / sth**2 + a * om * cth - 2.0 / sth**2
    L2S = dS + L2 * S
    L1L2S = d2S + L1 * dS + L2p * S + L2 * dS + L1 * L2 * S

    Ann0 = (-rho**(-2) * rhobar**(-1) * DeltaSqrt2**(-2)
            * (rho**(-1) * L1L2S + 3j * a * sth * L1 * S + 3j * a * cth * S
               + 2j * a * sth * dS - 1j * a * sth * L2 * S))
    Anmbar0 = (rho**(-3) / DeltaSqrt2
               * ((rho + rhobar - 1j * Kt / Delta) * L2S
                  + (rho - rhobar) * a * sth * Kt / Delta * S))
    Anmbar1 = -(rho**(-3) / DeltaSqrt2) * (L2S + 1j * (rho - rhobar)
                                           * a * sth * S)
    Ambarmbar0 = (rho**(-3) * rhobar * DeltaSqrt2**(-2)
                  * (0.5 * Kt**2 + 1j * Kt * (1 - rp + Delta * rho)
                     + 1j * rp * om * Delta) * S)
    Ambarmbar1 = -0.5 * rho**(-3) * rhobar * (1j * Kt / Delta - rho) * S
    Ambarmbar2 = -0.25 * rho**(-3) * rhobar * S

    Cr = (En * (rp**2 + a**2) - a * Lz) / (2 * Sigma)           # u_r = 0
    Cth = rho * (1j * sth * (a * En - Lz / sth**2) + uthp) / SQRT2
    Cnn, Cnmbar, Cmbarmbar = Cr * Cr, Cr * Cth, Cth * Cth       # (nq,)

    A0 = Ann0 * Cnn + Anmbar0 * Cnmbar + Ambarmbar0 * Cmbarmbar
    A1 = -(Anmbar1 * Cnmbar + Ambarmbar1 * Cmbarmbar)
    A2 = Ambarmbar2 * Cmbarmbar

    Zs = _Z_sigma(sig, kappa, s, ma, omega)         # (B,)
    dZs = _Z_sigma_deriv(sig, kappa, s, ma, omega)
    d2Zs = _Z_sigma_deriv2(sig, kappa, s, ma, omega)
    gs = -sig**2 / (2 * kappa)                      # scalar
    dgs = -sig / kappa

    B0 = (A0 * Zs[:, None] + A1 * (gs * dZs)[:, None]
          + A2 * (gs * (dgs * dZs + gs * d2Zs))[:, None])
    B1 = (A1 * (gs * Zs)[:, None]
          + A2 * (gs * (dgs * Zs + 2 * gs * dZs))[:, None])
    B2 = A2 * (gs * gs * Zs)[:, None]

    phase = jnp.exp(1j * (om * tth[None, :] - m[:, None] * phith[None, :]))
    J_up = (B0 * psi_up[:, None] + B1 * dpsi_up[:, None]
            + B2 * d2psi_up[:, None]) * phase
    J_in = (B0 * psi_in[:, None] + B1 * dpsi_in[:, None]
            + B2 * d2psi_in[:, None]) * phase
    J = jnp.stack([J_up, J_in])                     # (2, B, nq)

    if method == "fft":
        # amplitude = (1/N) sum_j J_j e^{+ i k qth_j} = ifft(J)[k mod N]
        spec = jnp.fft.ifft(J, axis=-1)
        c = jnp.take_along_axis(
            spec, (jnp.mod(k, nq))[None, :, None], axis=-1)[:, :, 0]
        freq = jnp.fft.fftfreq(nq, d=1.0 / nq)
        tail = jnp.max(jnp.where(jnp.abs(freq)[None, None, :] >= nq // 4,
                                 jnp.abs(spec), 0.0), axis=-1)
        int_err = tail / jnp.abs(c)
    else:
        fourier = jnp.exp(1j * k[:, None] * qth[None, :])
        c_even = jnp.mean((J * fourier)[:, :, ::2], axis=-1)
        c_odd = jnp.mean((J * fourier)[:, :, 1::2], axis=-1)
        c = 0.5 * (c_even + c_odd)
        int_err = 0.5 * jnp.abs(1 - c_odd / c_even)

    canc = 1e-15 / jnp.abs(c / jnp.mean(jnp.abs(J), axis=-1))
    err = jnp.maximum(int_err, canc)

    # Wronskian and the CPU rescale at the single radius
    W = psi_in * dpsi_up - psi_up * dpsi_in
    rescale = (_Z_sigma(sig, kappa, s, ma, omega)**2 * Delta**(-1)
               * (-2 * kappa / (rp - (1 - kappa))**2))
    const_w = jnp.stack([norm_in, norm_up]) * rescale[None, :] * W[None, :]

    amps = -8 * jnp.pi * c / const_w / gamma_mino   # (2, B): [In, Up]
    return SourceAmplitudes(amps.T, err.T)


def integrate_source_spherical_batch(samples, lam, m, k, omega, S, dS, d2S,
                                     hom, subdomains_up, subdomains_in,
                                     method="fft"):
    """Batched Teukolsky-mode amplitudes for a spherical-inclined orbit.

    samples: PolarSamples from polar_samples(orbit). lam/m/k/omega: (B,)
    mode table (s = -2 radiative, omega = m*Omega_phi + k*Omega_theta).
    S/dS/d2S: (B, nq) harmonic values from polar_swsh_values on samples.thp.
    hom: HomogeneousBatch on subdomain structures meeting at sigma(rp).
    method: "fft" or "quad", as in integrate_source_batch.

    Returns SourceAmplitudes with amplitudes (B, 2) ordered [In, Up] and
    relative error estimates (B, 2).
    """
    if method not in ("fft", "quad"):
        raise ValueError("method must be 'fft' or 'quad'")
    s = -2
    sig = np.array([2 * samples.kappa / (samples.rp - (1 - samples.kappa))])

    psi = []
    for coeffs, subs in [(hom.coeffs_up, subdomains_up),
                         (hom.coeffs_in, subdomains_in)]:
        for d in (0, 1, 2):
            psi.append(evaluate_multidomain(coeffs, subs, sig, deriv=d)[:, 0])

    En, Lz, _ = samples.constants
    polar_arrs = tuple(jnp.asarray(x) for x in
                       (samples.thp, samples.uthp, samples.tth,
                        samples.phith, samples.qth))
    psi_arrs = tuple(psi) + (jnp.asarray(S), jnp.asarray(dS), jnp.asarray(d2S))
    return _integrate_spherical_core(
        s, samples.kappa, samples.a, En, Lz, samples.gamma, samples.rp,
        polar_arrs,
        jnp.asarray(lam, dtype=jnp.complex128),
        jnp.asarray(m, dtype=jnp.float64),
        jnp.asarray(k, dtype=jnp.int64),
        jnp.asarray(omega, dtype=jnp.complex128),
        psi_arrs, method)
