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
from .chebyshev import evaluate_with_matrix, evaluation_matrix

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
    """S, dS/dtheta, d2S/dtheta2 on a shared theta grid -> (B, nth) each.

    Device Y_{L,|m|} ladder contracted with cached host spin-coupling tables
    (jax/swsh.py); d2S from the angular ODE. dev/JAX_PERF.md, Task 3.
    """
    from .summation import _swsh_deriv2
    from .swsh import ylm_contract, ylm_coupling_coeffs
    th = np.atleast_1d(np.asarray(th, dtype=np.float64))
    sth = jnp.sin(jnp.asarray(th))[None, :]
    S = sth**-abs(s) * ylm_contract(
        m_arr, ylm_coupling_coeffs(s, m_arr, lmin_arr, coeffs, 0), th)
    dS = -sth**-(abs(s) + 1) * ylm_contract(
        m_arr, ylm_coupling_coeffs(s, m_arr, lmin_arr, coeffs, 1), th)
    d2S = _swsh_deriv2(s, m_arr, gamma_arr, eigenvalues, th, S, dS)
    return S, dS, d2S


def _polar_swsh_values_reference(s, m_arr, lmin_arr, coeffs, eigenvalues,
                                 gamma_arr, th):
    # host Y-basis path (pre-Task 3); equivalence-test reference
    from .summation import swsh_theta_values
    S = swsh_theta_values(s, m_arr, lmin_arr, coeffs, th)
    dS = swsh_theta_values(s, m_arr, lmin_arr, coeffs, th, deriv=1)
    d2S = swsh_theta_values(s, m_arr, lmin_arr, coeffs, th, deriv=2,
                            eigenvalues=eigenvalues, gamma=gamma_arr)
    return S, dS, d2S


# ---- host-side generic sampling (e != 0, |x| < 1): both torus axes ----------

class GenericSamples(NamedTuple):
    qr: np.ndarray      # (nq,) uniform on [0, 2 pi), qr = 0 at periapsis
    rp: np.ndarray      # (nq,)
    urp: np.ndarray     # (nq,) signed radial velocity
    tr: np.ndarray      # (nq,)
    phir: np.ndarray    # (nq,)
    qth: np.ndarray     # (nth,) uniform on [0, 2 pi), qth = 0 at theta_min
    thp: np.ndarray     # (nth,)
    uthp: np.ndarray    # (nth,) signed d(theta)/d(lambda)
    tth: np.ndarray     # (nth,)
    phith: np.ndarray   # (nth,)
    constants: tuple    # (En, Lz, Q)
    gamma: float        # Mino time Gamma
    a: float
    kappa: float


def generic_samples(orbit, nsamples=2**10, ntheta=2**8):
    """Sample both torus axes on full-period uniform grids (host)."""
    if orbit.e == 0 or abs(orbit.x) == 1:
        raise ValueError("generic_samples requires e != 0 and |x| < 1; "
                         "degenerate orbits use the 1D integrators")
    qr = 2 * np.pi * np.arange(nsamples) / nsamples
    rp = orbit.r(qr)
    urp = orbit.ur(qr)
    urp[0] = 0.0
    if nsamples % 2 == 0:
        urp[nsamples // 2] = 0.0
    qth = 2 * np.pi * np.arange(ntheta) / ntheta
    thp = orbit.th(qth)
    uthp = orbit.uth(qth)
    uthp[0] = 0.0
    if ntheta % 2 == 0:
        uthp[ntheta // 2] = 0.0
    a = orbit.a
    return GenericSamples(qr, rp, urp, orbit.tr(qr), orbit.phir(qr),
                          qth, thp, uthp, orbit.tth(qth), orbit.phith(qth),
                          tuple(orbit.constants),
                          float(orbit.mino_frequencies[0]),
                          a, float(np.sqrt(1 - a**2)))


# ---- equatorial spheroidal-harmonic values (host basis, batched dot) ---------

@functools.lru_cache(maxsize=8192)
def _equatorial_basis(s, m, lmin, nterms):
    """Y-basis row [Y_{s, lmin+j, m}(pi/2)]_j and its theta-derivative.

    Depends only on (s, m, lmin, nterms), so it is cached: the table is
    otherwise rebuilt for every memory chunk and every adaptive flux round,
    and building it is ~99% of equatorial_swsh_values' cost (measured 2.2 s
    at lmax=60 vs 12 ms for the batch bookkeeping).
    """
    z = np.array([0.0])
    basis = np.array([sphericalY_expansion(s, lmin + j, m, z)[0]
                      for j in range(nterms)])
    dbasis = np.array([sphericalY_expansion_deriv(s, lmin + j, m, z)[0]
                       for j in range(nterms)])
    return basis, dbasis


def equatorial_swsh_values(s, m_arr, lmin_arr, coeffs, eigenvalues, gamma_arr):
    """S, dS/dtheta, d2S/dtheta2 at theta = pi/2 for a batch of modes.

    coeffs (B, nterms) from spheroidal_eigs_batched; the Y-basis rows are
    built (and cached) on host per unique (m, lmin), then gathered to the
    batch and contracted -- no per-key masking of the (B, nterms) array.
    d2S comes from the angular ODE (swsh.deriv2), whose cot(theta) term
    vanishes at the equator.
    """
    coeffs = np.asarray(coeffs)
    B, nterms = coeffs.shape
    pairs = np.stack([np.asarray(m_arr, dtype=np.int64),
                      np.asarray(lmin_arr, dtype=np.int64)], axis=1)
    uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
    tab = np.array([_equatorial_basis(s, int(m), int(lm), nterms)[0]
                    for m, lm in uniq])
    dtab = np.array([_equatorial_basis(s, int(m), int(lm), nterms)[1]
                     for m, lm in uniq])
    S = np.einsum("bj,bj->b", coeffs, tab[inv])
    dS = np.einsum("bj,bj->b", coeffs, dtab[inv])
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



def _psi_at(hom, subdomains_up, subdomains_in, sig):
    # [psi_up, dpsi_up, d2psi_up, psi_in, dpsi_in, d2psi_in], each (B, Np)
    n = hom.coeffs_up.shape[-1]
    sig = np.asarray(sig, dtype=np.float64)
    out = []
    for coeffs, subs in [(hom.coeffs_up, subdomains_up),
                         (hom.coeffs_in, subdomains_in)]:
        out.extend(evaluate_with_matrix(coeffs,
                                        evaluation_matrix(subs, sig, n)))
    return out

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

    psi = _psi_at(hom, subdomains_up, subdomains_in, sig)

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


# ---- shared Newman-Penrose A/C assembly (full-theta forms) --------------------

def _source_coefficients(a, En, Lz, m2, om2, rp, urp, sth, cth, uthp,
                         S, dS, d2S):
    """Newman-Penrose A/C assembly, elementwise in whatever shapes the
    caller broadcasts (sphinc: fixed rp, theta arrays; generic scan: rp
    arrays, fixed theta). Expressions are the full-theta forms of CPU
    source.py; returns (A0, A1, A2)."""
    Delta = rp**2 - 2 * rp + a**2
    DeltaSqrt2 = SQRT2 * Delta
    rho = -1.0 / (rp - 1j * a * cth)
    rhobar = jnp.conj(rho)
    Sigma = 1.0 / (rho * rhobar)
    Kt = (rp**2 + a**2) * om2 - (m2 * a)

    L1 = -m2 / sth + a * om2 * sth + cth / sth
    L2 = -m2 / sth + a * om2 * sth + 2 * cth / sth
    L2p = m2 * cth / sth**2 + a * om2 * cth - 2.0 / sth**2
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
                     + 1j * rp * om2 * Delta) * S)
    Ambarmbar1 = -0.5 * rho**(-3) * rhobar * (1j * Kt / Delta - rho) * S
    Ambarmbar2 = -0.25 * rho**(-3) * rhobar * S

    Cr = (En * (rp**2 + a**2) - a * Lz + urp) / (2 * Sigma)
    Cth = rho * (1j * sth * (a * En - Lz / sth**2) + uthp) / SQRT2
    Cnn, Cnmbar, Cmbarmbar = Cr * Cr, Cr * Cth, Cth * Cth

    A0 = Ann0 * Cnn + Anmbar0 * Cnmbar + Ambarmbar0 * Cmbarmbar
    A1 = -(Anmbar1 * Cnmbar + Ambarmbar1 * Cmbarmbar)
    A2 = Ambarmbar2 * Cmbarmbar
    return A0, A1, A2


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

    # general-theta Newman-Penrose projections via the shared assembly
    # (fixed rp scalar, theta arrays; u_r = 0 on spherical orbits)
    sth = jnp.sin(thp)
    cth = jnp.cos(thp)
    Delta = rp**2 - 2 * rp + a**2
    A0, A1, A2 = _source_coefficients(a, En, Lz, m[:, None], om,
                                      rp, 0.0, sth, cth, uthp, S, dS, d2S)

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

    psi = [p[:, 0] for p in _psi_at(hom, subdomains_up, subdomains_in, sig)]

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


# ---- generic-orbit integrand: 2D torus, memory-safe q_theta contraction ------

@functools.partial(jax.jit, static_argnums=(0,))
def _integrate_generic_core(s, kappa, a, En, Lz, gamma_mino, radial_arrs,
                            polar_arrs, lam, m, k, n, omega, psi_arrs):
    rp, urp, tr, phir, qr = radial_arrs
    thp, uthp, tth, phith, qth = polar_arrs
    (psi_up, dpsi_up, d2psi_up, psi_in, dpsi_in, d2psi_in,
     S, dS, d2S) = psi_arrs        # psi_* (B, nq); S* (B, nth)
    nq = rp.shape[0]
    nth = thp.shape[0]

    ma = m * a
    om = omega[:, None]

    norm_up = jnp.max(jnp.abs(jnp.stack([psi_up, dpsi_up, d2psi_up])),
                      axis=(0, 2))
    norm_in = jnp.max(jnp.abs(jnp.stack([psi_in, dpsi_in, d2psi_in])),
                      axis=(0, 2))
    psi_up, dpsi_up, d2psi_up = (x / norm_up[:, None]
                                 for x in (psi_up, dpsi_up, d2psi_up))
    psi_in, dpsi_in, d2psi_in = (x / norm_in[:, None]
                                 for x in (psi_in, dpsi_in, d2psi_in))

    # radial-only factors, precomputed once (Phase 3 shapes)
    sig = 2 * kappa / (rp - (1 - kappa))            # (nq,)
    sigB = sig[None, :]
    Zs = _Z_sigma(sigB, kappa, s, ma[:, None], om)
    dZs = _Z_sigma_deriv(sigB, kappa, s, ma[:, None], om)
    d2Zs = _Z_sigma_deriv2(sigB, kappa, s, ma[:, None], om)
    gs = -sig**2 / (2 * kappa)
    dgs = -sig / kappa
    phase_r = jnp.exp(1j * (om * tr[None, :] - m[:, None] * phir[None, :]))
    m2 = m[:, None]

    def theta_slice(carry, xs):
        Je_up, Jo_up, Je_in, Jo_in, absJ, idx = carry
        th_j, uth_j, tth_j, phith_j, qth_j, S_j, dS_j, d2S_j = xs
        sth = jnp.sin(th_j)
        cth = jnp.cos(th_j)
        A0, A1, A2 = _source_coefficients(a, En, Lz, m2, om, rp, urp,
                                          sth, cth, uth_j,
                                          S_j[:, None], dS_j[:, None],
                                          d2S_j[:, None])
        B0 = A0 * Zs + A1 * gs * dZs + A2 * gs * (dgs * dZs + gs * d2Zs)
        B1 = A1 * gs * Zs + A2 * gs * (dgs * Zs + 2 * gs * dZs)
        B2 = A2 * gs * gs * Zs
        # polar phase * q_theta Fourier weight: one scalar per mode
        w = jnp.exp(1j * (omega * tth_j - m * phith_j + k * qth_j))[:, None]
        Ju = (B0 * psi_up + B1 * dpsi_up + B2 * d2psi_up) * phase_r * w
        Ji = (B0 * psi_in + B1 * dpsi_in + B2 * d2psi_in) * phase_r * w
        even = (idx % 2 == 0)
        we = jnp.where(even, 1.0, 0.0)
        wo = 1.0 - we
        carry = (Je_up + we * Ju, Jo_up + wo * Ju,
                 Je_in + we * Ji, Jo_in + wo * Ji,
                 absJ + jnp.mean(jnp.abs(Ju), axis=-1)
                      + jnp.mean(jnp.abs(Ji), axis=-1),
                 idx + 1)
        return carry, None

    B = m.shape[0]
    zeros = jnp.zeros((B, nq), dtype=jnp.complex128)
    xs = (thp, uthp, tth, phith, qth,
          jnp.transpose(S), jnp.transpose(dS), jnp.transpose(d2S))
    (Je_up, Jo_up, Je_in, Jo_in, absJ, _), _ = jax.lax.scan(
        theta_slice, (zeros, zeros, zeros, zeros,
                      jnp.zeros((B,)), jnp.int32(0)), xs)

    # contracted J~(qr) per family: mean over theta samples
    J = jnp.stack([(Je_up + Jo_up) / nth, (Je_in + Jo_in) / nth])  # (2, B, nq)

    # qr-axis extraction: ifft bin n + aliasing tail (Phase 3)
    spec = jnp.fft.ifft(J, axis=-1)
    nn = (jnp.mod(n, nq))[None, :, None]
    c = jnp.take_along_axis(spec, nn, axis=-1)[:, :, 0]
    freq = jnp.fft.fftfreq(nq, d=1.0 / nq)
    tail_r = jnp.max(jnp.where(jnp.abs(freq)[None, None, :] >= nq // 4,
                               jnp.abs(spec), 0.0), axis=-1)
    err_r = tail_r / jnp.abs(c)

    # q_theta-axis estimate: even vs odd half-grid rules (staggered)
    halves = jnp.stack([2 * Je_up / nth, 2 * Jo_up / nth,
                        2 * Je_in / nth, 2 * Jo_in / nth])   # (4, B, nq)
    ch = jnp.take_along_axis(jnp.fft.ifft(halves, axis=-1),
                             (jnp.mod(n, nq))[None, :, None],
                             axis=-1)[:, :, 0]               # (4, B)
    err_th_up = 0.5 * jnp.abs(1 - ch[1] / ch[0])
    err_th_in = 0.5 * jnp.abs(1 - ch[3] / ch[2])
    err_th = jnp.stack([err_th_up, err_th_in])

    canc = 1e-15 / jnp.abs(c / (absJ / (2 * nth))[None, :])
    err = jnp.maximum(jnp.maximum(err_r, err_th), canc)

    # Wronskian and rescale at qr = 0 (periapsis), Phase 3 conventions
    W = psi_in[:, 0] * dpsi_up[:, 0] - psi_up[:, 0] * dpsi_in[:, 0]
    sig0 = sig[0]
    rp0 = rp[0]
    rescale = (_Z_sigma(sig0, kappa, s, ma, omega)**2
               * (rp0**2 - 2 * rp0 + a**2)**(-1)
               * (-2 * kappa / (rp0 - (1 - kappa))**2))
    const_w = jnp.stack([norm_in, norm_up]) * rescale[None, :] * W[None, :]

    amps = -8 * jnp.pi * c / const_w / gamma_mino   # (2, B): [In, Up]
    return SourceAmplitudes(amps.T, err.T)


def _integrate_generic_fft2(s, kappa, a, En, Lz, gamma_mino, radial_arrs,
                            polar_arrs, lam, m, k, n, omega, psi_arrs):
    # small-batch diagnostic: materializes (B, nth, nq); NOT for production
    rp, urp, tr, phir, qr = radial_arrs
    thp, uthp, tth, phith, qth = polar_arrs
    (psi_up, dpsi_up, d2psi_up, psi_in, dpsi_in, d2psi_in,
     S, dS, d2S) = psi_arrs
    nq, nth = rp.shape[0], thp.shape[0]
    ma = m * a
    om = omega[:, None, None]
    m3 = m[:, None, None]

    norm_up = jnp.max(jnp.abs(jnp.stack([psi_up, dpsi_up, d2psi_up])),
                      axis=(0, 2))
    norm_in = jnp.max(jnp.abs(jnp.stack([psi_in, dpsi_in, d2psi_in])),
                      axis=(0, 2))
    psi_up, dpsi_up, d2psi_up = (x[:, None, :] / norm_up[:, None, None]
                                 for x in (psi_up, dpsi_up, d2psi_up))
    psi_in, dpsi_in, d2psi_in = (x[:, None, :] / norm_in[:, None, None]
                                 for x in (psi_in, dpsi_in, d2psi_in))

    sig = 2 * kappa / (rp - (1 - kappa))
    Zs = _Z_sigma(sig[None, None, :], kappa, s, ma[:, None, None], om)
    dZs = _Z_sigma_deriv(sig[None, None, :], kappa, s, ma[:, None, None], om)
    d2Zs = _Z_sigma_deriv2(sig[None, None, :], kappa, s, ma[:, None, None], om)
    gs = -sig**2 / (2 * kappa)
    dgs = -sig / kappa

    A0, A1, A2 = _source_coefficients(
        a, En, Lz, m3, om, rp[None, None, :], urp[None, None, :],
        jnp.sin(thp)[None, :, None], jnp.cos(thp)[None, :, None],
        uthp[None, :, None], S[:, :, None], dS[:, :, None], d2S[:, :, None])
    B0 = A0 * Zs + A1 * gs * dZs + A2 * gs * (dgs * dZs + gs * d2Zs)
    B1 = A1 * gs * Zs + A2 * gs * (dgs * Zs + 2 * gs * dZs)
    B2 = A2 * gs * gs * Zs

    phase = jnp.exp(1j * (om * (tr[None, None, :] + tth[None, :, None])
                          - m3 * (phir[None, None, :] + phith[None, :, None])
                          + (jnp.asarray(k)[:, None, None] * qth[None, :, None])))
    J_up = (B0 * psi_up + B1 * dpsi_up + B2 * d2psi_up) * phase
    J_in = (B0 * psi_in + B1 * dpsi_in + B2 * d2psi_in) * phase
    J = jnp.stack([J_up, J_in])                     # (2, B, nth, nq)

    # the e^{ik qth} weight is already folded into the phase, so the
    # k-content of every mode sits at the qth-DC bin: take ifft2 bin (0, n)
    spec = jnp.fft.ifft2(J, axes=(-2, -1))
    nn = jnp.mod(n, nq)
    c = spec[:, jnp.arange(m.shape[0]), 0, nn]
    absJ = jnp.mean(jnp.abs(J), axis=(-2, -1))
    canc = 1e-15 / jnp.abs(c / absJ)
    freq_th = jnp.abs(jnp.fft.fftfreq(nth, 1 / nth))[None, None, :, None]
    freq_r = jnp.abs(jnp.fft.fftfreq(nq, 1 / nq))[None, None, None, :]
    tail = jnp.max(jnp.where((freq_th >= nth // 4) | (freq_r >= nq // 4),
                             jnp.abs(spec), 0.0), axis=(-2, -1))
    err = jnp.maximum(tail / jnp.abs(c), canc)

    W = psi_in[:, 0, 0] * dpsi_up[:, 0, 0] - psi_up[:, 0, 0] * dpsi_in[:, 0, 0]
    sig0, rp0 = sig[0], rp[0]
    rescale = (_Z_sigma(sig0, kappa, s, ma, omega)**2
               * (rp0**2 - 2 * rp0 + a**2)**(-1)
               * (-2 * kappa / (rp0 - (1 - kappa))**2))
    const_w = jnp.stack([norm_in, norm_up]) * rescale[None, :] * W[None, :]
    amps = -8 * jnp.pi * c / const_w / gamma_mino
    return SourceAmplitudes(amps.T, err.T)


def _generic_args(samples, lam, m, k, n, omega, S, dS, d2S, hom,
                  subdomains_up, subdomains_in):
    # positional arguments shared by every generic integrator core (s = -2)
    sig = 2 * samples.kappa / (samples.rp - (1 - samples.kappa))
    psi = _psi_at(hom, subdomains_up, subdomains_in, sig)
    En, Lz, _ = samples.constants
    radial_arrs = tuple(jnp.asarray(x) for x in
                        (samples.rp, samples.urp, samples.tr, samples.phir,
                         samples.qr))
    polar_arrs = tuple(jnp.asarray(x) for x in
                       (samples.thp, samples.uthp, samples.tth,
                        samples.phith, samples.qth))
    psi_arrs = tuple(psi) + (jnp.asarray(S), jnp.asarray(dS), jnp.asarray(d2S))
    return (-2, samples.kappa, samples.a, En, Lz, samples.gamma,
            radial_arrs, polar_arrs,
            jnp.asarray(lam, dtype=jnp.complex128),
            jnp.asarray(m, dtype=jnp.float64),
            jnp.asarray(k, dtype=jnp.float64),
            jnp.asarray(n, dtype=jnp.int64),
            jnp.asarray(omega, dtype=jnp.complex128),
            psi_arrs)


def integrate_source_generic_batch(samples, lam, m, k, n, omega, S, dS, d2S,
                                   hom, subdomains_up, subdomains_in,
                                   method="fft"):
    """Batched Teukolsky-mode amplitudes for a generic orbit (2D torus).

    samples: GenericSamples. lam/m/k/n/omega: (B,) mode table (s = -2,
    omega = m*Omega_phi + k*Omega_theta + n*Omega_r). S/dS/d2S: (B, nth)
    from polar_swsh_values on samples.thp. method: "fft" (lax.scan over
    q_theta, O(B*nq) working set, staggered-theta + FFT-tail error
    estimates) or "fft2" (materializes (B, nth, nq) -- small-batch
    diagnostic only).

    Returns SourceAmplitudes with amplitudes (B, 2) ordered [In, Up] and
    relative error estimates (B, 2).
    """
    if method not in ("fft", "fft2"):
        raise ValueError("method must be 'fft' or 'fft2'")
    args = _generic_args(samples, lam, m, k, n, omega, S, dS, d2S, hom,
                         subdomains_up, subdomains_in)
    if method == "fft":
        return _integrate_generic_core(*args)
    return _integrate_generic_fft2(*args)
