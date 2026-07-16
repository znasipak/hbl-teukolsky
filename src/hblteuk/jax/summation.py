# Batched mode summation: harmonic evaluation on theta grids, spheroidal ->
# spherical projection, and field/waveform assembly. See dev/PHASE4.md.
import jax.numpy as jnp
import numpy as np

from ..swsh import sphericalY_expansion
from .chebyshev import evaluate_multidomain


def swsh_theta_values(s, m_arr, lmin_arr, coeffs, th, deriv=0):
    """Batched S_{lmn}(theta) on a shared theta grid -> (B, ntheta).

    Y-basis rows are built on host per unique (m, lmin) (heavy special
    functions stay on CPU, GPU plan §2.2); the batched contraction
    coeffs @ basis runs on device.
    """
    from ..swsh import sphericalY_expansion_deriv
    coeffs = np.asarray(coeffs)
    B, nterms = coeffs.shape
    z = np.cos(np.atleast_1d(th))
    out = np.empty((B, len(z)))
    fn = sphericalY_expansion if deriv == 0 else sphericalY_expansion_deriv
    for key in {(int(m), int(lm)) for m, lm in zip(m_arr, lmin_arr)}:
        m, lmin = key
        sel = (np.asarray(m_arr) == m) & (np.asarray(lmin_arr) == lmin)
        basis = np.array([fn(s, lmin + j, m, z) for j in range(nterms)])
        out[sel] = coeffs[sel] @ basis
    return jnp.asarray(out)


def project_to_spherical(coupling, coeffs, m_arr, lmin_arr, l_arr):
    """Spheroidal -> spherical projection of radial coefficient grids.

    Batched analogue of the CPU cast_to_spherical contraction: for each m,
    the spherical-basis radial coefficients are

        R^{Ylm}_{j} = sum_L  b^{L}_{lmin+j}  R^{S_Lm},

    where b are the spherical-expansion coefficients (coupling rows) of the
    spheroidal harmonics. coupling (B, nterms); coeffs (B, D, n) from
    solve_homogeneous_batch (modes ordered arbitrarily; grouping by m is
    handled here). Returns (proj_coeffs (B_out, D, n), out_l, out_m) where
    row i is the spherical (out_l[i], out_m[i]) radial coefficient stack,
    with out_l = lmin..lmin+nterms-1 per m truncated to the input l-range.
    """
    coupling = np.asarray(coupling)
    m_arr = np.asarray(m_arr)
    l_arr = np.asarray(l_arr)
    lmin_arr = np.asarray(lmin_arr)
    lmax = int(l_arr.max())

    out_blocks, out_l, out_m = [], [], []
    for m in np.unique(m_arr):
        sel = m_arr == m
        lmin = int(lmin_arr[sel][0])
        jnum = min(coupling.shape[1], lmax - lmin + 1)
        # (B_m, jnum)^T @ (B_m, D, n) -> (jnum, D, n)
        block = jnp.einsum("bj,bdn->jdn",
                           jnp.asarray(coupling[sel][:, :jnum]),
                           jnp.asarray(coeffs)[np.where(sel)[0]])
        out_blocks.append(block)
        out_l.extend(range(lmin, lmin + jnum))
        out_m.extend([int(m)] * jnum)
    return (jnp.concatenate(out_blocks, axis=0),
            np.array(out_l), np.array(out_m))


def assemble_field(amplitudes, radial, angular, m_arr, omega, phi, t=0.0):
    """Sum the mode content: psi(sigma_i, theta_j, phi_k) at time t.

    amplitudes (B,), radial (B, nsigma), angular (B, ntheta), phases
    exp[i(m phi - omega t)] -> (nsigma, ntheta, nphi) complex field.
    """
    phase = jnp.exp(1j * (jnp.asarray(m_arr)[:, None] * jnp.asarray(phi)[None, :]
                          - jnp.asarray(omega)[:, None] * t))
    return jnp.einsum("b,bs,bh,bp->shp", jnp.asarray(amplitudes),
                      jnp.asarray(radial), jnp.asarray(angular), phase)


def waveform_timeseries(amplitudes, radial_scri, angular, m_arr, omega,
                        phi, t_arr):
    """Time series at a single field point/angles: (ntheta, nphi, nt).

    radial_scri (B,) — the radial factor at the evaluation sigma (e.g. the
    sigma -> 0 limit for waveforms at scri).
    """
    phase = jnp.exp(1j * (jnp.asarray(m_arr)[:, None, None]
                          * jnp.asarray(phi)[None, :, None]
                          - jnp.asarray(omega)[:, None, None]
                          * jnp.asarray(t_arr)[None, None, :]))
    return jnp.einsum("b,b,bh,bpt->hpt",
                      jnp.asarray(amplitudes), jnp.asarray(radial_scri),
                      jnp.asarray(angular), phase)
