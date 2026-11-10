# Bilinear-form generic source integral (source_bilinear.py). Design:
# docs/superpowers/specs/2026-11-10-bilinear-source-design.md
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from hblteuk.geo import MiniGeo
from hblteuk.jax.source import _source_coefficients, generic_samples


def test_terms_reproduce_source_coefficients():
    # every factored term recombines to _source_coefficients pointwise,
    # including large m and near-zero omega
    from hblteuk.jax.source_bilinear import (TERMS, kernels, polar_factors,
                                             radial_factors)
    smp = generic_samples(MiniGeo(0.9, 8.0, 0.3, 0.7), nsamples=64, ntheta=32)
    En, Lz, _ = smp.constants
    rp, urp = jnp.asarray(smp.rp), jnp.asarray(smp.urp)
    th, uth = jnp.asarray(smp.thp), jnp.asarray(smp.uthp)
    K = kernels(smp.a, En, Lz, rp, urp, th, uth)
    assert K.shape == (12, 64, 32)
    rng = np.random.default_rng(3)
    for m, om in ((14.0, 1.2 + 0j), (-3.0, 0.017 + 0j), (0.0, 1.3e-4 + 0j)):
        S, dS, d2S = (jnp.asarray(rng.normal(size=32) + 1j * rng.normal(size=32))
                      for _ in range(3))
        want = _source_coefficients(smp.a, En, Lz, m, om, rp[:, None],
                                    urp[:, None], jnp.sin(th)[None, :],
                                    jnp.cos(th)[None, :], uth[None, :],
                                    S[None, :], dS[None, :], d2S[None, :])
        R = radial_factors(smp.a, m, om, rp[:, None])
        P = polar_factors(smp.a, m, om, th[None, :], S[None, :], dS[None, :],
                          d2S[None, :])
        got = [0, 0, 0]
        for t, (c, rn, pn) in enumerate(TERMS):
            got[c] = got[c] + K[t] * R[rn] * P[pn]
        for c in range(3):
            w, g = np.asarray(want[c]), np.asarray(got[c])
            assert np.max(np.abs(g - w)) <= 1e-13 * np.max(np.abs(w)), (m, c)


MODES = [(13, 11, -1, -15), (14, 11, -1, -15), (2, 2, 0, 0), (2, 2, 0, 1),
         (4, 3, 1, -2), (8, 6, -5, -2), (12, 7, 5, 10), (3, 0, 1, 0)]


@pytest.fixture(scope="module")
def ref_case():
    # reference case (0.9, 8, 0.3, 0.7): near-zero omega at m = 11, negative n,
    # k = 0 and n = 0 modes
    from hblteuk.collocode import generate_subdomain_list
    from hblteuk.jax import solve_homogeneous_batch, spheroidal_eigs_batched
    from hblteuk.teuk import sigma_r

    orbit = MiniGeo(0.9, 8.0, 0.3, 0.7)
    l, m, k, n = (np.array(c) for c in zip(*MODES))
    om = np.array([orbit.mode_frequency(int(mm), int(kk), int(nn))
                   for mm, kk, nn in zip(m, k, n)])
    a = orbit.a
    kappa = np.sqrt(1 - a**2)
    lam, coeffs, lmin = spheroidal_eigs_batched(-2, l, m, a * om)
    su = generate_subdomain_list(
        [0, sigma_r(orbit.radial_roots[1], kappa)], 32, "arcsinh7")
    si = generate_subdomain_list(
        [1, sigma_r(orbit.radial_roots[0], kappa)], 32, "arcsinh7")
    hom = solve_homogeneous_batch(-2, kappa, np.asarray(lam), m * a, om, su, si)
    return orbit, (l, m, k, n, om, np.asarray(lam)), (np.asarray(coeffs),
                                                     lmin), hom, su, si


def _args(ref_case, nsamples, ntheta):
    from hblteuk.jax.source import _generic_args, polar_swsh_values
    orbit, (l, m, k, n, om, lam), (coeffs, lmin), hom, su, si = ref_case
    smp = generic_samples(orbit, nsamples=nsamples, ntheta=ntheta)
    S, dS, d2S = polar_swsh_values(-2, m, lmin, coeffs, lam, orbit.a * om,
                                   smp.thp)
    return _generic_args(smp, lam, m, k, n, om, S, dS, d2S, hom, su, si)


def _both(args):
    from hblteuk.jax.source import _integrate_generic_core
    from hblteuk.jax.source_bilinear import _integrate_generic_bilinear
    old, new = _integrate_generic_core(*args), _integrate_generic_bilinear(*args)
    return ((np.asarray(old.amplitudes), np.asarray(old.errors)),
            (np.asarray(new.amplitudes), np.asarray(new.errors)))


def _assert_agree(a_old, e_old, a_new, e_new):
    # 1e-10 where the amplitude is resolved; where both methods report a
    # cancellation-limited value (the near-zero-omega In amplitudes, ~1e-25,
    # error estimate ~0.4), they need only agree within their own estimates
    rel = np.abs(a_new - a_old) / np.abs(a_old)
    good = e_old < 1e-8
    assert np.all(rel[good] < 1e-10)
    assert np.all(rel[~good] <= np.maximum(e_old, e_new)[~good])


def test_bilinear_matches_fft_amplitudes(ref_case):
    (a_old, e_old), (a_new, e_new) = _both(_args(ref_case, 512, 128))
    assert np.all(np.isfinite(a_new))
    _assert_agree(a_old, e_old, a_new, e_new)


def test_bilinear_error_estimates_never_smaller(ref_case):
    # q_r tail and q_theta checks keep their definitions; the cancellation
    # floor is a triangle-inequality bound, so it can only be larger. The tail
    # and theta estimates are themselves differences of near-equal sums, so
    # they move by up to ~1% between summation orders (and are noise below
    # 1e-14): allow that slack, nothing more
    (_, e_old), (_, e_new) = _both(_args(ref_case, 512, 128))
    assert np.all(np.isfinite(e_new))
    assert np.all(e_new >= 0.99 * e_old - 1e-14)


def test_bilinear_non_pow2_grid(ref_case):
    (a_old, e_old), (a_new, e_new) = _both(_args(ref_case, 96, 40))
    _assert_agree(a_old, e_old, a_new, e_new)
