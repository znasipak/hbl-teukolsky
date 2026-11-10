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
