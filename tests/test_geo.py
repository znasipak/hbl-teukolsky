# MiniGeo polar-motion extension. kerrgeopy-vs-pybhpt conventions were
# gated by dev/check_polar_conventions.py; these are the env-local
# self-consistency checks (no pybhpt needed). See dev/PHASE5B.md.
import numpy as np
import pytest

from hblteuk.geo import MiniGeo


@pytest.fixture(scope="module")
def sph():
    return MiniGeo(0.9, 10.0, 0.0, 0.5)


@pytest.fixture(scope="module")
def sph_retro():
    return MiniGeo(0.9, 10.0, 0.0, -0.5)   # retrograde: Upsilon_theta < 0


def test_polar_turning_points_and_parity(sph):
    assert sph.th(0.0) == pytest.approx(np.arccos(np.sqrt(sph.polar_roots[0])),
                                        rel=1e-12)
    assert sph.th(np.pi) == pytest.approx(np.pi - sph.th(0.0), rel=1e-12)
    q = np.linspace(0.1, 3.0, 7)
    np.testing.assert_allclose(sph.th(-q), sph.th(q), rtol=1e-12)
    np.testing.assert_allclose(sph.tth(-q), -sph.tth(q), atol=1e-12)
    np.testing.assert_allclose(sph.phith(-q), -sph.phith(q), atol=1e-12)


def _check_uth_matches_dtheta_dlambda(geo):
    q = 2 * np.pi * np.arange(1, 32) / 32
    q = q[np.abs(np.sin(q)) > 0.1]        # keep away from turning points
    dq = 1e-6
    dth = (geo.th(q + dq) - geo.th(q - dq)) / (2 * dq)
    # mino_frequencies[2] stores |Upsilon_theta| (pybhpt convention); the
    # true signed Mino frequency for this identity is _uth_sign * that.
    upsTh_signed = geo._uth_sign * geo.mino_frequencies[2]
    np.testing.assert_allclose(geo.uth(q), upsTh_signed * dth, atol=1e-4)


def test_uth_matches_dtheta_dlambda(sph):
    _check_uth_matches_dtheta_dlambda(sph)


def test_uth_matches_dtheta_dlambda_retrograde(sph_retro):
    # locks in the sign(Upsilon_theta) factor found in Task 1: without it
    # this fails by O(1) for retrograde orbits, not by numerical noise
    assert sph_retro._uth_sign < 0
    _check_uth_matches_dtheta_dlambda(sph_retro)


def test_mode_frequency_uses_unsigned_upsilon_theta(sph_retro):
    # pybhpt's own mode_frequency uses |Upsilon_theta| even when retrograde
    # (verified against pybhpt in dev/check_polar_conventions.py); this is
    # the numpy-only regression guard for that convention
    assert sph_retro.mino_frequencies[2] > 0
    om = sph_retro.mode_frequency(2, 1, 0)
    # frequencies = (Omega_r, Omega_theta, Omega_phi); mode_frequency(m,k,n)
    # = dot(frequencies, (n, k, m))
    expected = 1 * sph_retro.frequencies[1] + 2 * sph_retro.frequencies[2]
    assert om == pytest.approx(expected, rel=1e-12)


def test_uth_sign_convention(sph):
    # theta runs theta_min -> theta_max on (0, pi): uth > 0 there, < 0 after
    # (this orbit is prograde: Upsilon_theta > 0, same sign as the radial case)
    assert sph.uth(1.0) > 0
    assert sph.uth(2 * np.pi - 1.0) < 0
    assert abs(sph.uth(0.0)) < 1e-12


def test_uth_sign_convention_retrograde(sph_retro):
    # retrograde: Upsilon_theta < 0 flips the sign relative to the prograde case
    assert sph_retro.uth(1.0) < 0
    assert sph_retro.uth(2 * np.pi - 1.0) > 0
    assert abs(sph_retro.uth(0.0)) < 1e-12


def test_equatorial_orbits_unchanged():
    geo = MiniGeo(0.9, 10.0, 0.2, 1.0)    # the shared test orbit
    q = np.linspace(0.0, 2 * np.pi, 9)
    np.testing.assert_allclose(geo.th(q), 0.5 * np.pi, atol=1e-12)
    np.testing.assert_allclose(geo.uth(q), 0.0, atol=1e-12)
    np.testing.assert_allclose(geo.tth(q), 0.0, atol=1e-12)
    np.testing.assert_allclose(geo.phith(q), 0.0, atol=1e-12)
    # the legacy positional call the CPU s=+2 source path makes
    assert geo.uth(0, 0.0) == 0.0
