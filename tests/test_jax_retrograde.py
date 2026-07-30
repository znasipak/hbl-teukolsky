# Retrograde (x < 0) validation against the independent pybhpt reference.
#
# Every orbit in pybhpt_reference_v1 was prograde (x = 1, 1, +0.5, +0.6), which
# is exactly how a retrograde-only bug survived: MiniGeo.uth applied
# sign(Upsilon_theta) to the polar velocity while mode_frequency used
# |Upsilon_theta|, mixing handedness in the source integral. Retrograde
# inclined amplitudes were wrong by 0.1-58x, strongly asymmetric in +k vs -k;
# prograde was untouched because sign(Upsilon_theta) = +1 there. See
# geo.MiniGeo.uth and dev/PHASE5B.md.
import pathlib

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from hblteuk.geo import MiniGeo
from hblteuk.jax import flux_modes, solve_modes

REF = pathlib.Path(__file__).parent / "golden" / "pybhpt_reference_v2.npz"
# spherical tables: l, m, k, omega, lam, Zin, Zup, Edot_I/H, Ldot_I/H
SPH = dict(l=0, m=1, k=2, omega=3, lam=4, zin_re=5, zin_im=6, zup_re=7,
           zup_im=8, edot_i=9, edot_h=10, ldot_i=11, ldot_h=12)
# generic tables carry an extra n column and the Carter flux
GEN = dict(l=0, m=1, k=2, n=3, omega=4, lam=5, zin_re=6, zin_im=7, zup_re=8,
           zup_im=9, edot_i=10, edot_h=11, ldot_i=12, ldot_h=13, qdot_i=14,
           qdot_h=15)


@pytest.fixture(scope="module")
def ref():
    return np.load(REF)


def _solve(ref, tag, cols, has_n):
    t = ref[f"{tag}_table"]
    orbit = MiniGeo(*ref[f"{tag}_params"])
    l = t[:, cols["l"]].astype(int)
    m = t[:, cols["m"]].astype(int)
    k = t[:, cols["k"]].astype(int)
    n = t[:, cols["n"]].astype(int) if has_n else np.zeros(len(l), int)
    kw = dict(max_memory_gb=2.0, return_radial=False)
    kw["ntheta"] = 2**11
    if has_n:
        kw["nsamples"] = 2**10
    return orbit, t, solve_modes(orbit, -2, l, m, k, n, **kw)


@pytest.mark.parametrize("tag", ["sph_retro", "polar_retro"])
def test_retrograde_spherical_amplitudes_vs_pybhpt(ref, tag):
    # magnitude AND phase -- the bug was largely a phase/handedness error, so a
    # magnitude-only check would have partially passed
    orbit, t, sol = _solve(ref, tag, SPH, has_n=False)
    np.testing.assert_allclose(np.asarray(sol.omega), t[:, SPH["omega"]],
                               rtol=1e-11)
    np.testing.assert_allclose(np.asarray(sol.eigenvalues), t[:, SPH["lam"]],
                               rtol=1e-9)
    zin = t[:, SPH["zin_re"]] + 1j * t[:, SPH["zin_im"]]
    zup = t[:, SPH["zup_re"]] + 1j * t[:, SPH["zup_im"]]
    amps = np.asarray(sol.amplitudes)
    np.testing.assert_allclose(amps[:, 0], zin, rtol=1e-6)
    np.testing.assert_allclose(amps[:, 1], zup, rtol=1e-6)


@pytest.mark.parametrize("tag", ["sph_retro", "polar_retro"])
def test_retrograde_spherical_fluxes_vs_pybhpt(ref, tag):
    orbit, t, sol = _solve(ref, tag, SPH, has_n=False)
    fx = flux_modes(sol.amplitudes, sol.m, sol.omega, sol.eigenvalues,
                    orbit.a, k=sol.k, n=sol.n,
                    carter_frequencies=orbit.carter_frequencies())
    Edot, Ldot = np.asarray(fx.Edot), np.asarray(fx.Ldot)
    np.testing.assert_allclose(Edot[:, 0], t[:, SPH["edot_i"]], rtol=1e-6)
    np.testing.assert_allclose(Edot[:, 1], t[:, SPH["edot_h"]], rtol=1e-6)
    np.testing.assert_allclose(Ldot[:, 0], t[:, SPH["ldot_i"]], rtol=1e-6)
    np.testing.assert_allclose(Ldot[:, 1], t[:, SPH["ldot_h"]], rtol=1e-6)


def test_retrograde_generic_amplitudes_vs_pybhpt(ref):
    # the generic (e != 0, |x| < 1) path uses the same polar velocity, so it
    # carried the same bug
    orbit, t, sol = _solve(ref, "gen_retro", GEN, has_n=True)
    zin = t[:, GEN["zin_re"]] + 1j * t[:, GEN["zin_im"]]
    zup = t[:, GEN["zup_re"]] + 1j * t[:, GEN["zup_im"]]
    amps = np.asarray(sol.amplitudes)
    np.testing.assert_allclose(amps[:, 0], zin, rtol=1e-6)
    np.testing.assert_allclose(amps[:, 1], zup, rtol=1e-6)


def test_retrograde_generic_fluxes_vs_pybhpt(ref):
    orbit, t, sol = _solve(ref, "gen_retro", GEN, has_n=True)
    fx = flux_modes(sol.amplitudes, sol.m, sol.omega, sol.eigenvalues,
                    orbit.a, k=sol.k, n=sol.n,
                    carter_frequencies=orbit.carter_frequencies())
    for arr, ci, ch in ((np.asarray(fx.Edot), "edot_i", "edot_h"),
                        (np.asarray(fx.Ldot), "ldot_i", "ldot_h"),
                        (np.asarray(fx.Qdot), "qdot_i", "qdot_h")):
        np.testing.assert_allclose(arr[:, 0], t[:, GEN[ci]], rtol=1e-5)
        np.testing.assert_allclose(arr[:, 1], t[:, GEN[ch]], rtol=1e-5)


def test_retrograde_k_asymmetry_is_correct(ref):
    # The bug was strongly +k/-k asymmetric: (2,2,-1) was off by 58x while
    # (2,2,+1) was off by 0.63x. Pin both signs against the reference so a
    # future handedness slip cannot pass by getting one sign right.
    t = ref["sph_retro_table"]
    orbit, _, sol = _solve(ref, "sph_retro", SPH, has_n=False)
    amps = np.asarray(sol.amplitudes)
    zup = t[:, SPH["zup_re"]] + 1j * t[:, SPH["zup_im"]]
    k = t[:, SPH["k"]].astype(int)
    for kk in (-2, -1, 1, 2):
        sel = k == kk
        assert sel.any(), f"reference has no k={kk} row"
        np.testing.assert_allclose(amps[sel, 1], zup[sel], rtol=1e-6,
                                   err_msg=f"k={kk}")
