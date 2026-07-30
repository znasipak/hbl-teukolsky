# Per-mode fluxes and amplitude-normalization gate vs the committed pybhpt
# snapshot (tests/generate_pybhpt_reference.py). See dev/PHASE5A.md.
import pathlib

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from hblteuk.geo import MiniGeo
from hblteuk.jax import flux_modes, solve_mode_table
from hblteuk.jax.fluxes import teukolsky_starobinsky_abs_C_squared
from hblteuk.teuk import teukolsky_starobinsky_constant_D_squared

REF_PATH = pathlib.Path(__file__).parent / "golden" / "pybhpt_reference_v2.npz"
COLS = dict(l=0, m=1, n=2, omega=3, lam=4, zin_re=5, zin_im=6,
            zup_re=7, zup_im=8, edot_i=9, edot_h=10, ldot_i=11, ldot_h=12)


@pytest.fixture(scope="module")
def ref():
    return np.load(REF_PATH)


def solve_table(params, table):
    orbit = MiniGeo(*params)
    l = table[:, COLS["l"]].astype(int)
    m = table[:, COLS["m"]].astype(int)
    n = table[:, COLS["n"]].astype(int)
    return orbit, solve_mode_table(orbit, -2, l, m, n)


@pytest.mark.parametrize("tag", ["ecc", "circ"])
def test_amplitude_normalization_gate(ref, tag):
    # THE gate: hblteuk Z must equal pybhpt Z in magnitude AND phase
    table = ref[f"{tag}_table"]
    orbit, sol = solve_table(ref[f"{tag}_params"], table)
    np.testing.assert_allclose(sol.omega, table[:, COLS["omega"]], rtol=1e-12)
    np.testing.assert_allclose(sol.eigenvalues, table[:, COLS["lam"]],
                               rtol=1e-10)
    zin = table[:, COLS["zin_re"]] + 1j * table[:, COLS["zin_im"]]
    zup = table[:, COLS["zup_re"]] + 1j * table[:, COLS["zup_im"]]
    amps = np.asarray(sol.amplitudes)
    np.testing.assert_allclose(amps[:, 0], zin, rtol=1e-7)
    np.testing.assert_allclose(amps[:, 1], zup, rtol=1e-7)


@pytest.mark.parametrize("tag", ["ecc", "circ"])
def test_fluxes_vs_pybhpt(ref, tag):
    table = ref[f"{tag}_table"]
    orbit, sol = solve_table(ref[f"{tag}_params"], table)
    fx = flux_modes(sol.amplitudes, sol.m, sol.omega, sol.eigenvalues,
                    orbit.a)
    Edot, Ldot = np.asarray(fx.Edot), np.asarray(fx.Ldot)
    np.testing.assert_allclose(Edot[:, 0], table[:, COLS["edot_i"]], rtol=1e-7)
    np.testing.assert_allclose(Edot[:, 1], table[:, COLS["edot_h"]], rtol=1e-7)
    np.testing.assert_allclose(Ldot[:, 0], table[:, COLS["ldot_i"]], rtol=1e-7)
    np.testing.assert_allclose(Ldot[:, 1], table[:, COLS["ldot_h"]], rtol=1e-7)


def test_ts_constant_identity():
    # |C|^2 (pybhpt lam convention) == teuk.py D^2(lamCH = lam + 2) + 144 w^2
    rng = np.random.default_rng(1)
    for _ in range(20):
        m, a, w = rng.integers(-5, 6), rng.uniform(0, 0.99), rng.uniform(-0.5, 0.5)
        lam = rng.uniform(0, 40)
        c2 = float(teukolsky_starobinsky_abs_C_squared(float(m), a, w, lam))
        d2 = teukolsky_starobinsky_constant_D_squared(m, a, w, lam + 2.0)
        assert c2 == pytest.approx(d2 + 144 * w**2, rel=1e-12)


def test_static_modes_zero_flux():
    amps = np.array([[1.0 + 1j, 2.0 - 1j], [1.0, 1.0]])
    fx = flux_modes(amps, [2, 0], [0.5, 0.0], [4.0, 4.0], 0.9)
    assert np.all(np.asarray(fx.Edot)[1] == 0)
    assert np.all(np.asarray(fx.Ldot)[1] == 0)
    # infinity flux positive; horizon flux may be negative (superradiance)
    assert np.asarray(fx.Edot)[0, 0] > 0
    assert np.asarray(fx.Edot)[0, 1] != 0


def test_newtonian_limit():
    # total Edot^inf ~ quadrupole 32/5 (M Omega)^{10/3} at large p (a = 0)
    orbit = MiniGeo(0.0, 500.0, 0.0, 1.0)
    modes = [(l, m) for l in (2, 3, 4) for m in range(1, l + 1)]
    l = np.array([c[0] for c in modes])
    m = np.array([c[1] for c in modes])
    sol = solve_mode_table(orbit, -2, l, m, np.zeros_like(l))
    fx = flux_modes(sol.amplitudes, sol.m, sol.omega, sol.eigenvalues, 0.0)
    total = 2 * float(np.asarray(fx.Edot)[:, 0].sum())   # x2 folds in -m
    newt = 32.0 / 5.0 * orbit.frequencies[2] ** (10.0 / 3.0)
    assert total == pytest.approx(newt, rel=2e-2)  # 1PN corrections ~ 1/p
