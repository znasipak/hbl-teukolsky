# Generate pybhpt cross-validation snapshot for the flux/amplitude tests.
# Requires the pybhpt package (run with the pybhpt-dev-env python):
#   ~/miniforge3/envs/pybhpt-dev-env/bin/python tests/generate_pybhpt_reference.py
# Writes tests/golden/pybhpt_reference_v2.npz (committed) so the pytest suite
# never needs pybhpt installed. See dev/PHASE5A.md.
import pathlib

import numpy as np
from pybhpt.flux import FluxMode
from pybhpt.geo import KerrGeodesic
from pybhpt.teuk import TeukolskyMode

OUT = pathlib.Path(__file__).parent / "golden" / "pybhpt_reference_v2.npz"

ORBITS = {
    "ecc": ((0.9, 10.0, 0.2, 1.0),
            [(2, 2, n) for n in range(-3, 4)] + [(3, 1, 1), (10, 7, 0), (4, -3, 1)]),
    "circ": ((0.9, 10.0, 0.0, 1.0),
             [(2, 2, 0), (2, 1, 0), (3, 3, 0), (5, 4, 0)]),
}

# spherical-inclined table (e = 0, |x| < 1): k-modes, n = 0. Column 2 is k.
SPH_ORBIT = (0.9, 10.0, 0.0, 0.5)
SPH_MODES = ([(2, 2, k) for k in range(-2, 3)]
             + [(2, 1, 1), (3, 3, 1), (3, 1, -2), (4, -3, 1), (5, 4, 2)])

# generic table (e != 0, |x| < 1): 16 columns incl. Carter flux.
GEN_ORBIT = (0.9, 8.0, 0.3, 0.6)
GEN_MODES = [(2, 2, 0, 0), (2, 2, 1, 0), (2, 2, 0, 1), (2, 2, 1, 1),
             (2, 2, -1, 1), (2, 2, 1, -1), (2, 1, 1, 1), (3, 2, 1, 1),
             (3, 3, -1, 2), (4, -3, 1, 1), (5, 4, 2, -1), (2, 2, 2, 2)]

# v2 (2026-07-30): RETROGRADE cases. Every v1 orbit was prograde (x = 1, 1,
# +0.5, +0.6), which is how a retrograde-only bug survived -- MiniGeo.uth
# applied sign(Upsilon_theta) while omega used |Upsilon_theta|, leaving
# retrograde inclined amplitudes wrong by 0.1-58x. Both signs of k are
# included because the error was strongly +k/-k asymmetric. polar_retro is a
# near-polar (95 deg) orbit reported as pathological by the flux driver.
SPH_RETRO_ORBIT = (0.9, 10.0, 0.0, -0.5)
GEN_RETRO_ORBIT = (0.9, 12.0, 0.3, -0.6)
POLAR_RETRO_ORBIT = (0.833625, 15.491158029478878, 0.0, -0.08655594470862094)
RETRO_SPH_MODES = ([(2, 2, k) for k in range(-2, 3)]
                   + [(2, 1, 1), (3, 2, 1), (3, 3, -1), (4, 2, 0),
                      (4, 3, -1), (5, 4, 2)])
RETRO_GEN_MODES = [(2, 2, 0, 0), (2, 2, 1, 0), (2, 2, -1, 0), (2, 2, 0, 1),
                   (2, 2, 1, 1), (2, 2, -1, 1), (3, 2, 1, -1), (4, 3, -1, 1)]


def main():
    data = {}
    for tag, (params, modes) in ORBITS.items():
        geo = KerrGeodesic(*params, nsamples=2**9)
        rows = []
        for (l, m, n) in modes:
            tm = TeukolskyMode(-2, l, m, 0, n, geo)
            tm.solve(geo)
            fx = FluxMode(geo, tm)
            zin, zup = tm.amplitude("In"), tm.amplitude("Up")
            rows.append([l, m, n, tm.frequency, tm.eigenvalue,
                         zin.real, zin.imag, zup.real, zup.imag,
                         fx.Edot["I"], fx.Edot["H"],
                         fx.Ldot["I"], fx.Ldot["H"]])
            print(f"{tag} ({l},{m},{n}): Edot_inf = {fx.Edot['I']:.6e}")
        data[f"{tag}_params"] = np.array(params)
        data[f"{tag}_table"] = np.array(rows)

    geo = KerrGeodesic(*SPH_ORBIT, nsamples=2**9)
    rows = []
    for (l, m, k) in SPH_MODES:
        tm = TeukolskyMode(-2, l, m, k, 0, geo)
        tm.solve(geo)
        fx = FluxMode(geo, tm)
        zin, zup = tm.amplitude("In"), tm.amplitude("Up")
        rows.append([l, m, k, tm.frequency, tm.eigenvalue,
                     zin.real, zin.imag, zup.real, zup.imag,
                     fx.Edot["I"], fx.Edot["H"],
                     fx.Ldot["I"], fx.Ldot["H"]])
        print(f"sph ({l},{m},{k}): |Zup| = {abs(zup):.6e}, Edot_inf = {fx.Edot['I']:.6e}")
    data["sph_params"] = np.array(SPH_ORBIT)
    data["sph_table"] = np.array(rows)

    geo = KerrGeodesic(*GEN_ORBIT, nsamples=2**9)
    rows = []
    for (l, m, k, n) in GEN_MODES:
        tm = TeukolskyMode(-2, l, m, k, n, geo)
        tm.solve(geo)
        fx = FluxMode(geo, tm)
        zin, zup = tm.amplitude("In"), tm.amplitude("Up")
        rows.append([l, m, k, n, tm.frequency, tm.eigenvalue,
                     zin.real, zin.imag, zup.real, zup.imag,
                     fx.Edot["I"], fx.Edot["H"],
                     fx.Ldot["I"], fx.Ldot["H"],
                     fx.Qdot["I"], fx.Qdot["H"]])
        print(f"gen ({l},{m},{k},{n}): |Zup| = {abs(zup):.6e}, "
              f"Edot_inf = {fx.Edot['I']:.6e}")
    data["gen_params"] = np.array(GEN_ORBIT)
    data["gen_table"] = np.array(rows)

    # --- retrograde spherical tables (v2) ---
    for tag, orbit in (("sph_retro", SPH_RETRO_ORBIT),
                       ("polar_retro", POLAR_RETRO_ORBIT)):
        geo = KerrGeodesic(*orbit, nsamples=2**9)
        rows = []
        for (l, m, k) in RETRO_SPH_MODES:
            tm = TeukolskyMode(-2, l, m, k, 0, geo)
            tm.solve(geo)
            fx = FluxMode(geo, tm)
            zin, zup = tm.amplitude("In"), tm.amplitude("Up")
            rows.append([l, m, k, tm.frequency, tm.eigenvalue,
                         zin.real, zin.imag, zup.real, zup.imag,
                         fx.Edot["I"], fx.Edot["H"],
                         fx.Ldot["I"], fx.Ldot["H"]])
            print(f"{tag} ({l},{m},{k}): |Zup| = {abs(zup):.6e}, "
                  f"Edot_inf = {fx.Edot['I']:.6e}")
        data[f"{tag}_params"] = np.array(orbit)
        data[f"{tag}_table"] = np.array(rows)

    # --- retrograde generic table (v2), incl. Carter flux ---
    geo = KerrGeodesic(*GEN_RETRO_ORBIT, nsamples=2**9)
    rows = []
    for (l, m, k, n) in RETRO_GEN_MODES:
        tm = TeukolskyMode(-2, l, m, k, n, geo)
        tm.solve(geo)
        fx = FluxMode(geo, tm)
        zin, zup = tm.amplitude("In"), tm.amplitude("Up")
        rows.append([l, m, k, n, tm.frequency, tm.eigenvalue,
                     zin.real, zin.imag, zup.real, zup.imag,
                     fx.Edot["I"], fx.Edot["H"],
                     fx.Ldot["I"], fx.Ldot["H"],
                     fx.Qdot["I"], fx.Qdot["H"]])
        print(f"gen_retro ({l},{m},{k},{n}): |Zup| = {abs(zup):.6e}, "
              f"Edot_inf = {fx.Edot['I']:.6e}")
    data["gen_retro_params"] = np.array(GEN_RETRO_ORBIT)
    data["gen_retro_table"] = np.array(rows)

    np.savez_compressed(OUT, **data)
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} kB)")


if __name__ == "__main__":
    main()
