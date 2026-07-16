# Generate pybhpt cross-validation snapshot for the flux/amplitude tests.
# Requires the pybhpt package (run with the pybhpt-dev-env python):
#   ~/miniforge3/envs/pybhpt-dev-env/bin/python tests/generate_pybhpt_reference.py
# Writes tests/golden/pybhpt_reference_v1.npz (committed) so the pytest suite
# never needs pybhpt installed. See dev/PHASE5A.md.
import pathlib

import numpy as np
from pybhpt.flux import FluxMode
from pybhpt.geo import KerrGeodesic
from pybhpt.teuk import TeukolskyMode

OUT = pathlib.Path(__file__).parent / "golden" / "pybhpt_reference_v1.npz"

ORBITS = {
    "ecc": ((0.9, 10.0, 0.2, 1.0),
            [(2, 2, n) for n in range(-3, 4)] + [(3, 1, 1), (10, 7, 0), (4, -3, 1)]),
    "circ": ((0.9, 10.0, 0.0, 1.0),
             [(2, 2, 0), (2, 1, 0), (3, 3, 0), (5, 4, 0)]),
}


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
    np.savez_compressed(OUT, **data)
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} kB)")


if __name__ == "__main__":
    main()
