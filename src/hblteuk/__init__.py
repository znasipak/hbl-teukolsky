"""hblteuk: Frequency-domain Teukolsky solver using hyperboloidal slicing,
spatial compactification, and pseudospectral (Chebyshev) collocation.

Submodules
----------
teuk
    Radial Teukolsky equation, hyperboloidal solvers, Teukolsky-Starobinsky relations.
mode
    Point-particle mode (and mode-grid) generators.
hertz
    Hertz potential / metric reconstruction mode generators.
metric
    Metric reconstruction (IRG/ORG) from Teukolsky/Hertz modes.
source
    Source integration for point-particle modes.
swsh
    Spin-weighted spheroidal/spherical harmonics helpers.
collocode, cheby
    Chebyshev spectral collocation machinery (ODE solvers, multi-domain bases).
geo, utils
    Geodesic helpers and small utilities.

The submodules are not imported eagerly: importing them triggers numba
compilation and pulls in the full dependency stack, so import only what you
need, e.g.::

    from hblteuk.teuk import HyperboloidalTeukolsky
    from hblteuk.mode import TeukolskyPointParticleModeGenerator
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hbl-teukolsky")
except PackageNotFoundError:  # package not installed (e.g. running from source tree)
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
