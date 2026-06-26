# hbl-teukolsky
Solving the Teukolsky equation via hyperboloidal slicing, spatial compactification, and pseudospectral collocation.

This code is still in development, so please do not re-use or replicate this code or any of its methods without consulting me first. If you are interested in using this code for research, please reach out to me znasipak[at]gmail[dot]com

A tutorial of the developmental version of this code is provided in the Jupyter notebook `tutorial.ipynb`

## Installation

The package uses a `src/` layout and is built with [hatchling](https://hatch.pypa.io/).
Install it (editable, for development) from the repository root:

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[dev]"    # pytest, build, ipykernel
pip install -e ".[cuda]"   # CuPy (experimental GPU backend; see note below)
pip install -e ".[jax]"    # JAX  (experimental GPU backend)
```

> **CuPy note:** CuPy wheels are CUDA-version specific. Prefer installing the
> matching prebuilt wheel directly, e.g. `pip install cupy-cuda12x`, rather than
> the generic `cupy` source build.

Runtime dependencies (`numpy`, `scipy`, `numba`, `kerrgeopy`, `spherical`,
`spheroidal`) are resolved automatically from PyPI.

## Usage

Submodules are imported explicitly (they are not pulled in eagerly, so that
`import hblteuk` does not trigger numba compilation of the whole stack):

```python
from hblteuk.teuk import HyperboloidalTeukolsky, HyperboloidalTeukolskySolver
from hblteuk.mode import TeukolskyPointParticleModeGenerator
from hblteuk.metric import MetricReconstructor
```

The earlier monolithic implementation is preserved under `legacy/hblteuk.py`.
It is **not** part of the installed package; import it only by running from a
source checkout.
