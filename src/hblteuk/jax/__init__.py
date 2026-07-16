"""JAX-accelerated batched backend. Requires `pip install hbl-teukolsky[jax]`.

Design notes: dev/GPU_PLAN.md and dev/PHASE1.md (not shipped with the wheel).
Public entry points are re-exported here; submodules mirror the CPU package
layout (swsh, teukolsky, source, ...).
"""

try:
    import jax as _jax
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "hblteuk.jax requires jax; install with `pip install hbl-teukolsky[jax]`"
    ) from _e

# full pipeline is float64/complex128; must be set before any array is created
_jax.config.update("jax_enable_x64", True)

from .swsh import spheroidal_eigs_batched, sphericalY_basis_matrix
from .chebyshev import evaluate_multidomain
from .teukolsky import solve_homogeneous_batch, HomogeneousBatch
from .source import (orbit_samples, equatorial_swsh_values,
                     integrate_source_batch, SourceAmplitudes)

__all__ = ["spheroidal_eigs_batched", "sphericalY_basis_matrix",
           "evaluate_multidomain", "solve_homogeneous_batch",
           "HomogeneousBatch", "orbit_samples", "equatorial_swsh_values",
           "integrate_source_batch", "SourceAmplitudes"]
