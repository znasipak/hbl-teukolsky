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
                     integrate_source_batch, SourceAmplitudes,
                     polar_samples, PolarSamples, polar_swsh_values,
                     integrate_source_spherical_batch,
                     generic_samples, GenericSamples,
                     integrate_source_generic_batch)
from .summation import (swsh_theta_values, project_to_spherical,
                        assemble_field, waveform_timeseries)
from .pipeline import (solve_mode_table, solve_circular_grid, ModeTableSolution,
                       solve_spherical_mode_table, SphericalModeSolution,
                       solve_generic_mode_table, GenericModeSolution)
from .fluxes import flux_modes, horizon_constant, FluxModes

__all__ = ["spheroidal_eigs_batched", "sphericalY_basis_matrix",
           "evaluate_multidomain", "solve_homogeneous_batch",
           "HomogeneousBatch", "orbit_samples", "equatorial_swsh_values",
           "integrate_source_batch", "SourceAmplitudes", "swsh_theta_values",
           "project_to_spherical", "assemble_field", "waveform_timeseries",
           "solve_mode_table", "solve_circular_grid", "ModeTableSolution",
           "flux_modes", "horizon_constant", "FluxModes",
           "polar_samples", "PolarSamples", "polar_swsh_values",
           "integrate_source_spherical_batch", "solve_spherical_mode_table",
           "SphericalModeSolution", "generic_samples", "GenericSamples",
           "integrate_source_generic_batch", "solve_generic_mode_table",
           "GenericModeSolution"]
