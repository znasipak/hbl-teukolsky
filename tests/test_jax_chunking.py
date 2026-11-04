# Memory-bounded chunking of solve_mode_table: chunked results must be
# mathematically equal to the unchunked path. See dev/MEMORY_CHUNKING_DESIGN.md.
import numpy as np
import pytest

jax = pytest.importorskip("jax")

from hblteuk.jax import solve_mode_table
from hblteuk.jax.pipeline import _chunk_size, _per_mode_bytes

MODES = [(l, m, n) for l in (2, 3, 4) for m in range(1, l + 1)
         for n in (-2, -1, 0, 1, 2)]      # B = 45
L = np.array([c[0] for c in MODES])
M = np.array([c[1] for c in MODES])
N = np.array([c[2] for c in MODES])


def budget_for_chunk(csize, nsamples=2**8, ndomains=40, nnodes=32,
                     stabilize=False):
    # budget (GB) that yields exactly csize modes/chunk (+0.5 mode of
    # headroom against float rounding in the division). stabilize must match
    # what the solver under test uses: TS stabilization costs
    # ndomains*nnodes^2 per mode, so the same budget buys ~6x fewer modes.
    return ((csize + 0.5)
            * _per_mode_bytes(nsamples, ndomains, nnodes, stabilize) / 2**30)


def test_chunk_size_arithmetic():
    assert _chunk_size(None, 2**10, 150, 32) is None
    assert _chunk_size(budget_for_chunk(7), 2**8, 40, 32) == 7
    assert _chunk_size(1e-9, 2**13, 150, 32) == 1      # floors at one mode
    # finer sampling costs more memory -> smaller chunks at fixed budget
    assert _chunk_size(8.0, 2**13, 150, 32) < _chunk_size(8.0, 2**10, 150, 32)


@pytest.fixture(scope="module")
def pair(orbit):
    kwargs = dict(nsamples=2**8, ndomains=40)
    full = solve_mode_table(orbit, -2, L, M, N, max_memory_gb=None, **kwargs)
    # ceil(45 / 10) = 5 chunks, ragged (padded) final chunk of 5 modes
    # solve_mode_table stabilizes by default, so size the budget with the
    # stabilized model or the intended 5-chunk split collapses to 45 chunks
    chunked = solve_mode_table(orbit, -2, L, M, N,
                               max_memory_gb=budget_for_chunk(
                                   10, stabilize=True), **kwargs)
    return full, chunked


def test_chunked_equals_unchunked(pair):
    # Chunked and unchunked solves are mathematically identical (modes are
    # independent), but not bit-identical: JAX's batched linear-algebra
    # kernels are not guaranteed bit-reproducible across different total
    # batch sizes (verified directly against solve_homogeneous_batch,
    # independent of the chunking loop -- see dev/MEMORY_CHUNKING_DESIGN.md
    # Results). rtol=1e-8 is still five orders of magnitude tighter than
    # the pybhpt physical-validation gate (1e-7) used elsewhere.
    full, chunked = pair
    np.testing.assert_allclose(np.asarray(chunked.amplitudes),
                               np.asarray(full.amplitudes), rtol=1e-8)
    # error ESTIMATES sit at the floating-point noise floor (~1e-14) for
    # these well-converged modes; an absolute floor avoids comparing noise
    np.testing.assert_allclose(np.asarray(chunked.errors),
                               np.asarray(full.errors), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(chunked.eigenvalues, full.eigenvalues,
                               rtol=1e-13)
    np.testing.assert_allclose(chunked.coupling, full.coupling,
                               rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(np.asarray(chunked.hom.coeffs_up),
                               np.asarray(full.hom.coeffs_up),
                               rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(np.asarray(chunked.hom.coeffs_in),
                               np.asarray(full.hom.coeffs_in),
                               rtol=1e-8, atol=1e-12)
    np.testing.assert_array_equal(chunked.omega, full.omega)
    np.testing.assert_array_equal(chunked.lmin, full.lmin)


def test_return_radial_false(orbit, pair):
    full, _ = pair
    slim = solve_mode_table(orbit, -2, L, M, N, nsamples=2**8, ndomains=40,
                            max_memory_gb=budget_for_chunk(10),
                            return_radial=False)
    assert slim.hom is None
    np.testing.assert_allclose(np.asarray(slim.amplitudes),
                               np.asarray(full.amplitudes), rtol=1e-8)
    np.testing.assert_allclose(slim.eigenvalues, full.eigenvalues, rtol=1e-13)


def test_chunk_model_accounts_for_stabilization():
    # Regression: the model omitted the TS collocation system, (B, ndomains,
    # nnodes, nnodes), so enabling stabilization by default understated the
    # working set several-fold and chunking stopped bounding memory (measured
    # 23 GB peak against a 2 GB budget before this term was added).
    # (Since the parallel radial solve, both paths carry an n^2 term; the
    # TS system is the extra one.)
    plain = _per_mode_bytes(2**8, 150, 32, False)
    stab = _per_mode_bytes(2**8, 150, 32, True)
    assert stab - plain == 3 * 150 * 32**2 * 16
    assert _chunk_size(8.0, 2**8, 150, 32, True) < \
        _chunk_size(8.0, 2**8, 150, 32, False) / 1.8


def test_per_mode_bytes_counts_parallel_solve():
    # all-subdomain parallel solve holds (ndomains, nnodes, nnodes) per mode
    assert (_per_mode_bytes(1024, 150, 32) - (58 * 1024 + 8 * 150 * 32) * 16
            == 3 * 150 * 32**2 * 16)


def test_generic_chunked_matches_unchunked():
    # Review Focus 1: padded last chunk through the generic path
    import pathlib
    from hblteuk.geo import MiniGeo
    from hblteuk.jax import solve_generic_mode_table
    ref = np.load(pathlib.Path(__file__).parent / "golden"
                  / "pybhpt_reference_v2.npz")
    orb = MiniGeo(*ref["gen_params"])
    modes = [(l, m, k, n) for l in (2, 3) for m in (-l, 1, l)
             for k in (-1, 0, 1) for n in (-1, 1)]      # B = 36
    l, m, k, n = (np.array(c) for c in zip(*modes))
    kw = dict(nsamples=2**6, ntheta=2**5, ndomains=40, return_radial=False)
    full = solve_generic_mode_table(orb, -2, l, m, k, n, max_memory_gb=None,
                                    **kw)
    # 10 modes/chunk -> 4 chunks, last padded with 4 real modes
    chunked = solve_generic_mode_table(
        orb, -2, l, m, k, n,
        max_memory_gb=budget_for_chunk(10, nsamples=2**6, stabilize=True),
        **kw)
    np.testing.assert_allclose(np.asarray(chunked.amplitudes),
                               np.asarray(full.amplitudes), rtol=1e-8)
