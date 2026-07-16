# Memory-bounded chunking of solve_mode_table: chunked results must equal
# the unchunked path to machine precision. See dev/MEMORY_CHUNKING_DESIGN.md.
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


def budget_for_chunk(csize, nsamples=2**8, ndomains=40, nnodes=32):
    # budget (GB) that yields exactly csize modes/chunk (+0.5 mode of
    # headroom against float rounding in the division)
    return (csize + 0.5) * _per_mode_bytes(nsamples, ndomains, nnodes) / 2**30


def test_chunk_size_arithmetic():
    assert _chunk_size(None, 2**10, 150, 32) is None
    assert _chunk_size(budget_for_chunk(7), 2**8, 40, 32) == 7
    assert _chunk_size(1e-9, 2**13, 150, 32) == 1      # floors at one mode
    # finer sampling costs more memory -> smaller chunks at fixed budget
    assert _chunk_size(8.0, 2**13, 150, 32) < _chunk_size(8.0, 2**10, 150, 32)
