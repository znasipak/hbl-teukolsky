import pathlib

import numpy as np
import pytest

from hblteuk.geo import MiniGeo

GOLDEN_PATH = pathlib.Path(__file__).parent / "golden" / "golden_v1.npz"

ORBIT = (0.9, 10.0, 0.2, 1.0)
SOLVER_KWARGS = {"subdomains": 150, "tol": 1e-13, "spacing": "arcsinh7"}


@pytest.fixture(scope="session")
def golden():
    return np.load(GOLDEN_PATH)


@pytest.fixture(scope="session")
def orbit():
    return MiniGeo(*ORBIT)
