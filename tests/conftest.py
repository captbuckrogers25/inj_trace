"""
Pytest configuration and fixtures for inj_trace tests.

Tests that require lgmpy or ptm are marked with @pytest.mark.needs_lgmpy
and @pytest.mark.needs_ptm respectively.  They are skipped automatically
when the upstream packages are not configured.
"""

import pytest
import numpy as np


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_lgmpy: requires lgmpy to be importable")
    config.addinivalue_line("markers", "needs_ptm: requires ptm executable")


def lgmpy_available():
    try:
        from inj_trace.config import load_config
        load_config()
        import lgmpy  # noqa
        return True
    except Exception:
        return False


def ptm_available():
    try:
        from inj_trace.config import load_config
        cfg = load_config()
        from pathlib import Path
        return Path(cfg.ptm_executable).is_file()
    except Exception:
        return False


@pytest.fixture(autouse=True)
def skip_if_no_lgmpy(request):
    if request.node.get_closest_marker("needs_lgmpy"):
        if not lgmpy_available():
            pytest.skip("lgmpy not available")


@pytest.fixture(autouse=True)
def skip_if_no_ptm(request):
    if request.node.get_closest_marker("needs_ptm"):
        if not ptm_available():
            pytest.skip("ptm executable not available")


@pytest.fixture
def simple_grid():
    """Small 5x5x3 FieldGrid for fast tests (without lgmpy)."""
    import numpy as np
    from inj_trace.fields.grid import FieldGrid
    grid = FieldGrid(
        xvec=np.linspace(-6, -2, 5),
        yvec=np.linspace(-2,  2, 5),
        zvec=np.linspace(-1,  1, 3),
    )
    # Manually populate with a simple dipole-like B
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    bz = np.zeros((nx, ny, nz))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                r = np.sqrt(grid.xvec[i]**2 + grid.yvec[j]**2 + grid.zvec[k]**2)
                bz[i, j, k] = 1000.0 / r**3 if r > 0.5 else 0.0
    grid.bx = np.zeros_like(bz)
    grid.by = np.zeros_like(bz)
    grid.bz = bz
    grid.set_zero_efield()
    return grid
