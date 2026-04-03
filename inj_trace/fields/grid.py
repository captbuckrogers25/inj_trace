"""
FieldGrid: evaluate an empirical magnetic field model on a regular 3-D
Cartesian grid in GSM coordinates (Earth radii).

The resulting B-field arrays are suitable as input to PTMFieldWriter.

Usage
-----
    from inj_trace.fields.grid import FieldGrid
    from datetime import datetime

    grid = FieldGrid.from_spec({'xmin': -12, 'xmax': 12, 'nx': 51,
                                'ymin': -12, 'ymax': 12, 'ny': 51,
                                'zmin':  -6, 'zmax':  6, 'nz': 25})
    grid.evaluate('T89', datetime(2013, 3, 17, 6), {'Kp': 5}, n_workers=4)
    print(grid.bmag().max())   # peak |B| in nT
"""

import itertools
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Top-level worker (must be at module scope to be picklable by multiprocessing)
# ---------------------------------------------------------------------------

def _eval_chunk(args):
    """Evaluate a list of grid points in a worker process.

    args = (model_name, positions, time_iso, model_params)
    Returns list of [Bx, By, Bz] for each position.
    """
    model_name, positions, time_iso, model_params = args
    # Ensure paths are injected in this worker process (Linux fork inherits
    # sys.path, but be defensive in case of spawn start method)
    try:
        from inj_trace.config import load_config
        load_config()
    except Exception:
        pass
    time = datetime.fromisoformat(time_iso)
    from inj_trace.fields.models import MODELS
    fn = MODELS[model_name]
    results = []
    for pos in positions:
        try:
            B = fn(pos, time, **model_params)
            results.append(B)
        except Exception:
            results.append([0.0, 0.0, 0.0])
    return results


# ---------------------------------------------------------------------------
# FieldGrid
# ---------------------------------------------------------------------------

class FieldGrid:
    """3-D GSM magnetic field grid evaluated from an empirical lgmpy model.

    Attributes (available after evaluate())
    ----------------------------------------
    bx, by, bz : ndarray, shape (nx, ny, nz)   nT
    ex, ey, ez : ndarray, shape (nx, ny, nz)   mV/m  (always zero for static runs)
    xvec, yvec, zvec : 1-D arrays              Re (GSM)
    """

    def __init__(
        self,
        xvec: np.ndarray,
        yvec: np.ndarray,
        zvec: np.ndarray,
    ) -> None:
        self.xvec = np.asarray(xvec, dtype=float)
        self.yvec = np.asarray(yvec, dtype=float)
        self.zvec = np.asarray(zvec, dtype=float)
        self.nx = len(self.xvec)
        self.ny = len(self.yvec)
        self.nz = len(self.zvec)
        self.bx: Optional[np.ndarray] = None
        self.by: Optional[np.ndarray] = None
        self.bz: Optional[np.ndarray] = None
        self.ex: Optional[np.ndarray] = None
        self.ey: Optional[np.ndarray] = None
        self.ez: Optional[np.ndarray] = None

    @classmethod
    def from_spec(cls, spec: Dict) -> "FieldGrid":
        """Construct from a grid specification dict.

        Required keys: xmin, xmax, nx, ymin, ymax, ny, zmin, zmax, nz
        """
        x = np.linspace(spec["xmin"], spec["xmax"], int(spec["nx"]))
        y = np.linspace(spec["ymin"], spec["ymax"], int(spec["ny"]))
        z = np.linspace(spec["zmin"], spec["zmax"], int(spec["nz"]))
        return cls(x, y, z)

    def evaluate(
        self,
        model: str,
        time: datetime,
        model_params: dict,
        mask_inner: float = 1.2,
        n_workers: int = 1,
    ) -> None:
        """Evaluate the field model at every grid point.

        Parameters
        ----------
        model       : one of 'T89', 'TS04', 'OP77'
        time        : UTC datetime
        model_params: kwargs forwarded to the model function
                      (e.g. {'Kp': 5} for T89)
        mask_inner  : zero B inside this radius (Re) to avoid unphysical values
        n_workers   : >1 uses multiprocessing (ProcessPoolExecutor) to
                      parallelise calls to lgmpy. Each process imports lgmpy
                      independently (required: lgmpy ctypes is not thread-safe).
        """
        # Build flat list of (index_tuple, position) pairs
        index_tuples = list(itertools.product(
            range(self.nx), range(self.ny), range(self.nz)
        ))
        positions = [
            [self.xvec[i], self.yvec[j], self.zvec[k]]
            for i, j, k in index_tuples
        ]
        time_iso = time.isoformat()

        # Evaluate
        if n_workers <= 1:
            flat_args = (model, positions, time_iso, model_params)
            B_flat = _eval_chunk(flat_args)
        else:
            B_flat = self._eval_parallel(
                model, positions, time_iso, model_params, n_workers
            )

        # Assemble into (nx, ny, nz) arrays, applying inner-boundary mask
        bx = np.zeros((self.nx, self.ny, self.nz))
        by = np.zeros_like(bx)
        bz = np.zeros_like(bx)

        for idx, (i, j, k) in enumerate(index_tuples):
            r = np.sqrt(
                self.xvec[i] ** 2
                + self.yvec[j] ** 2
                + self.zvec[k] ** 2
            )
            if r >= mask_inner:
                bx[i, j, k] = B_flat[idx][0]
                by[i, j, k] = B_flat[idx][1]
                bz[i, j, k] = B_flat[idx][2]

        self.bx = bx
        self.by = by
        self.bz = bz
        self.set_zero_efield()

    def _eval_parallel(
        self,
        model: str,
        positions,
        time_iso: str,
        model_params: dict,
        n_workers: int,
    ):
        """Split positions into chunks and evaluate in parallel processes."""
        chunk_size = max(1, len(positions) // n_workers)
        chunks = [
            positions[i: i + chunk_size]
            for i in range(0, len(positions), chunk_size)
        ]
        args_list = [
            (model, chunk, time_iso, model_params) for chunk in chunks
        ]
        B_flat = []
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for chunk_result in pool.map(_eval_chunk, args_list):
                B_flat.extend(chunk_result)
        return B_flat

    def set_zero_efield(self) -> None:
        """Zero-fill electric field arrays (valid for static empirical models)."""
        shape = (self.nx, self.ny, self.nz)
        self.ex = np.zeros(shape)
        self.ey = np.zeros(shape)
        self.ez = np.zeros(shape)

    def set_efield_from_potential(self, potential: np.ndarray) -> None:
        """Extension hook: set E = -grad(Phi) from a given potential array.

        potential : ndarray, shape (nx, ny, nz), in kV
        E arrays set in mV/m (1 kV/Re ≈ 0.157 mV/m)
        """
        Re_km = 6371.0
        kV_per_Re_to_mV_per_m = 1e3 / (Re_km * 1e3)  # = 1/Re in mV/m per kV/Re
        # Negative gradient via central differences
        dx = self.xvec[1] - self.xvec[0] if self.nx > 1 else 1.0
        dy = self.yvec[1] - self.yvec[0] if self.ny > 1 else 1.0
        dz = self.zvec[1] - self.zvec[0] if self.nz > 1 else 1.0
        self.ex = -np.gradient(potential, dx, axis=0) * kV_per_Re_to_mV_per_m
        self.ey = -np.gradient(potential, dy, axis=1) * kV_per_Re_to_mV_per_m
        self.ez = -np.gradient(potential, dz, axis=2) * kV_per_Re_to_mV_per_m

    def bmag(self) -> np.ndarray:
        """Return |B| array, shape (nx, ny, nz), nT."""
        if self.bx is None:
            raise RuntimeError("Call evaluate() before bmag()")
        return np.sqrt(self.bx ** 2 + self.by ** 2 + self.bz ** 2)
