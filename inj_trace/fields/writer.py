"""
PTMFieldWriter: write LANLGeoMag-derived field data in the PTM input format.

PTM field file format (ptm_fields_XXXX.dat)
-------------------------------------------
Line 1 : "  nx   ny   nz"
Line 2 : nx x-coordinates formatted as "{:12.5e} " * nx
Line 3 : ny y-coordinates
Line 4 : nz z-coordinates
Body   : one line per grid point:
         "{i+1:4} {j+1:4} {k+1:4}  {Bx:12.5e}  {By:12.5e}  {Bz:12.5e}  {Ex:12.5e}  {Ey:12.5e}  {Ez:12.5e}"
         (1-based indices, fields loop over i,j,k in C order: outermost=i)

Time grid file (tgrid.dat)
--------------------------
Plain text column of float seconds since the run epoch (one value per line),
written by np.savetxt.

PTM requires at least TWO snapshots for its internal time interpolation.
For static (single-time) runs use write_static(), which duplicates the snapshot.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .grid import FieldGrid


class PTMFieldWriter:
    """Write PTM-format field files from a FieldGrid.

    Parameters
    ----------
    output_dir : directory where ptm_fields_XXXX.dat and tgrid.dat are written.
                 Created automatically if it does not exist.
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def write_snapshot(
        self,
        grid: FieldGrid,
        snapshot_index: int,
    ) -> Path:
        """Write a single ptm_fields_{snapshot_index:04d}.dat file.

        Parameters
        ----------
        grid           : evaluated FieldGrid (bx/by/bz must not be None)
        snapshot_index : 1-based integer used in the filename

        Returns
        -------
        Path to the written file.
        """
        self._check_grid(grid)
        fname = self.output_dir / f"ptm_fields_{snapshot_index:04d}.dat"
        _write_ptm_fields_fast(grid, str(fname))
        return fname

    def write_tgrid(self, times_seconds: List[float]) -> Path:
        """Write tgrid.dat from a list of epoch offsets in seconds.

        Parameters
        ----------
        times_seconds : e.g. [0.0, 3600.0] for a static run

        Returns
        -------
        Path to tgrid.dat.
        """
        path = self.output_dir / "tgrid.dat"
        np.savetxt(str(path), np.array(times_seconds, dtype=float))
        return path

    def write_static(
        self,
        grid: FieldGrid,
        duration_s: float = 3600.0,
    ) -> Tuple[Path, Path, Path]:
        """Write two identical snapshots + tgrid for a static-field PTM run.

        PTM requires at least two field snapshots for its time interpolation
        even when the field does not change.  This method writes snapshot
        indices 0001 and 0002 with identical content, and a tgrid.dat
        containing [0.0, duration_s].

        Parameters
        ----------
        grid       : evaluated FieldGrid
        duration_s : run duration in seconds (sets second tgrid entry)

        Returns
        -------
        (field_path_1, field_path_2, tgrid_path)
        """
        p1 = self.write_snapshot(grid, 1)
        p2 = self.write_snapshot(grid, 2)
        tg = self.write_tgrid([0.0, duration_s])
        return p1, p2, tg

    def write_time_series(
        self,
        grids: List[FieldGrid],
        dt: float,
        t0: float = 0.0,
    ) -> Tuple[List[Path], Path]:
        """Write multiple field snapshots for a time-varying run.

        Parameters
        ----------
        grids : list of evaluated FieldGrids, one per snapshot
        dt    : seconds between snapshots
        t0    : start time in seconds (default 0)

        Returns
        -------
        (list of field file paths, tgrid path)
        """
        paths = []
        for idx, grid in enumerate(grids, start=1):
            paths.append(self.write_snapshot(grid, idx))
        times = [t0 + i * dt for i in range(len(grids))]
        tg = self.write_tgrid(times)
        return paths, tg

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def times_from_datetimes(
        datetimes: List[datetime],
        t0: Optional[datetime] = None,
    ) -> List[float]:
        """Convert a list of datetimes to float seconds since t0.

        Parameters
        ----------
        datetimes : list of datetime objects
        t0        : reference epoch (default: first element of datetimes)

        Returns
        -------
        List of float seconds.
        """
        if t0 is None:
            t0 = datetimes[0]
        return [(dt - t0).total_seconds() for dt in datetimes]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _check_grid(grid: FieldGrid) -> None:
        if grid.bx is None:
            raise RuntimeError(
                "FieldGrid has not been evaluated yet — call grid.evaluate() first."
            )


# ---------------------------------------------------------------------------
# Fast low-level writer (avoids slow Python itertools loop)
# ---------------------------------------------------------------------------

def _write_ptm_fields_fast(grid: FieldGrid, filename: str) -> None:
    """Write PTM fields file using numpy for speed.

    For a 75x75x75 grid this is ~30x faster than the itertools loop
    used in ptm_preprocessing.PTMfields.write_file().
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    with open(filename, "w") as fh:
        # Header: dimensions
        fh.write(f"{nx:4d} {ny:4d} {nz:4d}\n")

        # Coordinate lines
        fh.write((" ".join(f"{x:12.5e}" for x in grid.xvec)).strip() + "\n")
        fh.write((" ".join(f"{y:12.5e}" for y in grid.yvec)).strip() + "\n")
        fh.write((" ".join(f"{z:12.5e}" for z in grid.zvec)).strip() + "\n")

        # Field values: build index arrays and flatten
        ii, jj, kk = np.meshgrid(
            np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
        )
        # Shape (nx*ny*nz, 9): i j k Bx By Bz Ex Ey Ez
        data = np.column_stack([
            (ii.ravel() + 1).astype(int),
            (jj.ravel() + 1).astype(int),
            (kk.ravel() + 1).astype(int),
            grid.bx.ravel(),
            grid.by.ravel(),
            grid.bz.ravel(),
            grid.ex.ravel(),
            grid.ey.ravel(),
            grid.ez.ravel(),
        ])

    # Append body using savetxt (fast)
    with open(filename, "a") as fh:
        np.savetxt(
            fh,
            data,
            fmt=["%4d", "%4d", "%4d"] + ["%12.5e"] * 6,
            delimiter=" ",
        )
