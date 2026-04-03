"""
TrajectoryData: load and analyze particle trajectory output from PTM.

PTM writes trajectory data as formatted ASCII with 8 columns per step:
    TIME  XPOS  YPOS  ZPOS  VPERP  VPARA  ENERGY  PITCHANGLE

Each particle section is delimited by a line starting with '#'.
Parsing is handled by ptm_tools.parse_trajectory_file().

After loading, TrajectoryData can compute L-shell (L*) at particle
final positions using lgmpy.Lstar.get_Lstar().

Usage
-----
    traj = TrajectoryData.from_file('run001/ptm_output/ptm_0001.dat')
    positions = traj.final_positions()     # (N, 3) GSM in Re
    energies  = traj.final_energies()      # (N,) keV
    lstar     = traj.compute_lstar(time, model='Lgm_B_T89', Kp=5, n_workers=4)
"""

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

# Column indices in the 8-column PTM output array
COL_TIME  = 0
COL_X     = 1
COL_Y     = 2
COL_Z     = 3
COL_VPERP = 4
COL_VPARA = 5
COL_E     = 6
COL_PA    = 7

# lgmpy sentinel for an open field line
_LSTAR_OPEN = -1e31
_LSTAR_OPEN_THRESHOLD = -1e30  # any value below this → NaN


# ---------------------------------------------------------------------------
# Top-level worker for parallel L* computation (must be at module scope)
# ---------------------------------------------------------------------------

def _lstar_chunk(args):
    """Evaluate L* for a list of positions in a worker process.

    args = (positions, time_iso, model, Kp, alpha, quality)
    Returns list of float L* values (np.nan where field is open).
    """
    positions, time_iso, model, Kp, alpha, quality = args
    try:
        from inj_trace.config import load_config
        load_config()
    except Exception:
        pass
    time = datetime.fromisoformat(time_iso)
    from lgmpy import Lstar as LstarMod
    results = []
    for pos in positions:
        try:
            res = LstarMod.get_Lstar(
                pos, time,
                alpha=alpha,
                Kp=Kp,
                Bfield=model,
                LstarQuality=quality,
            )
            val = res[float(alpha)]["Lstar"]
            if val < _LSTAR_OPEN_THRESHOLD:
                val = np.nan
            results.append(float(val))
        except Exception:
            results.append(np.nan)
    return results


# ---------------------------------------------------------------------------
# TrajectoryData
# ---------------------------------------------------------------------------

class TrajectoryData:
    """Container for PTM particle trajectory output.

    Parameters
    ----------
    raw : dict returned by ptm_tools.parse_trajectory_file()
          Keys are integer particle IDs (1-based as written in the file).
          Values are (N, 8) float arrays: TIME X Y Z VPERP VPARA ENERGY PITCHANGLE
    """

    def __init__(self, raw: dict) -> None:
        self.particles = raw

    @classmethod
    def from_file(cls, path: str) -> "TrajectoryData":
        """Load trajectory data from a PTM output file.

        Parameters
        ----------
        path : path to ptm_output/ptm_XXXX.dat
        """
        from ptm_python.ptm_tools import parse_trajectory_file
        return cls(parse_trajectory_file(str(path)))

    # ------------------------------------------------------------------
    # Basic accessors
    # ------------------------------------------------------------------

    def particle_ids(self) -> List[int]:
        """Return sorted list of particle IDs."""
        return sorted(self.particles.keys())

    def n_particles(self) -> int:
        return len(self.particles)

    def get_track(self, particle_id: int) -> np.ndarray:
        """Return (N, 8) array for one particle.

        Columns: TIME X Y Z VPERP VPARA ENERGY PITCHANGLE
        """
        return self.particles[particle_id]

    def final_positions(self) -> np.ndarray:
        """Return (n_particles, 3) array of final GSM positions (Re)."""
        ids = self.particle_ids()
        return np.array([self.particles[pid][-1, COL_X:COL_Z + 1] for pid in ids])

    def initial_positions(self) -> np.ndarray:
        """Return (n_particles, 3) array of initial GSM positions (Re)."""
        ids = self.particle_ids()
        return np.array([self.particles[pid][0, COL_X:COL_Z + 1] for pid in ids])

    def final_energies(self) -> np.ndarray:
        """Return (n_particles,) array of final particle energies (keV)."""
        return np.array([self.particles[pid][-1, COL_E] for pid in self.particle_ids()])

    def final_pitch_angles(self) -> np.ndarray:
        """Return (n_particles,) array of final pitch angles (degrees)."""
        return np.array([self.particles[pid][-1, COL_PA] for pid in self.particle_ids()])

    def all_positions(self) -> List[np.ndarray]:
        """Return list of (N_i, 3) position arrays, one per particle."""
        return [self.particles[pid][:, COL_X:COL_Z + 1] for pid in self.particle_ids()]

    # ------------------------------------------------------------------
    # L* computation
    # ------------------------------------------------------------------

    def compute_lstar(
        self,
        time: datetime,
        model: str = "Lgm_B_T89",
        Kp: int = 2,
        alpha: float = 90.0,
        quality: int = 3,
        n_workers: int = 1,
        lstar_array: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute L* at each particle's final position.

        Parameters
        ----------
        time      : UTC datetime for the field model epoch
        model     : lgmpy Bfield string, e.g. 'Lgm_B_T89', 'Lgm_B_OP77'
        Kp        : geomagnetic activity index (used by T89)
        alpha     : local pitch angle in degrees (default 90 = equatorial)
        quality   : LstarQuality 0 (fast) – 7 (accurate); 3 is a good default
        n_workers : >1 uses ProcessPoolExecutor (lgmpy ctypes is not thread-safe)
        lstar_array : if provided, return this directly (skips recomputation)

        Returns
        -------
        (n_particles,) float array; np.nan where field line is open.
        """
        if lstar_array is not None:
            return np.asarray(lstar_array)

        positions = self.final_positions().tolist()
        time_iso = time.isoformat()
        chunk_args = (positions, time_iso, model, Kp, alpha, quality)

        if n_workers <= 1:
            vals = _lstar_chunk(chunk_args)
        else:
            vals = self._lstar_parallel(positions, time_iso, model, Kp, alpha, quality, n_workers)

        return np.array(vals, dtype=float)

    def _lstar_parallel(self, positions, time_iso, model, Kp, alpha, quality, n_workers):
        chunk_size = max(1, len(positions) // n_workers)
        chunks = [
            positions[i: i + chunk_size]
            for i in range(0, len(positions), chunk_size)
        ]
        args_list = [
            (chunk, time_iso, model, Kp, alpha, quality) for chunk in chunks
        ]
        flat = []
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for result in pool.map(_lstar_chunk, args_list):
                flat.extend(result)
        return flat
