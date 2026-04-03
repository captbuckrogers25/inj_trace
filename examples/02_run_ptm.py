"""
Example 2: End-to-end PTM run — build fields, configure run, execute, post-process, plot.

Run from the inj_trace project root after 'pip install -e .':
    python examples/02_run_ptm.py
"""

import os
import matplotlib.pyplot as plt
from datetime import datetime

from inj_trace.config import load_config
load_config()

from inj_trace.fields.grid import FieldGrid
from inj_trace.fields.writer import PTMFieldWriter
from inj_trace.runner.ptm_setup import PTMRunConfig
from inj_trace.runner.executor import PTMExecutor
from inj_trace.postprocess.trajectories import TrajectoryData
from inj_trace.visualization.trajectories3d import plot_trajectory_3d
from inj_trace.visualization.lshell import plot_lshell_map

RUN_DIR = "example_run2"
RUN_ID  = 1
TIME    = datetime(2013, 3, 17, 6, 0, 0)

# --- Create directory structure ---
for sub in ("ptm_data", "ptm_input", "ptm_output"):
    os.makedirs(os.path.join(RUN_DIR, sub), exist_ok=True)

# --- 1. Field grid ---
print("Building T89 field grid…")
grid = FieldGrid.from_spec({
    "xmin": -12, "xmax": 12, "nx": 51,
    "ymin": -12, "ymax": 12, "ny": 51,
    "zmin":  -6, "zmax":  6, "nz": 25,
})
grid.evaluate("T89", TIME, {"Kp": 5}, n_workers=2)
print(f"  |B| max = {grid.bmag().max():.1f} nT")

writer = PTMFieldWriter(os.path.join(RUN_DIR, "ptm_data"))
writer.write_static(grid, duration_s=3600.0)
print("Field files written.")

# --- 2. PTM configuration ---
ptm_cfg = PTMRunConfig(
    run_id=RUN_ID,
    n_particles=200,
    trace_direction=-1,   # backward tracing
    tlo=0.0,
    thi=3600.0,
    n_snapshots=2,
)
ptm_cfg.set_electron_defaults()
ptm_cfg.set_spatial_distribution(3, r0=6.6, mltmin=-6.0, mltmax=6.0)
ptm_cfg.set_velocity_distribution(3, emin=1.0, emax=500.0, nenergy=20, npitch=15,
                                   pamin=0.0, pamax=90.0, xsource=-12.0)
ptm_cfg.write(os.path.join(RUN_DIR, "ptm_input"))
print("PTM input files written.")

# --- 3. Execute PTM ---
print("Launching PTM…")
executor = PTMExecutor(RUN_DIR)
result = executor.run_single(RUN_ID)
print(f"PTM done (returncode={result.returncode})")

# --- 4. Load trajectories ---
traj_path = os.path.join(RUN_DIR, "ptm_output", f"ptm_{RUN_ID:04d}.dat")
traj = TrajectoryData.from_file(traj_path)
print(f"Loaded {traj.n_particles()} particle trajectories")

# --- 5. Compute L* ---
print("Computing L*…")
lstar = traj.compute_lstar(TIME, model="Lgm_B_T89", Kp=5, quality=3)
print(f"  L* range: {lstar[~__import__('numpy').isnan(lstar)].min():.2f} – "
      f"{lstar[~__import__('numpy').isnan(lstar)].max():.2f}")

# --- 6. Visualize ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 3D trajectories
plot_trajectory_3d(traj, particle_ids=traj.particle_ids()[:50],
                   color_by="energy", ax=axes[0])

# L-shell map
import matplotlib
axes[1].remove()
axes[1] = fig.add_subplot(122)
plot_lshell_map(lstar, traj.final_energies(), ax=axes[1])

plt.tight_layout()
plt.savefig(os.path.join(RUN_DIR, "results.png"), dpi=150)
print(f"Saved {RUN_DIR}/results.png")
plt.show()
