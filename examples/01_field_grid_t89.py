"""
Example 1: Generate a T89 magnetic field grid and inspect it.

Run from the inj_trace project root after 'pip install -e .':
    python examples/01_field_grid_t89.py
"""

import matplotlib.pyplot as plt
from datetime import datetime

# Configure paths before importing (or use inj-config set first)
from inj_trace.config import load_config
load_config()

from inj_trace.fields.grid import FieldGrid
from inj_trace.fields.writer import PTMFieldWriter
from inj_trace.visualization.equatorial import plot_equatorial_bfield

# --- Build a medium-resolution grid ---
grid = FieldGrid.from_spec({
    "xmin": -10, "xmax": 10, "nx": 41,
    "ymin": -10, "ymax": 10, "ny": 41,
    "zmin":  -5, "zmax":  5, "nz": 21,
})

time = datetime(2013, 3, 17, 6, 0, 0)
print(f"Evaluating T89 (Kp=5) on {grid.nx}×{grid.ny}×{grid.nz} grid…")
grid.evaluate("T89", time, {"Kp": 5}, n_workers=1)
print(f"  |B| min={grid.bmag().min():.2f}  max={grid.bmag().max():.2f} nT")

# --- Write PTM field files ---
import os
os.makedirs("example_run/ptm_data", exist_ok=True)
writer = PTMFieldWriter("example_run/ptm_data")
p1, p2, tg = writer.write_static(grid, duration_s=3600.0)
print(f"Written: {p1.name}, {p2.name}, {tg.name}")

# --- Plot equatorial B ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plot_equatorial_bfield(grid, component="bmag", ax=axes[0])
plot_equatorial_bfield(grid, component="bz",   ax=axes[1], log_scale=False, cmap="RdBu_r")
plt.tight_layout()
plt.savefig("example_run/equatorial_bfield.png", dpi=150)
print("Saved example_run/equatorial_bfield.png")
plt.show()
