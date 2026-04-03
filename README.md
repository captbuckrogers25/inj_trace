# inj_trace

A Python library for simulating and visualizing energetic electron injections
to the inner magnetosphere.  `inj_trace` wraps two existing packages —
**LANLGeoMag** (`lgmpy`) for empirical magnetic field models and L-shell
calculations, and **SHIELDS-PTM** for relativistic particle tracing — and
provides a unified interface from field grid generation through simulation
execution to publication-quality matplotlib figures and animations.

---

## Table of contents

- [Overview](#overview)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick start](#quick-start)
- [CLI reference](#cli-reference)
- [Python API reference](#python-api-reference)
- [Data flow](#data-flow)
- [Examples](#examples)
- [Testing](#testing)
- [Project layout](#project-layout)

---

## Overview

Energetic particle injections are sudden enhancements of electron fluxes in the
inner magnetosphere, typically associated with substorm activity.  Studying them
requires (1) a realistic background magnetic field, (2) a particle tracing code
that can follow guiding-centre or full-orbit trajectories through that field, and
(3) tools to map the resulting flux distributions back onto physically meaningful
coordinates such as L-shell.

`inj_trace` glues these three components together:

| Stage | Provided by |
|---|---|
| Empirical B-field model (T89, TS04, OP77) | LANLGeoMag / `lgmpy` |
| 3-D grid construction and PTM field-file I/O | `inj_trace.fields` |
| Particle distribution setup and simulation launch | `inj_trace.runner` (wraps SHIELDS-PTM) |
| Trajectory analysis, L-shell mapping | `inj_trace.postprocess` |
| Equatorial maps, L*–flux plots, animations | `inj_trace.visualization` |

---

## Dependencies

### Python (≥ 3.9)

| Package | Version | Notes |
|---|---|---|
| numpy | ≥ 1.23 | |
| scipy | ≥ 1.9 | Used for spatial interpolation in plots |
| matplotlib | ≥ 3.6 | All figures and animations |
| spacepy | ≥ 0.4 | Required by `lgmpy.Lstar` for IGRF |

Install with:
```bash
pip3 install --user numpy scipy matplotlib spacepy
```

### External packages (local installs, not on PyPI)

Both packages must be built/installed separately before using `inj_trace`.
Their filesystem paths are registered via `inj-config` (see
[Configuration](#configuration)).

**LANLGeoMag** — magnetic field models, L-shell calculations, field-line tracing.
```bash
cd ~/LANLGeoMag
sudo apt install libgsl0-dev libhdf5-serial-dev autoconf automake libtool
autoreconf -i && ./configure && make -j && make install
```

**SHIELDS-PTM** — relativistic particle tracing (Fortran + Python).
```bash
cd ~/SHIELDS-PTM
make ptm        # build Fortran executable
make python     # build Python module
```

### System requirements for animation export

MP4 export requires `ffmpeg`.  GIF export requires the `Pillow` package.

```bash
sudo apt install ffmpeg
pip3 install --user Pillow
```

---

## Installation

```bash
git clone <repo-url>
cd inj_trace
pip3 install --user -e .
```

This installs the package in editable mode and registers the four CLI entry
points (`inj-config`, `inj-make-fields`, `inj-run`, `inj-plot`).

---

## Configuration

`inj_trace` needs to know where LANLGeoMag and SHIELDS-PTM are installed.
Run `inj-config set` once after installation:

```bash
inj-config set \
    --lgmpy-path      ~/LANLGeoMag/Python \
    --ptm-python-path ~/SHIELDS-PTM \
    --ptm-executable  ~/SHIELDS-PTM/ptm
```

Settings are saved to `~/.inj_trace.cfg`.  Verify with:

```bash
inj-config validate
```

You can also override any path with environment variables, which take priority
over the config file:

| Variable | Overrides |
|---|---|
| `INJ_LGMPY_PATH` | path to `LANLGeoMag/Python/` |
| `INJ_PTM_PYTHON_PATH` | path to the parent of `ptm_python/` |
| `INJ_PTM_EXE` | path to the compiled `ptm` executable |

---

## Quick start

### Python API

```python
import os
from datetime import datetime
import inj_trace

# ── 1. Create run directory ──────────────────────────────────────────────────
RUN_DIR = "./my_run"
for sub in ("ptm_data", "ptm_input", "ptm_output"):
    os.makedirs(os.path.join(RUN_DIR, sub), exist_ok=True)

# ── 2. Build a T89 field grid ────────────────────────────────────────────────
grid = inj_trace.FieldGrid.from_spec({
    "xmin": -12, "xmax": 12, "nx": 51,
    "ymin": -12, "ymax": 12, "ny": 51,
    "zmin":  -6, "zmax":  6, "nz": 25,
})
grid.evaluate("T89", datetime(2013, 3, 17, 6), {"Kp": 5}, n_workers=4)
print(f"|B| max = {grid.bmag().max():.1f} nT")

# ── 3. Write PTM field files (two identical snapshots for a static run) ──────
writer = inj_trace.PTMFieldWriter(os.path.join(RUN_DIR, "ptm_data"))
writer.write_static(grid, duration_s=3600.0)

# ── 4. Configure PTM ─────────────────────────────────────────────────────────
cfg = inj_trace.PTMRunConfig(
    run_id=1, n_particles=1000, tlo=0, thi=3600, n_snapshots=2
)
cfg.set_electron_defaults()
cfg.set_spatial_distribution(3, r0=6.6, mltmin=-6.0, mltmax=6.0)
cfg.set_velocity_distribution(3, emin=1, emax=500, nenergy=34, npitch=31,
                               pamin=0, pamax=90, xsource=-12.0)
cfg.write(os.path.join(RUN_DIR, "ptm_input"))

# ── 5. Run PTM ───────────────────────────────────────────────────────────────
inj_trace.PTMExecutor(RUN_DIR).run_single(1)

# ── 6. Load trajectories and compute L* ──────────────────────────────────────
traj = inj_trace.TrajectoryData.from_file(f"{RUN_DIR}/ptm_output/ptm_0001.dat")
lstar = traj.compute_lstar(datetime(2013, 3, 17, 6), model="Lgm_B_T89", Kp=5)

# ── 7. Visualize ─────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
inj_trace.viz.lshell.plot_lshell_map(lstar, traj.final_energies(),
                                      save_path="lshell.png")
inj_trace.viz.trajectories3d.plot_trajectory_3d(traj, save_path="traj3d.png")
plt.show()
```

### One-call convenience function

For simple runs, `run_injection()` handles all five stages in a single call:

```python
result = inj_trace.run_injection(
    model="T89",
    time=datetime(2013, 3, 17, 6),
    model_params={"Kp": 5},
    grid_spec={"xmin": -12, "xmax": 12, "nx": 51,
               "ymin": -12, "ymax": 12, "ny": 51,
               "zmin":  -6, "zmax":  6, "nz": 25},
    run_config={"run_id": 1, "n_particles": 1000, "tlo": 0, "thi": 3600,
                "Ec": 10.0, "n": 0.01},
    run_dir="./my_run",
    n_workers=4,
)
print(f"Peak flux energy: {result.peak_energy():.1f} keV")
```

---

## CLI reference

### `inj-config`

Manage path configuration.

```
inj-config set --lgmpy-path PATH --ptm-python-path PATH --ptm-executable PATH
inj-config validate
inj-config show
```

### `inj-make-fields`

Evaluate a magnetic field model on a 3-D Cartesian grid and write PTM-format
field files.

```
inj-make-fields --model T89 --time 2013-03-17T06:00:00 --Kp 5 \
                --xmin -12 --xmax 12 --nx 51 \
                --ymin -12 --ymax 12 --ny 51 \
                --zmin  -6 --zmax  6 --nz 25 \
                --output-dir ./my_run/ptm_data \
                --static --duration 3600 \
                --workers 4
```

Key options:

| Flag | Description |
|---|---|
| `--model` | `T89` (default), `TS04`, or `OP77` |
| `--time` | UTC time as ISO 8601 string |
| `--time-file` | Text file with one ISO time per line (time-series runs) |
| `--Kp` | Kp index 0–5 for T89 |
| `--P`, `--Dst`, `--By`, `--Bz`, `--W` | TS04 solar wind parameters |
| `--nx/ny/nz` | Grid resolution in each dimension |
| `--static` | Duplicate snapshot for static PTM run (required if only one time) |
| `--workers` | Parallel processes for grid evaluation |

### `inj-run`

Configure input files and launch PTM.

```
inj-run --run-id 1 --run-dir ./my_run \
        --n-particles 1000 --tlo 0 --thi 3600 \
        --electron \
        --spatial-mode 3 --r0 6.6 --mltmin -6 --mltmax 6 \
        --velocity-mode 3 --emin 1 --emax 500 --nenergy 34 --npitch 31
```

Key options:

| Flag | Description |
|---|---|
| `--electron` | Preset electron defaults (charge=−1, guiding-centre mode) |
| `--spatial-mode` | `1`=point, `2`=box, `3`=radial ring |
| `--velocity-mode` | `1`=ring (single E/PA), `2`=bi-Maxwellian, `3`=uniform flux map |
| `--trace-dir` | `+1` forward, `−1` backward (default; recommended for injection mapping) |
| `--parallel N` | Launch N simultaneous PTM processes |

### `inj-plot`

Visualize PTM results.

```bash
# Equatorial flux map
inj-plot equatorial --run-dir ./my_run --run-id 1 --energy 100 --save eq.png

# Flux vs L* (requires --time for L* computation)
inj-plot lshell --run-dir ./my_run --run-id 1 --time 2013-03-17T06:00:00 \
                --Kp 5 --workers 4 --save lshell.png

# 3-D particle trajectories
inj-plot trajectory --run-dir ./my_run --run-id 1 --n-particles 200 \
                    --color-by energy --save traj.png

# Time series across runs
inj-plot timeseries --run-dir ./my_run --run-ids 1,2,3,4,5 --save ts.png

# Time-lapse animation across runs
inj-plot animate --run-dir ./my_run --run-ids 1,2,3,4,5 \
                 --type equatorial --fps 5 --save injection.mp4
```

---

## Python API reference

### `inj_trace.FieldGrid`

Evaluate an empirical B-field model on a regular 3-D Cartesian (GSM) grid.

```python
FieldGrid(xvec, yvec, zvec)
FieldGrid.from_spec(spec)           # spec: dict with xmin/xmax/nx etc.

grid.evaluate(model, time, model_params,
              mask_inner=1.2, n_workers=1)
grid.set_zero_efield()
grid.set_efield_from_potential(phi)  # extension hook for Volland–Stern
grid.bmag()                          # → ndarray (nx, ny, nz)

# Attributes after evaluate()
grid.bx, grid.by, grid.bz   # nT, shape (nx, ny, nz)
grid.ex, grid.ey, grid.ez   # mV/m (zero for static empirical runs)
grid.xvec, grid.yvec, grid.zvec   # Re (GSM)
```

**Supported models** and their required `model_params` keys:

| `model` | Required keys |
|---|---|
| `"T89"` | `Kp` (int 0–5) |
| `"TS04"` | `P`, `Dst`, `By`, `Bz`, `W` (list of 6) |
| `"OP77"` | *(none)* |

All models accept an optional `internal_model` key:
`"LGM_IGRF"` (default), `"LGM_CDIP"`, or `"LGM_EDIP"`.

### `inj_trace.PTMFieldWriter`

Write PTM-format field files from a `FieldGrid`.

```python
writer = PTMFieldWriter(output_dir)

writer.write_snapshot(grid, snapshot_index)      # → Path
writer.write_tgrid(times_seconds)                # → Path
writer.write_static(grid, duration_s=3600.0)     # → (p1, p2, tgrid)
writer.write_time_series(grids, dt, t0=0.0)      # → (paths, tgrid)

PTMFieldWriter.times_from_datetimes(datetimes, t0=None)  # → List[float]
```

> **Note:** PTM requires at least **two** field snapshots for internal time
> interpolation, even for static runs.  `write_static()` automatically writes
> two identical snapshots.

### `inj_trace.PTMRunConfig`

Configure and write PTM input files.

```python
cfg = PTMRunConfig(run_id, n_particles, trace_direction=-1,
                   tlo=0.0, thi=3600.0, n_snapshots=2)
cfg.set_electron_defaults()
cfg.set_spatial_distribution(mode, **params)
cfg.set_velocity_distribution(mode, **params)
cfg.write(input_dir)
cfg.print_settings()
cfg.set_parameters(**kwargs)          # direct escape hatch
```

**Spatial distribution modes:**

| `mode` | Description | Required `params` |
|---|---|---|
| `1` | Single point | `x0, y0, z0` |
| `2` | Random box | `xmin, xmax, ymin, ymax, zmin, zmax` |
| `3` | Radial ring (MLT) | `r0, mltmin, mltmax` (hours) |

**Velocity distribution modes:**

| `mode` | Description | Required `params` |
|---|---|---|
| `1` | Ring (mono-energetic) | `ekev, alpha` [, `phi`] |
| `2` | Bi-Maxwellian | `vtperp, vtpara` [, `phi`] |
| `3` | Uniform flux map | `nenergy, npitch, emin, emax, pamin, pamax, xsource` |

### `inj_trace.PTMExecutor`

Launch the PTM Fortran executable.

```python
exe = PTMExecutor(run_dir, ptm_executable=None)
exe.run_single(run_id, timeout=None, stdout_callback=None)
exe.run_parallel(run_ids, max_workers=4, timeout=None)
exe.check_output_exists(run_id)
exe.output_path(run_id)   # → Path to ptm_output/ptm_XXXX.dat
exe.map_path(run_id)      # → Path to ptm_output/map_XXXX.dat
```

### `inj_trace.TrajectoryData`

Load and analyse PTM trajectory output.

```python
traj = TrajectoryData.from_file(path)

traj.particle_ids()           # → List[int]
traj.n_particles()            # → int
traj.get_track(particle_id)   # → (N, 8) ndarray
                              #   cols: TIME X Y Z VPERP VPARA ENERGY PITCHANGLE
traj.final_positions()        # → (N, 3) GSM in Re
traj.initial_positions()      # → (N, 3) GSM in Re
traj.final_energies()         # → (N,) keV
traj.final_pitch_angles()     # → (N,) degrees
traj.all_positions()          # → List of (N_i, 3) arrays

traj.compute_lstar(time, model="Lgm_B_T89", Kp=2, alpha=90.0,
                   quality=3, n_workers=1)   # → (N,) L* array; NaN = open field
```

### `inj_trace.FluxMapResult`

Compute differential flux from PTM map output (requires `idist=3` or `4`).

```python
result = FluxMapResult.from_run(run_id, run_dir, Ec, n,
                                 mc2=511.0, kind="kappa", kap=2.5)
result = FluxMapResult.from_map_file(fnames, Ec, n, ...)

result.energies        # (n_e,) keV
result.angles          # (n_pa,) degrees
result.flux            # (n_e, n_pa)  keV⁻¹ cm⁻² s⁻¹ sr⁻¹
result.omni            # (n_e,)       keV⁻¹ cm⁻² s⁻¹
result.position        # (3,) GSM source position

result.peak_energy()              # → float keV
result.flux_at_energy(energy_kev) # → (n_pa,)
result.omni_at_energy(energy_kev) # → float
result.save(path)
FluxMapResult.load(path)
```

**Source distribution parameters** (`kind`, `Ec`, `n`, `kap`) are passed to
`ptm_tools.energy_to_flux()`.  See that function's docstring for physical
interpretation.  A value of `kap=2.5` corresponds to electrons during a strong
substorm (Gabrielse et al. 2014).

### `inj_trace.viz`

All visualization functions share the same call pattern:
- Data arrays as positional/keyword arguments
- Optional `ax` (existing `matplotlib.Axes`; created if `None`)
- Optional `save_path` (figure saved and closed if provided; shown interactively otherwise)
- Returns the `Axes` object

```python
# Equatorial plane
inj_trace.viz.equatorial.plot_equatorial_bfield(grid, component="bmag",
    z_slice=0.0, cmap="viridis", log_scale=True)
inj_trace.viz.equatorial.plot_equatorial_flux(positions, flux_values,
    energy_kev, cmap="hot_r", log_scale=True, grid_res=100)

# L-shell
inj_trace.viz.lshell.plot_lshell_map(lstar_values, flux_values,
    energy_kev=None, pitch_angle_deg=90.0, log_scale=True)
inj_trace.viz.lshell.plot_lshell_energy_map(energies_kev, lstar_values,
    flux_matrix, cmap="hot_r")

# Time series
inj_trace.viz.timeseries.plot_flux_timeseries(times, flux_vs_time,
    energies_kev=None, log_scale=True)

# 3-D trajectories
inj_trace.viz.trajectories3d.plot_trajectory_3d(traj_data,
    particle_ids=None, color_by="energy", cmap="plasma")
# color_by options: "energy", "time", "pitchangle", "particle"

# Animations (return FuncAnimation; set save_path to export)
inj_trace.viz.animation.animate_equatorial_flux(
    flux_snapshots, positions, times,
    fps=5, save_path="injection.mp4")
inj_trace.viz.animation.animate_lshell(
    lstar_snapshots, flux_snapshots, times,
    fps=5, save_path="lshell_anim.gif")
```

---

## Data flow

```
User input: model name, time, model_params, grid spec
        │
        ▼
FieldGrid.evaluate()
  calls lgmpy at each (x,y,z) via ProcessPoolExecutor
  produces bx/by/bz [nT] on (nx,ny,nz) grid; ex=ey=ez=0
        │
        ▼
PTMFieldWriter.write_static()
  ptm_data/ptm_fields_0001.dat  ← ASCII field file
  ptm_data/ptm_fields_0002.dat  ← identical duplicate (PTM needs ≥2)
  ptm_data/tgrid.dat            ← [0.0, duration_s]
        │
        ▼
PTMRunConfig → .write("ptm_input/")
  ptm_input/ptm_parameters_0001.txt
  ptm_input/dist_density_0001.txt
  ptm_input/dist_velocity_0001.txt
        │
        ▼
PTMExecutor.run_single(1)
  subprocess: cd run_dir && ./ptm 1
  ptm_output/ptm_0001.dat   ← particle trajectories (ASCII)
  ptm_output/map_0001.dat   ← flux map (if idist=3 or 4)
        │
        ├─▶ TrajectoryData.from_file()
        │     .compute_lstar() via lgmpy.Lstar.get_Lstar() (parallel)
        │
        └─▶ FluxMapResult.from_run()
              ptm_tools.parse_map_file() + energy_to_flux()
        │
        ▼
viz.lshell / viz.equatorial / viz.animation
```

---

## Examples

Three example scripts are in `examples/`:

| Script | Description |
|---|---|
| `01_field_grid_t89.py` | Build a T89 grid, write PTM files, plot equatorial |B| |
| `02_run_ptm.py` | Full end-to-end: fields → PTM → L* → trajectory + L-shell plots |
| `03_animate.py` | Time-lapse animation across multiple PTM runs |

Run from the project root:

```bash
python3 examples/01_field_grid_t89.py
python3 examples/02_run_ptm.py
python3 examples/03_animate.py --run-dir ./multi_run --run-ids 1 2 3 4 5
```

---

## Testing

Tests use `pytest`.  Tests marked `needs_lgmpy` or `needs_ptm` are skipped
automatically when the upstream packages are not configured.

```bash
pip3 install --user pytest
cd /path/to/inj_trace
pytest tests/ -v
```

To run only the tests that do not require the external libraries:
```bash
pytest tests/ -v -m "not needs_lgmpy and not needs_ptm"
```

---

## Project layout

```
inj_trace/
├── pyproject.toml
├── README.md
├── PLAN.md                          ← detailed implementation guide
├── inj_trace/
│   ├── config.py                    ← path config, sys.path injection
│   ├── fields/
│   │   ├── models.py                ← lgmpy model wrappers
│   │   ├── grid.py                  ← FieldGrid
│   │   └── writer.py                ← PTMFieldWriter
│   ├── runner/
│   │   ├── ptm_setup.py             ← PTMRunConfig
│   │   └── executor.py              ← PTMExecutor
│   ├── postprocess/
│   │   ├── trajectories.py          ← TrajectoryData
│   │   └── fluxmap.py               ← FluxMapResult
│   ├── visualization/
│   │   ├── equatorial.py
│   │   ├── lshell.py
│   │   ├── timeseries.py
│   │   ├── trajectories3d.py
│   │   └── animation.py
│   └── cli/
│       ├── config_cmd.py
│       ├── make_fields.py
│       ├── run_ptm.py
│       └── plot.py
├── examples/
│   ├── 01_field_grid_t89.py
│   ├── 02_run_ptm.py
│   └── 03_animate.py
└── tests/
    ├── conftest.py
    ├── test_fields.py
    ├── test_ptm_driver.py
    └── test_visualization.py
```

---

## Physical notes

**Backward tracing** (`trace_direction=-1`, the default) is the standard approach
for injection mapping.  Particles are started at a target location (e.g. GEO)
and traced backward in time to determine which source regions they can access.
This is equivalent to forward-tracing by Liouville's theorem.

**Electric field** is set to zero for all static empirical model runs.  This is
physically consistent for guiding-centre electrons: gradient and curvature drifts
are captured by the B-field gradients alone.  A stub
`FieldGrid.set_efield_from_potential()` is provided for future Volland–Stern
convection electric field support.

**L\*** (the third adiabatic invariant shell parameter) is computed via
`lgmpy.Lstar.get_Lstar()`.  Computation is expensive (~1–5 s per point at
quality=3).  Use `n_workers > 1` in `compute_lstar()` for large particle
populations.  Open-field-line particles are returned as `NaN`.

**Inner boundary masking**: lgmpy empirical models are not valid inside ≈1.0–1.5 Re.
`FieldGrid.evaluate()` zeros B inside `mask_inner` (default 1.2 Re).

---

## License

All original work licensed under Apache 2.0 license (See `LICENSE` file).  
`inj_trace` is an interface library; the physics computations are performed by 
[LANLGeoMag](https://github.com/drsteve/LANLGeoMag)  and [SHIELDS-PTM](https://github.com/lanl/SHIELDS-PTM)
, which carry their own licenses.  


---

## Acknowledgements

- **LANLGeoMag** — J. Roeder, M. Henderson, et al., Los Alamos National Laboratory
- **SHIELDS-PTM** — J. Woodroffe, S. Morley, et al., Los Alamos National Laboratory
- Gabrielse et al. (2014) [DOI:10.1002/2013JA019638](https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2013JA019638) for kappa distribution parameters
