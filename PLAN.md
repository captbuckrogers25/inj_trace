# inj_trace — Implementation Plan

## Overview

`inj_trace` is a Python library that wraps LANLGeoMag (`lgmpy`) and SHIELDS-PTM
(`ptm_python` + Fortran `ptm` executable) to run end-to-end energetic electron
injection simulations and produce matplotlib visualizations and animations.

## Upstream dependencies (local installs, not pip packages)

| Package       | Path                              | Key entry points |
|---------------|-----------------------------------|-----------------|
| lgmpy         | `~/LANLGeoMag/Python/`            | `Lgm_T89.T89()`, `Lgm_TS04.TS04()`, `Lgm_OP77.OP77()`, `Lstar.get_Lstar()`, `Lgm_CTrans.coordTrans()` |
| ptm_python    | `~/SHIELDS-PTM/` (parent of pkg)  | `ptm_python.ptm_input.ptm_input_creator`, `ptm_python.ptm_tools.parse_trajectory_file`, `ptm_python.ptm_tools.energy_to_flux`, `ptm_python.ptm_preprocessing.PTMfields` |
| ptm           | `~/SHIELDS-PTM/ptm`               | Fortran executable: `./ptm RUNID` |

Both Python packages are injected into `sys.path` at runtime by `inj_trace/config.py`.
They are **never** listed in `pyproject.toml` as pip dependencies.

## Project layout

```
inj_trace/
├── pyproject.toml
├── PLAN.md
├── inj_trace/
│   ├── __init__.py          ← exposes top-level API + run_injection()
│   ├── config.py            ← InjTraceConfig, load_config(), ConfigError
│   ├── fields/
│   │   ├── __init__.py
│   │   ├── models.py        ← eval_t89(), eval_ts04(), eval_op77(), MODELS dict
│   │   ├── grid.py          ← FieldGrid class
│   │   └── writer.py        ← PTMFieldWriter class
│   ├── runner/
│   │   ├── __init__.py
│   │   ├── ptm_setup.py     ← PTMRunConfig class
│   │   └── executor.py      ← PTMExecutor class
│   ├── postprocess/
│   │   ├── __init__.py
│   │   ├── trajectories.py  ← TrajectoryData class
│   │   └── fluxmap.py       ← FluxMapResult class
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── equatorial.py
│   │   ├── lshell.py
│   │   ├── timeseries.py
│   │   ├── trajectories3d.py
│   │   └── animation.py
│   └── cli/
│       ├── __init__.py
│       ├── config_cmd.py    ← inj-config
│       ├── make_fields.py   ← inj-make-fields
│       ├── run_ptm.py       ← inj-run
│       └── plot.py          ← inj-plot
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

## Module-by-module implementation guide

### `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "inj_trace"
version = "0.1.0"
requires-python = ">=3.9"
dependencies = ["numpy>=1.23", "scipy>=1.9", "matplotlib>=3.6", "spacepy>=0.4"]

[project.scripts]
inj-config      = "inj_trace.cli.config_cmd:main"
inj-make-fields = "inj_trace.cli.make_fields:main"
inj-run         = "inj_trace.cli.run_ptm:main"
inj-plot        = "inj_trace.cli.plot:main"
```

---

### `inj_trace/config.py`

**Purpose**: Central path configuration; injects lgmpy/ptm_python into sys.path.

**Key design**:
- `InjTraceConfig` dataclass: `lgmpy_path`, `ptm_python_path`, `ptm_executable`
- Config file: `~/.inj_trace.cfg` using `configparser`, section `[paths]`
- Env vars (override file): `INJ_LGMPY_PATH`, `INJ_PTM_PYTHON_PATH`, `INJ_PTM_EXE`
- `load_config()`: reads config, calls `_inject_paths()`, returns singleton `InjTraceConfig`
- `_inject_paths()`: `sys.path.insert(0, ...)` for both package dirs; idempotent
- `validate()`: attempts `import lgmpy`, checks `ptm_executable` is executable
- `ConfigError(Exception)`: raised with clear message on failure
- `save_config(cfg)`: writes `~/.inj_trace.cfg`

**Called in `inj_trace/__init__.py` at import time** so all submodules can `import lgmpy`.

---

### `inj_trace/fields/models.py`

**Purpose**: Uniform wrappers around lgmpy field model functions.

```python
def eval_t89(pos_gsm, time, Kp, internal_model='LGM_IGRF') -> list:
    """Return [Bx, By, Bz] nT. Kp must be int 0-5."""
    from lgmpy import Lgm_T89
    return list(Lgm_T89.T89(pos_gsm, time, Kp=int(Kp), INTERNAL_MODEL=internal_model))

def eval_ts04(pos_gsm, time, P, Dst, By, Bz, W, internal_model='LGM_IGRF') -> list:
    from lgmpy import Lgm_TS04
    return list(Lgm_TS04.TS04(pos_gsm, time, P=P, Dst=Dst, By=By, Bz=Bz, W=W, INTERNAL_MODEL=internal_model))

def eval_op77(pos_gsm, time, internal_model='LGM_IGRF') -> list:
    from lgmpy import Lgm_OP77
    return list(Lgm_OP77.OP77(pos_gsm, time, INTERNAL_MODEL=internal_model))

MODELS = {'T89': eval_t89, 'TS04': eval_ts04, 'OP77': eval_op77}
```

**Note**: Imports are deferred (inside functions) so config.py's sys.path injection
runs first before lgmpy is loaded.

---

### `inj_trace/fields/grid.py`

**Purpose**: Evaluate B on a regular 3D Cartesian (GSM) grid.

```python
class FieldGrid:
    def __init__(self, xvec, yvec, zvec):
        # Store 1D coordinate arrays; bx/by/bz/ex/ey/ez initially None

    @classmethod
    def from_spec(cls, spec: dict) -> 'FieldGrid':
        # spec keys: xmin, xmax, nx, ymin, ymax, ny, zmin, zmax, nz
        # Uses np.linspace to build coordinate arrays

    def evaluate(self, model: str, time: datetime, model_params: dict,
                 mask_inner: float = 1.2, n_workers: int = 1) -> None:
        # Build flat list of all (x,y,z) grid points
        # If n_workers == 1: call _eval_sequential()
        # If n_workers > 1: split into chunks, use ProcessPoolExecutor(_eval_chunk)
        # Reshape results into (nx, ny, nz) arrays
        # Apply inner-boundary mask: zero B where r < mask_inner
        # Call set_zero_efield()

    def set_zero_efield(self) -> None:
        # self.ex = self.ey = self.ez = np.zeros((nx, ny, nz))

    def bmag(self) -> np.ndarray:
        return np.sqrt(self.bx**2 + self.by**2 + self.bz**2)
```

**Parallel worker** (module-level function, picklable):
```python
def _eval_chunk(args):
    model_name, positions, time_iso, model_params = args
    # Re-run load_config() to ensure paths set in worker process
    from datetime import datetime
    time = datetime.fromisoformat(time_iso)
    from inj_trace.fields.models import MODELS
    fn = MODELS[model_name]
    return [fn(pos, time, **model_params) for pos in positions]
```

**Implementation note**: Use `fork` start method (Linux default).
Child processes inherit sys.path so lgmpy is importable without re-injection.
Still call `load_config()` defensively in the worker.

---

### `inj_trace/fields/writer.py`

**Purpose**: Write PTM-format field files.

```python
class PTMFieldWriter:
    def __init__(self, output_dir: str):
        # Create output_dir if needed

    def write_snapshot(self, grid: FieldGrid, snapshot_index: int) -> Path:
        # Use PTMfields from ptm_preprocessing (for format compatibility) OR
        # write directly using numpy (faster):
        #   Line 1: "{nx:4} {ny:4} {nz:4}\n"
        #   Line 2: nx x-coords formatted "{:12.5e} " * nx
        #   Line 3: ny y-coords
        #   Line 4: nz z-coords
        #   Body: "{i+1:4} {j+1:4} {k+1:4} {bx:12.5e} {by:12.5e} {bz:12.5e} {ex:12.5e} {ey:12.5e} {ez:12.5e}\n"
        # Use numpy approach for speed (np.savetxt on pre-formed array)
        # Returns path to written file

    def write_tgrid(self, times_seconds: list) -> Path:
        # np.savetxt(os.path.join(output_dir, 'tgrid.dat'), np.array(times_seconds))
        # Returns path

    def write_static(self, grid: FieldGrid, duration_s: float = 3600.0) -> tuple:
        # Convenience: write two identical snapshots (0001, 0002) + tgrid [0, duration]
        # Returns (field_path_1, field_path_2, tgrid_path)

    @staticmethod
    def times_from_datetimes(datetimes, t0=None) -> list:
        # Returns float seconds since t0 (default: first element)
```

**Critical**: PTM needs ≥ 2 field snapshots for time interpolation even for static runs.
`write_static()` handles this automatically.

---

### `inj_trace/runner/ptm_setup.py`

**Purpose**: Configure and write PTM input files.

```python
class PTMRunConfig:
    def __init__(self, run_id=1, n_particles=1000, trace_direction=-1,
                 tlo=0.0, thi=3600.0, n_snapshots=2, **kwargs):
        # Create ptm_input_creator(runid=run_id, idensity=2, ivelocity=3)
        # Call set_parameters with all relevant kwargs

    def set_electron_defaults(self):
        # self._creator.set_parameters(charge=-1.0, mass=1.0, iswitch=-1)

    def set_spatial_distribution(self, mode: int, **params):
        # mode 1: point (x0, y0, z0)
        # mode 2: box (xmin, xmax, ymin, ymax, zmin, zmax)
        # mode 3: ring (r0, mltmin, mltmax)
        # Calls self._creator.set_parameters(idens=mode, **params)

    def set_velocity_distribution(self, mode: int, **params):
        # mode 1: ring (ekev, alpha, phi)
        # mode 2: bi-maxwellian (vtperp, vtpara, phi)
        # mode 3: uniform fluxmap (nenergy, npitch, emin, emax, pamin, pamax, xsource)
        # Calls self._creator.set_parameters(idist=mode, **params)

    def write(self, input_dir: str = 'ptm_input') -> None:
        # self._creator.create_input_files(input_dir)

    def print_settings(self) -> None:
        # self._creator.print_settings()
```

**Key**: `ptm_input_creator` defaults to `itrace=-1` (backward tracing) — correct for injection studies.
`n_snapshots` maps to `ntot` and `ilast` in ptm_input_creator.

---

### `inj_trace/runner/executor.py`

**Purpose**: Launch PTM Fortran executable.

```python
class PTMExecutor:
    def __init__(self, run_dir: str, ptm_executable: str = None):
        # ptm_executable from config if not given

    def run_single(self, run_id: int, timeout=None,
                   stdout_callback=None) -> subprocess.CompletedProcess:
        # os.chdir(run_dir)  ← PTM resolves files relative to CWD
        # subprocess.run([ptm_executable, str(run_id)], ...)
        # Uses ptm_tools.cd context manager for safe chdir

    def run_parallel(self, run_ids: list, max_workers: int = 4,
                     timeout=None) -> dict:
        # ThreadPoolExecutor: each PTM is its own OS process, no shared memory
        # Returns {run_id: CompletedProcess}

    def check_output_exists(self, run_id: int) -> bool:
        # os.path.exists(os.path.join(run_dir, 'ptm_output', f'ptm_{run_id:04d}.dat'))
```

**Critical**: Must `os.chdir(run_dir)` before calling PTM. Use `ptm_tools.cd()` context manager
from `~/SHIELDS-PTM/ptm_python/ptm_tools.py` for safe restore of original directory.

---

### `inj_trace/postprocess/trajectories.py`

**Purpose**: Load and analyze PTM trajectory output.

```python
class TrajectoryData:
    def __init__(self, raw: dict):
        # raw is from parse_trajectory_file()
        # self.particles = raw (newDict keyed by int particle ID)

    @classmethod
    def from_file(cls, path: str) -> 'TrajectoryData':
        from ptm_python.ptm_tools import parse_trajectory_file
        return cls(parse_trajectory_file(path))

    def particle_ids(self) -> list:
        return sorted(self.particles.keys())

    def get_track(self, particle_id: int) -> np.ndarray:
        # Returns (N, 8) array: TIME XPOS YPOS ZPOS VPERP VPARA ENERGY PITCHANGLE
        return self.particles[particle_id]

    def final_positions(self) -> np.ndarray:
        # Returns (n_particles, 3): last row x,y,z for each particle
        return np.array([self.particles[pid][-1, 1:4] for pid in self.particle_ids()])

    def final_energies(self) -> np.ndarray:
        # Returns (n_particles,): last energy in keV for each particle
        return np.array([self.particles[pid][-1, 6] for pid in self.particle_ids()])

    def compute_lstar(self, time: datetime, model: str = 'Lgm_B_T89',
                      Kp: int = 2, alpha: float = 90.0,
                      quality: int = 3, n_workers: int = 1) -> np.ndarray:
        # Call lgmpy.Lstar.get_Lstar() at each particle's final position
        # With n_workers > 1, use ProcessPoolExecutor (ctypes not thread-safe)
        # Return (n_particles,) array of L* values; NaN where field is open
        # Return structure: result[alpha]['Lstar']
```

**Lstar return structure** (from lgmpy docs):
```python
result = Lstar.get_Lstar(pos, date, alpha=90., Kp=2, Bfield='Lgm_B_T89', LstarQuality=3)
lstar_val = result[90.0]['Lstar']  # float or -1e31 if open field
```
Map `-1e31` (lgmpy's sentinel for open field) to `np.nan`.

---

### `inj_trace/postprocess/fluxmap.py`

**Purpose**: Compute differential flux from PTM map output.

```python
class FluxMapResult:
    # Attributes: energies (keV), angles (deg), flux (n_e, n_pa), omni (n_e,), position (3,)

    @classmethod
    def from_map_file(cls, fnames, Ec: float, n: float,
                      mc2: float = 511.0, kind: str = 'kappa',
                      kap: float = 2.5) -> 'FluxMapResult':
        # Uses ptm_tools.parse_map_file() and ptm_tools.energy_to_flux()
        # fnames: str or list of str paths to ptm map output files
        # Ec: characteristic energy (keV), n: density (cm^-3)

    def peak_energy(self) -> float:
        return float(self.energies[np.argmax(self.omni)])

    def to_dict(self) -> dict:
        return {'energies': self.energies, 'angles': self.angles,
                'flux': self.flux, 'omni': self.omni, 'position': self.position}
```

**`energy_to_flux` signature** (from ptm_tools.py):
```python
energy_to_flux(ei, ef, ec, n, mc2=511.0, kind='kappa', kap=2.5, energyFlux=False)
# ei = initial energy (source), ef = final energy (obs point)
# Returns differential flux in keV-1 cm-2 s-1 sr-1
```

---

### `inj_trace/visualization/equatorial.py`

```python
def plot_equatorial_bfield(grid, component='bmag', z_slice=0.0,
                            ax=None, cmap='viridis', save_path=None) -> plt.Axes:
    # Interpolate to z_slice, filled contour plot
    # component: 'bx','by','bz','bmag'

def plot_equatorial_flux(positions, flux_values, energy_kev,
                          ax=None, cmap='hot_r', vmin=None, vmax=None,
                          save_path=None) -> plt.Axes:
    # Scatter plot or 2D interpolated flux map in equatorial plane
    # positions: (N, 3) GSM in Re, flux_values: (N,)
    # Adds Earth circle (r=1 Re), labels axes in Re
```

### `inj_trace/visualization/lshell.py`

```python
def plot_lshell_map(lstar_values, flux_values, energy_kev=None,
                    pitch_angle_deg=90.0, ax=None, save_path=None) -> plt.Axes:
    # flux vs L* line plot; optionally color-code by energy
```

### `inj_trace/visualization/timeseries.py`

```python
def plot_flux_timeseries(times, flux_vs_time, energies_kev=None,
                          ax=None, save_path=None) -> plt.Axes:
    # times: seconds or datetime; flux: (N_times,) or (N_times, n_energy)
    # Multiple energies → separate lines with colorbar
```

### `inj_trace/visualization/trajectories3d.py`

```python
def plot_trajectory_3d(traj_data, particle_ids=None, earth_radius=1.0,
                        color_by='energy', ax=None, save_path=None) -> plt.Axes:
    # 3D line plot of guiding-center paths in GSM
    # Adds Earth sphere (wireframe)
    # color_by: 'energy', 'time', 'pitchangle'
    # Template: ~/SHIELDS-PTM/scripts/trajectory_quicklook.py
```

### `inj_trace/visualization/animation.py`

```python
def animate_equatorial_flux(flux_snapshots, positions, times,
                              interval_ms=200, fps=5, save_path=None,
                              cmap='hot_r', vmin=None, vmax=None):
    # Returns FuncAnimation
    # flux_snapshots: list of (N,) arrays; positions: (N, 3)
    # save_path: .mp4 or .gif via matplotlib writer

def animate_lshell(lstar_snapshots, flux_snapshots, times,
                    energy_kev=None, interval_ms=200, fps=5,
                    save_path=None):
    # Returns FuncAnimation
```

---

### CLI entry points

All use `argparse`. Defined in `pyproject.toml` `[project.scripts]`.

#### `inj-config`
```
inj-config set --lgmpy-path PATH --ptm-python-path PATH --ptm-executable PATH
inj-config validate
inj-config show
```

#### `inj-make-fields`
```
inj-make-fields --model T89 --time 2013-03-17T06:00:00 --Kp 5
                --xmin -12 --xmax 12 --nx 75
                --ymin -12 --ymax 12 --ny 75
                --zmin -6  --zmax 6  --nz 51
                --output-dir ./run001/ptm_data
                --workers 8
                [--static]          # duplicate for static PTM run
                [--duration 3600]   # duration_s for tgrid when --static
```

#### `inj-run`
```
inj-run --run-id 1 --run-dir ./run001
        --n-particles 5000 --tlo 0 --thi 3600
        --spatial-mode 3 --r0 6.6 --mltmin -6 --mltmax 6
        --velocity-mode 3 --emin 1 --emax 500 --nenergy 34 --npitch 31
        [--parallel 4]
        [--electron]                # shorthand for set_electron_defaults()
```

#### `inj-plot`
```
inj-plot equatorial  --run-dir DIR --run-id N --energy KEV --save FILE
inj-plot lshell      --run-dir DIR --run-id N --save FILE
inj-plot timeseries  --run-dir DIR --run-ids N,N,N --save FILE
inj-plot trajectory  --run-dir DIR --run-id N [--n-particles N] --save FILE
inj-plot animate     --run-dir DIR --run-ids N,N,N --fps N --save FILE
                     [--type equatorial|lshell]
```

---

## Data flow summary

```
1. FieldGrid.from_spec(spec)
   .evaluate(model='T89', time=dt, model_params={'Kp': 5}, n_workers=8)
       → calls lgmpy.Lgm_T89.T89() at each grid point (parallel)
       → populates .bx/.by/.bz [nT], zeros .ex/.ey/.ez

2. PTMFieldWriter(output_dir)
   .write_static(grid, duration_s=3600)
       → writes ptm_data/ptm_fields_0001.dat + 0002.dat (identical)
       → writes ptm_data/tgrid.dat ([0.0, 3600.0])

3. PTMRunConfig(run_id=1, n_particles=5000, tlo=0, thi=3600, n_snapshots=2)
   .set_electron_defaults()
   .set_spatial_distribution(mode=3, r0=6.6, mltmin=-6, mltmax=6)
   .set_velocity_distribution(mode=3, emin=1, emax=500, nenergy=34, npitch=31)
   .write('ptm_input/')
       → writes ptm_input/ptm_parameters_0001.txt
       → writes ptm_input/dist_density_0001.txt
       → writes ptm_input/dist_velocity_0001.txt

4. PTMExecutor(run_dir='./run001')
   .run_single(run_id=1)
       → subprocess: cd run001 && ./ptm 1
       → writes ptm_output/ptm_0001.dat  (trajectories)
       → writes ptm_output/map_0001.dat  (flux map, if idist=3)

5. TrajectoryData.from_file('ptm_output/ptm_0001.dat')
   .final_positions()    → (N, 3) GSM positions
   .compute_lstar(time, model='Lgm_B_T89', Kp=5)
       → lgmpy.Lstar.get_Lstar() per particle; returns (N,) L* array

6. FluxMapResult.from_map_file('ptm_output/map_0001.dat', Ec=10.0, n=0.01)
       → parse_map_file + energy_to_flux
       → .energies, .angles, .flux, .omni

7. Visualization:
   plot_equatorial_flux(positions, omni_at_energy, energy_kev=100)
   plot_lshell_map(lstar_values, flux)
   animate_equatorial_flux(flux_snapshots, positions, times, save_path='anim.mp4')
```

---

## Critical implementation notes

1. **sys.path injection**: `inj_trace/__init__.py` calls `load_config()` at import time.
   All lgmpy/ptm imports inside functions (deferred) so this runs first.

2. **PTM static field duplication**: Always write ≥ 2 snapshots. `write_static()` does this.

3. **PTM CWD requirement**: `PTMExecutor.run_single()` must `os.chdir(run_dir)` before calling `./ptm`.
   Use `ptm_tools.cd()` context manager for safe restore.

4. **ProcessPoolExecutor for lgmpy**: Use for both field grid evaluation and L* computation.
   lgmpy's ctypes library is not thread-safe. Linux `fork` inherits sys.path.

5. **Lstar sentinel**: lgmpy returns `-1e31` for open field lines. Map to `np.nan`.

6. **T89 Kp validation**: Must be int 0–5. Validate in CLI and `eval_t89()`.

7. **Electric field**: Always zero for static empirical model runs. Stub `set_efield_from_potential()`
   as a hook for future Volland–Stern extension.

8. **PTM parameters**: `ifirst=1, ilast=2, ntot=2` for static runs with 2 snapshots.
   `dtin` should be set to the snapshot interval (e.g., `thi - tlo`).

---

## Verification steps

```bash
# 1. Configure paths
inj-config set --lgmpy-path ~/LANLGeoMag/Python \
               --ptm-python-path ~/SHIELDS-PTM \
               --ptm-executable ~/SHIELDS-PTM/ptm
inj-config validate

# 2. Generate field grid
mkdir -p run001/{ptm_data,ptm_input,ptm_output}
inj-make-fields --model T89 --time 2013-03-17T06:00:00 --Kp 5 \
                --xmin -12 --xmax 12 --nx 51 \
                --ymin -12 --ymax 12 --ny 51 \
                --zmin -6  --zmax 6  --nz 25 \
                --output-dir run001/ptm_data --static --workers 4
# Verify: head -4 run001/ptm_data/ptm_fields_0001.dat  # should show "  51   51   25"

# 3. Run PTM
inj-run --run-id 1 --run-dir run001 --n-particles 500 \
        --tlo 0 --thi 3600 --electron \
        --spatial-mode 3 --r0 6.6 --mltmin -6 --mltmax 6 \
        --velocity-mode 3 --emin 1 --emax 500
# Verify: ls -la run001/ptm_output/

# 4. Plot
inj-plot equatorial --run-dir run001 --run-id 1 --energy 100 --save eq.png
inj-plot trajectory --run-dir run001 --run-id 1 --save traj.png
```
