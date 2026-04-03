"""
inj_trace — energetic electron injection visualization

Wraps LANLGeoMag (lgmpy) and SHIELDS-PTM to run end-to-end
particle injection simulations and produce matplotlib figures
and animations.

Quick start
-----------
    from inj_trace import FieldGrid, PTMFieldWriter, PTMRunConfig, PTMExecutor
    from inj_trace import TrajectoryData, FluxMapResult
    from inj_trace import viz
    from datetime import datetime

    # 1. Build field grid
    grid = FieldGrid.from_spec({'xmin': -12, 'xmax': 12, 'nx': 51,
                                'ymin': -12, 'ymax': 12, 'ny': 51,
                                'zmin':  -6, 'zmax':  6, 'nz': 25})
    grid.evaluate('T89', datetime(2013, 3, 17, 6), {'Kp': 5})

    # 2. Write PTM files
    writer = PTMFieldWriter('run001/ptm_data')
    writer.write_static(grid, duration_s=3600)

    # 3. Configure and run PTM
    cfg = PTMRunConfig(run_id=1, n_particles=500, tlo=0, thi=3600)
    cfg.set_electron_defaults()
    cfg.set_spatial_distribution(3, r0=6.6, mltmin=-6, mltmax=6)
    cfg.set_velocity_distribution(3, emin=1, emax=500)
    cfg.write('run001/ptm_input')
    PTMExecutor('run001').run_single(1)

    # 4. Post-process
    traj = TrajectoryData.from_file('run001/ptm_output/ptm_0001.dat')
    lstar = traj.compute_lstar(datetime(2013, 3, 17, 6))

    # 5. Visualize
    viz.lshell.plot_lshell_map(lstar, traj.final_energies(), save_path='lshell.png')

Configuration
-------------
Run 'inj-config set ...' to set paths to LANLGeoMag and SHIELDS-PTM,
or set environment variables INJ_LGMPY_PATH, INJ_PTM_PYTHON_PATH, INJ_PTM_EXE.
"""

# Inject upstream package paths before any lgmpy / ptm_python imports
from .config import load_config as _load_config
_load_config()

# Public API
from .config import InjTraceConfig, load_config, save_config, ConfigError
from .fields.grid import FieldGrid
from .fields.writer import PTMFieldWriter
from .runner.ptm_setup import PTMRunConfig
from .runner.executor import PTMExecutor
from .postprocess.trajectories import TrajectoryData
from .postprocess.fluxmap import FluxMapResult
from . import visualization as viz


def run_injection(
    model: str,
    time,
    model_params: dict,
    grid_spec: dict,
    run_config: dict,
    run_dir: str,
    n_workers: int = 1,
) -> "FluxMapResult":
    """End-to-end convenience function.

    Builds a field grid, writes PTM input files, launches PTM,
    and returns a FluxMapResult.

    Parameters
    ----------
    model        : 'T89', 'TS04', or 'OP77'
    time         : UTC datetime (or list of datetimes for a time series)
    model_params : dict of model-specific kwargs, e.g. {'Kp': 5}
    grid_spec    : dict with xmin/xmax/nx/ymin/ymax/ny/zmin/zmax/nz keys
    run_config   : dict passed to PTMRunConfig, plus optional keys:
                   'Ec', 'n', 'kind', 'kap' for FluxMapResult.from_run()
    run_dir      : directory for all PTM subdirectories
    n_workers    : parallel workers for field grid evaluation

    Returns
    -------
    FluxMapResult
    """
    import os
    from datetime import datetime as _dt

    os.makedirs(run_dir, exist_ok=True)
    for sub in ("ptm_data", "ptm_input", "ptm_output"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)

    # --- Field grid ---
    grid = FieldGrid.from_spec(grid_spec)
    t = time if isinstance(time, _dt) else time[0]
    grid.evaluate(model, t, model_params, n_workers=n_workers)

    duration = run_config.get("thi", 3600.0) - run_config.get("tlo", 0.0)
    writer = PTMFieldWriter(os.path.join(run_dir, "ptm_data"))
    writer.write_static(grid, duration_s=duration)

    # --- PTM config ---
    rc_keys = {
        k: v for k, v in run_config.items()
        if k not in ("Ec", "n", "kind", "kap")
    }
    ptm_cfg = PTMRunConfig(**rc_keys)
    ptm_cfg.set_electron_defaults()
    ptm_cfg.write(os.path.join(run_dir, "ptm_input"))

    # --- Run PTM ---
    PTMExecutor(run_dir).run_single(run_config.get("run_id", 1))

    # --- Post-process ---
    return FluxMapResult.from_run(
        run_id=run_config.get("run_id", 1),
        run_dir=run_dir,
        Ec=run_config.get("Ec", 10.0),
        n=run_config.get("n", 0.01),
        kind=run_config.get("kind", "kappa"),
        kap=run_config.get("kap", 2.5),
    )


__all__ = [
    "InjTraceConfig", "load_config", "save_config", "ConfigError",
    "FieldGrid", "PTMFieldWriter",
    "PTMRunConfig", "PTMExecutor",
    "TrajectoryData", "FluxMapResult",
    "viz",
    "run_injection",
]
