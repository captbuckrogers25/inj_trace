"""
CLI entry point: inj-plot

Subcommands
-----------
inj-plot equatorial  --run-dir DIR --run-id N --energy KEV --save FILE
inj-plot lshell      --run-dir DIR --run-id N --Kp N --save FILE
inj-plot timeseries  --run-dirs DIR ... --run-ids N ... --save FILE
inj-plot trajectory  --run-dir DIR --run-id N [--n-particles N] --save FILE
inj-plot animate     --run-dir DIR --run-ids N,N,N --type equatorial|lshell --save FILE
"""

import argparse
import sys
from pathlib import Path


def _add_common(p):
    p.add_argument("--run-dir",  default=".", help="Run directory (default: .)")
    p.add_argument("--run-id",   type=int, default=1, help="PTM run ID")
    p.add_argument("--save",     default=None, help="Output file path (shown if not given)")


def _cmd_equatorial(args):
    from inj_trace.config import load_config
    load_config()
    from inj_trace.postprocess.trajectories import TrajectoryData
    from inj_trace.visualization.equatorial import plot_equatorial_flux
    import matplotlib.pyplot as plt

    path = Path(args.run_dir) / "ptm_output" / f"ptm_{args.run_id:04d}.dat"
    if not path.is_file():
        print(f"ERROR: trajectory file not found: {path}", file=sys.stderr)
        sys.exit(1)

    traj = TrajectoryData.from_file(str(path))
    positions = traj.final_positions()
    energies  = traj.final_energies()

    # Simple: colour points by their energy; no full flux-map postprocessing needed
    from inj_trace.visualization.equatorial import plot_equatorial_flux
    energy_kev = args.energy

    # Filter to particles near the requested energy (within a factor of 2)
    mask = (energies > energy_kev / 2) & (energies < energy_kev * 2)
    if mask.sum() == 0:
        print("WARNING: no particles near the requested energy; plotting all.")
        mask = np.ones(len(energies), dtype=bool)

    import numpy as np
    ax = plot_equatorial_flux(
        positions[mask],
        energies[mask],
        energy_kev=energy_kev,
        save_path=args.save,
    )
    if args.save:
        print(f"Saved to {args.save}")
    else:
        plt.show()


def _cmd_lshell(args):
    from inj_trace.config import load_config
    load_config()
    from datetime import datetime
    from inj_trace.postprocess.trajectories import TrajectoryData
    from inj_trace.visualization.lshell import plot_lshell_map
    import matplotlib.pyplot as plt
    import numpy as np

    path = Path(args.run_dir) / "ptm_output" / f"ptm_{args.run_id:04d}.dat"
    traj = TrajectoryData.from_file(str(path))
    energies = traj.final_energies()

    if args.time is None:
        print("ERROR: --time required for L* computation (ISO format)", file=sys.stderr)
        sys.exit(1)
    t = datetime.fromisoformat(args.time)
    print(f"Computing L* for {traj.n_particles()} particles (quality={args.quality})…")
    lstar = traj.compute_lstar(t, model=args.lstar_model, Kp=args.Kp,
                               quality=args.quality, n_workers=args.workers)

    ax = plot_lshell_map(lstar, energies, save_path=args.save)
    if args.save:
        print(f"Saved to {args.save}")
    else:
        plt.show()


def _cmd_trajectory(args):
    from inj_trace.config import load_config
    load_config()
    from inj_trace.postprocess.trajectories import TrajectoryData
    from inj_trace.visualization.trajectories3d import plot_trajectory_3d
    import matplotlib.pyplot as plt

    path = Path(args.run_dir) / "ptm_output" / f"ptm_{args.run_id:04d}.dat"
    traj = TrajectoryData.from_file(str(path))

    pids = traj.particle_ids()
    if args.n_particles and args.n_particles < len(pids):
        import random
        pids = sorted(random.sample(pids, args.n_particles))

    ax = plot_trajectory_3d(traj, particle_ids=pids,
                            color_by=args.color_by, save_path=args.save)
    if args.save:
        print(f"Saved to {args.save}")
    else:
        plt.show()


def _cmd_timeseries(args):
    from inj_trace.config import load_config
    load_config()
    from inj_trace.postprocess.trajectories import TrajectoryData
    from inj_trace.visualization.timeseries import plot_flux_timeseries
    import matplotlib.pyplot as plt
    import numpy as np

    run_ids = [int(r) for r in args.run_ids.split(",")]
    times_s = list(range(len(run_ids)))  # placeholder seconds

    omni_list = []
    for rid in run_ids:
        path = Path(args.run_dir) / "ptm_output" / f"ptm_{rid:04d}.dat"
        traj = TrajectoryData.from_file(str(path))
        omni_list.append(traj.final_energies().mean())

    ax = plot_flux_timeseries(
        np.array(times_s), np.array(omni_list),
        save_path=args.save,
    )
    if args.save:
        print(f"Saved to {args.save}")
    else:
        plt.show()


def _cmd_animate(args):
    from inj_trace.config import load_config
    load_config()
    from inj_trace.postprocess.trajectories import TrajectoryData
    import numpy as np

    run_ids = [int(r) for r in args.run_ids.split(",")]

    trajs = []
    for rid in run_ids:
        path = Path(args.run_dir) / "ptm_output" / f"ptm_{rid:04d}.dat"
        trajs.append(TrajectoryData.from_file(str(path)))

    # Collect all final positions (use first run's positions as reference)
    positions = trajs[0].final_positions()
    flux_snapshots = [t.final_energies() for t in trajs]  # use energy as proxy
    times_s = np.arange(len(run_ids)) * args.dt

    if args.type == "equatorial":
        from inj_trace.visualization.animation import animate_equatorial_flux
        anim = animate_equatorial_flux(
            flux_snapshots, positions, times_s,
            fps=args.fps, save_path=args.save,
        )
    else:
        print("ERROR: lshell animation requires --time; not yet implemented in CLI", file=sys.stderr)
        sys.exit(1)

    if args.save:
        print(f"Saved animation to {args.save}")
    else:
        import matplotlib.pyplot as plt
        plt.show()


def main():
    import numpy as np  # noqa: needed in subcommand closures

    parser = argparse.ArgumentParser(prog="inj-plot", description="Visualize PTM results")
    sub = parser.add_subparsers(dest="command", required=True)

    # equatorial
    p_eq = sub.add_parser("equatorial", help="Equatorial flux map")
    _add_common(p_eq)
    p_eq.add_argument("--energy", type=float, default=100.0, help="Energy in keV")
    p_eq.set_defaults(func=_cmd_equatorial)

    # lshell
    p_ls = sub.add_parser("lshell", help="Flux vs L*")
    _add_common(p_ls)
    p_ls.add_argument("--time",       default=None, help="UTC ISO time for L* computation")
    p_ls.add_argument("--Kp",         type=int, default=2)
    p_ls.add_argument("--lstar-model", default="Lgm_B_T89")
    p_ls.add_argument("--quality",    type=int, default=3)
    p_ls.add_argument("--workers",    type=int, default=1)
    p_ls.set_defaults(func=_cmd_lshell)

    # trajectory
    p_tr = sub.add_parser("trajectory", help="3-D particle trajectories")
    _add_common(p_tr)
    p_tr.add_argument("--n-particles", type=int, default=None,
                      help="Max particles to plot (default: all)")
    p_tr.add_argument("--color-by", default="energy",
                      choices=["energy", "time", "pitchangle", "particle"])
    p_tr.set_defaults(func=_cmd_trajectory)

    # timeseries
    p_ts = sub.add_parser("timeseries", help="Flux time series across runs")
    p_ts.add_argument("--run-dir",  default=".", help="Run directory")
    p_ts.add_argument("--run-ids",  required=True, help="Comma-separated run IDs, e.g. 1,2,3")
    p_ts.add_argument("--save",     default=None)
    p_ts.set_defaults(func=_cmd_timeseries)

    # animate
    p_an = sub.add_parser("animate", help="Time-lapse animation across runs")
    p_an.add_argument("--run-dir",  default=".")
    p_an.add_argument("--run-ids",  required=True, help="Comma-separated run IDs")
    p_an.add_argument("--type",     default="equatorial", choices=["equatorial", "lshell"])
    p_an.add_argument("--dt",       type=float, default=300., help="Seconds between runs")
    p_an.add_argument("--fps",      type=int, default=5)
    p_an.add_argument("--save",     default=None)
    p_an.set_defaults(func=_cmd_animate)

    args = parser.parse_args()
    args.func(args)
