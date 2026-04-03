"""
Example 3: Create a time-lapse animation of injection events across multiple PTM runs.

This example assumes you have already run several PTM simulations (run IDs 1..N)
with the same spatial grid but potentially different field conditions (e.g. varying Kp).

Run from the inj_trace project root:
    python examples/03_animate.py --run-dir ./multi_run --run-ids 1 2 3 4 5
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from inj_trace.config import load_config
load_config()

from inj_trace.postprocess.trajectories import TrajectoryData
from inj_trace.visualization.animation import animate_equatorial_flux, animate_lshell


def main():
    parser = argparse.ArgumentParser(description="Animate injection events")
    parser.add_argument("--run-dir",  default="multi_run")
    parser.add_argument("--run-ids",  type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--dt",       type=float, default=300.0,
                        help="Seconds between runs (default: 300)")
    parser.add_argument("--time0",    default="2013-03-17T06:00:00",
                        help="Reference time for first run")
    parser.add_argument("--save-eq",  default="injection_equatorial.mp4")
    parser.add_argument("--fps",      type=int, default=5)
    args = parser.parse_args()

    t0 = datetime.fromisoformat(args.time0)
    times = [t0 + timedelta(seconds=i * args.dt) for i in range(len(args.run_ids))]
    times_s = np.array([i * args.dt for i in range(len(args.run_ids))])

    # Load all trajectories
    flux_snapshots = []
    all_positions = None

    for i, rid in enumerate(args.run_ids):
        path = os.path.join(args.run_dir, "ptm_output", f"ptm_{rid:04d}.dat")
        if not os.path.isfile(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
        traj = TrajectoryData.from_file(path)
        if all_positions is None:
            all_positions = traj.final_positions()
        flux_snapshots.append(traj.final_energies())
        print(f"  Loaded run {rid}: {traj.n_particles()} particles")

    if all_positions is None or len(flux_snapshots) == 0:
        print("No data found. Check --run-dir and --run-ids.")
        return

    # Equatorial animation
    print(f"Creating equatorial animation ({len(flux_snapshots)} frames)…")
    anim = animate_equatorial_flux(
        flux_snapshots=flux_snapshots,
        positions=all_positions,
        times=times_s[:len(flux_snapshots)],
        energy_kev=None,
        fps=args.fps,
        save_path=args.save_eq,
    )
    print(f"Saved {args.save_eq}")


if __name__ == "__main__":
    main()
