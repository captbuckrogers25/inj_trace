"""
CLI entry point: inj-run

Configures and launches a SHIELDS-PTM simulation.

Example
-------
inj-run --run-id 1 --run-dir ./run001 \\
        --n-particles 5000 --tlo 0 --thi 3600 \\
        --electron \\
        --spatial-mode 3 --r0 6.6 --mltmin -6 --mltmax 6 \\
        --velocity-mode 3 --emin 1 --emax 500 --nenergy 34 --npitch 31 \\
        --parallel 4
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="inj-run",
        description="Configure and launch a PTM simulation",
    )

    # Run identification
    parser.add_argument("--run-id",   type=int, default=1,    help="PTM run ID (default: 1)")
    parser.add_argument("--run-dir",  default=".",             help="Run directory (default: .)")
    parser.add_argument("--n-particles", type=int, default=1000, help="Number of test particles")
    parser.add_argument("--tlo",      type=float, default=0.0,  help="Start time in seconds")
    parser.add_argument("--thi",      type=float, default=3600., help="End time in seconds")
    parser.add_argument("--n-snapshots", type=int, default=2,   help="Number of field snapshots")
    parser.add_argument("--trace-dir", type=int, default=-1,    help="+1=forward, -1=backward")
    parser.add_argument("--electron", action="store_true", help="Preset electron defaults")

    # Spatial distribution
    parser.add_argument("--spatial-mode", type=int, default=2,
                        help="Spatial mode: 1=point, 2=box, 3=ring (default: 2)")
    # Mode 1 params
    parser.add_argument("--x0", type=float, default=-6.6)
    parser.add_argument("--y0", type=float, default=0.0)
    parser.add_argument("--z0", type=float, default=0.0)
    # Mode 2 params
    parser.add_argument("--xmin-s", type=float, default=-7.0, dest="xmin_s")
    parser.add_argument("--xmax-s", type=float, default=-6.0, dest="xmax_s")
    parser.add_argument("--ymin-s", type=float, default=-0.5, dest="ymin_s")
    parser.add_argument("--ymax-s", type=float, default=0.5,  dest="ymax_s")
    parser.add_argument("--zmin-s", type=float, default=-0.5, dest="zmin_s")
    parser.add_argument("--zmax-s", type=float, default=0.5,  dest="zmax_s")
    # Mode 3 params
    parser.add_argument("--r0",     type=float, default=6.6)
    parser.add_argument("--mltmin", type=float, default=-6.0)
    parser.add_argument("--mltmax", type=float, default=6.0)

    # Velocity distribution
    parser.add_argument("--velocity-mode", type=int, default=3,
                        help="Velocity mode: 1=ring, 2=bimax, 3=fluxmap (default: 3)")
    # Mode 1 params
    parser.add_argument("--ekev",  type=float, default=100.0)
    parser.add_argument("--alpha", type=float, default=90.0)
    parser.add_argument("--phi",   type=float, default=180.0)
    # Mode 3 params
    parser.add_argument("--emin",     type=float, default=1.0)
    parser.add_argument("--emax",     type=float, default=500.0)
    parser.add_argument("--nenergy",  type=int,   default=34)
    parser.add_argument("--npitch",   type=int,   default=31)
    parser.add_argument("--pamin",    type=float, default=0.0)
    parser.add_argument("--pamax",    type=float, default=90.0)
    parser.add_argument("--xsource", type=float, default=-12.0)

    # Parallel
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel PTM processes (default: 1)")
    parser.add_argument("--timeout",  type=float, default=None,
                        help="Per-run timeout in seconds")

    # PTM executable override
    parser.add_argument("--ptm-exe", default=None, help="Override PTM executable path")

    args = parser.parse_args()

    from inj_trace.config import load_config
    load_config()
    from inj_trace.runner.ptm_setup import PTMRunConfig
    from inj_trace.runner.executor import PTMExecutor

    # --- Build run IDs for parallel ---
    if args.parallel > 1:
        run_ids = list(range(args.run_id, args.run_id + args.parallel))
    else:
        run_ids = [args.run_id]

    for rid in run_ids:
        cfg = PTMRunConfig(
            run_id=rid,
            n_particles=args.n_particles,
            trace_direction=args.trace_dir,
            tlo=args.tlo,
            thi=args.thi,
            n_snapshots=args.n_snapshots,
        )

        if args.electron:
            cfg.set_electron_defaults()

        # Spatial distribution
        mode = args.spatial_mode
        if mode == 1:
            cfg.set_spatial_distribution(1, x0=args.x0, y0=args.y0, z0=args.z0)
        elif mode == 2:
            cfg.set_spatial_distribution(
                2,
                xmin=args.xmin_s, xmax=args.xmax_s,
                ymin=args.ymin_s, ymax=args.ymax_s,
                zmin=args.zmin_s, zmax=args.zmax_s,
            )
        elif mode == 3:
            cfg.set_spatial_distribution(3, r0=args.r0, mltmin=args.mltmin, mltmax=args.mltmax)
        else:
            print(f"ERROR: Unknown spatial mode {mode}", file=sys.stderr)
            sys.exit(1)

        # Velocity distribution
        vmode = args.velocity_mode
        if vmode == 1:
            cfg.set_velocity_distribution(1, ekev=args.ekev, alpha=args.alpha, phi=args.phi)
        elif vmode == 2:
            import scipy.constants as const
            ckm = const.c / 1e3
            cfg.set_velocity_distribution(2, vtperp=0.25*ckm, vtpara=0.25*ckm, phi=args.phi)
        elif vmode == 3:
            cfg.set_velocity_distribution(
                3,
                nenergy=args.nenergy, npitch=args.npitch,
                phi=args.phi,
                emin=args.emin, emax=args.emax,
                pamin=args.pamin, pamax=args.pamax,
                xsource=args.xsource,
            )
        else:
            print(f"ERROR: Unknown velocity mode {vmode}", file=sys.stderr)
            sys.exit(1)

        import os
        input_dir = os.path.join(args.run_dir, "ptm_input")
        cfg.write(input_dir)
        print(f"Wrote PTM input files for run {rid} to {input_dir}/")

    # --- Execute ---
    executor = PTMExecutor(run_dir=args.run_dir, ptm_executable=args.ptm_exe)

    if len(run_ids) == 1:
        print(f"Launching PTM run {run_ids[0]}…", flush=True)
        result = executor.run_single(run_ids[0], timeout=args.timeout)
        print(f"Run {run_ids[0]} complete (returncode={result.returncode})")
        if result.stdout:
            print(result.stdout[-2000:])  # last 2000 chars of stdout
    else:
        print(f"Launching {len(run_ids)} PTM runs in parallel…", flush=True)
        results = executor.run_parallel(run_ids, max_workers=len(run_ids), timeout=args.timeout)
        for rid, res in results.items():
            print(f"  Run {rid}: returncode={res.returncode}")
