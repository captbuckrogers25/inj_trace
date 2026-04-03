"""
CLI entry point: inj-make-fields

Evaluates an empirical magnetic field model on a 3-D Cartesian GSM grid
and writes PTM-format field files.

Examples
--------
# Static single-time run (writes two identical snapshots)
inj-make-fields --model T89 --time 2013-03-17T06:00:00 --Kp 5 \\
                --xmin -12 --xmax 12 --nx 51 \\
                --ymin -12 --ymax 12 --ny 51 \\
                --zmin -6  --zmax 6  --nz 25 \\
                --output-dir ./run001/ptm_data --static --workers 4

# Time series (one snapshot per entry in times.txt)
inj-make-fields --model T89 --time-file times.txt --Kp 5 \\
                --xmin -12 --xmax 12 --nx 51 ... \\
                --output-dir ./run001/ptm_data --dt 300
"""

import argparse
import sys
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        prog="inj-make-fields",
        description="Generate PTM field files from an empirical lgmpy model",
    )

    # Field model
    parser.add_argument("--model", default="T89", choices=["T89", "TS04", "OP77"],
                        help="Field model (default: T89)")
    parser.add_argument("--time",  default=None,
                        help="UTC time as ISO string, e.g. 2013-03-17T06:00:00")
    parser.add_argument("--time-file", default=None,
                        help="Text file with one ISO time per line (for time series)")

    # T89 parameters
    parser.add_argument("--Kp", type=int, default=2,
                        help="Kp index 0-5 for T89 (default: 2)")

    # TS04 parameters
    parser.add_argument("--P",   type=float, default=2.0,  help="Solar wind pressure nPa")
    parser.add_argument("--Dst", type=float, default=-30., help="Dst index nT")
    parser.add_argument("--By",  type=float, default=0.0,  help="IMF By nT")
    parser.add_argument("--Bz",  type=float, default=-5.0, help="IMF Bz nT")
    parser.add_argument("--W",   type=float, nargs=6, default=[0.1]*6,
                        help="W1-W6 coupling coefficients (6 values)")

    # Grid spec
    parser.add_argument("--xmin", type=float, default=-12.)
    parser.add_argument("--xmax", type=float, default=12.)
    parser.add_argument("--nx",   type=int,   default=51)
    parser.add_argument("--ymin", type=float, default=-12.)
    parser.add_argument("--ymax", type=float, default=12.)
    parser.add_argument("--ny",   type=int,   default=51)
    parser.add_argument("--zmin", type=float, default=-6.)
    parser.add_argument("--zmax", type=float, default=6.)
    parser.add_argument("--nz",   type=int,   default=25)
    parser.add_argument("--mask-inner", type=float, default=1.2,
                        help="Zero B inside this radius in Re (default: 1.2)")

    # Output
    parser.add_argument("--output-dir", default="ptm_data",
                        help="Output directory for field files (default: ptm_data)")
    parser.add_argument("--static", action="store_true",
                        help="Duplicate single snapshot for static PTM run")
    parser.add_argument("--duration", type=float, default=3600.0,
                        help="Run duration in seconds for tgrid when --static (default: 3600)")
    parser.add_argument("--dt", type=float, default=300.0,
                        help="Seconds between snapshots for time series (default: 300)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers for grid evaluation (default: 1)")
    parser.add_argument("--internal-model", default="LGM_IGRF",
                        choices=["LGM_IGRF", "LGM_CDIP", "LGM_EDIP"],
                        help="Internal field model (default: LGM_IGRF)")

    args = parser.parse_args()

    # Build model params
    model_params = {"internal_model": args.internal_model}
    if args.model == "T89":
        model_params["Kp"] = args.Kp
    elif args.model == "TS04":
        model_params.update(P=args.P, Dst=args.Dst, By=args.By, Bz=args.Bz, W=args.W)
    # OP77 needs no extra params beyond internal_model

    # Build grid spec
    spec = {
        "xmin": args.xmin, "xmax": args.xmax, "nx": args.nx,
        "ymin": args.ymin, "ymax": args.ymax, "ny": args.ny,
        "zmin": args.zmin, "zmax": args.zmax, "nz": args.nz,
    }

    from inj_trace.config import load_config
    load_config()
    from inj_trace.fields.grid import FieldGrid
    from inj_trace.fields.writer import PTMFieldWriter

    writer = PTMFieldWriter(args.output_dir)

    if args.time_file:
        # Time series mode
        with open(args.time_file) as fh:
            iso_times = [line.strip() for line in fh if line.strip()]
        times = [datetime.fromisoformat(t) for t in iso_times]
        grids = []
        for i, t in enumerate(times, start=1):
            print(f"  Evaluating snapshot {i}/{len(times)}: {t.isoformat()}", flush=True)
            grid = FieldGrid.from_spec(spec)
            grid.evaluate(args.model, t, model_params,
                          mask_inner=args.mask_inner, n_workers=args.workers)
            grids.append(grid)
        seconds = PTMFieldWriter.times_from_datetimes(times)
        field_paths, tgrid_path = writer.write_time_series(grids, dt=args.dt)
        print(f"Wrote {len(field_paths)} field snapshots to {args.output_dir}/")
        print(f"Wrote {tgrid_path}")
    else:
        if args.time is None:
            print("ERROR: provide --time or --time-file", file=sys.stderr)
            sys.exit(1)
        t = datetime.fromisoformat(args.time)
        n_pts = args.nx * args.ny * args.nz
        print(f"Evaluating {args.model} on {n_pts} grid points ({args.workers} workers)…", flush=True)
        grid = FieldGrid.from_spec(spec)
        grid.evaluate(args.model, t, model_params,
                      mask_inner=args.mask_inner, n_workers=args.workers)
        print(f"  |B| max = {grid.bmag().max():.1f} nT")

        if args.static:
            p1, p2, tg = writer.write_static(grid, duration_s=args.duration)
            print(f"Wrote {p1}")
            print(f"Wrote {p2}  (duplicate for static run)")
            print(f"Wrote {tg}")
        else:
            p = writer.write_snapshot(grid, snapshot_index=1)
            tg = writer.write_tgrid([0.0])
            print(f"Wrote {p}")
            print(f"Wrote {tg}")
            print("NOTE: PTM requires ≥2 snapshots. Use --static or provide multiple times.")
