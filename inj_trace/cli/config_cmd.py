"""
CLI entry point: inj-config

Commands
--------
inj-config set    --lgmpy-path PATH --ptm-python-path PATH --ptm-executable PATH
inj-config validate
inj-config show
"""

import argparse
import sys


def _cmd_set(args):
    from inj_trace.config import InjTraceConfig, save_config, load_config, _DEFAULTS

    # Start from current config (or defaults) and overlay provided values
    try:
        cfg = load_config()
    except Exception:
        from inj_trace.config import InjTraceConfig
        cfg = InjTraceConfig(**_DEFAULTS)

    if args.lgmpy_path:
        cfg = InjTraceConfig(
            lgmpy_path=args.lgmpy_path,
            ptm_python_path=cfg.ptm_python_path,
            ptm_executable=cfg.ptm_executable,
        )
    if args.ptm_python_path:
        cfg = InjTraceConfig(
            lgmpy_path=cfg.lgmpy_path,
            ptm_python_path=args.ptm_python_path,
            ptm_executable=cfg.ptm_executable,
        )
    if args.ptm_executable:
        cfg = InjTraceConfig(
            lgmpy_path=cfg.lgmpy_path,
            ptm_python_path=cfg.ptm_python_path,
            ptm_executable=args.ptm_executable,
        )

    save_config(cfg)
    print("Configuration saved to ~/.inj_trace.cfg")
    print(f"  lgmpy_path:      {cfg.lgmpy_path}")
    print(f"  ptm_python_path: {cfg.ptm_python_path}")
    print(f"  ptm_executable:  {cfg.ptm_executable}")


def _cmd_validate(args):
    from inj_trace.config import load_config, ConfigError
    try:
        cfg = load_config()
        cfg.validate()
        print("Configuration OK:")
        print(f"  lgmpy_path:      {cfg.lgmpy_path}")
        print(f"  ptm_python_path: {cfg.ptm_python_path}")
        print(f"  ptm_executable:  {cfg.ptm_executable}")
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


def _cmd_show(args):
    from inj_trace.config import load_config
    cfg = load_config()
    print(f"lgmpy_path:      {cfg.lgmpy_path}")
    print(f"ptm_python_path: {cfg.ptm_python_path}")
    print(f"ptm_executable:  {cfg.ptm_executable}")


def main():
    parser = argparse.ArgumentParser(
        prog="inj-config",
        description="Manage inj_trace path configuration",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # set
    p_set = sub.add_parser("set", help="Set configuration paths")
    p_set.add_argument("--lgmpy-path",      default=None, help="Path to LANLGeoMag/Python/")
    p_set.add_argument("--ptm-python-path", default=None, help="Path to SHIELDS-PTM/ptm_python/")
    p_set.add_argument("--ptm-executable",  default=None, help="Path to ptm Fortran executable")
    p_set.set_defaults(func=_cmd_set)

    # validate
    p_val = sub.add_parser("validate", help="Validate paths and test imports")
    p_val.set_defaults(func=_cmd_validate)

    # show
    p_show = sub.add_parser("show", help="Print current configuration")
    p_show.set_defaults(func=_cmd_show)

    args = parser.parse_args()
    args.func(args)
