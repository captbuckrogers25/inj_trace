"""
Central configuration for inj_trace.

Manages paths to the locally-installed (non-pip) upstream packages:
  - lgmpy   (~/LANLGeoMag/Python/)
  - ptm_python (~/SHIELDS-PTM/ptm_python/)
  - ptm Fortran executable (~/SHIELDS-PTM/ptm)

Configuration is read from (in priority order):
  1. Environment variables: INJ_LGMPY_PATH, INJ_PTM_PYTHON_PATH, INJ_PTM_EXE
  2. ~/.inj_trace.cfg  (INI format, section [paths])
  3. Hard-coded defaults based on ~/LANLGeoMag and ~/SHIELDS-PTM locations

Call load_config() once at import time (done by inj_trace/__init__.py).
Subsequent calls are cheap (cached singleton).
"""

import configparser
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class ConfigError(Exception):
    """Raised when inj_trace cannot locate its upstream dependencies."""


_CONFIG_FILE = Path.home() / ".inj_trace.cfg"
_SECTION = "paths"

# Default paths — adjust if the user keeps packages elsewhere
_DEFAULTS = {
    "lgmpy_path":       str(Path.home() / "LANLGeoMag" / "Python"),
    "ptm_python_path":  str(Path.home() / "SHIELDS-PTM"),
    "ptm_executable":   str(Path.home() / "SHIELDS-PTM" / "ptm"),
}


@dataclass
class InjTraceConfig:
    lgmpy_path: str
    ptm_python_path: str
    ptm_executable: str

    def validate(self) -> None:
        """Raise ConfigError if any path is missing or lgmpy cannot be imported."""
        errors = []

        lgmpy_dir = Path(self.lgmpy_path)
        if not lgmpy_dir.is_dir():
            errors.append(f"lgmpy_path does not exist: {self.lgmpy_path}")
        else:
            lgmpy_pkg = lgmpy_dir / "lgmpy"
            if not lgmpy_pkg.is_dir():
                errors.append(
                    f"lgmpy package not found under lgmpy_path: expected {lgmpy_pkg}"
                )

        ptm_dir = Path(self.ptm_python_path)
        if not ptm_dir.is_dir():
            errors.append(f"ptm_python_path does not exist: {self.ptm_python_path}")
        else:
            ptm_pkg = ptm_dir / "ptm_python"
            if not ptm_pkg.is_dir():
                errors.append(
                    f"ptm_python package not found under ptm_python_path: expected {ptm_pkg}\n"
                    "  ptm_python_path should be the PARENT of the ptm_python/ directory,\n"
                    "  i.e. ~/SHIELDS-PTM, not ~/SHIELDS-PTM/ptm_python"
                )

        ptm_exe = Path(self.ptm_executable)
        if not ptm_exe.is_file():
            errors.append(f"ptm_executable not found: {self.ptm_executable}")
        elif not os.access(str(ptm_exe), os.X_OK):
            errors.append(f"ptm_executable is not executable: {self.ptm_executable}")

        if errors:
            raise ConfigError(
                "inj_trace configuration errors:\n  "
                + "\n  ".join(errors)
                + "\n\nRun 'inj-config set ...' to fix, or set environment variables:\n"
                + "  INJ_LGMPY_PATH, INJ_PTM_PYTHON_PATH, INJ_PTM_EXE"
            )

        # Try importing lgmpy as a final check
        _inject_paths(self)
        try:
            import lgmpy  # noqa: F401
        except ImportError as exc:
            raise ConfigError(
                f"lgmpy path is set ({self.lgmpy_path}) but 'import lgmpy' failed: {exc}\n"
                "Check that LANLGeoMag was built and libLanlGeoMag.so is visible "
                "(LD_LIBRARY_PATH or rpath)."
            ) from exc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _inject_paths(cfg: InjTraceConfig) -> None:
    """Inject lgmpy_path and ptm_python_path into sys.path (idempotent).

    ptm_python_path is inserted at position 0 (not pip-installed; must take precedence).
    lgmpy_path is appended (lgmpy is already installed to site-packages; source dir
    must not shadow the installed version, which contains the generated Lgm_Wrap.py).
    """
    ptm = str(cfg.ptm_python_path)
    if ptm not in sys.path:
        sys.path.insert(0, ptm)

    lgm = str(cfg.lgmpy_path)
    if lgm not in sys.path:
        sys.path.append(lgm)


def _read_cfg_file(path: Path) -> dict:
    cp = configparser.ConfigParser()
    cp.read(str(path))
    if _SECTION not in cp:
        return {}
    return dict(cp[_SECTION])


def _build_config() -> InjTraceConfig:
    file_vals = _read_cfg_file(_CONFIG_FILE)
    result = {}
    for key, default in _DEFAULTS.items():
        env_key = "INJ_" + key.upper()
        if env_key in os.environ:
            result[key] = os.environ[env_key]
        elif key in file_vals:
            result[key] = file_vals[key]
        else:
            result[key] = default
    return InjTraceConfig(**result)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_singleton: Optional[InjTraceConfig] = None


def load_config() -> InjTraceConfig:
    """Return (and cache) the active InjTraceConfig, injecting sys.path."""
    global _singleton
    if _singleton is None:
        _singleton = _build_config()
        _inject_paths(_singleton)
    return _singleton


def save_config(cfg: InjTraceConfig) -> None:
    """Write cfg to ~/.inj_trace.cfg."""
    cp = configparser.ConfigParser()
    cp[_SECTION] = {
        "lgmpy_path":      cfg.lgmpy_path,
        "ptm_python_path": cfg.ptm_python_path,
        "ptm_executable":  cfg.ptm_executable,
    }
    with open(str(_CONFIG_FILE), "w") as fh:
        cp.write(fh)
    # Reset singleton so next load_config() picks up new values
    global _singleton
    _singleton = None
