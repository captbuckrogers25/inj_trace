"""
Time-lapse animations of injection events.

animate_equatorial_flux : particle flux in the equatorial plane vs time
animate_lshell          : flux vs L* vs time
"""

from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.interpolate import griddata


def animate_equatorial_flux(
    flux_snapshots: List[np.ndarray],
    positions: np.ndarray,
    times: Union[np.ndarray, List],
    energy_kev: Optional[float] = None,
    interval_ms: int = 200,
    fps: int = 5,
    cmap: str = "hot_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    log_scale: bool = True,
    grid_res: int = 80,
    save_path: Optional[str] = None,
) -> FuncAnimation:
    """Time-lapse animation of equatorial-plane flux.

    Parameters
    ----------
    flux_snapshots : list of (N,) flux arrays, one per time step
    positions      : (N, 3) fixed GSM positions in Re
    times          : (n_times,) seconds or datetime objects
    energy_kev     : label energy in keV
    interval_ms    : delay between frames in milliseconds
    fps            : frames per second (for video export)
    cmap           : matplotlib colormap
    vmin, vmax     : colour scale limits (in log units if log_scale=True)
    log_scale      : use log10 of flux
    grid_res       : interpolation grid resolution
    save_path      : .mp4 or .gif path; None → display interactively

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    x = positions[:, 0]
    y = positions[:, 1]
    xi = np.linspace(x.min(), x.max(), grid_res)
    yi = np.linspace(y.min(), y.max(), grid_res)
    Xi, Yi = np.meshgrid(xi, yi)

    times_arr = np.asarray(times)

    # Precompute log-flux grids
    def _prep(flux_arr):
        v = np.asarray(flux_arr, dtype=float)
        if log_scale:
            with np.errstate(divide="ignore", invalid="ignore"):
                v = np.log10(np.where(v > 0, v, np.nan))
        return griddata((x, y), v, (Xi, Yi), method="linear")

    grids = [_prep(f) for f in flux_snapshots]

    # Determine colour scale
    all_vals = np.concatenate([g[np.isfinite(g)] for g in grids])
    _vmin = vmin if vmin is not None else float(np.nanpercentile(all_vals, 5))
    _vmax = vmax if vmax is not None else float(np.nanpercentile(all_vals, 99))

    fig, ax = plt.subplots(figsize=(7, 6))
    cf_init = ax.contourf(Xi, Yi, grids[0], levels=40, cmap=cmap, vmin=_vmin, vmax=_vmax)
    plt.colorbar(
        cf_init, ax=ax,
        label="log₁₀ flux  [keV⁻¹ cm⁻² s⁻¹ sr⁻¹]" if log_scale else "flux",
    )

    # Earth circle
    earth = plt.Circle((0, 0), 1.0, color="steelblue", zorder=5)
    ax.add_patch(earth)

    ax.set_xlabel("X GSM (Re)")
    ax.set_ylabel("Y GSM (Re)")
    ax.set_aspect("equal")
    e_label = f"  E={energy_kev:.0f} keV" if energy_kev is not None else ""

    time_text = ax.set_title("")

    def _update(frame_idx):
        ax.collections.clear()
        cf = ax.contourf(Xi, Yi, grids[frame_idx], levels=40, cmap=cmap, vmin=_vmin, vmax=_vmax)
        earth = plt.Circle((0, 0), 1.0, color="steelblue", zorder=5)
        ax.add_patch(earth)
        t = times_arr[frame_idx]
        t_str = str(t) if hasattr(t, "isoformat") else f"t={float(t):.0f} s"
        time_text.set_text(f"Equatorial Flux{e_label}  [{t_str}]")
        return ax.collections

    anim = FuncAnimation(
        fig, _update,
        frames=len(flux_snapshots),
        interval=interval_ms,
        blit=False,
    )

    if save_path:
        _save_animation(anim, save_path, fps)
    return anim


def animate_lshell(
    lstar_snapshots: List[np.ndarray],
    flux_snapshots: List[np.ndarray],
    times: Union[np.ndarray, List],
    energy_kev: Optional[float] = None,
    lstar_range: Optional[tuple] = None,
    interval_ms: int = 200,
    fps: int = 5,
    log_scale: bool = True,
    save_path: Optional[str] = None,
) -> FuncAnimation:
    """Time-lapse animation of flux vs L*.

    Parameters
    ----------
    lstar_snapshots : list of (N,) L* arrays, one per time step
    flux_snapshots  : list of (N,) flux arrays, one per time step
    times           : (n_times,) seconds or datetime objects
    energy_kev      : label energy
    lstar_range     : (lmin, lmax) x-axis range
    interval_ms     : frame delay in milliseconds
    fps             : frames per second for export
    log_scale       : log10 y-axis
    save_path       : .mp4 or .gif; None → interactive

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    times_arr = np.asarray(times)

    fig, ax = plt.subplots(figsize=(7, 5))
    (line,) = ax.plot([], [], "-o", markersize=3, lw=1.5, color="firebrick")

    if log_scale:
        ax.set_yscale("log")
        ylabel = "Flux  [keV⁻¹ cm⁻² s⁻¹ sr⁻¹]"
    else:
        ylabel = "Flux"

    ax.set_xlabel("L*")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    all_lstar = np.concatenate([np.asarray(l)[np.isfinite(l)] for l in lstar_snapshots])
    all_flux  = np.concatenate([np.asarray(f)[np.isfinite(f) & (np.asarray(f) > 0)] for f in flux_snapshots])

    if lstar_range is None:
        ax.set_xlim(float(all_lstar.min()), float(all_lstar.max()))
    else:
        ax.set_xlim(*lstar_range)

    if log_scale and all_flux.size > 0:
        ax.set_ylim(10 ** (np.floor(np.log10(all_flux.min())) - 0.5),
                    10 ** (np.ceil(np.log10(all_flux.max())) + 0.5))

    e_label = f"  E={energy_kev:.0f} keV" if energy_kev is not None else ""

    def _update(frame_idx):
        ls = np.asarray(lstar_snapshots[frame_idx], dtype=float)
        fl = np.asarray(flux_snapshots[frame_idx], dtype=float)
        mask = np.isfinite(ls) & np.isfinite(fl) & (fl > 0)
        ls, fl = ls[mask], fl[mask]
        if ls.size > 0:
            order = np.argsort(ls)
            line.set_data(ls[order], fl[order])
        else:
            line.set_data([], [])
        t = times_arr[frame_idx]
        t_str = str(t) if hasattr(t, "isoformat") else f"t={float(t):.0f} s"
        ax.set_title(f"Flux vs L*{e_label}  [{t_str}]")
        return (line,)

    anim = FuncAnimation(
        fig, _update,
        frames=len(lstar_snapshots),
        interval=interval_ms,
        blit=True,
    )

    if save_path:
        _save_animation(anim, save_path, fps)
    return anim


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _save_animation(anim: FuncAnimation, path: str, fps: int) -> None:
    """Save animation to .mp4 or .gif."""
    if path.endswith(".gif"):
        writer = "pillow"
    else:
        writer = "ffmpeg"
    anim.save(path, writer=writer, fps=fps, dpi=150)
