"""
3-D particle trajectory visualization.

Adapted from ~/SHIELDS-PTM/scripts/trajectory_quicklook.py.
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)


_COLOR_CHOICES = ("energy", "time", "pitchangle", "particle")

# Column indices in PTM 8-column output
_COL_X  = 1
_COL_Y  = 2
_COL_Z  = 3
_COL_E  = 6
_COL_PA = 7


def plot_trajectory_3d(
    traj_data,
    particle_ids: Optional[List[int]] = None,
    earth_radius: float = 1.0,
    color_by: str = "energy",
    cmap: str = "plasma",
    alpha: float = 0.7,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """3-D line traces of particle guiding-centre paths in GSM coordinates.

    Parameters
    ----------
    traj_data    : TrajectoryData instance
    particle_ids : subset of particle IDs to plot; default = all
    earth_radius : radius of the Earth wireframe sphere (Re)
    color_by     : how to colour each trajectory segment:
                   'energy'     — keV (varies along track)
                   'time'       — seconds since start of track
                   'pitchangle' — degrees
                   'particle'   — one distinct colour per particle
    cmap         : matplotlib colormap name
    alpha        : line transparency
    ax           : existing 3D Axes; created if None
    save_path    : if given, save figure to this path

    Returns
    -------
    plt.Axes (3D)
    """
    if color_by not in _COLOR_CHOICES:
        raise ValueError(f"color_by must be one of {_COLOR_CHOICES}, got '{color_by}'")

    if ax is None:
        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")

    ids = particle_ids if particle_ids is not None else traj_data.particle_ids()
    n_particles = len(ids)

    # Determine global colour scale
    all_vals = []
    col_idx = {"energy": _COL_E, "pitchangle": _COL_PA}.get(color_by)

    if color_by == "particle":
        particle_colors = plt.get_cmap(cmap)(np.linspace(0, 1, n_particles))
    else:
        for pid in ids:
            track = traj_data.get_track(pid)
            if col_idx is not None:
                all_vals.extend(track[:, col_idx].tolist())
            elif color_by == "time":
                t0 = track[0, 0]
                all_vals.extend((track[:, 0] - t0).tolist())
        vmin = min(all_vals) if all_vals else 0.0
        vmax = max(all_vals) if all_vals else 1.0
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=vmin, vmax=vmax),
        )
        sm.set_array([])

    for k, pid in enumerate(ids):
        track = traj_data.get_track(pid)
        x = track[:, _COL_X]
        y = track[:, _COL_Y]
        z = track[:, _COL_Z]

        if color_by == "particle":
            _plot_line_3d(ax, x, y, z, particle_colors[k], alpha)
        else:
            if col_idx is not None:
                vals = track[:, col_idx]
            else:
                vals = track[:, 0] - track[0, 0]
            _plot_line_colored_3d(ax, x, y, z, vals, cmap, vmin, vmax, alpha)

    # Earth sphere
    _add_earth_3d(ax, earth_radius)

    ax.set_xlabel("X GSM (Re)")
    ax.set_ylabel("Y GSM (Re)")
    ax.set_zlabel("Z GSM (Re)")
    ax.set_title(f"Particle Trajectories (coloured by {color_by})")

    if color_by != "particle":
        units = {"energy": "keV", "pitchangle": "deg", "time": "s"}
        label = f"{color_by}  [{units.get(color_by, '')}]"
        ax.figure.colorbar(sm, ax=ax, shrink=0.6, label=label)

    if save_path:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plot_line_3d(ax, x, y, z, color, alpha):
    ax.plot(x, y, z, color=color, alpha=alpha, lw=0.8)


def _plot_line_colored_3d(ax, x, y, z, vals, cmap_name, vmin, vmax, alpha):
    """Plot a polyline coloured by vals using matplotlib line collections."""
    import matplotlib
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)
    # Segment the line and colour each segment
    n = len(x)
    for i in range(n - 1):
        c = cmap(norm(0.5 * (vals[i] + vals[i + 1])))
        ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=c, alpha=alpha, lw=0.8)


def _add_earth_3d(ax, radius: float = 1.0) -> None:
    """Draw a wireframe sphere representing Earth."""
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    xs = radius * np.outer(np.cos(u), np.sin(v))
    ys = radius * np.outer(np.sin(u), np.sin(v))
    zs = radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color="steelblue", alpha=0.3, lw=0.4)
