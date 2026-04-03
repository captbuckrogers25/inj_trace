"""
Equatorial-plane visualization functions.

plot_equatorial_bfield : filled-contour map of B-field in the equatorial plane
plot_equatorial_flux   : scatter / interpolated flux map in the equatorial plane
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy.interpolate import griddata


def plot_equatorial_bfield(
    grid,
    component: str = "bmag",
    z_slice: float = 0.0,
    ax: Optional[plt.Axes] = None,
    cmap: str = "viridis",
    log_scale: bool = True,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Filled contour map of a B-field component in the equatorial plane.

    Parameters
    ----------
    grid      : inj_trace FieldGrid with evaluated B-field
    component : 'bmag' (default), 'bx', 'by', 'bz'
    z_slice   : Z-GSM value in Re at which to slice (default 0 = equatorial)
    ax        : existing Axes (3D not required); created if None
    cmap      : matplotlib colormap name
    log_scale : use logarithmic colour scale (recommended for |B|)
    save_path : if given, save figure to this path
    """
    # Find nearest z index
    z_idx = int(np.argmin(np.abs(grid.zvec - z_slice)))

    if component == "bmag":
        data = np.sqrt(
            grid.bx[:, :, z_idx] ** 2
            + grid.by[:, :, z_idx] ** 2
            + grid.bz[:, :, z_idx] ** 2
        )
        label = "|B| (nT)"
    elif component in ("bx", "by", "bz"):
        field_arr = {"bx": grid.bx, "by": grid.by, "bz": grid.bz}[component]
        data = field_arr[:, :, z_idx]
        label = f"B{component[1]} (nT)"
    else:
        raise ValueError(f"component must be 'bmag', 'bx', 'by', or 'bz', got '{component}'")

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    X, Y = np.meshgrid(grid.xvec, grid.yvec, indexing="ij")
    plot_data = data.copy()
    plot_data[plot_data <= 0] = np.nan

    if log_scale and np.nanmax(plot_data) > 0:
        plot_data = np.log10(plot_data)
        label = f"log₁₀ {label}"

    levels = 40
    cf = ax.contourf(X, Y, plot_data, levels=levels, cmap=cmap)
    plt.colorbar(cf, ax=ax, label=label)

    _add_earth(ax)
    ax.set_xlabel("X GSM (Re)")
    ax.set_ylabel("Y GSM (Re)")
    ax.set_title(f"{component} at Z={grid.zvec[z_idx]:.2f} Re")
    ax.set_aspect("equal")

    if save_path:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax


def plot_equatorial_flux(
    positions: np.ndarray,
    flux_values: np.ndarray,
    energy_kev: float,
    ax: Optional[plt.Axes] = None,
    cmap: str = "hot_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    log_scale: bool = True,
    grid_res: int = 100,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Interpolated differential flux map in the equatorial plane.

    Parameters
    ----------
    positions   : (N, 3) GSM positions in Re (only X,Y used)
    flux_values : (N,) differential or omnidirectional flux
    energy_kev  : label energy in keV
    ax          : existing Axes; created if None
    cmap        : matplotlib colormap name
    vmin, vmax  : colour scale limits (in log units if log_scale=True)
    log_scale   : use log10 of flux values
    grid_res    : number of points in each dimension for interpolation grid
    save_path   : if given, save figure to this path
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    x = positions[:, 0]
    y = positions[:, 1]

    vals = np.asarray(flux_values, dtype=float)
    if log_scale:
        with np.errstate(divide="ignore", invalid="ignore"):
            vals = np.log10(np.where(vals > 0, vals, np.nan))
        flux_label = f"log₁₀ flux  [keV⁻¹ cm⁻² s⁻¹ sr⁻¹]  E={energy_kev:.0f} keV"
    else:
        flux_label = f"flux  [keV⁻¹ cm⁻² s⁻¹ sr⁻¹]  E={energy_kev:.0f} keV"

    # Interpolate scattered points onto a regular grid for a smooth map
    xi = np.linspace(x.min(), x.max(), grid_res)
    yi = np.linspace(y.min(), y.max(), grid_res)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), vals, (Xi, Yi), method="linear")

    cf = ax.contourf(Xi, Yi, Zi, levels=40, cmap=cmap, vmin=vmin, vmax=vmax)
    sc = ax.scatter(x, y, c=vals, cmap=cmap, s=4, vmin=vmin, vmax=vmax, alpha=0.6)
    plt.colorbar(cf, ax=ax, label=flux_label)

    _add_earth(ax)
    ax.set_xlabel("X GSM (Re)")
    ax.set_ylabel("Y GSM (Re)")
    ax.set_title(f"Equatorial Flux  E={energy_kev:.0f} keV")
    ax.set_aspect("equal")

    if save_path:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _add_earth(ax: plt.Axes, radius: float = 1.0) -> None:
    """Draw a filled Earth circle at the origin."""
    earth = Circle((0, 0), radius, color="steelblue", zorder=5)
    ax.add_patch(earth)
    ax.plot(0, 0, "k.", markersize=2, zorder=6)
