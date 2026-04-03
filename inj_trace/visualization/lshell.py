"""
L-shell visualization functions.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_lshell_map(
    lstar_values: np.ndarray,
    flux_values: np.ndarray,
    energy_kev: Optional[float] = None,
    pitch_angle_deg: float = 90.0,
    ax: Optional[plt.Axes] = None,
    log_scale: bool = True,
    lstar_range: Optional[tuple] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Plot flux as a function of L*.

    Parameters
    ----------
    lstar_values    : (N,) L* values (NaN for open field lines)
    flux_values     : (N,) corresponding flux values
    energy_kev      : label energy in keV (for title/legend)
    pitch_angle_deg : pitch angle for the label
    ax              : existing Axes; created if None
    log_scale       : log10 y-axis
    lstar_range     : (lmin, lmax) limits on x-axis
    save_path       : if given, save figure to this path

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    lstar = np.asarray(lstar_values, dtype=float)
    flux  = np.asarray(flux_values,  dtype=float)

    # Remove NaN / open field lines
    mask = np.isfinite(lstar) & np.isfinite(flux) & (flux > 0)
    lstar = lstar[mask]
    flux  = flux[mask]

    if lstar.size == 0:
        ax.text(0.5, 0.5, "No closed-field particles", transform=ax.transAxes,
                ha="center", va="center")
        return ax

    # Sort by L* for a line plot
    order = np.argsort(lstar)
    lstar = lstar[order]
    flux  = flux[order]

    label = ""
    if energy_kev is not None:
        label = f"E={energy_kev:.0f} keV, α={pitch_angle_deg:.0f}°"

    ax.plot(lstar, flux, "-o", markersize=3, lw=1.5, label=label)

    if log_scale:
        ax.set_yscale("log")
        ylabel = "Flux  [keV⁻¹ cm⁻² s⁻¹ sr⁻¹]"
    else:
        ylabel = "Flux"

    if lstar_range is not None:
        ax.set_xlim(lstar_range)

    ax.set_xlabel("L*")
    ax.set_ylabel(ylabel)
    title = "Flux vs L*"
    if energy_kev is not None:
        title += f"  (E={energy_kev:.0f} keV, α={pitch_angle_deg:.0f}°)"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend()

    if save_path:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax


def plot_lshell_energy_map(
    energies_kev: np.ndarray,
    lstar_values: np.ndarray,
    flux_matrix: np.ndarray,
    ax: Optional[plt.Axes] = None,
    cmap: str = "hot_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Colour-coded L*–energy spectrogram.

    Parameters
    ----------
    energies_kev : (n_e,) energy array
    lstar_values : (n_l,) L* array
    flux_matrix  : (n_e, n_l) flux array
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    with np.errstate(divide="ignore", invalid="ignore"):
        log_flux = np.log10(np.where(flux_matrix > 0, flux_matrix, np.nan))

    cf = ax.pcolormesh(
        lstar_values, energies_kev, log_flux,
        cmap=cmap, vmin=vmin, vmax=vmax, shading="auto",
    )
    plt.colorbar(cf, ax=ax, label="log₁₀ flux  [keV⁻¹ cm⁻² s⁻¹ sr⁻¹]")
    ax.set_xlabel("L*")
    ax.set_ylabel("Energy (keV)")
    ax.set_yscale("log")
    ax.set_title("L*–Energy Flux Spectrogram")

    if save_path:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax
