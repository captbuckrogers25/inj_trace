"""
Time-series flux visualization.
"""

from datetime import datetime
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


def plot_flux_timeseries(
    times: Union[np.ndarray, List],
    flux_vs_time: np.ndarray,
    energies_kev: Optional[np.ndarray] = None,
    energy_indices: Optional[List[int]] = None,
    ax: Optional[plt.Axes] = None,
    log_scale: bool = True,
    cmap: str = "plasma",
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Plot omnidirectional or energy-resolved flux time series.

    Parameters
    ----------
    times         : (N_times,) array of seconds since epoch or datetime objects
    flux_vs_time  : (N_times,) omnidirectional flux OR (N_times, n_energy) matrix
    energies_kev  : (n_energy,) energy array (required when flux_vs_time is 2D)
    energy_indices: subset of energy indices to plot (default: all, up to 8)
    ax            : existing Axes; created if None
    log_scale     : log10 y-axis
    cmap          : colormap for multi-energy lines
    save_path     : if given, save figure to this path

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    times_arr = np.asarray(times)
    use_datetimes = isinstance(times_arr.flat[0], datetime)
    flux_arr = np.asarray(flux_vs_time, dtype=float)

    if flux_arr.ndim == 1:
        # Single omnidirectional trace
        ax.plot(times_arr, flux_arr, lw=1.5, color="steelblue", label="Omni")
    else:
        # Multi-energy traces
        n_e = flux_arr.shape[1]
        if energy_indices is None:
            # Pick up to 8 evenly spaced energies
            step = max(1, n_e // 8)
            energy_indices = list(range(0, n_e, step))

        colors = plt.get_cmap(cmap)(np.linspace(0.1, 0.9, len(energy_indices)))
        for k, ei in enumerate(energy_indices):
            label = (
                f"{energies_kev[ei]:.0f} keV" if energies_kev is not None else f"E[{ei}]"
            )
            ax.plot(times_arr, flux_arr[:, ei], lw=1.5, color=colors[k], label=label)
        ax.legend(fontsize=8, ncol=2, loc="upper right")

    if log_scale:
        ax.set_yscale("log")
        ylabel = "Flux  [keV⁻¹ cm⁻² s⁻¹ sr⁻¹]"
    else:
        ylabel = "Flux"

    if use_datetimes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.figure.autofmt_xdate()
        ax.set_xlabel("Time (UT)")
    else:
        ax.set_xlabel("Time (s)")

    ax.set_ylabel(ylabel)
    ax.set_title("Flux Time Series")
    ax.grid(True, alpha=0.3)

    if save_path:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax
