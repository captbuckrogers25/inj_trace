"""
PTMRunConfig: configure and write SHIELDS-PTM input files.

Wraps ptm_python.ptm_input.ptm_input_creator.  Does not re-implement
any file format logic — all writes are delegated to ptm_input_creator.create_input_files().

PTM input file layout (per run ID XXXX)
----------------------------------------
ptm_input/ptm_parameters_XXXX.txt  — global simulation parameters
ptm_input/dist_density_XXXX.txt    — spatial distribution of seed particles
ptm_input/dist_velocity_XXXX.txt   — energy / pitch-angle distribution

Usage
-----
    cfg = PTMRunConfig(run_id=1, n_particles=5000, tlo=0, thi=3600, n_snapshots=2)
    cfg.set_electron_defaults()
    cfg.set_spatial_distribution(mode=3, r0=6.6, mltmin=-6, mltmax=6)
    cfg.set_velocity_distribution(mode=3, emin=1, emax=500, nenergy=34, npitch=31)
    cfg.write('run001/ptm_input')
"""

from typing import Any


class PTMRunConfig:
    """High-level wrapper around ptm_input_creator.

    Parameters
    ----------
    run_id          : PTM run identifier (XXXX in filenames)
    n_particles     : number of test particles
    trace_direction : +1 = forward in time, -1 = backward (default; for injection mapping)
    tlo             : start of particle trace in seconds
    thi             : end of particle trace in seconds
    n_snapshots     : number of available field snapshots (maps to ntot and ilast)
    **kwargs        : additional parameters forwarded to ptm_input_creator.set_parameters()
    """

    def __init__(
        self,
        run_id: int = 1,
        n_particles: int = 1000,
        trace_direction: int = -1,
        tlo: float = 0.0,
        thi: float = 3600.0,
        n_snapshots: int = 2,
        **kwargs: Any,
    ) -> None:
        from ptm_python.ptm_input import ptm_input_creator
        self._creator = ptm_input_creator(runid=run_id, idensity=2, ivelocity=3)
        self._creator.set_parameters(
            runid=run_id,
            nparticles=n_particles,
            itrace=trace_direction,
            tlo=tlo,
            thi=thi,
            ntot=n_snapshots,
            ifirst=1,
            ilast=n_snapshots,
            dtin=thi - tlo,   # interval between snapshots (static: equals run duration)
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Convenience presets
    # ------------------------------------------------------------------

    def set_electron_defaults(self) -> None:
        """Set particle as an electron: charge=-1, mass=1, guiding-centre mode."""
        self._creator.set_parameters(
            charge=-1.0,
            mass=1.0,
            iswitch=-1,    # -1 = guiding centre always
            iphase=3,      # gradient method for gyrophase
        )

    # ------------------------------------------------------------------
    # Spatial distribution
    # ------------------------------------------------------------------

    def set_spatial_distribution(self, mode: int, **params: Any) -> None:
        """Set the spatial seeding distribution.

        mode=1 : single point       — params: x0, y0, z0
        mode=2 : random box         — params: xmin, xmax, ymin, ymax, zmin, zmax
        mode=3 : radial ring (MLT)  — params: r0, mltmin, mltmax

        MLT values are in hours (e.g. mltmin=-6, mltmax=6 covers dusk flank).
        """
        self._creator.set_parameters(idens=mode, **params)

    # ------------------------------------------------------------------
    # Velocity (energy/pitch-angle) distribution
    # ------------------------------------------------------------------

    def set_velocity_distribution(self, mode: int, **params: Any) -> None:
        """Set the velocity-space distribution.

        mode=1 : ring (single energy + pitch angle)
                 params: ekev, alpha[, phi]
        mode=2 : bi-Maxwellian
                 params: vtperp, vtpara[, phi]
        mode=3 : uniform flux map (energies × pitch angles)
                 params: nenergy, npitch, emin, emax, pamin, pamax, xsource
                 xsource is the tailward X-GSM boundary where tracing stops (Re)
        mode=4 : user-specified flux map from binary files
                 params: nenergy, npitch[, phi, xsource]
        """
        self._creator.set_parameters(idist=mode, **params)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def write(self, input_dir: str = "ptm_input") -> None:
        """Write the three PTM parameter files to input_dir.

        Creates the directory if it does not exist.
        """
        self._creator.create_input_files(filedir=input_dir, verbose=False)

    def print_settings(self) -> None:
        """Print a summary of all parameter values."""
        self._creator.print_settings()

    # ------------------------------------------------------------------
    # Direct parameter access (escape hatch)
    # ------------------------------------------------------------------

    def set_parameters(self, **kwargs: Any) -> None:
        """Directly pass keyword arguments to ptm_input_creator.set_parameters()."""
        self._creator.set_parameters(**kwargs)
