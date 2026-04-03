"""
FluxMapResult: compute differential flux from PTM map file output.

PTM writes map output (ptm_output/map_XXXX.dat) when running in flux-map mode
(idist=3 or 4).  The file lists one line per (energy, pitch angle) particle,
recording where it came from and its initial energy.

ptm_tools.parse_map_file() reads the file and assembles energy × pitch-angle
grids.  ptm_tools.energy_to_flux() converts the adiabatic-invariant–mapped
particle energies into physical differential flux using a source distribution
model (kappa or Maxwell–Jüttner).

Usage
-----
    result = FluxMapResult.from_map_file(
        fnames='run001/ptm_output/map_0001.dat',
        Ec=10.0,   # characteristic energy in keV
        n=0.01,    # source density in cm^-3
        kind='kappa',
        kap=2.5,
    )
    print(result.peak_energy(), 'keV')
    print(result.flux.shape)   # (n_energy, n_pitchangle)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


class FluxMapResult:
    """Differential flux computed from a PTM map run.

    Attributes
    ----------
    energies : (n_e,) ndarray   — observation-point energies in keV
    angles   : (n_pa,) ndarray  — pitch angles in degrees
    flux     : (n_e, n_pa) ndarray  — differential flux  [keV^-1 cm^-2 s^-1 sr^-1]
    omni     : (n_e,) ndarray   — omnidirectional flux   [keV^-1 cm^-2 s^-1]
    position : (3,) ndarray     — source position in GSM Re (from map file header)
    """

    def __init__(
        self,
        energies: np.ndarray,
        angles: np.ndarray,
        flux: np.ndarray,
        omni: np.ndarray,
        position: np.ndarray,
    ) -> None:
        self.energies = energies
        self.angles   = angles
        self.flux     = flux
        self.omni     = omni
        self.position = position

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_map_file(
        cls,
        fnames: Union[str, List[str]],
        Ec: float,
        n: float,
        mc2: float = 511.0,
        kind: str = "kappa",
        kap: float = 2.5,
    ) -> "FluxMapResult":
        """Load flux from one or more PTM map files.

        Parameters
        ----------
        fnames : path (str) or list of paths to ptm_output/map_XXXX.dat
        Ec     : characteristic energy of source distribution in keV
                 (analogous to temperature; see ptm_tools.energy_to_flux)
        n      : source density in cm^-3
        mc2    : rest-mass energy in keV (default 511 = electron)
        kind   : distribution type: 'kappa' or 'maxwell'
        kap    : kappa index (only used when kind='kappa'; default 2.5)

        Returns
        -------
        FluxMapResult
        """
        from ptm_python.ptm_tools import (
            parse_map_file,
            energy_to_flux,
            calculate_omnidirectional_flux,
        )

        if isinstance(fnames, str):
            fnames = [fnames]

        fluxmap = parse_map_file(fnames)
        envec  = fluxmap["energies"]    # final (observation) energies
        pavec  = fluxmap["angles"]      # pitch angles
        ei_arr = fluxmap["init_E"]      # initial (source) energies, shape (n_e, n_pa)
        ef_arr = fluxmap["final_E"]     # final energies (same grid, shape (n_e, n_pa))
        position = fluxmap.attrs.get("position", np.zeros(3))

        # Compute differential flux at each (E, PA) grid point
        diff_flux = energy_to_flux(
            ei_arr, ef_arr, Ec, n,
            mc2=mc2,
            kind=kind,
            kap=kap,
            energyFlux=False,
        )

        # Omnidirectional flux: integrate over pitch angle
        omni = calculate_omnidirectional_flux(pavec, diff_flux, angleDegrees=True)

        return cls(
            energies=envec,
            angles=pavec,
            flux=diff_flux,
            omni=omni,
            position=np.asarray(position, dtype=float),
        )

    @classmethod
    def from_run(
        cls,
        run_id: int,
        run_dir: str,
        Ec: float,
        n: float,
        mc2: float = 511.0,
        kind: str = "kappa",
        kap: float = 2.5,
    ) -> "FluxMapResult":
        """Load flux from a run directory.

        Parameters
        ----------
        run_id  : integer run identifier
        run_dir : directory containing ptm_output/
        Ec, n, mc2, kind, kap : forwarded to from_map_file()
        """
        map_path = Path(run_dir) / "ptm_output" / f"map_{run_id:04d}.dat"
        if not map_path.is_file():
            raise FileNotFoundError(
                f"PTM map file not found: {map_path}\n"
                "Was PTM run with idist=3 or idist=4 (flux map mode)?"
            )
        return cls.from_map_file(str(map_path), Ec=Ec, n=n, mc2=mc2, kind=kind, kap=kap)

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def peak_energy(self) -> float:
        """Return the energy of peak omnidirectional flux (keV)."""
        return float(self.energies[np.argmax(self.omni)])

    def flux_at_energy(self, energy_kev: float) -> np.ndarray:
        """Return the (n_pa,) differential flux slice nearest to energy_kev."""
        idx = np.argmin(np.abs(self.energies - energy_kev))
        return self.flux[idx, :]

    def omni_at_energy(self, energy_kev: float) -> float:
        """Return the omnidirectional flux nearest to energy_kev."""
        idx = np.argmin(np.abs(self.energies - energy_kev))
        return float(self.omni[idx])

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "energies": self.energies,
            "angles":   self.angles,
            "flux":     self.flux,
            "omni":     self.omni,
            "position": self.position,
        }

    def save(self, path: str) -> None:
        """Save to a numpy .npz archive."""
        np.savez(path, **self.to_dict())

    @classmethod
    def load(cls, path: str) -> "FluxMapResult":
        """Load from a numpy .npz archive previously written by save()."""
        d = np.load(path)
        return cls(
            energies=d["energies"],
            angles=d["angles"],
            flux=d["flux"],
            omni=d["omni"],
            position=d["position"],
        )
