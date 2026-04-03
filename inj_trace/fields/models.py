"""
Thin wrappers around lgmpy magnetic field model functions.

All functions return plain Python lists [Bx, By, Bz] in nanoTesla (GSM),
hiding the lgmpy object/list duality and per-model call quirks.

Imports are deferred (inside each function) so that inj_trace.config.load_config()
has already run before lgmpy is imported.

Available models
----------------
T89   — Tsyganenko 1989; driven by Kp (integer 0–5)
TS04  — Tsyganenko–Sitnov 2004; driven by solar wind and IMF parameters
OP77  — Olsen–Pfitzer 1977; static (no driver parameters)
"""

from datetime import datetime
from typing import List


def eval_t89(
    pos_gsm: List[float],
    time: datetime,
    Kp: int,
    internal_model: str = "LGM_IGRF",
) -> List[float]:
    """Return [Bx, By, Bz] nT at pos_gsm via the T89 model.

    Parameters
    ----------
    pos_gsm : [x, y, z]  in Earth radii (GSM)
    time    : UTC datetime
    Kp      : geomagnetic index, integer 0–5
    internal_model : 'LGM_IGRF' (default), 'LGM_CDIP', or 'LGM_EDIP'
    """
    Kp = int(Kp)
    if Kp < 0 or Kp > 5:
        raise ValueError(f"T89 requires Kp in {{0,...,5}}, got {Kp}")
    from lgmpy import Lgm_T89
    B = Lgm_T89.T89(pos_gsm, time, Kp=Kp, INTERNAL_MODEL=internal_model)
    return list(B)


def eval_ts04(
    pos_gsm: List[float],
    time: datetime,
    P: float,
    Dst: float,
    By: float,
    Bz: float,
    W: List[float],
    internal_model: str = "LGM_IGRF",
) -> List[float]:
    """Return [Bx, By, Bz] nT at pos_gsm via the TS04 model.

    Parameters
    ----------
    pos_gsm : [x, y, z]  in Earth radii (GSM)
    time    : UTC datetime
    P       : solar wind dynamic pressure (nPa)
    Dst     : Dst index (nT)
    By      : IMF By component (nT)
    Bz      : IMF Bz component (nT)
    W       : 6-element list W1–W6 coupling coefficients
    internal_model : 'LGM_IGRF' (default), 'LGM_CDIP', or 'LGM_EDIP'
    """
    from lgmpy import Lgm_TS04
    B = Lgm_TS04.TS04(
        pos_gsm, time,
        P=P, Dst=Dst, By=By, Bz=Bz, W=list(W),
        INTERNAL_MODEL=internal_model,
    )
    return list(B)


def eval_op77(
    pos_gsm: List[float],
    time: datetime,
    internal_model: str = "LGM_IGRF",
) -> List[float]:
    """Return [Bx, By, Bz] nT at pos_gsm via the OP77 static model.

    Parameters
    ----------
    pos_gsm : [x, y, z]  in Earth radii (GSM)
    time    : UTC datetime (used only for IGRF epoch)
    internal_model : 'LGM_IGRF' (default), 'LGM_CDIP', or 'LGM_EDIP'
    """
    from lgmpy import Lgm_OP77
    B = Lgm_OP77.OP77(pos_gsm, time, INTERNAL_MODEL=internal_model)
    return list(B)


# Dispatcher: map model name string → evaluation function
MODELS = {
    "T89":  eval_t89,
    "TS04": eval_ts04,
    "OP77": eval_op77,
}
