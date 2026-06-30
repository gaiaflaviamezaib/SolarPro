"""Soiling losses via a simplified HSU (Humboldt State University) model.

The HSU model estimates the soiling ratio from airborne particulate matter
(PM2.5 and PM10) deposition, tilt-dependent settling, and rain-based cleaning.
This is a representative daily mass-balance implementation of that framework:

* particulate mass accumulates each day proportionally to PM concentration and a
  tilt factor (steeper modules accumulate less),
* rainfall above a threshold cleans the surface (mass reset),
* the soiling ratio is 1 minus a linear loss in accumulated mass, floored at a
  physical minimum.

Reference:
    Coello, M. & Boyle, L. (2019). "Simple Model for Predicting Time Series
    Soiling of Photovoltaic Panels." IEEE Journal of Photovoltaics, 9(5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Indicative dry-deposition contribution per unit PM concentration (g/m^2 per
# (ug/m^3) per day). Representative default; override via ModelConfig.
_DEPOSITION_COEFF = 1.0e-3
_MIN_SOILING_RATIO = 0.80


def hsu_soiling_ratio(
    daily_pm: pd.Series,
    daily_rain_mm: pd.Series,
    tilt_deg: float,
    rain_cleaning_threshold_mm: float = 1.0,
    soiling_loss_per_gram: float = 0.0015,
    deposition_coeff: float = _DEPOSITION_COEFF,
) -> pd.Series:
    """Compute a daily soiling ratio in (0, 1] (1 = perfectly clean).

    Args:
        daily_pm: Daily mean particulate concentration (ug/m^3), e.g. PM10.
        daily_rain_mm: Daily precipitation total (mm), aligned to ``daily_pm``.
        tilt_deg: Module tilt; steeper tilt reduces deposition (cos factor).
        rain_cleaning_threshold_mm: Rain at/above this resets accumulated mass.
        soiling_loss_per_gram: Soiling-ratio loss per g/m^2 accumulated.
        deposition_coeff: Deposition contribution per PM unit per day.
    """
    tilt_factor = np.cos(np.radians(tilt_deg))
    accumulated = 0.0
    ratios = []
    rain = daily_rain_mm.reindex(daily_pm.index).fillna(0.0)
    for day, pm in daily_pm.items():
        if rain.loc[day] >= rain_cleaning_threshold_mm:
            accumulated = 0.0
        else:
            accumulated += deposition_coeff * float(pm) * tilt_factor
        ratio = max(1.0 - soiling_loss_per_gram * accumulated, _MIN_SOILING_RATIO)
        ratios.append(ratio)
    return pd.Series(ratios, index=daily_pm.index, name="soiling_ratio")
