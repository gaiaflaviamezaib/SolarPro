"""Robust calibration of PVUSA coefficients (pipeline stage 3).

Coefficients are fitted with **Huber regression**, which is robust to the
heavy-tailed residuals typical of operating PV data (shading spikes, sensor
glitches). The fit uses *training data only* to avoid leakage; validation is
performed separately on the held-out test set (see ``metrics.py``).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import HuberRegressor

from solarpro.prediction.pvusa import PVUSACoefficients, design_matrix


def calibrate_pvusa(
    gti,
    temp,
    wind,
    power,
    epsilon: float = 1.35,
    max_iter: int = 1000,
) -> PVUSACoefficients:
    """Fit PVUSA coefficients (a, b, c, d) from operating data.

    Args:
        gti: Plane-of-array irradiance, W/m^2.
        temp: Air (or cell) temperature, °C.
        wind: Wind speed, m/s.
        power: Measured power output (same units as predictions).
        epsilon: Huber threshold; smaller is more robust to outliers.
        max_iter: Maximum solver iterations.
    """
    features = design_matrix(gti, temp, wind)
    target = np.asarray(power, dtype=float)

    # The PVUSA features (G, G^2, G*T, G*W) span very different magnitudes, which
    # destabilises the robust solver. Scale each column to unit std, fit, then
    # map the coefficients back to physical units.
    scales = features.std(axis=0)
    scales[scales == 0] = 1.0

    # fit_intercept=False: the intercept is absorbed by the PVUSA structure.
    model = HuberRegressor(epsilon=epsilon, max_iter=max_iter, fit_intercept=False)
    model.fit(features / scales, target)
    a, b, c, d = model.coef_ / scales
    return PVUSACoefficients(a=a, b=b, c=c, d=d)
