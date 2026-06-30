"""PVUSA empirical production model.

Reference:
    Dows, R.N. & Gough, E.J. (1995). PVUSA Procurement, Acceptance, and Rating
    Practices for Photovoltaic Power Plants. (Photovoltaics for Utility Scale
    Applications.)

The PVUSA model predicts AC/DC power as a polynomial in plane-of-array
irradiance ``G`` with linear corrections for ambient temperature ``T`` and
wind speed ``W``::

    P = G * (a + b * G + c * T + d * W)

The four coefficients (a, b, c, d) are fitted from operating data
(see ``calibration.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PVUSACoefficients:
    """Fitted PVUSA coefficients."""

    a: float
    b: float
    c: float
    d: float

    def as_array(self) -> np.ndarray:
        return np.array([self.a, self.b, self.c, self.d], dtype=float)


def design_matrix(gti, temp, wind) -> np.ndarray:
    """Build the PVUSA feature matrix [G, G^2, G*T, G*W] (P is linear in coeffs)."""
    gti = np.asarray(gti, dtype=float)
    temp = np.asarray(temp, dtype=float)
    wind = np.asarray(wind, dtype=float)
    return np.column_stack([gti, gti**2, gti * temp, gti * wind])


def pvusa_power(coeffs: PVUSACoefficients, gti, temp, wind) -> np.ndarray:
    """Predict power for given weather using fitted PVUSA coefficients."""
    return design_matrix(gti, temp, wind) @ coeffs.as_array()
