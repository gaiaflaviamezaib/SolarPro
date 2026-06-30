"""Humidity-driven degradation via the Hallberg-Peck model.

The Hallberg-Peck (a.k.a. temperature-humidity) acceleration factor combines a
relative-humidity power law with an Arrhenius temperature term::

    AF = (RH / RH_ref) ** n * exp[ (Ea / k_B) * (1 / T_ref - 1 / T) ]

It captures moisture-induced corrosion and delamination.

Reference:
    Hallberg, O. & Peck, D.S. (1991). "Recent Humidity Accelerations, a Base for
    Testing Standards." Quality and Reliability Engineering International, 7(3).
"""

from __future__ import annotations

import numpy as np

from solarpro.degradation.arrhenius import BOLTZMANN_EV_PER_K


def hallberg_peck_acceleration(
    relative_humidity_pct,
    cell_temperature_celsius,
    activation_energy_ev: float = 0.65,
    humidity_exponent: float = 3.0,
    reference_humidity_pct: float = 50.0,
    reference_temperature_k: float = 298.15,
):
    """Hallberg-Peck acceleration factor (dimensionless, > 1 means accelerated)."""
    rh = np.asarray(relative_humidity_pct, dtype=float)
    temp_k = np.asarray(cell_temperature_celsius, dtype=float) + 273.15
    humidity_term = (rh / reference_humidity_pct) ** humidity_exponent
    temperature_term = np.exp(
        (activation_energy_ev / BOLTZMANN_EV_PER_K)
        * (1.0 / reference_temperature_k - 1.0 / temp_k)
    )
    return humidity_term * temperature_term
