"""Thermal degradation via the Arrhenius relation.

The Arrhenius acceleration factor relates a degradation reaction rate at cell
temperature ``T`` to the rate at a reference temperature ``T_ref``::

    AF(T) = exp[ (Ea / k_B) * (1 / T_ref - 1 / T) ]

with ``Ea`` the activation energy (eV) and ``k_B`` Boltzmann's constant.
A factor > 1 means degradation is accelerated relative to STC.

Reference:
    Jordan, D.C. & Kurtz, S.R. (2013). "Photovoltaic Degradation Rates — An
    Analytical Review." Progress in Photovoltaics, 21(1), 12-29.
"""

from __future__ import annotations

import numpy as np

# Boltzmann constant in eV/K.
BOLTZMANN_EV_PER_K = 8.617333262e-5


def arrhenius_acceleration(
    cell_temperature_celsius,
    activation_energy_ev: float = 0.65,
    reference_temperature_k: float = 298.15,
):
    """Arrhenius acceleration factor for one or many cell temperatures (°C)."""
    temp_k = np.asarray(cell_temperature_celsius, dtype=float) + 273.15
    exponent = (activation_energy_ev / BOLTZMANN_EV_PER_K) * (
        1.0 / reference_temperature_k - 1.0 / temp_k
    )
    return np.exp(exponent)
