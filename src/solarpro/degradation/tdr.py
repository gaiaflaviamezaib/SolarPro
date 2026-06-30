"""Transitory Deterioration Rate (TDR).

The TDR aggregates the instantaneous degradation/stress mechanisms into a single
multiplicative performance factor over time:

    TDR(t) = soiling_ratio(t) * long_term_factor(t)

* ``soiling_ratio``     — reversible loss from the HSU model.
* ``long_term_factor``  — irreversible ageing: first-year light-induced
  degradation (LID) plus a constant annual degradation rate. The thermal
  (Arrhenius) and humidity (Hallberg-Peck) acceleration factors modulate the
  *rate* of this irreversible ageing relative to STC.

The result is a time series in (0, 1] that scales clean-condition predictions
down to expected real-world output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from solarpro.config import ModelConfig
from solarpro.degradation.arrhenius import arrhenius_acceleration
from solarpro.degradation.hallberg_peck import hallberg_peck_acceleration


def long_term_factor(
    ages_years,
    stress_acceleration=1.0,
    first_year_lid_pct: float = 2.0,
    annual_degradation_pct: float = 0.5,
):
    """Irreversible performance factor as a function of module age (years)."""
    ages = np.asarray(ages_years, dtype=float)
    lid = first_year_lid_pct / 100.0
    annual = (annual_degradation_pct / 100.0) * stress_acceleration
    # LID is applied progressively over the first year, then linear ageing.
    lid_loss = lid * np.clip(ages, 0.0, 1.0)
    linear_loss = annual * ages
    return np.clip(1.0 - lid_loss - linear_loss, 0.0, 1.0)


def transitory_deterioration_rate(
    cell_temperature_celsius: pd.Series,
    relative_humidity_pct: pd.Series,
    soiling_ratio: pd.Series,
    ages_years: pd.Series,
    config: ModelConfig | None = None,
) -> pd.DataFrame:
    """Combine all mechanisms into a TDR time series.

    Returns a DataFrame with the per-mechanism factors and the combined ``tdr``
    column, indexed like the inputs.
    """
    config = config or ModelConfig()

    thermal_af = arrhenius_acceleration(
        cell_temperature_celsius,
        activation_energy_ev=config.activation_energy_ev,
        reference_temperature_k=config.reference_temperature_k,
    )
    humidity_af = hallberg_peck_acceleration(
        relative_humidity_pct,
        cell_temperature_celsius,
        activation_energy_ev=config.activation_energy_ev,
        humidity_exponent=config.humidity_exponent,
        reference_humidity_pct=config.reference_humidity_pct,
        reference_temperature_k=config.reference_temperature_k,
    )
    # Combined stress accelerates irreversible ageing (geometric mean keeps it bounded).
    stress = np.sqrt(np.asarray(thermal_af) * np.asarray(humidity_af))

    long_term = long_term_factor(
        ages_years,
        stress_acceleration=stress,
        first_year_lid_pct=config.first_year_lid_pct,
        annual_degradation_pct=config.annual_degradation_pct,
    )

    soiling = soiling_ratio.reindex(cell_temperature_celsius.index).ffill().fillna(1.0)
    tdr = soiling.to_numpy() * long_term

    return pd.DataFrame(
        {
            "thermal_af": thermal_af,
            "humidity_af": humidity_af,
            "soiling_ratio": soiling.to_numpy(),
            "long_term_factor": long_term,
            "tdr": tdr,
        },
        index=cell_temperature_celsius.index,
    )
