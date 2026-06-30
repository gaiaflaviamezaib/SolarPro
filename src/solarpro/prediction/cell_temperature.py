"""Cell-temperature estimation using the Skoplaki model.

Reference:
    Skoplaki, E. & Palyvos, J.A. (2009). "Operating temperature of photovoltaic
    modules: A survey of pertinent correlations." Renewable Energy, 34(1), 23-29.

The explicit Skoplaki correlation expresses cell temperature as a function of
ambient temperature, plane-of-array irradiance and wind speed, the latter
through the ratio of free to forced convection heat-transfer coefficients.
"""

from __future__ import annotations

import numpy as np

# Reference irradiance for the NOCT definition (W/m^2).
_G_NOCT = 800.0
# Free/forced convection coefficient parameters from Skoplaki & Palyvos (2009).
_H_FREE = 9.5
_H_FORCED_A = 5.7
_H_FORCED_B = 3.8


def skoplaki_cell_temperature(
    temp_air,
    gti,
    wind_speed,
    noct_celsius: float = 45.0,
    module_efficiency_stc: float = 0.18,
    transmittance_absorptance: float = 0.90,
):
    """Estimate module/cell temperature (°C).

    Args:
        temp_air: Ambient air temperature, °C.
        gti: Global tilted (plane-of-array) irradiance, W/m^2.
        wind_speed: Wind speed at module height, m/s.
        noct_celsius: Nominal operating cell temperature, °C.
        module_efficiency_stc: Module efficiency at STC (fraction).
        transmittance_absorptance: Optical transmittance-absorptance product.
    """
    temp_air = np.asarray(temp_air, dtype=float)
    gti = np.asarray(gti, dtype=float)
    wind_speed = np.asarray(wind_speed, dtype=float)

    wind_term = _H_FREE / (_H_FORCED_A + _H_FORCED_B * wind_speed)
    efficiency_term = 1.0 - module_efficiency_stc / transmittance_absorptance
    return temp_air + (gti / _G_NOCT) * (noct_celsius - 20.0) * wind_term * efficiency_term
