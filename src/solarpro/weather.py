"""Open-Meteo weather retrieval (pipeline stage 2).

Two endpoints are used:

* **Archive API** (``archive-api.open-meteo.com``) for multi-year hourly history
  used to calibrate and validate the model.
* **Forecast API** (``api.open-meteo.com``) for the 15-day forward forecast.

Hourly variables: global tilted irradiance, air temperature, wind speed,
relative humidity and precipitation.

In the browser build, ``requests`` is transparently routed through
``pyodide-http``; no code changes are required.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import requests

from solarpro.config import PlantConfig

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo hourly variable names mapped to the column names used internally.
_HOURLY_VARS = {
    "global_tilted_irradiance": "gti",
    "temperature_2m": "temp_air",
    "wind_speed_10m": "wind_speed",
    "relative_humidity_2m": "humidity",
    "precipitation": "precipitation",
}

_REQUEST_TIMEOUT = 60


def _hourly_frame(payload: dict) -> pd.DataFrame:
    """Convert an Open-Meteo ``hourly`` payload into a tidy DataFrame."""
    hourly = payload["hourly"]
    frame = pd.DataFrame({"time": pd.to_datetime(hourly["time"])})
    for api_name, column in _HOURLY_VARS.items():
        if api_name in hourly:
            frame[column] = hourly[api_name]
    return frame.set_index("time")


def _common_params(plant: PlantConfig) -> dict:
    return {
        "latitude": plant.latitude,
        "longitude": plant.longitude,
        "hourly": ",".join(_HOURLY_VARS),
        "tilt": plant.tilt,
        "azimuth": plant.azimuth,
        "wind_speed_unit": "ms",
        "timezone": plant.timezone,
    }


def fetch_history(
    plant: PlantConfig,
    start: date,
    end: date,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch hourly historical weather between ``start`` and ``end`` (inclusive)."""
    params = _common_params(plant)
    params.update({"start_date": start.isoformat(), "end_date": end.isoformat()})
    http = session or requests
    response = http.get(ARCHIVE_URL, params=params, timeout=_REQUEST_TIMEOUT)
    response.raise_for_status()
    return _hourly_frame(response.json())


def fetch_history_years(plant: PlantConfig, years: int, session=None) -> pd.DataFrame:
    """Convenience wrapper: fetch the last ``years`` of hourly history."""
    end = date.today()
    start = end - timedelta(days=365 * years)
    return fetch_history(plant, start, end, session=session)


def fetch_forecast(plant: PlantConfig, days: int = 15, session=None) -> pd.DataFrame:
    """Fetch the hourly forecast for the next ``days`` days (max 16 on Open-Meteo)."""
    params = _common_params(plant)
    params["forecast_days"] = min(days, 16)
    http = session or requests
    response = http.get(FORECAST_URL, params=params, timeout=_REQUEST_TIMEOUT)
    response.raise_for_status()
    return _hourly_frame(response.json())
