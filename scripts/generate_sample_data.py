"""Regenerate the synthetic sample dataset in ``data/sample/``.

The data is deliberately synthetic and reproducible (fixed seed): weather follows
seasonal/diurnal patterns and power is produced from a known PVUSA relation plus
soiling, ageing, an injected ~5-week string-fault anomaly and measurement noise.
This lets the demo, tests and browser app run end-to-end with no real data.

Usage::

    python scripts/generate_sample_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "sample"

# Known "true" PVUSA coefficients used to synthesize healthy production (kW).
TRUE_COEFFS = (0.045, -3.0e-6, -1.5e-4, 5.0e-4)


def generate() -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    idx = pd.date_range("2023-01-01", periods=24 * 365 * 2, freq="h")
    n = len(idx)
    hour = idx.hour.to_numpy()
    doy = idx.dayofyear.to_numpy()

    daylight = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None)
    seasonal = 0.65 + 0.35 * np.cos((doy - 172) / 365 * 2 * np.pi)
    gti = np.clip(1000 * daylight * seasonal * rng.uniform(0.35, 1.0, n), 0, None)

    temp_air = (
        12
        + 10 * np.cos((doy - 200) / 365 * 2 * np.pi)
        + 5 * np.sin((hour - 9) / 24 * 2 * np.pi)
        + rng.normal(0, 1.5, n)
    )
    wind_speed = np.clip(rng.gamma(2.0, 1.4, n), 0.2, None)
    humidity = np.clip(85 - 1.2 * (temp_air - 12) + rng.normal(0, 6, n), 20, 100)
    precipitation = np.where(rng.random(n) < 0.04, rng.gamma(2.0, 1.5, n), 0.0)
    pm10 = np.clip(18 + 8 * np.sin((doy - 30) / 365 * 2 * np.pi) + rng.normal(0, 4, n), 2, None)

    a, b, c, d = TRUE_COEFFS
    clean = np.clip(gti * (a + b * gti + c * temp_air + d * wind_speed), 0, None)

    age_years = (idx - idx[0]).days / 365.25
    ageing = 1 - 0.02 * np.clip(age_years, 0, 1) - 0.005 * age_years

    soiling = np.ones(n)
    acc = 0.0
    for i in range(n):
        if precipitation[i] >= 1.0:
            acc = 0.0
        else:
            acc += 1e-3 * pm10[i] * np.cos(np.radians(30))
        soiling[i] = max(1 - 0.0015 * acc, 0.85)

    fault = np.ones(n)
    fault[(idx >= "2024-06-01") & (idx < "2024-07-07")] = 0.82  # injected string fault

    power = np.clip(clean * ageing * soiling * fault * rng.normal(1.0, 0.03, n), 0, None)

    df = pd.DataFrame(
        {
            "time": idx,
            "gti": gti.round(1),
            "temp_air": temp_air.round(2),
            "wind_speed": wind_speed.round(2),
            "humidity": humidity.round(1),
            "precipitation": precipitation.round(2),
            "pm10": pm10.round(1),
            "power": power.round(3),
        }
    )
    df.to_csv(OUT / "weather_power.csv", index=False)

    plant = {
        "name": "Eiffage Demo Plant (50 kWc)",
        "latitude": 48.8566,
        "longitude": 2.3522,
        "tilt": 30.0,
        "azimuth": 180.0,
        "installed_power_kwc": 50.0,
        "roof_height_m": 12.0,
        "temperature_coefficient": -0.0035,
        "timezone": "Europe/Paris",
    }
    (OUT / "plant_config.json").write_text(json.dumps(plant, indent=2), encoding="utf-8")
    return df


if __name__ == "__main__":
    frame = generate()
    print(f"Wrote {len(frame)} rows to {OUT / 'weather_power.csv'}")
