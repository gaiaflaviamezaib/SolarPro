"""Weather/power preprocessing (part of pipeline stages 2-3).

Steps mirror the SolarPro demo pipeline:

* correct 10 m wind speed to roof height (logarithmic wind profile),
* remove outliers (IQR and Hampel filters),
* keep only meaningful daylight rows (GTI > 50 W/m^2, midday hours),
* split the series temporally into train/test sets (no data leakage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Roughness length for open/suburban terrain (m), logarithmic wind profile.
_ROUGHNESS_LENGTH_M = 0.25
_MEASUREMENT_HEIGHT_M = 10.0


def correct_wind_speed(
    wind_speed: pd.Series,
    roof_height_m: float,
    roughness_length_m: float = _ROUGHNESS_LENGTH_M,
) -> pd.Series:
    """Scale 10 m wind speed to roof height using the log wind profile."""
    if roof_height_m <= 0:
        return wind_speed
    factor = np.log(roof_height_m / roughness_length_m) / np.log(
        _MEASUREMENT_HEIGHT_M / roughness_length_m
    )
    return wind_speed * factor


def remove_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    """Mask values outside [Q1 - k*IQR, Q3 + k*IQR] (returns NaN for outliers)."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return series.where((series >= lower) & (series <= upper))


def hampel_filter(series: pd.Series, window: int = 11, n_sigma: float = 3.0) -> pd.Series:
    """Hampel filter: replace points that deviate > n_sigma MADs from the local median."""
    rolling_median = series.rolling(window, center=True, min_periods=1).median()
    mad = (series - rolling_median).abs().rolling(window, center=True, min_periods=1).median()
    threshold = n_sigma * 1.4826 * mad  # 1.4826 scales MAD to std for normal data
    diff = (series - rolling_median).abs()
    return series.where(diff <= threshold, rolling_median)


def filter_operating_rows(
    frame: pd.DataFrame,
    min_gti: float = 50.0,
    midday_hours: tuple[int, int] = (9, 16),
) -> pd.DataFrame:
    """Keep daylight, midday rows with usable irradiance."""
    hours = frame.index.hour
    mask = (frame["gti"] > min_gti) & (hours >= midday_hours[0]) & (hours <= midday_hours[1])
    return frame.loc[mask]


def train_test_split_temporal(
    frame: pd.DataFrame, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: earliest data trains, most recent data tests."""
    frame = frame.sort_index()
    cutoff = int(len(frame) * (1 - test_fraction))
    return frame.iloc[:cutoff], frame.iloc[cutoff:]


def clean(frame: pd.DataFrame, roof_height_m: float) -> pd.DataFrame:
    """Apply the full cleaning chain and drop residual NaNs."""
    out = frame.copy()
    out["wind_speed"] = correct_wind_speed(out["wind_speed"], roof_height_m)
    for column in ("gti", "temp_air", "wind_speed"):
        if column in out:
            out[column] = hampel_filter(remove_outliers_iqr(out[column]))
    return out.dropna(subset=[c for c in ("gti", "temp_air", "wind_speed") if c in out])
