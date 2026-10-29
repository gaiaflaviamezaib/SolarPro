"""Tests for preprocessing.py: wind correction, outlier filters, operating rows, split."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solarpro import preprocessing as pp


def _hourly(values, start="2024-06-01 00:00"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="h"), dtype=float)


# Wind -----------------------------------------------------------------------

def test_wind_unchanged_at_measurement_height():
    wind = _hourly([1.0, 4.0])
    pd.testing.assert_series_equal(pp.correct_wind_speed(wind, 10.0), wind)


def test_wind_log_profile_value():
    # ln(20 / 0.25) / ln(10 / 0.25) = ln(80) / ln(40)
    out = pp.correct_wind_speed(_hourly([5.0]), 20.0)
    assert out.iloc[0] == pytest.approx(5.0 * np.log(80) / np.log(40))
    assert pp.correct_wind_speed(_hourly([5.0]), 5.0).iloc[0] < 5.0  # lower roof, less wind


def test_wind_left_alone_without_roof_height():
    wind = _hourly([3.0])
    assert pp.correct_wind_speed(wind, 0.0) is wind


# Outliers -------------------------------------------------------------------

def test_iqr_masks_only_the_outlier():
    s = _hourly([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
    out = pp.remove_outliers_iqr(s)
    assert np.isnan(out.iloc[-1])
    assert out.iloc[:-1].tolist() == list(range(1, 10))


def test_iqr_k_controls_how_strict_it_is():
    s = _hourly([1, 2, 3, 4, 5, 6, 7, 8, 9, 20])
    assert np.isnan(pp.remove_outliers_iqr(s, k=1.0).iloc[-1])
    assert pp.remove_outliers_iqr(s, k=3.0).iloc[-1] == 20


def test_hampel_replaces_a_spike_with_the_local_median():
    s = _hourly([10, 11, 10, 12, 500, 11, 10, 12, 11, 10, 11])
    out = pp.hampel_filter(s, window=5)
    assert out.iloc[4] == pytest.approx(s.iloc[2:7].median())
    pd.testing.assert_series_equal(out.drop(out.index[4]), s.drop(s.index[4]))


def test_hampel_keeps_a_smooth_signal():
    s = _hourly(np.sin(np.linspace(0, 3, 24)) * 100)
    pd.testing.assert_series_equal(pp.hampel_filter(s), s)


def test_hampel_fills_gaps_left_by_iqr():
    # Points masked by the IQR step come back as the local median, so clean()
    # replaces outliers rather than dropping their rows.
    s = _hourly([10, 11, 10, np.nan, 11, 10, 12])
    out = pp.hampel_filter(s, window=5)
    assert out.iloc[3] == pytest.approx(10.5)
    assert out.notna().all()


# Operating rows and split ---------------------------------------------------

def test_operating_rows_keep_daylight_midday_only():
    frame = pd.DataFrame({"gti": _hourly([100.0] * 24)})
    frame.loc[frame.index[12], "gti"] = 50.0  # exactly at the threshold: dropped
    kept = pp.filter_operating_rows(frame)
    assert list(kept.index.hour) == [9, 10, 11, 13, 14, 15, 16]


def test_split_sorts_before_cutting():
    frame = pd.DataFrame({"x": range(10)}, index=pd.date_range("2024-01-01", periods=10, freq="D"))
    train, test = pp.train_test_split_temporal(frame.sample(frac=1, random_state=0), 0.3)
    assert len(train) == 7 and len(test) == 3
    assert train.index.max() < test.index.min()


# Full cleaning chain --------------------------------------------------------

def test_clean_replaces_spikes_and_corrects_wind():
    n = 24
    frame = pd.DataFrame({
        "gti": _hourly(np.full(n, 600.0) + np.arange(n)),
        "temp_air": _hourly(np.full(n, 20.0)),
        "wind_speed": _hourly(np.full(n, 3.0)),
    })
    frame.loc[frame.index[10], "gti"] = 10_000.0  # sensor spike

    out = pp.clean(frame, roof_height_m=20.0)
    assert len(out) == n
    assert out["gti"].max() < 700
    np.testing.assert_allclose(out["wind_speed"], 3.0 * np.log(80) / np.log(40))
    assert frame.loc[frame.index[10], "gti"] == 10_000.0  # input not modified


def test_clean_keeps_sunny_hours_despite_night_zeros():
    # Clear-sky days: half the hours are 0 at night. Before the fix the IQR
    # range was computed on all hours and clipped every bright midday value.
    hours = pd.date_range("2024-06-01", periods=24 * 10, freq="h")
    sun = np.clip(np.sin((hours.hour - 6) / 12 * np.pi), 0, None) * 900
    frame = pd.DataFrame({"gti": sun, "temp_air": 20.0, "wind_speed": 3.0}, index=hours)

    out = pp.clean(frame, roof_height_m=10.0)
    np.testing.assert_allclose(out["gti"], frame["gti"])
