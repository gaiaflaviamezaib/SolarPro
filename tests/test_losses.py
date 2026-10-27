"""Tests for the loss analysis (losses.py) and the export helpers (exports.py)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from solarpro.exports import export_csv, export_json
from solarpro.losses import analyze_losses


def _hours(values):
    return pd.Series(values, index=pd.date_range("2024-06-01 08:00", periods=len(values), freq="h"))


def test_loss_summary_by_hand():
    expected = _hours([10.0, 10.0, 10.0, 10.0])
    measured = _hours([10.0, 9.5, 6.0, 12.0])
    frame, summary = analyze_losses(expected, measured)

    np.testing.assert_allclose(frame["performance_ratio"], [1.0, 0.95, 0.6, 1.2])
    assert frame["anomaly"].tolist() == [False, False, True, False]  # only below 0.90
    np.testing.assert_allclose(frame["loss_kwh"], [0.0, 0.5, 4.0, 0.0])  # never negative

    assert summary.expected_energy_kwh == 40.0
    assert summary.measured_energy_kwh == 37.5
    assert summary.energy_gap_kwh == 2.5
    assert summary.recoverable_energy_kwh == 4.0  # only the flagged hour counts
    assert summary.performance_ratio == pytest.approx(37.5 / 40.0)
    assert summary.anomaly_fraction == 0.25


def test_threshold_is_strict():
    _, summary = analyze_losses(_hours([10.0]), _hours([9.0]), anomaly_threshold=0.90)
    assert summary.anomaly_fraction == 0.0
    _, summary = analyze_losses(_hours([10.0]), _hours([9.0]), anomaly_threshold=0.91)
    assert summary.anomaly_fraction == 1.0


def test_zero_expected_power_is_not_an_anomaly():
    # e.g. a dark hour: no ratio, so it cannot be flagged
    frame, summary = analyze_losses(_hours([0.0, 10.0]), _hours([0.0, 5.0]))
    assert np.isnan(frame["performance_ratio"].iloc[0])
    assert frame["anomaly"].tolist() == [False, True]
    assert summary.recoverable_energy_kwh == 5.0


def test_missing_values_are_dropped():
    frame, summary = analyze_losses(_hours([10.0, np.nan, 10.0]), _hours([10.0, 3.0, np.nan]))
    assert len(frame) == 1
    assert summary.expected_energy_kwh == 10.0


def test_empty_input():
    _, summary = analyze_losses(_hours([]), _hours([]))
    assert summary.anomaly_fraction == 0.0
    assert np.isnan(summary.performance_ratio)


def test_to_dict_rounds_for_display():
    _, summary = analyze_losses(_hours([3.0, 3.0, 3.0]), _hours([1.0, 1.0, 1.0]))
    d = summary.to_dict()
    assert d["performance_ratio"] == 0.333
    assert d["anomaly_fraction"] == 1.0
    assert json.dumps(d)  # plain floats, safe to export


def test_exports_create_folders_and_round_trip(tmp_path):
    frame = pd.DataFrame({"a": [1.0, 2.0]}, index=pd.Index(["x", "y"], name="key"))
    csv_path = export_csv(frame, tmp_path / "out" / "frame.csv")
    pd.testing.assert_frame_equal(pd.read_csv(csv_path, index_col="key"), frame)

    json_path = export_json({"r2": 0.96, "ok": True}, tmp_path / "nested" / "dir" / "m.json")
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"r2": 0.96, "ok": True}
