"""Smoke and unit tests for the SolarPro core.

Run with::

    pip install -e ".[dev]"
    pytest
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from solarpro import metrics, preprocessing
from solarpro.config import ModelConfig, PlantConfig
from solarpro.degradation.arrhenius import arrhenius_acceleration
from solarpro.degradation.hsu_soiling import hsu_soiling_ratio
from solarpro.prediction.calibration import calibrate_pvusa
from solarpro.prediction.cell_temperature import skoplaki_cell_temperature
from solarpro.prediction.pvusa import PVUSACoefficients, pvusa_power

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE = _REPO_ROOT / "data" / "sample"


def test_calibration_recovers_known_coefficients():
    """Calibration on noise-free synthetic data should recover the true coeffs."""
    rng = np.random.default_rng(0)
    gti = rng.uniform(50, 1000, 5000)
    temp = rng.uniform(0, 40, 5000)
    wind = rng.uniform(0, 8, 5000)
    true = PVUSACoefficients(a=0.045, b=-3e-6, c=-1.5e-4, d=5e-4)
    power = pvusa_power(true, gti, temp, wind)

    fitted = calibrate_pvusa(gti, temp, wind, power)
    np.testing.assert_allclose(fitted.as_array(), true.as_array(), rtol=1e-2, atol=1e-6)


def test_metrics_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    result = metrics.evaluate(y, y)
    assert result.r2 == pytest.approx(1.0)
    assert result.rmse == pytest.approx(0.0)
    assert result.is_strong


def test_arrhenius_monotonic_in_temperature():
    """Higher cell temperature -> larger acceleration factor."""
    cool = arrhenius_acceleration(25.0)
    hot = arrhenius_acceleration(60.0)
    assert hot > cool
    assert arrhenius_acceleration(25.0) == pytest.approx(1.0, abs=1e-6)  # at reference


def test_soiling_ratio_bounds_and_rain_cleaning():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    pm = pd.Series(40.0, index=idx)
    rain = pd.Series(0.0, index=idx)
    rain.iloc[5] = 5.0  # heavy rain cleans on day 5
    sr = hsu_soiling_ratio(pm, rain, tilt_deg=30)
    assert (sr <= 1.0).all() and (sr >= 0.80).all()
    assert sr.iloc[5] > sr.iloc[4]  # cleaning event recovers performance


def test_skoplaki_above_ambient_in_sun():
    tc = skoplaki_cell_temperature(temp_air=20.0, gti=800.0, wind_speed=1.0)
    assert tc > 20.0


def test_train_test_split_is_temporal():
    idx = pd.date_range("2024-01-01", periods=100, freq="h")
    frame = pd.DataFrame({"x": range(100)}, index=idx)
    train, test = preprocessing.train_test_split_temporal(frame, 0.2)
    assert train.index.max() < test.index.min()


@pytest.mark.skipif(
    not (_SAMPLE / "weather_power.csv").exists(), reason="sample data not present"
)
def test_pipeline_runs_on_sample_data():
    from solarpro import pipeline

    plant = PlantConfig.from_json(_SAMPLE / "plant_config.json")
    dataset = pipeline.load_sample_dataset()
    # use_ollama=False keeps the test offline and deterministic.
    result = pipeline.run(plant, dataset, ModelConfig(), use_ollama=False)

    assert result.test_metrics.r2 > 0.85  # clears the documented quality bar
    assert result.loss_summary.anomaly_fraction > 0  # the injected fault is detected
    assert isinstance(result.recommendations, str) and result.recommendations
