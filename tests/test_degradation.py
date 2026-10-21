"""Tests for the degradation models (Arrhenius, Hallberg-Peck, HSU soiling, TDR).

Expected values are worked out by hand from the formulas in each module's
docstring, with the default parameters (Ea = 0.65 eV, T_ref = 25 C, RH_ref = 50%,
n = 3, LID 2%, 0.5%/year).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solarpro.config import ModelConfig
from solarpro.degradation import (
    arrhenius_acceleration,
    hallberg_peck_acceleration,
    hsu_soiling_ratio,
    transitory_deterioration_rate,
)
from solarpro.degradation.tdr import long_term_factor

# Arrhenius ------------------------------------------------------------------

@pytest.mark.parametrize("temp_c, expected", [(0.0, 0.0987), (60.0, 14.27), (85.0, 69.29)])
def test_arrhenius_known_values(temp_c, expected):
    assert arrhenius_acceleration(temp_c) == pytest.approx(expected, rel=1e-3)


def test_arrhenius_vectorised_and_slower_when_cold():
    af = arrhenius_acceleration([10.0, 25.0, 40.0])
    assert af.shape == (3,)
    assert af[0] < 1.0 < af[2]


# Hallberg-Peck --------------------------------------------------------------

def test_hallberg_peck_is_one_at_reference_conditions():
    assert hallberg_peck_acceleration(50.0, 25.0) == pytest.approx(1.0)


def test_hallberg_peck_humidity_power_law():
    # At the reference temperature only the (RH / 50) ** 3 term is left.
    assert hallberg_peck_acceleration(100.0, 25.0) == pytest.approx(8.0)
    assert hallberg_peck_acceleration(25.0, 25.0) == pytest.approx(0.125)


def test_hallberg_peck_reduces_to_arrhenius_at_reference_humidity():
    temps = np.array([5.0, 45.0, 70.0])
    np.testing.assert_allclose(
        hallberg_peck_acceleration(50.0, temps), arrhenius_acceleration(temps)
    )


def test_hallberg_peck_damp_heat_test():
    # 85 C / 85 % RH, the IEC damp-heat test condition.
    assert hallberg_peck_acceleration(85.0, 85.0) == pytest.approx(340.4, rel=1e-3)


# HSU soiling ----------------------------------------------------------------

def _days(n):
    return pd.date_range("2024-01-01", periods=n, freq="D")


def test_soiling_accumulates_linearly_when_dry():
    idx = _days(5)
    sr = hsu_soiling_ratio(pd.Series(40.0, index=idx), pd.Series(0.0, index=idx), tilt_deg=0)
    # 1e-3 * 40 g/m^2 per day, 0.0015 loss per g/m^2 -> 6e-5 per day.
    np.testing.assert_allclose(sr, 1.0 - 6e-5 * np.arange(1, 6))


def test_rain_at_threshold_cleans():
    idx = _days(3)
    rain = pd.Series([0.0, 1.0, 0.99], index=idx)
    sr = hsu_soiling_ratio(pd.Series(40.0, index=idx), rain, tilt_deg=0)
    assert sr.iloc[1] == 1.0  # exactly 1 mm cleans
    assert sr.iloc[2] < 1.0  # just under the threshold does not


def test_steeper_tilt_soils_less():
    idx = _days(30)
    pm, rain = pd.Series(60.0, index=idx), pd.Series(0.0, index=idx)
    flat = hsu_soiling_ratio(pm, rain, tilt_deg=0)
    steep = hsu_soiling_ratio(pm, rain, tilt_deg=60)
    assert (steep >= flat).all()
    # cos(60) = 0.5, so a 60 degree module loses half as much.
    assert 1 - steep.iloc[-1] == pytest.approx(0.5 * (1 - flat.iloc[-1]))


def test_soiling_has_a_floor():
    idx = _days(10)
    sr = hsu_soiling_ratio(
        pd.Series(1e6, index=idx), pd.Series(0.0, index=idx), tilt_deg=0
    )
    assert sr.min() == pytest.approx(0.80)


def test_missing_rain_days_count_as_dry():
    idx = _days(4)
    rain = pd.Series([5.0], index=idx[:1])  # only the first day has data
    sr = hsu_soiling_ratio(pd.Series(40.0, index=idx), rain, tilt_deg=0)
    assert sr.iloc[0] == 1.0
    assert (np.diff(sr.to_numpy()) < 0).all()


# Long-term ageing and TDR ---------------------------------------------------

def test_long_term_factor_lid_then_linear():
    ages = [0.0, 0.5, 1.0, 3.0]
    expected = [1.0, 1 - 0.01 - 0.0025, 1 - 0.02 - 0.005, 1 - 0.02 - 0.015]
    np.testing.assert_allclose(long_term_factor(ages), expected)


def test_long_term_factor_stress_speeds_up_only_linear_ageing():
    base = long_term_factor(2.0)
    stressed = long_term_factor(2.0, stress_acceleration=3.0)
    assert 1 - stressed == pytest.approx(0.02 + 3 * 0.01)
    assert stressed < base


def test_long_term_factor_never_negative():
    assert long_term_factor(500.0) == 0.0


def test_tdr_combines_soiling_and_ageing():
    hours = pd.date_range("2024-01-01", periods=48, freq="h")
    temp = pd.Series(25.0, index=hours)
    rh = pd.Series(50.0, index=hours)
    daily_soiling = pd.Series([0.95, 0.90], index=_days(2))
    ages = pd.Series(np.linspace(0, 1, 48), index=hours)

    out = transitory_deterioration_rate(temp, rh, daily_soiling, ages, ModelConfig())

    # Reference conditions: no extra stress on ageing.
    np.testing.assert_allclose(out["thermal_af"], 1.0)
    np.testing.assert_allclose(out["humidity_af"], 1.0)
    # Daily soiling is carried over to every hour of that day.
    assert (out["soiling_ratio"].iloc[:24] == 0.95).all()
    assert (out["soiling_ratio"].iloc[24:] == 0.90).all()
    np.testing.assert_allclose(out["long_term_factor"], long_term_factor(ages))
    np.testing.assert_allclose(out["tdr"], out["soiling_ratio"] * out["long_term_factor"])
    assert ((out["tdr"] > 0) & (out["tdr"] <= 1)).all()
