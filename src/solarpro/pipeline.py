"""End-to-end SolarPro pipeline (the six-stage workflow).

1. Load plant configuration (JSON).
2. Obtain hourly weather + measured power (sample CSV or live Open-Meteo).
3. Preprocess and calibrate PVUSA coefficients (Huber regression, train only).
4. Predict production across all periods.
5. Validate with train/test metrics (R^2, RMSE, MAE, MAPE).
6. Run degradation + loss analysis and generate maintenance recommendations.

Run as a module for an offline demo on the bundled sample data::

    python -m solarpro
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from solarpro import metrics, preprocessing
from solarpro.config import ModelConfig, PlantConfig
from solarpro.degradation.hsu_soiling import hsu_soiling_ratio
from solarpro.degradation.tdr import transitory_deterioration_rate
from solarpro.losses import analyze_losses
from solarpro.prediction.calibration import calibrate_pvusa
from solarpro.prediction.cell_temperature import skoplaki_cell_temperature
from solarpro.prediction.pvusa import pvusa_power
from solarpro.recommendations import DiagnosisContext, generate_recommendations

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SAMPLE_DIR = _REPO_ROOT / "data" / "sample"


@dataclass
class PipelineResult:
    coefficients: object
    train_metrics: metrics.ValidationMetrics
    test_metrics: metrics.ValidationMetrics
    predictions: pd.DataFrame
    loss_summary: object
    recommendations: str


def load_sample_dataset() -> pd.DataFrame:
    """Load the bundled synthetic weather+power sample (hourly)."""
    frame = pd.read_csv(_SAMPLE_DIR / "weather_power.csv", parse_dates=["time"])
    return frame.set_index("time")


def run(
    plant: PlantConfig,
    dataset: pd.DataFrame,
    model_config: ModelConfig | None = None,
    use_ollama: bool = True,
) -> PipelineResult:
    """Run the full pipeline on an hourly weather+power dataset."""
    model_config = model_config or ModelConfig()

    # Stage 3a: select operating rows first (daylight, GTI > 50, midday), then clean.
    # Order matters: outlier removal on the full series would clip legitimate
    # high-irradiance peaks, so we restrict to operating rows before cleaning.
    operating = preprocessing.filter_operating_rows(dataset)
    operating = preprocessing.clean(operating, plant.roof_height_m)
    train, test = preprocessing.train_test_split_temporal(
        operating, model_config.test_fraction
    )

    # Stage 3b: calibrate PVUSA coefficients on training data only.
    coeffs = calibrate_pvusa(
        train["gti"], train["temp_air"], train["wind_speed"], train["power"]
    )

    # Stage 4: predict across all operating periods.
    operating = operating.copy()
    operating["clean_power"] = pvusa_power(
        coeffs, operating["gti"], operating["temp_air"], operating["wind_speed"]
    )

    # Stage 5: validate on train and held-out test.
    train_pred = pvusa_power(coeffs, train["gti"], train["temp_air"], train["wind_speed"])
    test_pred = pvusa_power(coeffs, test["gti"], test["temp_air"], test["wind_speed"])
    train_metrics = metrics.evaluate(train["power"], train_pred)
    test_metrics = metrics.evaluate(test["power"], test_pred)

    # Stage 6a: degradation (TDR) and soiling.
    cell_temp = pd.Series(
        skoplaki_cell_temperature(
            operating["temp_air"],
            operating["gti"],
            operating["wind_speed"],
            noct_celsius=model_config.noct_celsius,
            module_efficiency_stc=model_config.module_efficiency_stc,
            transmittance_absorptance=model_config.transmittance_absorptance,
        ),
        index=operating.index,
    )
    pm_column = operating["pm10"] if "pm10" in operating else pd.Series(20.0, index=operating.index)
    daily_pm = pm_column.resample("D").mean()
    daily_rain = (
        operating["precipitation"].resample("D").sum()
        if "precipitation" in operating
        else pd.Series(0.0, index=daily_pm.index)
    )
    soiling = hsu_soiling_ratio(
        daily_pm,
        daily_rain,
        plant.tilt,
        rain_cleaning_threshold_mm=model_config.rain_cleaning_threshold_mm,
        soiling_loss_per_gram=model_config.soiling_loss_per_gram,
    )
    ages_years = (operating.index - operating.index[0]).days / 365.25
    tdr = transitory_deterioration_rate(
        cell_temp,
        operating["humidity"],
        soiling,
        pd.Series(ages_years, index=operating.index),
        config=model_config,
    )
    operating["tdr"] = tdr["tdr"]

    # Stage 6b: loss analysis vs. measured (PVUSA baseline; TDR kept as diagnostic).
    loss_frame, loss_summary = analyze_losses(
        operating["clean_power"], operating["power"]
    )

    # Stage 6c: AI maintenance recommendations.
    ctx = DiagnosisContext(
        plant_name=plant.name,
        performance_ratio=loss_summary.performance_ratio,
        recoverable_energy_kwh=loss_summary.recoverable_energy_kwh,
        anomaly_fraction=loss_summary.anomaly_fraction,
        mean_soiling_ratio=float(tdr["soiling_ratio"].mean()),
        mean_cell_temperature=float(cell_temp.mean()),
    )
    recommendations = (
        generate_recommendations(ctx) if use_ollama else _placeholder(ctx)
    )

    return PipelineResult(
        coefficients=coeffs,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        predictions=operating,
        loss_summary=loss_summary,
        recommendations=recommendations,
    )


def _placeholder(ctx: DiagnosisContext) -> str:
    from solarpro.recommendations import _rule_based

    return _rule_based(ctx)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the SolarPro pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=_SAMPLE_DIR / "plant_config.json",
        help="Path to a plant configuration JSON file.",
    )
    parser.add_argument(
        "--no-ollama",
        action="store_true",
        help="Skip the Ollama call and use the rule-based recommendation fallback.",
    )
    args = parser.parse_args(argv)

    plant = PlantConfig.from_json(args.config)
    dataset = load_sample_dataset()
    result = run(plant, dataset, use_ollama=not args.no_ollama)

    print(f"\n=== SolarPro - {plant.name} ===")
    print(f"PVUSA coefficients: {result.coefficients}")
    print(f"Train metrics: {result.train_metrics.to_dict()}")
    print(f"Test  metrics: {result.test_metrics.to_dict()}")
    strong = "PASS" if result.test_metrics.is_strong else "BELOW THRESHOLD"
    print(f"R^2 > 0.85 quality bar: {strong}")
    print(f"Loss summary: {result.loss_summary.to_dict()}")
    print("\nMaintenance recommendations:")
    print(result.recommendations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
