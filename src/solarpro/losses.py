"""Performance-loss analysis and anomaly flagging.

The PVUSA model, calibrated on healthy training data, defines the **expected**
production for the observed weather. Comparing it against the **measured**
production isolates losses that weather alone does not explain — candidate
maintenance events.

The TDR factor (thermal / humidity / soiling) is produced separately as a
diagnostic and forecasting signal; it explains *why* losses occur rather than
being subtracted from the baseline (which would double-count effects the
data-driven PVUSA fit already absorbs).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LossSummary:
    expected_energy_kwh: float
    measured_energy_kwh: float
    energy_gap_kwh: float
    recoverable_energy_kwh: float
    performance_ratio: float
    anomaly_fraction: float

    def to_dict(self) -> dict:
        return {
            "expected_energy_kwh": round(self.expected_energy_kwh, 1),
            "measured_energy_kwh": round(self.measured_energy_kwh, 1),
            "energy_gap_kwh": round(self.energy_gap_kwh, 1),
            "recoverable_energy_kwh": round(self.recoverable_energy_kwh, 1),
            "performance_ratio": round(self.performance_ratio, 3),
            "anomaly_fraction": round(self.anomaly_fraction, 3),
        }


def analyze_losses(
    expected_power: pd.Series,
    measured_power: pd.Series,
    anomaly_threshold: float = 0.90,
) -> tuple[pd.DataFrame, LossSummary]:
    """Build a per-timestamp loss frame and an aggregate summary.

    Args:
        expected_power: PVUSA-predicted production (healthy baseline).
        measured_power: Actual measured production.
        anomaly_threshold: Performance ratio below which a point is flagged.
    """
    frame = pd.DataFrame(
        {"expected_power": expected_power, "measured_power": measured_power}
    ).dropna()
    frame["performance_ratio"] = frame["measured_power"] / frame["expected_power"].replace(
        0, np.nan
    )
    frame["anomaly"] = frame["performance_ratio"] < anomaly_threshold
    # Positive shortfall (expected above measured), per timestamp.
    frame["loss_kwh"] = (frame["expected_power"] - frame["measured_power"]).clip(lower=0)

    # Hourly power summed over hourly samples approximates energy in kWh.
    expected_energy = float(frame["expected_power"].sum())
    measured_energy = float(frame["measured_power"].sum())
    recoverable = float(frame.loc[frame["anomaly"], "loss_kwh"].sum())
    summary = LossSummary(
        expected_energy_kwh=expected_energy,
        measured_energy_kwh=measured_energy,
        energy_gap_kwh=expected_energy - measured_energy,
        recoverable_energy_kwh=recoverable,
        performance_ratio=(
            measured_energy / expected_energy if expected_energy else float("nan")
        ),
        anomaly_fraction=float(frame["anomaly"].mean()) if len(frame) else 0.0,
    )
    return frame, summary
