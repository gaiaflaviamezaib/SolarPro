"""Validation metrics for predicted vs. measured production (pipeline stage 5)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class ValidationMetrics:
    r2: float
    rmse: float
    mae: float
    mape: float

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_strong(self) -> bool:
        """Whether the fit clears the SolarPro R^2 > 0.85 quality bar."""
        return self.r2 > 0.85


def r2_score(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot else float("nan")


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred, eps: float = 1e-6) -> float:
    """Mean absolute percentage error (%), guarding against division by zero."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.clip(np.abs(y_true), eps, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def evaluate(y_true, y_pred) -> ValidationMetrics:
    return ValidationMetrics(
        r2=r2_score(y_true, y_pred),
        rmse=rmse(y_true, y_pred),
        mae=mae(y_true, y_pred),
        mape=mape(y_true, y_pred),
    )
