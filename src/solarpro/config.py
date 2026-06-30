"""Configuration models for a photovoltaic installation and the degradation engine.

Plant configuration is loaded from a JSON file (workflow stage 1). Degradation
constants default to values documented in the literature and on the SolarPro
demo (https://lozanobosch.k3pler.org/); every constant is overridable so nothing
is hard-coded into the physics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class PlantConfig:
    """Physical description of a photovoltaic installation.

    Attributes:
        name: Human-readable plant identifier.
        latitude: Decimal degrees (north positive).
        longitude: Decimal degrees (east positive).
        tilt: Module tilt from horizontal, degrees.
        azimuth: Module azimuth, degrees (180 = due south in Open-Meteo's convention).
        installed_power_kwc: Installed DC capacity, kWc.
        roof_height_m: Mounting height above ground, used for wind-speed correction.
        temperature_coefficient: Power temperature coefficient, 1/°C (typically negative).
        timezone: IANA timezone string used for weather queries.
    """

    name: str
    latitude: float
    longitude: float
    tilt: float
    azimuth: float
    installed_power_kwc: float
    roof_height_m: float = 10.0
    temperature_coefficient: float = -0.0035
    timezone: str = "auto"

    @classmethod
    def from_json(cls, path: str | Path) -> "PlantConfig":
        """Load a plant configuration from a JSON file (pipeline stage 1)."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


@dataclass
class ModelConfig:
    """Tunable constants for prediction and degradation models.

    Defaults follow the references cited on the SolarPro demo and in
    Jordan & Kurtz (2013); see ``docs/methodology.md`` for the rationale.
    """

    # --- Calibration / validation ---
    training_years: int = 2
    test_fraction: float = 0.2
    r2_threshold: float = 0.85

    # --- Cell temperature (Skoplaki) ---
    noct_celsius: float = 45.0
    module_efficiency_stc: float = 0.18
    transmittance_absorptance: float = 0.90

    # --- Arrhenius (thermal) ---
    activation_energy_ev: float = 0.65  # EVA encapsulant, typical range 0.6–1.1 eV
    reference_temperature_k: float = 298.15  # STC

    # --- Hallberg-Peck (humidity) ---
    humidity_exponent: float = 3.0
    reference_humidity_pct: float = 50.0

    # --- HSU soiling ---
    rain_cleaning_threshold_mm: float = 1.0
    soiling_loss_per_gram: float = 0.0015  # SR loss per g/m^2 accumulated mass

    # --- Long-term degradation ---
    first_year_lid_pct: float = 2.0
    annual_degradation_pct: float = 0.5

    overrides: dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})
