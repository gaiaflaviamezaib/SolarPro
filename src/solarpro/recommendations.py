"""AI-assisted maintenance recommendations via Ollama.

This is the server-side feature of SolarPro (the cloud "Team" tier in the
product). A local Ollama instance runs an open-source LLM that turns the
numerical diagnosis into plain-language maintenance guidance.

The module degrades gracefully: if no Ollama server is reachable, a deterministic
rule-based fallback produces useful recommendations so the pipeline always runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import requests

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3"  # configurable; any local Ollama model works

_SYSTEM_PROMPT = (
    "You are a photovoltaic maintenance advisor. Given a numerical performance "
    "diagnosis, produce a short, prioritized list of concrete maintenance actions. "
    "Be specific and practical. Do not invent data beyond what is provided."
)


@dataclass
class DiagnosisContext:
    """Numerical inputs handed to the LLM (or the fallback)."""

    plant_name: str
    performance_ratio: float
    recoverable_energy_kwh: float
    anomaly_fraction: float
    mean_soiling_ratio: float
    mean_cell_temperature: float

    def to_prompt(self) -> str:
        return (
            f"Plant: {self.plant_name}\n"
            f"Performance ratio (measured/expected): {self.performance_ratio:.2f}\n"
            f"Recoverable energy from detected anomalies: {self.recoverable_energy_kwh:.0f} kWh\n"
            f"Fraction of under-performing hours: {self.anomaly_fraction:.0%}\n"
            f"Mean soiling ratio: {self.mean_soiling_ratio:.2f}\n"
            f"Mean cell temperature: {self.mean_cell_temperature:.1f} C\n"
        )


def _rule_based(ctx: DiagnosisContext) -> str:
    """Deterministic fallback used when Ollama is unavailable."""
    lines: list[str] = []
    if ctx.mean_soiling_ratio < 0.95:
        lines.append(
            "- Schedule panel cleaning: soiling is measurably reducing output "
            "(dust/pollen build-up between rain events)."
        )
    if ctx.performance_ratio < 0.90 or ctx.anomaly_fraction > 0.03:
        recoverable = ""
        if ctx.recoverable_energy_kwh > 0:
            recoverable = f" (~{ctx.recoverable_energy_kwh:.0f} kWh recoverable)"
        lines.append(
            "- Investigate sustained under-performance" + recoverable + ": check for "
            "string faults, inverter clipping, partial shading or failing bypass diodes "
            "during the flagged periods."
        )
    if ctx.mean_cell_temperature > 45:
        lines.append(
            "- Improve thermal management: verify rear ventilation and mounting "
            "clearance to limit Arrhenius-driven ageing."
        )
    if not lines:
        lines.append("- No action required: the system is performing within expected bounds.")
    return "\n".join(lines)


def generate_recommendations(
    ctx: DiagnosisContext,
    model: str = DEFAULT_MODEL,
    url: str = DEFAULT_OLLAMA_URL,
    timeout: int = 60,
) -> str:
    """Return maintenance recommendations, preferring Ollama, falling back to rules."""
    payload = {
        "model": model,
        "prompt": f"{_SYSTEM_PROMPT}\n\n{ctx.to_prompt()}\nRecommendations:",
        "stream": False,
    }
    try:
        response = requests.post(url, data=json.dumps(payload), timeout=timeout)
        response.raise_for_status()
        text = response.json().get("response", "").strip()
        if text:
            return text
    except (requests.RequestException, ValueError):
        pass  # Ollama not running / unreachable — use fallback.
    return _rule_based(ctx)
