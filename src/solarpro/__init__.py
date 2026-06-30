"""SolarPro — decision support for photovoltaic performance.

A modular Python core that predicts photovoltaic production (PVUSA), models
multi-physics degradation (Arrhenius / Hallberg-Peck / HSU soiling), diagnoses
performance losses, and generates maintenance recommendations.

The core is dependency-light and Pyodide-compatible so the exact same code runs
both as an importable library (notebooks, tests, servers) and client-side in the
browser via PyScript / WebAssembly.
"""

from solarpro.config import ModelConfig, PlantConfig

__version__ = "0.1.0"

__all__ = ["PlantConfig", "ModelConfig", "__version__"]
