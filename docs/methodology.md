# Methodology

This document describes the models behind SolarPro and the parameters used.
Every constant is a documented default and is overridable through
[`ModelConfig`](../src/solarpro/config.py) — nothing is hard-coded into the physics.

## 1. Cell temperature — Skoplaki

Module temperature drives both efficiency and degradation. SolarPro uses the
explicit Skoplaki correlation, which adds a wind-dependent convection term:

$$T_c = T_a + \frac{G}{G_{NOCT}}\,(T_{NOCT}-20)\cdot\frac{h_{free}}{h_{forced,a}+h_{forced,b}\,V_w}\cdot\left(1-\frac{\eta_{STC}}{\tau\alpha}\right)$$

- $G_{NOCT}=800\ \mathrm{W/m^2}$, convection coefficients $9.5 / (5.7 + 3.8\,V_w)$.
- Reference: Skoplaki & Palyvos (2009), *Renewable Energy* 34(1).

## 2. Production — PVUSA

The PVUSA empirical model expresses power as a polynomial in plane-of-array
irradiance with linear temperature and wind corrections:

$$P = G\,(a + b\,G + c\,T + d\,V_w)$$

The four coefficients are fitted from operating data. Because the four feature
columns ($G, G^2, GT, GV_w$) span very different magnitudes, they are scaled to
unit standard deviation before fitting and the coefficients are mapped back to
physical units afterwards.

**Calibration** uses **Huber regression** on the *training split only* to avoid
data leakage; validation is reported on a held-out, chronologically-later test
split. Reference: Dows & Gough (1995).

## 3. Degradation models

The three mechanisms below produce instantaneous stress/loss factors that are
combined into a single **Transitory Deterioration Rate (TDR)**.

### 3a. Thermal — Arrhenius

$$AF(T) = \exp\!\left[\frac{E_a}{k_B}\left(\frac{1}{T_{ref}}-\frac{1}{T}\right)\right]$$

- $E_a \approx 0.65\ \mathrm{eV}$ for EVA encapsulants (typical range 0.6–1.1 eV).
- $T_{ref}=298.15\ \mathrm{K}$ (STC), $k_B = 8.617\times10^{-5}\ \mathrm{eV/K}$.

### 3b. Humidity — Hallberg-Peck

$$AF = \left(\frac{RH}{RH_{ref}}\right)^{n}\exp\!\left[\frac{E_a}{k_B}\left(\frac{1}{T_{ref}}-\frac{1}{T}\right)\right]$$

- Humidity exponent $n \approx 3$; captures corrosion and delamination.
- Reference: Hallberg & Peck (1991), *QREI* 7(3).

### 3c. Soiling — HSU

A daily mass-balance implementation of the Humboldt State University framework:
particulate matter (PM10/PM2.5) accumulates with a tilt-dependent settling
factor, and rainfall above a threshold cleans the surface. The soiling ratio is
floored at a physical minimum.

- Reference: Coello & Boyle (2019), *IEEE J. Photovoltaics* 9(5).

### 3d. Aggregation — TDR

$$\mathrm{TDR}(t) = \mathrm{soiling\_ratio}(t)\times\mathrm{long\_term\_factor}(t)$$

`long_term_factor` combines first-year light-induced degradation (LID, ~2%) and a
constant annual degradation rate (~0.5%/yr); the thermal and humidity
acceleration factors modulate the *rate* of this irreversible ageing relative to
STC. Reference for degradation rates: Jordan & Kurtz (2013).

## 4. Validation metrics

| Metric | Meaning | Quality bar |
|---|---|---|
| R² | Variance explained | **> 0.85** = strong |
| RMSE | Root-mean-square error (kW) | lower is better |
| MAE | Mean absolute error (kW) | lower is better |
| MAPE | Mean absolute percentage error (%) | lower is better |

## 5. Loss & anomaly analysis

The calibrated PVUSA model defines the expected healthy production. The
performance ratio (measured / expected) flags hours below a threshold (default
0.90); the positive shortfall over flagged hours is reported as **recoverable
energy**. TDR is reported alongside as the physical explanation and is used for
the 15-day forward forecast.

## A note on parameters

Defaults follow the cited literature and the figures shown on the
[SolarPro demo](https://lozanobosch.k3pler.org/). They are starting points, not
plant-specific truth — calibrate against real production history for any given
installation.
