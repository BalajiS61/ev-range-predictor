# Changelog

## v1.0.0 - 2026-10-17
First complete release.

- **Step 7** - README, technical report, MIT licence, figure export script.
- **Step 6** - Streamlit dashboard (`app/app.py`) and command-line predictor (`scripts/predict_range.py`).
- **Step 5** - ML surrogate: 3,000 physics-labelled synthetic trips, five models compared,
  physics-informed features, leave-one-cycle-family-out extrapolation test, saved predictor.
- **Step 4** - Ambient temperature (air density, cabin HVAC, battery Arrhenius resistance and
  cold capacity), payload, hills, constant speed, battery ageing, real-world scenarios,
  EPA-style label estimate.
- **Step 3** - PMSM + inverter loss map with torque/power envelope, single-speed gearbox, regen
  blending with low-speed fade-out, battery pack ECM (OCV + R0 + RC), closed energy balance,
  full-depletion range test.
- **Step 2** - Drive-cycle library (UDDS, HWFET, ECE-15, EUDC, NEDC, MIDC), EPA downloader,
  cycle statistics, validation against published distances.
- **Step 1** - Vehicle parameters, longitudinal road-load dynamics, energy and range, tests.
