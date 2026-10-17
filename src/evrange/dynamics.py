"""Longitudinal vehicle dynamics and battery energy calculation.

Road-load forces:
    F_aero    = 0.5 * rho * Cd * A * v^2
    F_roll    = m * g * Crr * cos(theta)     (only while moving)
    F_grade   = m * g * sin(theta)
    F_inertia = m_eff * dv/dt
Wheel power:  P_wheel = (F_aero + F_roll + F_grade + F_inertia) * v
Battery power:
    traction (P_wheel > 0): P_batt = P_wheel / eta_drivetrain + P_aux
    braking  (P_wheel < 0): P_batt = P_wheel * regen_fraction * eta_regen + P_aux
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from evrange.cycles import DriveCycle, trapezoid
from evrange.vehicle import GRAVITY, Vehicle

MOVING_THRESHOLD_MS = 0.01


def _cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y, dtype=float)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def simulate(vehicle: Vehicle, cycle: DriveCycle) -> pd.DataFrame:
    """Run the backward-facing road-load simulation over a drive cycle."""
    t = cycle.time_s
    v = cycle.speed_ms
    theta = cycle.grade_rad
    a = np.gradient(v, t)

    moving = v > MOVING_THRESHOLD_MS
    f_aero = 0.5 * vehicle.air_density * vehicle.drag_coefficient * vehicle.frontal_area_m2 * v**2
    f_roll = np.where(
        moving, vehicle.total_mass_kg * GRAVITY * vehicle.rolling_resistance * np.cos(theta), 0.0
    )
    f_grade = vehicle.total_mass_kg * GRAVITY * np.sin(theta)
    f_inertia = vehicle.effective_mass_kg * a
    f_total = f_aero + f_roll + f_grade + f_inertia

    p_wheel = f_total * v
    p_traction = np.where(p_wheel > 0, p_wheel / vehicle.drivetrain_efficiency, 0.0)
    p_regen = np.where(
        p_wheel < 0, p_wheel * vehicle.regen_fraction * vehicle.regen_efficiency, 0.0
    )
    p_friction_brake = np.where(p_wheel < 0, p_wheel * (1.0 - vehicle.regen_fraction), 0.0)
    p_batt = p_traction + p_regen + vehicle.aux_power_w

    return pd.DataFrame(
        {
            "time_s": t,
            "speed_ms": v,
            "speed_kmh": v * 3.6,
            "accel_ms2": a,
            "grade_rad": theta,
            "f_aero_n": f_aero,
            "f_roll_n": f_roll,
            "f_grade_n": f_grade,
            "f_inertia_n": f_inertia,
            "f_total_n": f_total,
            "p_wheel_w": p_wheel,
            "p_traction_w": p_traction,
            "p_regen_w": p_regen,
            "p_friction_brake_w": p_friction_brake,
            "p_batt_w": p_batt,
            "energy_batt_wh": _cumulative_trapezoid(p_batt, t) / 3600.0,
            "distance_km": _cumulative_trapezoid(v, t) / 1000.0,
        }
    )


def summarize(result: pd.DataFrame, vehicle: Vehicle) -> dict:
    """Key energy metrics and the range implied by the cycle's average consumption."""
    t = result["time_s"].to_numpy()
    duration_s = float(t[-1] - t[0])
    distance_km = float(result["distance_km"].iloc[-1])
    energy_kwh = float(result["energy_batt_wh"].iloc[-1]) / 1000.0
    traction_kwh = trapezoid(result["p_traction_w"].to_numpy(), t) / 3.6e6
    regen_kwh = -trapezoid(result["p_regen_w"].to_numpy(), t) / 3.6e6
    aux_kwh = vehicle.aux_power_w * duration_s / 3.6e6

    if distance_km > 0:
        consumption = energy_kwh * 1000.0 / distance_km
        range_km = vehicle.battery_usable_kwh * 1000.0 / consumption if consumption > 0 else float("inf")
        avg_speed = distance_km / (duration_s / 3600.0)
    else:
        consumption = range_km = avg_speed = float("nan")

    return {
        "distance_km": distance_km,
        "duration_s": duration_s,
        "avg_speed_kmh": avg_speed,
        "max_speed_kmh": float(result["speed_kmh"].max()),
        "energy_kwh": energy_kwh,
        "traction_kwh": traction_kwh,
        "regen_kwh": regen_kwh,
        "aux_kwh": aux_kwh,
        "regen_share_pct": 100.0 * regen_kwh / traction_kwh if traction_kwh > 0 else 0.0,
        "consumption_wh_per_km": consumption,
        "range_km": range_km,
    }
