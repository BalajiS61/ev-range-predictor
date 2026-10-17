"""Drive-cycle statistics (kinematic parameters used to compare cycles)."""

from __future__ import annotations

import numpy as np

from evrange.cycles import DriveCycle

STOP_THRESHOLD_KMH = 0.5   # below this the vehicle is considered stopped
ACCEL_THRESHOLD_MS2 = 0.1  # |a| below this while moving counts as cruising


def cycle_statistics(cycle: DriveCycle) -> dict:
    """Return distance, speeds, idle share, stops, accelerations and PKE."""
    t = cycle.time_s
    v = cycle.speed_ms
    dt = np.diff(t)
    a = np.diff(v) / dt                          # forward-difference acceleration per interval
    v_mid = 0.5 * (v[1:] + v[:-1])

    stop_ms = STOP_THRESHOLD_KMH / 3.6
    stopped = v_mid < stop_ms
    moving = ~stopped
    accelerating = moving & (a > ACCEL_THRESHOLD_MS2)
    decelerating = moving & (a < -ACCEL_THRESHOLD_MS2)
    cruising = moving & ~accelerating & ~decelerating

    duration = float(t[-1] - t[0])
    distance_m = float(np.sum(v_mid * dt))
    moving_time = float(np.sum(dt[moving]))

    is_stopped_sample = v < stop_ms
    stops = int(np.sum(~is_stopped_sample[:-1] & is_stopped_sample[1:]))

    # Positive kinetic energy per unit distance [m/s^2]: sum of positive v^2 increments / distance
    dv2 = np.diff(v**2)
    pke = float(np.sum(dv2[dv2 > 0]) / distance_m) if distance_m > 0 else 0.0

    def share(mask: np.ndarray) -> float:
        return 100.0 * float(np.sum(dt[mask])) / duration if duration > 0 else 0.0

    pos_a = a[accelerating]
    neg_a = a[decelerating]
    return {
        "duration_s": duration,
        "distance_km": distance_m / 1000.0,
        "avg_speed_kmh": 3.6 * distance_m / duration if duration > 0 else 0.0,
        "avg_moving_speed_kmh": 3.6 * distance_m / moving_time if moving_time > 0 else 0.0,
        "max_speed_kmh": float(np.max(v) * 3.6),
        "idle_pct": share(stopped),
        "accel_pct": share(accelerating),
        "cruise_pct": share(cruising),
        "decel_pct": share(decelerating),
        "stops": stops,
        "stops_per_km": stops / (distance_m / 1000.0) if distance_m > 0 else 0.0,
        "max_accel_ms2": float(np.max(a)) if len(a) else 0.0,
        "max_decel_ms2": float(np.min(a)) if len(a) else 0.0,
        "mean_pos_accel_ms2": float(np.mean(pos_a)) if len(pos_a) else 0.0,
        "mean_neg_accel_ms2": float(np.mean(neg_a)) if len(neg_a) else 0.0,
        "pke_ms2": pke,
    }
