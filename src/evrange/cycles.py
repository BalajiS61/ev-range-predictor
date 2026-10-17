"""Drive-cycle container. Standard cycles (UDDS, HWFET, WLTP, MIDC) are added in Step 2."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

KMH_TO_MS = 1.0 / 3.6


def trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integral (works on all NumPy versions)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


@dataclass
class DriveCycle:
    name: str
    time_s: np.ndarray
    speed_ms: np.ndarray
    grade_rad: np.ndarray | None = field(default=None)

    def __post_init__(self) -> None:
        self.time_s = np.asarray(self.time_s, dtype=float)
        self.speed_ms = np.asarray(self.speed_ms, dtype=float)
        if self.grade_rad is None:
            self.grade_rad = np.zeros_like(self.time_s)
        else:
            self.grade_rad = np.asarray(self.grade_rad, dtype=float)

        if self.time_s.ndim != 1 or len(self.time_s) < 2:
            raise ValueError("time_s must be a 1-D array with at least 2 samples")
        if not (len(self.time_s) == len(self.speed_ms) == len(self.grade_rad)):
            raise ValueError("time_s, speed_ms and grade_rad must have equal length")
        if np.any(np.diff(self.time_s) <= 0):
            raise ValueError("time_s must be strictly increasing")
        if np.any(self.speed_ms < 0):
            raise ValueError("speed_ms must be non-negative")

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0])

    @property
    def distance_km(self) -> float:
        return trapezoid(self.speed_ms, self.time_s) / 1000.0


def piecewise_cycle(
    name: str,
    waypoints: list[tuple[float, float]],
    dt: float = 1.0,
) -> DriveCycle:
    """Build a cycle by linear interpolation between (time_s, speed_kmh) waypoints."""
    pts = np.asarray(waypoints, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("waypoints must be a list of (time_s, speed_kmh) pairs")
    n = int(round((pts[-1, 0] - pts[0, 0]) / dt)) + 1
    t = pts[0, 0] + dt * np.arange(n)
    v_kmh = np.interp(t, pts[:, 0], pts[:, 1])
    return DriveCycle(name=name, time_s=t, speed_ms=v_kmh * KMH_TO_MS)
