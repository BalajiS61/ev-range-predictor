"""Drive-cycle container, file loading, concatenation and resampling."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

KMH_TO_MS = 1.0 / 3.6
MPH_TO_MS = 0.44704
UNIT_FACTORS = {"kmh": KMH_TO_MS, "mph": MPH_TO_MS, "ms": 1.0}


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

    @property
    def speed_kmh(self) -> np.ndarray:
        return self.speed_ms * 3.6

    def resample(self, dt: float) -> "DriveCycle":
        """Linear resampling onto a uniform time grid with step dt [s]."""
        if dt <= 0:
            raise ValueError("dt must be > 0")
        n = int(np.floor(self.duration_s / dt + 1e-9)) + 1
        t = self.time_s[0] + dt * np.arange(n)
        return DriveCycle(
            name=self.name,
            time_s=t,
            speed_ms=np.interp(t, self.time_s, self.speed_ms),
            grade_rad=np.interp(t, self.time_s, self.grade_rad),
        )


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


def load_cycle_file(path: str | Path, name: str | None = None, unit: str = "auto") -> DriveCycle:
    """Load a time/speed text file (EPA, DieselNet or plain CSV).

    The first two numeric columns are read as time [s] and speed. Non-numeric
    lines before the data are treated as header. With unit="auto", the speed
    unit is mph if the header mentions "mph", otherwise km/h.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    header_lines: list[str] = []
    times: list[float] = []
    speeds: list[float] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = [p for p in re.split(r"[\s,;]+", stripped) if p]
        try:
            t_val = float(parts[0])
            v_val = float(parts[1])
        except (ValueError, IndexError):
            if not times:
                header_lines.append(stripped)
            continue
        times.append(t_val)
        speeds.append(v_val)

    if len(times) < 2:
        raise ValueError(f"No time/speed data found in {path}")

    if unit == "auto":
        header = " ".join(header_lines).lower()
        unit = "mph" if "mph" in header else "kmh"
    if unit not in UNIT_FACTORS:
        raise ValueError(f"unit must be one of {sorted(UNIT_FACTORS)} or 'auto', got {unit!r}")

    speed_ms = np.clip(np.asarray(speeds) * UNIT_FACTORS[unit], 0.0, None)
    return DriveCycle(name=name or path.stem, time_s=np.asarray(times), speed_ms=speed_ms)


def concat_cycles(name: str, cycles: list[DriveCycle]) -> DriveCycle:
    """Join cycles end to end on one continuous time axis.

    When a cycle starts at the same speed the previous one ended at (normally
    0 km/h), the duplicated boundary sample is dropped so no time is added.
    """
    if not cycles:
        raise ValueError("cycles must not be empty")

    first = cycles[0]
    t_parts = [first.time_s - first.time_s[0]]
    v_parts = [first.speed_ms]
    g_parts = [first.grade_rad]

    for cyc in cycles[1:]:
        t_rel = cyc.time_s - cyc.time_s[0]
        t_end = t_parts[-1][-1]
        if np.isclose(cyc.speed_ms[0], v_parts[-1][-1]):
            t_parts.append(t_rel[1:] + t_end)
            v_parts.append(cyc.speed_ms[1:])
            g_parts.append(cyc.grade_rad[1:])
        else:
            step = float(np.median(np.diff(cyc.time_s)))
            t_parts.append(t_rel + t_end + step)
            v_parts.append(cyc.speed_ms)
            g_parts.append(cyc.grade_rad)

    return DriveCycle(
        name=name,
        time_s=np.concatenate(t_parts),
        speed_ms=np.concatenate(v_parts),
        grade_rad=np.concatenate(g_parts),
    )
