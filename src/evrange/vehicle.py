"""Vehicle parameter definition, validation and YAML loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

GRAVITY = 9.81  # m/s^2


@dataclass(frozen=True)
class Vehicle:
    name: str
    mass_kg: float
    payload_kg: float
    drag_coefficient: float
    frontal_area_m2: float
    rolling_resistance: float
    rotational_mass_factor: float
    drivetrain_efficiency: float
    regen_efficiency: float
    regen_fraction: float
    aux_power_w: float
    battery_usable_kwh: float
    air_density: float = 1.225

    def __post_init__(self) -> None:
        positive = {
            "mass_kg": self.mass_kg,
            "frontal_area_m2": self.frontal_area_m2,
            "battery_usable_kwh": self.battery_usable_kwh,
            "air_density": self.air_density,
            "drivetrain_efficiency": self.drivetrain_efficiency,
        }
        for key, val in positive.items():
            if val <= 0:
                raise ValueError(f"{key} must be > 0, got {val}")

        non_negative = {
            "payload_kg": self.payload_kg,
            "drag_coefficient": self.drag_coefficient,
            "rolling_resistance": self.rolling_resistance,
            "rotational_mass_factor": self.rotational_mass_factor,
            "aux_power_w": self.aux_power_w,
        }
        for key, val in non_negative.items():
            if val < 0:
                raise ValueError(f"{key} must be >= 0, got {val}")

        fractions = {
            "drivetrain_efficiency": self.drivetrain_efficiency,
            "regen_efficiency": self.regen_efficiency,
            "regen_fraction": self.regen_fraction,
        }
        for key, val in fractions.items():
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{key} must be in [0, 1], got {val}")

    @property
    def total_mass_kg(self) -> float:
        """Curb mass + payload (used for rolling resistance and grade)."""
        return self.mass_kg + self.payload_kg

    @property
    def effective_mass_kg(self) -> float:
        """Total mass including equivalent rotating inertia (used for acceleration)."""
        return self.total_mass_kg * (1.0 + self.rotational_mass_factor)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Vehicle":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def with_updates(self, **changes) -> "Vehicle":
        """Return a copy with some parameters changed (for sensitivity studies)."""
        data = asdict(self)
        data.update(changes)
        return Vehicle(**data)

    def to_dict(self) -> dict:
        return asdict(self)
