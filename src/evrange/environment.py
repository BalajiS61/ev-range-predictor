"""Ambient-condition effects: air density, cabin HVAC power and battery temperature.

- Air density from the ideal gas law at sea-level pressure.
- HVAC electrical power from a steady-state cabin heat balance:
      heating load = UA (T_on_heat - T_amb),  cooling load = UA (T_amb - T_on_cool) + solar
      P_elec = load / COP, capped at max_electrical_w
- Battery resistance follows Arrhenius: R(T) = R_ref exp(Ea/Rg (1/T - 1/T_ref)).
- Available capacity falls in the cold (interpolated table).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

GAS_CONSTANT = 8.314          # J/(mol K)
AIR_GAS_CONSTANT = 287.05     # J/(kg K)
SEA_LEVEL_PA = 101_325.0


def air_density(ambient_c: float, pressure_pa: float = SEA_LEVEL_PA) -> float:
    return pressure_pa / (AIR_GAS_CONSTANT * (ambient_c + 273.15))


@dataclass(frozen=True)
class CabinHVAC:
    setpoint_c: float
    deadband_c: float
    heat_loss_w_per_k: float
    solar_load_w: float
    heater_cop: float
    cooling_cop: float
    max_electrical_w: float

    def __post_init__(self) -> None:
        if self.deadband_c < 0 or self.heat_loss_w_per_k < 0 or self.solar_load_w < 0:
            raise ValueError("deadband, heat loss and solar load must be >= 0")
        if self.heater_cop <= 0 or self.cooling_cop <= 0 or self.max_electrical_w < 0:
            raise ValueError("COPs must be > 0 and max_electrical_w >= 0")

    def thermal_load_w(self, ambient_c: float) -> tuple[float, float]:
        """(heating_load, cooling_load) in thermal watts."""
        heat_on = self.setpoint_c - self.deadband_c
        cool_on = self.setpoint_c + self.deadband_c
        heating = self.heat_loss_w_per_k * max(0.0, heat_on - ambient_c)
        cooling = 0.0
        if ambient_c > cool_on:
            cooling = self.heat_loss_w_per_k * (ambient_c - cool_on) + self.solar_load_w
        return heating, cooling

    def electrical_power_w(self, ambient_c: float) -> float:
        heating, cooling = self.thermal_load_w(ambient_c)
        p = heating / self.heater_cop + cooling / self.cooling_cop
        return float(min(p, self.max_electrical_w))


@dataclass(frozen=True)
class BatteryThermal:
    activation_energy_j_mol: float
    reference_c: float
    capacity_temp_c: tuple = field(default=())
    capacity_factor: tuple = field(default=())

    def __post_init__(self) -> None:
        object.__setattr__(self, "capacity_temp_c", tuple(float(x) for x in self.capacity_temp_c))
        object.__setattr__(self, "capacity_factor", tuple(float(x) for x in self.capacity_factor))
        if len(self.capacity_temp_c) < 2 or len(self.capacity_temp_c) != len(self.capacity_factor):
            raise ValueError("capacity_temp_c and capacity_factor must have equal length >= 2")
        if np.any(np.diff(self.capacity_temp_c) <= 0):
            raise ValueError("capacity_temp_c must be increasing")
        if self.activation_energy_j_mol < 0:
            raise ValueError("activation_energy_j_mol must be >= 0")

    def resistance_factor(self, cell_temp_c: float) -> float:
        t, t_ref = cell_temp_c + 273.15, self.reference_c + 273.15
        return float(np.exp(self.activation_energy_j_mol / GAS_CONSTANT * (1.0 / t - 1.0 / t_ref)))

    def capacity_factor_at(self, cell_temp_c: float) -> float:
        return float(np.interp(cell_temp_c, self.capacity_temp_c, self.capacity_factor))


@dataclass(frozen=True)
class Environment:
    cabin: CabinHVAC
    battery_thermal: BatteryThermal

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Environment":
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls(cabin=CabinHVAC(**cfg["cabin"]), battery_thermal=BatteryThermal(**cfg["battery_thermal"]))
