"""Real-world scenarios and one-at-a-time sensitivity sweeps for range.

A Scenario only changes what it sets; fields left as None keep the base model
exactly as configured, so Scenario() reproduces the Step 3 range.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from evrange.cycles import DriveCycle, piecewise_cycle
from evrange.environment import Environment, air_density
from evrange.ev_model import EVModel


@dataclass(frozen=True)
class Scenario:
    name: str = "baseline"
    extra_payload_kg: float = 0.0        # passengers + luggage on top of the base payload
    ambient_c: float | None = None       # sets air density and battery temperature
    hvac: bool = False                   # cabin heating/cooling at ambient_c
    battery_temp_c: float | None = None  # defaults to ambient_c when ambient_c is set
    hill_amplitude_pct: float = 0.0      # sinusoidal grade amplitude, zero net elevation
    hill_wavelength_km: float = 2.0
    soh: float = 1.0                     # capacity state of health (1.0 = new)
    resistance_growth: float = 1.0       # ageing multiplier on R0 and R1
    extra_aux_w: float = 0.0             # other electrical loads (lights, seat heaters, ...)

    def __post_init__(self) -> None:
        if not 0 < self.soh <= 1:
            raise ValueError("soh must be in (0, 1]")
        if self.resistance_growth <= 0 or self.extra_aux_w < 0 or self.hill_wavelength_km <= 0:
            raise ValueError("resistance_growth, hill_wavelength_km must be > 0; extra_aux_w >= 0")
        if self.hvac and self.ambient_c is None:
            raise ValueError("hvac=True needs ambient_c")


def hvac_power_w(env: Environment, scenario: Scenario) -> float:
    if not scenario.hvac or scenario.ambient_c is None:
        return 0.0
    return env.cabin.electrical_power_w(scenario.ambient_c)


def build_model(base: EVModel, env: Environment, scenario: Scenario) -> EVModel:
    veh = base.vehicle
    changes = {
        "payload_kg": veh.payload_kg + scenario.extra_payload_kg,
        "aux_power_w": veh.aux_power_w + scenario.extra_aux_w + hvac_power_w(env, scenario),
    }
    if scenario.ambient_c is not None:
        changes["air_density"] = air_density(scenario.ambient_c)
    vehicle = veh.with_updates(**changes)

    bat = base.battery
    cell_temp = scenario.battery_temp_c if scenario.battery_temp_c is not None else scenario.ambient_c
    r_factor, c_factor = scenario.resistance_growth, scenario.soh
    if cell_temp is not None:
        r_factor *= env.battery_thermal.resistance_factor(cell_temp)
        c_factor *= env.battery_thermal.capacity_factor_at(cell_temp)
    battery = replace(
        bat,
        cell_capacity_ah=bat.cell_capacity_ah * c_factor,
        cell_r0_ohm=bat.cell_r0_ohm * r_factor,
        cell_r1_ohm=bat.cell_r1_ohm * r_factor,
    )
    return EVModel(vehicle, base.drivetrain, base.motor, battery, base.regen)


def hilly(cycle: DriveCycle, amplitude_pct: float, wavelength_km: float = 2.0) -> DriveCycle:
    """Add a sinusoidal grade along distance. The wavelength is adjusted so each
    cycle covers a whole number of hills, giving zero net elevation per cycle."""
    if amplitude_pct == 0:
        return cycle
    v, t = cycle.speed_ms, cycle.time_s
    dist_m = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))])
    total_km = dist_m[-1] / 1000.0
    n_hills = max(1, round(total_km / wavelength_km))
    lam_m = dist_m[-1] / n_hills
    grade = np.arctan(amplitude_pct / 100.0 * np.sin(2 * np.pi * dist_m / lam_m))
    return DriveCycle(f"{cycle.name}_hills{amplitude_pct:g}", t, v, grade)


def constant_speed_cycle(speed_kmh: float, duration_s: float = 1800.0) -> DriveCycle:
    return piecewise_cycle(f"const_{speed_kmh:g}", [(0, speed_kmh), (duration_s, speed_kmh)])


def evaluate(base: EVModel, env: Environment, cycle: DriveCycle, scenario: Scenario) -> dict:
    model = build_model(base, env, scenario)
    cyc = hilly(cycle, scenario.hill_amplitude_pct, scenario.hill_wavelength_km)
    rng = model.range_test(cyc)
    return {
        "scenario": scenario.name,
        "cycle": cycle.name,
        "range_km": rng["range_km"],
        "consumption_wh_per_km": rng["consumption_wh_per_km"],
        "energy_used_kwh": rng["energy_used_kwh"],
        "hvac_w": hvac_power_w(env, scenario),
        "usable_kwh": model.battery.usable_energy_kwh,
        "r0_mohm": 1000 * model.battery.r0_ohm,
        "total_mass_kg": model.vehicle.total_mass_kg,
    }


def sweep(base: EVModel, env: Environment, cycle: DriveCycle, field_name: str,
          values, template: Scenario | None = None) -> pd.DataFrame:
    """Vary one Scenario field over values, everything else from template."""
    template = template or Scenario()
    rows = []
    for val in values:
        sc = replace(template, name=f"{field_name}={val}", **{field_name: val})
        row = evaluate(base, env, cycle, sc)
        row[field_name] = val
        rows.append(row)
    return pd.DataFrame(rows)


def epa_label_range(base: EVModel, env: Environment, city: DriveCycle, highway: DriveCycle,
                    city_weight: float = 0.55, adjustment: float = 0.70) -> dict:
    """EPA-style 2-cycle label estimate: 55/45 city/highway consumption, x0.7 adjustment."""
    c = evaluate(base, env, city, Scenario("city"))
    h = evaluate(base, env, highway, Scenario("highway"))
    combined_whkm = city_weight * c["consumption_wh_per_km"] + (1 - city_weight) * h["consumption_wh_per_km"]
    unadjusted = 1000 * base.battery.usable_energy_kwh / combined_whkm
    return {
        "city_range_km": c["range_km"],
        "highway_range_km": h["range_km"],
        "combined_wh_per_km": combined_whkm,
        "unadjusted_range_km": unadjusted,
        "label_range_km": adjustment * unadjusted,
    }
