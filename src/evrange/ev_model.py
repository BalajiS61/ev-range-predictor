"""Detailed EV model: road load -> drivetrain -> motor map -> battery ECM.

The road-load part (forces, wheel power) comes from evrange.dynamics. On top of
it this model adds braking blending with speed fade-out, gear losses, a motor
efficiency map with torque/power limits, and a battery pack whose voltage, SOC
and losses evolve in time. `range_test` repeats a cycle until the usable SOC
window is used up, like a chassis-dyno range test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from evrange.battery import BatteryPack
from evrange.cycles import DriveCycle
from evrange.dynamics import simulate as road_load
from evrange.powertrain import Drivetrain, Motor
from evrange.vehicle import Vehicle


@dataclass(frozen=True)
class RegenStrategy:
    max_power_w: float
    fade_start_kmh: float
    fade_end_kmh: float

    def __post_init__(self) -> None:
        if self.max_power_w < 0:
            raise ValueError("max_power_w must be >= 0")
        if self.fade_end_kmh > self.fade_start_kmh:
            raise ValueError("fade_end_kmh must be <= fade_start_kmh")

    def fade(self, speed_ms: np.ndarray) -> np.ndarray:
        """Share of the regen request kept: 0 below fade_end, 1 above fade_start."""
        v = np.asarray(speed_ms) * 3.6
        if self.fade_start_kmh == self.fade_end_kmh:
            return (v >= self.fade_start_kmh).astype(float)
        return np.clip((v - self.fade_end_kmh) / (self.fade_start_kmh - self.fade_end_kmh), 0.0, 1.0)


def _trap(y: np.ndarray, t: np.ndarray) -> float:
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(t)))


def _regen_torque_for_power(motor: Motor, w: np.ndarray, p_limit: float) -> np.ndarray:
    """Negative torque at which electrical power equals -p_limit (root nearest zero)."""
    kc = motor.loss_copper
    c = motor.loss_iron * w + motor.loss_windage * w**3 + motor.loss_constant_w + p_limit
    if kc == 0:
        return -c / np.maximum(w, 1e-9)
    disc = np.maximum(w**2 - 4.0 * kc * c, 0.0)
    return (-w + np.sqrt(disc)) / (2.0 * kc)


class EVModel:
    def __init__(
        self,
        vehicle: Vehicle,
        drivetrain: Drivetrain,
        motor: Motor,
        battery: BatteryPack,
        regen: RegenStrategy,
    ) -> None:
        self.vehicle = vehicle
        self.drivetrain = drivetrain
        self.motor = motor
        self.battery = battery
        self.regen = regen

    @classmethod
    def from_yaml(cls, vehicle_path: str | Path, powertrain_path: str | Path) -> "EVModel":
        vehicle = Vehicle.from_yaml(vehicle_path)
        with open(powertrain_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls(
            vehicle=vehicle,
            drivetrain=Drivetrain(**cfg["drivetrain"]),
            motor=Motor(**cfg["motor"]),
            battery=BatteryPack(**cfg["battery"]),
            regen=RegenStrategy(**cfg["regen"]),
        )

    # ---- SOC-independent part: road load, braking split, gear, motor ------------
    def powertrain_demand(self, cycle: DriveCycle) -> pd.DataFrame:
        veh, dtr, mot = self.vehicle, self.drivetrain, self.motor
        df = road_load(veh, cycle)
        v = df["speed_ms"].to_numpy()
        p_wheel = df["p_wheel_w"].to_numpy()
        w = dtr.motor_speed_rads(v)
        moving = w > 1e-6

        p_trac_req = np.maximum(p_wheel, 0.0)
        p_regen_req = np.where(p_wheel < 0, p_wheel * veh.regen_fraction * self.regen.fade(v), 0.0)
        p_shaft_req = dtr.shaft_power(p_trac_req + p_regen_req)

        t_req = np.where(moving, p_shaft_req / np.where(moving, w, 1.0), 0.0)
        t_max = mot.max_torque(w)
        torque = np.clip(t_req, -t_max, t_max)
        if self.regen.max_power_w > 0:
            t_regen_lim = _regen_torque_for_power(mot, w, self.regen.max_power_w)
            torque = np.where(torque < 0, np.maximum(torque, t_regen_lim), torque)
        else:
            torque = np.maximum(torque, 0.0)
        torque = np.where(moving, torque, 0.0)

        p_shaft = torque * w
        p_wheel_motor = np.where(p_shaft >= 0, p_shaft * dtr.gear_efficiency, p_shaft / dtr.gear_efficiency)
        p_gear_loss = np.abs(p_shaft - p_wheel_motor)
        p_motor_loss = mot.losses(torque, w)
        p_motor_elec = p_shaft + p_motor_loss

        p_unmet = np.maximum(p_trac_req - np.maximum(p_wheel_motor, 0.0), 0.0)
        p_friction = np.where(p_wheel < 0, p_wheel - np.minimum(p_wheel_motor, 0.0), 0.0)

        df["motor_speed_rpm"] = w * 60.0 / (2.0 * np.pi)
        df["motor_torque_nm"] = torque
        df["torque_limited"] = np.abs(t_req - torque) > 1e-6
        df["p_shaft_w"] = p_shaft
        df["p_wheel_motor_w"] = p_wheel_motor
        df["p_gear_loss_w"] = p_gear_loss
        df["p_motor_loss_w"] = p_motor_loss
        df["p_motor_elec_w"] = p_motor_elec
        df["p_friction_brake_w"] = p_friction
        df["p_unmet_w"] = p_unmet
        df["p_aux_w"] = veh.aux_power_w
        df["p_batt_req_w"] = p_motor_elec + veh.aux_power_w
        return df

    # ---- battery: run one cycle from a given state ----------------------------
    def _run_battery(self, df: pd.DataFrame, soc0: float, v_rc0: float) -> dict:
        t = df["time_s"].to_numpy()
        p = df["p_batt_req_w"].to_numpy()
        p_mid = 0.5 * (p[1:] + p[:-1])             # interval-average power (trapezoid-consistent)
        return self.battery.run(np.append(p_mid, p_mid[-1]), t, soc0, v_rc0) | {"p_req_mid_w": p_mid}

    def run_cycle(self, cycle: DriveCycle, soc0: float = 0.8, v_rc0: float = 0.0) -> pd.DataFrame:
        """One pass over the cycle with the full model. Interval values are put on sample k."""
        df = self.powertrain_demand(cycle)
        bat = self._run_battery(df, soc0, v_rc0)
        pad = lambda x: np.append(x, np.nan)  # noqa: E731  (N-1 interval values -> N rows)
        df["soc"] = bat["soc"]
        df["v_rc"] = bat["v_rc"]
        df["current_a"] = pad(bat["current_a"])
        df["v_terminal"] = pad(bat["v_terminal"])
        df["p_batt_w"] = pad(bat["p_terminal_w"])
        df["p_batt_loss_w"] = pad(bat["p_loss_w"])
        df["p_chemical_w"] = pad(bat["p_chemical_w"])
        df["p_req_mid_w"] = pad(bat["p_req_mid_w"])
        df["batt_limited"] = pad(bat["limited"].astype(float))
        return df

    # ---- energy accounting ----------------------------------------------------
    def energy_breakdown(self, df: pd.DataFrame) -> dict:
        """Energy flows over a run in kWh (sample signals by trapezoid, intervals by sum)."""
        t = df["time_s"].to_numpy()
        dt = np.diff(t)
        to_kwh = 1.0 / 3.6e6

        def interval(col: str) -> float:
            return float(np.nansum(df[col].to_numpy()[:-1] * dt)) * to_kwh

        p_req = df["p_req_mid_w"].to_numpy()[:-1]
        p_act = df["p_batt_w"].to_numpy()[:-1]
        curtailed = max(0.0, float(np.sum(np.where(p_req < 0, p_act - p_req, 0.0) * dt)) * to_kwh)
        batt_short = max(0.0, float(np.sum(np.where(p_req > 0, p_req - p_act, 0.0) * dt)) * to_kwh)

        out = {
            "aero_kwh": _trap((df["f_aero_n"] * df["speed_ms"]).to_numpy(), t) * to_kwh,
            "rolling_kwh": _trap((df["f_roll_n"] * df["speed_ms"]).to_numpy(), t) * to_kwh,
            "grade_kwh": _trap((df["f_grade_n"] * df["speed_ms"]).to_numpy(), t) * to_kwh,
            "inertia_net_kwh": _trap((df["f_inertia_n"] * df["speed_ms"]).to_numpy(), t) * to_kwh,
            "friction_brake_kwh": -_trap(df["p_friction_brake_w"].to_numpy(), t) * to_kwh,
            "gear_loss_kwh": _trap(df["p_gear_loss_w"].to_numpy(), t) * to_kwh,
            "motor_loss_kwh": _trap(df["p_motor_loss_w"].to_numpy(), t) * to_kwh,
            "aux_kwh": _trap(df["p_aux_w"].to_numpy(), t) * to_kwh,
            "regen_to_battery_kwh": -float(np.sum(np.minimum(p_act, 0.0) * dt)) * to_kwh,
            "regen_curtailed_kwh": curtailed,
            "unmet_traction_kwh": _trap(df["p_unmet_w"].to_numpy(), t) * to_kwh,
            "battery_shortfall_kwh": batt_short,
            "battery_loss_kwh": interval("p_batt_loss_w"),
            "battery_terminal_kwh": interval("p_batt_w"),
            "battery_chemical_kwh": interval("p_chemical_w"),
        }
        return out

    def summarize(self, df: pd.DataFrame) -> dict:
        e = self.energy_breakdown(df)
        dist = float(df["distance_km"].iloc[-1])
        moving = df["motor_torque_nm"].abs() > 0
        shaft = df.loc[moving, "p_shaft_w"].to_numpy()
        elec = df.loc[moving, "p_motor_elec_w"].to_numpy()
        pos = shaft > 0
        mot_eff = float(np.sum(shaft[pos]) / np.sum(elec[pos])) if np.any(pos) else float("nan")
        return {
            "distance_km": dist,
            "consumption_wh_per_km": 1000 * e["battery_terminal_kwh"] / dist if dist > 0 else float("nan"),
            "motor_eff_motoring_pct": 100 * mot_eff,
            "regen_share_pct": 100 * e["regen_to_battery_kwh"] / max(e["battery_terminal_kwh"] + e["regen_to_battery_kwh"], 1e-12),
            "soc_used_pct": 100 * float(df["soc"].iloc[0] - df["soc"].iloc[-1]),
            "min_voltage_v": float(np.nanmin(df["v_terminal"])),
            "peak_current_a": float(np.nanmax(df["current_a"])),
            "torque_limited_samples": int(df["torque_limited"].sum()),
            **e,
        }

    # ---- full-depletion range test ---------------------------------------------
    def range_test(self, cycle: DriveCycle, soc_start: float | None = None, max_cycles: int = 500) -> dict:
        """Repeat the cycle from soc_start until SOC reaches battery.soc_min."""
        bat = self.battery
        soc0 = bat.soc_max if soc_start is None else soc_start
        df = self.powertrain_demand(cycle)
        dist = df["distance_km"].to_numpy()
        t = df["time_s"].to_numpy()
        cycle_km = float(dist[-1])

        soc, v_rc = soc0, 0.0
        energy_kwh, total_km, n_full = 0.0, 0.0, 0
        trace_km, trace_soc, trace_v = [0.0], [soc0], [float(bat.ocv(soc0))]

        for _ in range(max_cycles):
            res = self._run_battery(df, soc, v_rc)
            socs = res["soc"]
            e_int = res["p_terminal_w"] * np.diff(t) / 3.6e6
            below = np.nonzero(socs <= bat.soc_min)[0]
            if below.size == 0:
                energy_kwh += float(np.sum(e_int))
                total_km += cycle_km
                n_full += 1
                soc, v_rc = float(socs[-1]), float(res["v_rc"][-1])
                trace_km.append(total_km)
                trace_soc.append(soc)
                trace_v.append(float(res["v_terminal"][-1]))
                continue
            j = int(below[0])                          # first sample at/below soc_min
            frac = (socs[j - 1] - bat.soc_min) / (socs[j - 1] - socs[j]) if socs[j - 1] != socs[j] else 1.0
            part_km = float(dist[j - 1]) + frac * (dist[j] - dist[j - 1])
            energy_kwh += float(np.sum(e_int[: j - 1]) + frac * e_int[j - 1])
            total_km += part_km
            trace_km.append(total_km)
            trace_soc.append(bat.soc_min)
            trace_v.append(float(res["v_terminal"][j - 1]))
            break
        else:
            raise RuntimeError("SOC window not used up within max_cycles")

        return {
            "range_km": float(total_km),
            "full_cycles": n_full,
            "energy_used_kwh": energy_kwh,
            "consumption_wh_per_km": float(1000 * energy_kwh / total_km),
            "soc_start": soc0,
            "soc_end": bat.soc_min,
            "trace": pd.DataFrame({"distance_km": trace_km, "soc": trace_soc, "v_terminal": trace_v}),
        }
