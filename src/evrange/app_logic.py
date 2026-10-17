"""Computation and plotting behind the dashboard and the CLI (no Streamlit imports here)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from evrange.cycle_library import describe, get_cycle  # noqa: E402
from evrange.cycles import DriveCycle  # noqa: E402
from evrange.environment import Environment  # noqa: E402
from evrange.ev_model import EVModel  # noqa: E402
from evrange.ml import RangePredictor, trip_features  # noqa: E402
from evrange.scenarios import Scenario, build_model, constant_speed_cycle, hilly, hvac_power_w  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "cycles"
MODEL_PATH = ROOT / "results" / "models" / "range_model.joblib"
CYCLE_KEYS = ["udds", "hwfet", "nedc", "midc", "wltc3b"]
CONSTANT = "constant"

LOSS_ITEMS = [
    ("aero_kwh", "Aerodynamic"),
    ("rolling_kwh", "Rolling"),
    ("grade_kwh", "Climbing (net)"),
    ("friction_brake_kwh", "Friction brakes"),
    ("motor_loss_kwh", "Motor + inverter"),
    ("gear_loss_kwh", "Gearbox"),
    ("battery_loss_kwh", "Battery (I^2R)"),
    ("aux_kwh", "Auxiliaries + HVAC"),
]


@dataclass(frozen=True)
class TripInputs:
    cycle_key: str = "udds"
    constant_speed_kmh: float = 100.0
    ambient_c: float = 20.0
    hvac: bool = True
    heater_cop: float = 1.0
    extra_payload_kg: float = 0.0
    hill_amplitude_pct: float = 0.0
    soh: float = 1.0
    extra_aux_w: float = 0.0

    def __post_init__(self) -> None:
        if self.constant_speed_kmh <= 0:
            raise ValueError("constant_speed_kmh must be > 0")
        if self.heater_cop <= 0:
            raise ValueError("heater_cop must be > 0")


# ---- loading -----------------------------------------------------------------------
def load_base(root: Path = ROOT) -> tuple[EVModel, Environment]:
    model = EVModel.from_yaml(root / "configs" / "compact_ev.yaml", root / "configs" / "compact_ev_powertrain.yaml")
    env = Environment.from_yaml(root / "configs" / "environment.yaml")
    return model, env


def available_cycles(data_dir: Path = DATA_DIR) -> dict[str, DriveCycle]:
    cycles = {}
    for key in CYCLE_KEYS:
        try:
            cycles[key] = get_cycle(key, data_dir)
        except FileNotFoundError:
            pass
    return cycles


def load_predictor(path: Path = MODEL_PATH) -> RangePredictor | None:
    return RangePredictor.load(path) if Path(path).exists() else None


def cycle_label(key: str) -> str:
    if key == CONSTANT:
        return "Constant speed (motorway)"
    return f"{key.upper()} - {describe(key)}"


# ---- inputs -> model objects -----------------------------------------------------
def make_cycle(inputs: TripInputs, cycles: dict[str, DriveCycle]) -> DriveCycle:
    if inputs.cycle_key == CONSTANT:
        return constant_speed_cycle(inputs.constant_speed_kmh, duration_s=900.0)
    if inputs.cycle_key not in cycles:
        raise KeyError(f"Cycle {inputs.cycle_key!r} not available. Run scripts/download_cycles.py")
    return cycles[inputs.cycle_key]


def make_scenario(inputs: TripInputs) -> Scenario:
    return Scenario(
        name="dashboard",
        ambient_c=inputs.ambient_c,
        hvac=inputs.hvac,
        extra_payload_kg=inputs.extra_payload_kg,
        hill_amplitude_pct=inputs.hill_amplitude_pct,
        soh=inputs.soh,
        resistance_growth=1.0 + 2.5 * (1.0 - inputs.soh),   # same ageing link as the ML dataset
        extra_aux_w=inputs.extra_aux_w,
    )


def environment_for(env: Environment, inputs: TripInputs) -> Environment:
    return replace(env, cabin=replace(env.cabin, heater_cop=inputs.heater_cop))


# ---- computations ----------------------------------------------------------------
def run_trip(base: EVModel, env: Environment, cycles: dict, inputs: TripInputs) -> dict:
    """Physics model: full-depletion range + one detailed pass (start SOC 80%)."""
    env_i = environment_for(env, inputs)
    sc = make_scenario(inputs)
    model = build_model(base, env_i, sc)
    cycle = make_cycle(inputs, cycles)
    trip = hilly(cycle, sc.hill_amplitude_pct, sc.hill_wavelength_km)
    rng = model.range_test(trip)
    run = model.run_cycle(trip, soc0=0.8)
    summary = model.summarize(run)
    dist = summary["distance_km"]
    breakdown = pd.Series({label: 1000 * summary[col] / dist for col, label in LOSS_ITEMS})
    return {
        "range_km": rng["range_km"],
        "consumption_wh_per_km": rng["consumption_wh_per_km"],
        "hvac_w": hvac_power_w(env_i, sc),
        "usable_kwh": model.battery.usable_energy_kwh,
        "regen_wh_per_km": 1000 * summary["regen_to_battery_kwh"] / dist,
        "motor_eff_pct": summary["motor_eff_motoring_pct"],
        "trace_misses": summary["torque_limited_samples"],
        "breakdown_wh_per_km": breakdown,
        "run": run,
        "range_trace": rng["trace"],
        "cycle": cycle,
        "model": model,
    }


def ml_range(predictor: RangePredictor | None, base: EVModel, env: Environment, cycles: dict,
             inputs: TripInputs, usable_kwh: float) -> float | None:
    if predictor is None:
        return None
    env_i = environment_for(env, inputs)
    feats = trip_features(make_cycle(inputs, cycles), make_scenario(inputs), env_i, base.vehicle.aux_power_w)
    return float(predictor.predict_range(pd.DataFrame([feats]), usable_kwh)[0])


def temperature_curve(base: EVModel, env: Environment, cycles: dict, inputs: TripInputs,
                      temps=tuple(range(-20, 41, 5))) -> pd.DataFrame:
    rows = []
    for t in temps:
        res = run_trip(base, env, cycles, replace(inputs, ambient_c=float(t)))
        rows.append({"ambient_c": float(t), "range_km": res["range_km"], "hvac_w": res["hvac_w"]})
    return pd.DataFrame(rows)


# ---- figures -----------------------------------------------------------------------
def fig_trip_signals(run: pd.DataFrame, title: str):
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    t = run["time_s"]
    axes[0].plot(t, run["speed_kmh"])
    axes[0].set_ylabel("Speed\n[km/h]")
    axes[1].plot(t, run["motor_torque_nm"], color="tab:orange")
    axes[1].set_ylabel("Motor torque\n[N m]")
    axes[2].plot(t, run["current_a"], color="tab:red")
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set_ylabel("Pack current\n[A]")
    axes[3].plot(t, 100 * run["soc"], color="tab:green")
    axes[3].set_ylabel("SOC [%]")
    axes[3].set_xlabel("Time [s]")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def fig_breakdown(breakdown: pd.Series, regen_wh_per_km: float):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b = breakdown[breakdown.abs() > 0.05]
    ax.barh(b.index, b.values, color=plt.cm.tab10(np.arange(len(b)) % 10))
    ax.invert_yaxis()
    ax.set_xlabel("Wh/km (energy drawn from the cells)")
    ax.set_title(f"Energy breakdown - total {breakdown.sum():.0f} Wh/km, regen recovered {regen_wh_per_km:.0f} Wh/km")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def fig_range_trace(trace: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(trace["distance_km"], 100 * trace["soc"], "o-", ms=3)
    ax.set_xlabel("Distance [km]")
    ax.set_ylabel("SOC [%]")
    ax.set_title("Full-depletion range test")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_temperature(curve: pd.DataFrame, current_c: float):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(curve["ambient_c"], curve["range_km"], "o-")
    ax.axvline(current_c, color="tab:red", ls="--", label="selected temperature")
    ax.set_xlabel("Ambient temperature [C]")
    ax.set_ylabel("Range [km]")
    ax.set_title("Range vs temperature (other inputs as selected)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def fig_efficiency_map(model: EVModel, run: pd.DataFrame):
    S, T, eff = model.motor.efficiency_map()
    fig, ax = plt.subplots(figsize=(8, 5))
    levels = [0.70, 0.80, 0.85, 0.88, 0.90, 0.92, 0.93, 0.94, 0.95, 0.96]
    cs = ax.contourf(S, T, eff, levels=levels, cmap="viridis", extend="both")
    fig.colorbar(cs, ax=ax, label="Efficiency")
    w = np.linspace(1, model.motor.max_speed_rads, 300)
    ax.plot(w * 60 / (2 * np.pi), model.motor.max_torque(w), "k-", lw=2)
    drive = run["motor_torque_nm"] > 0
    ax.scatter(run.loc[drive, "motor_speed_rpm"], run.loc[drive, "motor_torque_nm"],
               s=8, color="white", edgecolors="k", linewidths=0.2, label="Trip operating points")
    ax.set_xlabel("Motor speed [rpm]")
    ax.set_ylabel("Motor torque [N m]")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


# ---- ML validity checks ------------------------------------------------------------
TRAINING_LIMITS = {
    "ambient_c": (-20.0, 40.0),
    "extra_payload_kg": (0.0, 450.0),
    "hill_amplitude_pct": (0.0, 8.0),
    "soh": (0.7, 1.0),
    "extra_aux_w": (0.0, 400.0),
    "constant_speed_kmh": (30.0, 140.0),
}


def ml_warnings(inputs: TripInputs, env: Environment | None = None) -> list[str]:
    """Reasons the ML prediction may be less reliable (outside the training data)."""
    notes = []
    for name, (lo, hi) in TRAINING_LIMITS.items():
        if name == "constant_speed_kmh" and inputs.cycle_key != CONSTANT:
            continue
        val = getattr(inputs, name)
        if not lo <= val <= hi:
            notes.append(f"{name} = {val:g} is outside the training range [{lo:g}, {hi:g}]")
    heating = env is None or env.cabin.thermal_load_w(inputs.ambient_c)[0] > 0
    if inputs.hvac and inputs.heater_cop != 1.0 and heating:
        notes.append("the ML model was trained with a PTC heater only (COP 1.0)")
    return notes
