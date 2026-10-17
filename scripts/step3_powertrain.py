"""Step 3: detailed powertrain + battery model - losses, efficiency map and full-depletion range."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from evrange import EVModel, get_cycle, simulate, summarize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "cycles"
RESULTS = ROOT / "results"
CYCLES = ["udds", "hwfet", "nedc", "midc", "wltc3b"]
LOSS_ITEMS = [
    ("aero_kwh", "Aerodynamic"),
    ("rolling_kwh", "Rolling"),
    ("friction_brake_kwh", "Friction brakes"),
    ("motor_loss_kwh", "Motor + inverter"),
    ("gear_loss_kwh", "Gearbox"),
    ("battery_loss_kwh", "Battery (I^2R)"),
    ("aux_kwh", "Auxiliaries"),
]


def load_available() -> dict:
    cycles = {}
    for key in CYCLES:
        try:
            cycles[key] = get_cycle(key, DATA_DIR)
        except FileNotFoundError:
            print(f"[skip] {key}: cycle file not found")
    return cycles


def compare_models(model: EVModel, cycles: dict) -> tuple[pd.DataFrame, dict, dict]:
    rows, runs, ranges = {}, {}, {}
    for key, cyc in cycles.items():
        simple = summarize(simulate(model.vehicle, cyc), model.vehicle)
        run = model.run_cycle(cyc, soc0=0.5)
        detail = model.summarize(run)
        rng = model.range_test(cyc)
        runs[key], ranges[key] = run, rng
        rows[key] = {
            "simple_wh_per_km": simple["consumption_wh_per_km"],
            "simple_range_km": simple["range_km"],
            "detailed_wh_per_km": detail["consumption_wh_per_km"],
            "detailed_range_km": rng["range_km"],
            "range_change_pct": 100 * (rng["range_km"] - simple["range_km"]) / simple["range_km"],
            "motor_eff_pct": detail["motor_eff_motoring_pct"],
            "regen_share_pct": detail["regen_share_pct"],
            "peak_current_a": detail["peak_current_a"],
            "min_voltage_v": detail["min_voltage_v"],
            "trace_misses": detail["torque_limited_samples"],
            "range_cycles": rng["full_cycles"],
        }
    return pd.DataFrame(rows).T, runs, ranges


def loss_table(model: EVModel, runs: dict) -> pd.DataFrame:
    rows = {}
    for key, run in runs.items():
        e = model.energy_breakdown(run)
        dist = float(run["distance_km"].iloc[-1])
        rows[key] = {label: 1000 * e[col] / dist for col, label in LOSS_ITEMS}
        rows[key]["Regen recovered"] = -1000 * e["regen_to_battery_kwh"] / dist
        rows[key]["Battery net (Wh/km)"] = 1000 * e["battery_terminal_kwh"] / dist
    return pd.DataFrame(rows).T


def plot_efficiency_map(model: EVModel, runs: dict, path: Path) -> None:
    S, T, eff = model.motor.efficiency_map()
    fig, ax = plt.subplots(figsize=(9, 6))
    levels = [0.70, 0.80, 0.85, 0.88, 0.90, 0.92, 0.93, 0.94, 0.95, 0.96]
    cs = ax.contourf(S, T, eff, levels=levels, cmap="viridis", extend="both")
    ax.contour(S, T, eff, levels=levels, colors="k", linewidths=0.3)
    fig.colorbar(cs, ax=ax, label="Motor + inverter efficiency")
    w = np.linspace(1, model.motor.max_speed_rads, 300)
    ax.plot(w * 60 / (2 * np.pi), model.motor.max_torque(w), "k-", lw=2, label="Torque envelope")
    markers = {"udds": ("tab:red", "UDDS"), "hwfet": ("tab:orange", "HWFET"),
               "nedc": ("tab:pink", "NEDC"), "midc": ("white", "MIDC"), "wltc3b": ("tab:cyan", "WLTC")}
    for key, run in runs.items():
        drive = run["motor_torque_nm"] > 0
        color, label = markers.get(key, ("gray", key))
        ax.scatter(run.loc[drive, "motor_speed_rpm"], run.loc[drive, "motor_torque_nm"],
                   s=8, color=color, edgecolors="k", linewidths=0.2, alpha=0.8,
                   label=f"{label} operating points")
    ax.set_xlabel("Motor speed [rpm]")
    ax.set_ylabel("Motor torque [N m]")
    ax.set_title("PMSM + inverter efficiency map (motoring)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_losses(losses: pd.DataFrame, path: Path) -> None:
    """Stacked sinks = energy drawn from the cells; regen is recirculated energy, shown as text."""
    fig, ax = plt.subplots(figsize=(11.5, 5))
    bottom = np.zeros(len(losses))
    labels = [k.upper() for k in losses.index]
    for _, label in LOSS_ITEMS:
        ax.bar(labels, losses[label], bottom=bottom, label=label)
        bottom += losses[label].to_numpy()
    for x, top, regen in zip(labels, bottom, losses["Regen recovered"]):
        ax.annotate(f"total {top:.0f} Wh/km\nregen recovered {-regen:.0f} Wh/km",
                    (x, top), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
    ax.set_ylim(0, bottom.max() * 1.18)
    ax.set_ylabel("Wh/km (energy drawn from the cells)")
    ax.set_title("Where the battery energy goes")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_range_tests(ranges: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for key, rng in ranges.items():
        tr = rng["trace"]
        axes[0].plot(tr["distance_km"], 100 * tr["soc"], label=f"{key.upper()} ({rng['range_km']:.0f} km)")
        axes[1].plot(tr["distance_km"], tr["v_terminal"], label=key.upper())
    axes[0].set_ylabel("SOC [%]")
    axes[1].set_ylabel("Pack voltage at end of each cycle [V]")
    for ax in axes:
        ax.set_xlabel("Distance [km]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Full-depletion range test (cycle repeated from SOC max to SOC min)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cycle_detail(run: pd.DataFrame, name: str, path: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    t = run["time_s"]
    axes[0].plot(t, run["speed_kmh"])
    axes[0].set_ylabel("Speed [km/h]")
    axes[1].plot(t, run["motor_torque_nm"])
    axes[1].set_ylabel("Motor torque [N m]")
    axes[2].plot(t, run["current_a"])
    axes[2].set_ylabel("Pack current [A]")
    axes[3].plot(t, run["v_terminal"])
    axes[3].set_ylabel("Pack voltage [V]")
    axes[3].set_xlabel("Time [s]")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(f"{name.upper()} - detailed model signals (start SOC 50%)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    model = EVModel.from_yaml(ROOT / "configs" / "compact_ev.yaml", ROOT / "configs" / "compact_ev_powertrain.yaml")
    b = model.battery
    print(f"Battery: {b.cells_series}s{b.cells_parallel}p, {b.capacity_ah:.0f} Ah, "
          f"{b.nominal_energy_kwh:.1f} kWh nominal, {b.usable_energy_kwh:.1f} kWh usable "
          f"(SOC {100 * b.soc_min:.0f}-{100 * b.soc_max:.0f}%), R0 {1000 * b.r0_ohm:.0f} mOhm")
    _, _, eff = model.motor.efficiency_map()
    print(f"Motor: {model.motor.peak_power_w / 1000:.0f} kW, {model.motor.peak_torque_nm:.0f} N m, "
          f"peak efficiency {100 * np.nanmax(eff):.1f}%")

    cycles = load_available()
    if not cycles:
        print("No cycles found. Run: python scripts/download_cycles.py")
        return

    comp, runs, ranges = compare_models(model, cycles)
    print("\n=== 1. Simple (Step 1) vs detailed (Step 3) model ===")
    print(comp.astype(float).round(1).to_string())

    losses = loss_table(model, runs)
    print("\n=== 2. Energy breakdown [Wh/km] ===")
    print(losses.round(1).to_string())

    comp.to_csv(RESULTS / "step3_model_comparison.csv")
    losses.to_csv(RESULTS / "step3_loss_breakdown.csv")
    plot_efficiency_map(model, runs, RESULTS / "step3_motor_efficiency_map.png")
    plot_losses(losses, RESULTS / "step3_loss_breakdown.png")
    plot_range_tests(ranges, RESULTS / "step3_range_tests.png")
    first = "udds" if "udds" in runs else next(iter(runs))
    plot_cycle_detail(runs[first], first, RESULTS / f"step3_{first}_signals.png")
    print(f"\nSaved CSVs and plots to {RESULTS}")


if __name__ == "__main__":
    main()
