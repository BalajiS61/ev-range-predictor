"""Step 2: standard drive cycles - statistics, validation and energy/range per cycle."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from evrange import Vehicle, cycle_statistics, get_cycle, simulate, summarize  # noqa: E402
from evrange.cycle_library import describe, reference  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "cycles"
RESULTS = ROOT / "results"
CYCLES = ["udds", "hwfet", "nedc", "midc", "wltc3b"]
DISTANCE_TOLERANCE_PCT = 2.0


def load_available() -> dict:
    cycles = {}
    for key in CYCLES:
        try:
            cycles[key] = get_cycle(key, DATA_DIR)
        except FileNotFoundError as exc:
            print(f"[skip] {key}: {str(exc).splitlines()[0]}")
    return cycles


def validation_table(cycles: dict) -> pd.DataFrame:
    rows = []
    for key, cyc in cycles.items():
        ref_t, ref_d = reference(key)
        stats = cycle_statistics(cyc)
        diff = None if ref_d is None else 100 * (stats["distance_km"] - ref_d) / ref_d
        if diff is None:
            check = "n/a"
        else:
            check = "OK" if abs(diff) <= DISTANCE_TOLERANCE_PCT else "CHECK"
        rows.append({
            "cycle": key,
            "duration_s": stats["duration_s"],
            "ref_duration_s": ref_t,
            "distance_km": stats["distance_km"],
            "ref_distance_km": ref_d,
            "distance_diff_pct": diff,
            "check": check,
        })
    return pd.DataFrame(rows).set_index("cycle")


def statistics_table(cycles: dict) -> pd.DataFrame:
    rows = {key: cycle_statistics(cyc) for key, cyc in cycles.items()}
    return pd.DataFrame(rows).T


def energy_table(vehicle: Vehicle, cycles: dict) -> tuple[pd.DataFrame, dict]:
    rows, results = {}, {}
    for key, cyc in cycles.items():
        res = simulate(vehicle, cyc)
        results[key] = res
        rows[key] = summarize(res, vehicle)
    return pd.DataFrame(rows).T, results


def plot_speed_traces(cycles: dict, path: Path) -> None:
    fig, axes = plt.subplots(len(cycles), 1, figsize=(11, 2.3 * len(cycles)), squeeze=False)
    for ax, (key, cyc) in zip(axes[:, 0], cycles.items()):
        ax.plot(cyc.time_s, cyc.speed_kmh, lw=1)
        ax.set_title(f"{key.upper()} - {describe(key)}", fontsize=9, loc="left")
        ax.set_ylabel("km/h")
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_energy(energy: pd.DataFrame, vehicle: Vehicle, path: Path) -> None:
    labels = [k.upper() for k in energy.index]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].bar(labels, energy["consumption_wh_per_km"])
    axes[0].set_ylabel("Wh/km")
    axes[0].set_title("Battery consumption")
    axes[1].bar(labels, energy["range_km"], color="tab:green")
    axes[1].set_ylabel("km")
    axes[1].set_title(f"Range ({vehicle.battery_usable_kwh:.0f} kWh usable)")
    axes[2].bar(labels, energy["regen_share_pct"], color="tab:orange")
    axes[2].set_ylabel("% of traction energy")
    axes[2].set_title("Regen recovery")
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(vehicle.name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)

    cycles = load_available()
    if not cycles:
        print("No cycles found. Run: python scripts/download_cycles.py")
        return

    print("\n=== 1. Validation against published values ===")
    val = validation_table(cycles)
    print(val.round(3).to_string())

    print("\n=== 2. Cycle statistics ===")
    stats = statistics_table(cycles)
    cols = ["duration_s", "distance_km", "avg_speed_kmh", "max_speed_kmh", "idle_pct",
            "stops", "max_accel_ms2", "max_decel_ms2", "pke_ms2"]
    print(stats[cols].astype(float).round(2).to_string())

    vehicle = Vehicle.from_yaml(ROOT / "configs" / "compact_ev.yaml")
    print(f"\n=== 3. Energy and range: {vehicle.name} ===")
    energy, _ = energy_table(vehicle, cycles)
    cols = ["distance_km", "energy_kwh", "regen_share_pct", "consumption_wh_per_km", "range_km"]
    print(energy[cols].astype(float).round(2).to_string())

    val.to_csv(RESULTS / "step2_validation.csv")
    stats.to_csv(RESULTS / "step2_cycle_stats.csv")
    energy.to_csv(RESULTS / "step2_energy_by_cycle.csv")
    plot_speed_traces(cycles, RESULTS / "step2_speed_traces.png")
    plot_energy(energy, vehicle, RESULTS / "step2_energy_by_cycle.png")
    print(f"\nSaved CSVs and plots to {RESULTS}")


if __name__ == "__main__":
    main()
