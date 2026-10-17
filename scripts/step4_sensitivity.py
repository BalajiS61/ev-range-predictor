"""Step 4: real-world range sensitivity - temperature/HVAC, speed, payload, hills, ageing."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from evrange import EVModel, get_cycle  # noqa: E402
from evrange.environment import Environment  # noqa: E402
from evrange.scenarios import (  # noqa: E402
    Scenario, constant_speed_cycle, epa_label_range, evaluate, sweep,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "cycles"
RESULTS = ROOT / "results"

SCENARIOS = [
    Scenario("Mild day 25 C", ambient_c=25.0, hvac=True),
    Scenario("Summer 35 C + A/C", ambient_c=35.0, hvac=True),
    Scenario("Winter -10 C + heater", ambient_c=-10.0, hvac=True, extra_payload_kg=75.0),
    Scenario("Family trip +300 kg, 4% hills", ambient_c=25.0, hvac=True,
             extra_payload_kg=300.0, hill_amplitude_pct=4.0),
    Scenario("Aged battery SOH 80%", ambient_c=25.0, hvac=True, soh=0.8, resistance_growth=1.5),
    Scenario("Worst case: winter, loaded, hills, aged", ambient_c=-10.0, hvac=True,
             extra_payload_kg=300.0, hill_amplitude_pct=4.0, soh=0.8, resistance_growth=1.5),
]


def progress(msg: str) -> None:
    print(f"  ... {msg}", flush=True)


def temperature_sweep(model, env, cycles) -> pd.DataFrame:
    temps = np.arange(-20, 41, 5)
    frames = []
    for key, cyc in cycles.items():
        for hvac in (False, True):
            progress(f"temperature sweep {key.upper()} HVAC {'on' if hvac else 'off'}")
            df = sweep(model, env, cyc, "ambient_c", [float(t) for t in temps], Scenario(hvac=hvac, ambient_c=25.0))
            df["hvac_on"] = hvac
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def speed_sweep(model, env) -> pd.DataFrame:
    progress("constant-speed sweep")
    rows = []
    for v in range(40, 141, 10):
        row = evaluate(model, env, constant_speed_cycle(v), Scenario(ambient_c=25.0, hvac=True))
        row["speed_kmh"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def simple_sweeps(model, env, cycles) -> dict:
    out = {}
    for field, values in [
        ("extra_payload_kg", [0.0, 75.0, 150.0, 225.0, 300.0, 375.0, 450.0]),
        ("hill_amplitude_pct", [0.0, 2.0, 4.0, 6.0, 8.0]),
        ("soh", [1.0, 0.9, 0.8, 0.7]),
    ]:
        frames = []
        for key, cyc in cycles.items():
            progress(f"{field} sweep {key.upper()}")
            df = sweep(model, env, cyc, field, values)
            frames.append(df)
        out[field] = pd.concat(frames, ignore_index=True)
    return out


def scenario_table(model, env, cycles) -> pd.DataFrame:
    rows = []
    for sc in SCENARIOS:
        progress(f"scenario '{sc.name}'")
        for key, cyc in cycles.items():
            r = evaluate(model, env, cyc, sc)
            r["cycle"] = key
            rows.append(r)
    df = pd.DataFrame(rows)
    table = df.pivot(index="scenario", columns="cycle", values="range_km").reindex([s.name for s in SCENARIOS])
    base = table.iloc[0]
    for col in list(table.columns):
        table[f"{col}_vs_mild_pct"] = 100 * (table[col] - base[col]) / base[col]
    hvac = df.groupby("scenario")["hvac_w"].first().reindex(table.index)
    table.insert(0, "hvac_w", hvac)
    return table


# ---- plots ---------------------------------------------------------------------
def plot_temperature(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for (cycle, hvac), g in df.groupby(["cycle", "hvac_on"]):
        style = "-" if hvac else "--"
        ax.plot(g["ambient_c"], g["range_km"], style, marker="o", ms=3,
                label=f"{cycle.upper()} - {'HVAC on' if hvac else 'HVAC off (battery + air only)'}")
    ax.axvspan(18, 24, color="green", alpha=0.08, label="HVAC deadband")
    ax.set_xlabel("Ambient temperature [C]")
    ax.set_ylabel("Range [km]")
    ax.set_title("Range vs ambient temperature")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_speed(df: pd.DataFrame, path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(df["speed_kmh"], df["range_km"], "o-", color="tab:green", label="Range")
    ax1.set_xlabel("Constant speed [km/h]")
    ax1.set_ylabel("Range [km]", color="tab:green")
    ax2 = ax1.twinx()
    ax2.plot(df["speed_kmh"], df["consumption_wh_per_km"], "s--", color="tab:blue", label="Consumption")
    ax2.set_ylabel("Battery consumption [Wh/km]", color="tab:blue")
    ax1.set_title("Steady-speed range (25 C, flat road)")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_simple_sweeps(sweeps: dict, path: Path) -> None:
    titles = {"extra_payload_kg": ("Extra payload [kg]", "Payload"),
              "hill_amplitude_pct": ("Hill grade amplitude [%] (zero net elevation)", "Hills"),
              "soh": ("Battery state of health [-]", "Battery ageing")}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (field, df) in zip(axes, sweeps.items()):
        for cycle, g in df.groupby("cycle"):
            ax.plot(g[field], g["range_km"], "o-", label=cycle.upper())
        xlabel, title = titles[field]
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Range [km]")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[2].invert_xaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scenarios(table: pd.DataFrame, cycles: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    y = np.arange(len(table))
    h = 0.8 / len(cycles)
    for i, cyc in enumerate(cycles):
        ax.barh(y + i * h, table[cyc], height=h, label=cyc.upper())
    ax.set_yticks(y + h * (len(cycles) - 1) / 2)
    ax.set_yticklabels(table.index)
    ax.invert_yaxis()
    ax.set_xlabel("Range [km]")
    ax.set_title("Range under real-world scenarios")
    ax.grid(axis="x", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    model = EVModel.from_yaml(ROOT / "configs" / "compact_ev.yaml", ROOT / "configs" / "compact_ev_powertrain.yaml")
    env = Environment.from_yaml(ROOT / "configs" / "environment.yaml")

    try:
        cycles = {"udds": get_cycle("udds", DATA_DIR), "hwfet": get_cycle("hwfet", DATA_DIR)}
    except FileNotFoundError:
        print("UDDS/HWFET not found. Run: python scripts/download_cycles.py")
        return
    scenario_cycles = dict(cycles, motorway_120=constant_speed_cycle(120.0))

    print("Running sweeps (about 1 minute):")
    temp = temperature_sweep(model, env, cycles)
    speed = speed_sweep(model, env)
    sweeps = simple_sweeps(model, env, cycles)
    table = scenario_table(model, env, scenario_cycles)
    label = epa_label_range(model, env, cycles["udds"], cycles["hwfet"])

    print("\n=== 1. EPA-style label estimate (2-cycle, 55% city / 45% highway, x0.70) ===")
    for k, v in label.items():
        print(f"  {k:<22} {v:8.1f}")

    print("\n=== 2. Range [km] vs ambient temperature ===")
    pivot = temp.pivot_table(index="ambient_c", columns=["cycle", "hvac_on"], values="range_km")
    pivot.columns = [f"{c.upper()}_{'hvac' if h else 'nohvac'}" for c, h in pivot.columns]
    hv = temp[temp["hvac_on"]].drop_duplicates("ambient_c").set_index("ambient_c")["hvac_w"]
    pivot.insert(0, "hvac_w", hv)
    print(pivot.round(0).to_string())

    print("\n=== 3. Constant-speed range (25 C) ===")
    print(speed[["speed_kmh", "consumption_wh_per_km", "range_km"]].round(1).to_string(index=False))

    print("\n=== 4. One-at-a-time sweeps [range km] ===")
    for field, df in sweeps.items():
        print(f"\n{field}:")
        print(df.pivot(index=field, columns="cycle", values="range_km").round(0).to_string())

    print("\n=== 5. Real-world scenarios [range km] ===")
    print(table.round(0).to_string())

    temp.to_csv(RESULTS / "step4_temperature_sweep.csv", index=False)
    speed.to_csv(RESULTS / "step4_speed_sweep.csv", index=False)
    for field, df in sweeps.items():
        df.to_csv(RESULTS / f"step4_{field}_sweep.csv", index=False)
    table.to_csv(RESULTS / "step4_scenarios.csv")
    pd.Series(label).to_csv(RESULTS / "step4_epa_label_estimate.csv", header=["value"])

    plot_temperature(temp, RESULTS / "step4_range_vs_temperature.png")
    plot_speed(speed, RESULTS / "step4_range_vs_speed.png")
    plot_simple_sweeps(sweeps, RESULTS / "step4_payload_hills_ageing.png")
    plot_scenarios(table, list(scenario_cycles), RESULTS / "step4_scenarios.png")
    print(f"\nSaved CSVs and plots to {RESULTS}")


if __name__ == "__main__":
    main()
