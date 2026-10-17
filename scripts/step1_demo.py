"""Step 1 demo: simulate a simple synthetic cycle and check against a hand calculation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from evrange import Vehicle, piecewise_cycle, simulate, summarize  # noqa: E402
from evrange.vehicle import GRAVITY  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def analytic_constant_speed(vehicle: Vehicle, speed_kmh: float) -> float:
    """Battery consumption (Wh/km) at constant speed on a flat road, by hand."""
    v = speed_kmh / 3.6
    f_aero = 0.5 * vehicle.air_density * vehicle.drag_coefficient * vehicle.frontal_area_m2 * v**2
    f_roll = vehicle.total_mass_kg * GRAVITY * vehicle.rolling_resistance
    p_batt = (f_aero + f_roll) * v / vehicle.drivetrain_efficiency + vehicle.aux_power_w
    return p_batt / v / 3.6  # J/m -> Wh/km


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    vehicle = Vehicle.from_yaml(ROOT / "configs" / "compact_ev.yaml")
    print(f"Vehicle: {vehicle.name}, total mass {vehicle.total_mass_kg:.0f} kg")

    print("\nConstant-speed check (flat road):")
    print(f"{'speed':>10} {'sim Wh/km':>10} {'hand Wh/km':>11} {'error %':>8} {'range km':>9}")
    for speed in (30, 50, 80, 100, 120):
        cyc = piecewise_cycle(f"const_{speed}", [(0, speed), (600, speed)])
        s = summarize(simulate(vehicle, cyc), vehicle)
        hand = analytic_constant_speed(vehicle, speed)
        err = 100 * (s["consumption_wh_per_km"] - hand) / hand
        print(f"{speed:>5} km/h {s['consumption_wh_per_km']:>10.1f} {hand:>11.1f} "
              f"{err:>8.3f} {s['range_km']:>9.0f}")

    cycle = piecewise_cycle(
        "synthetic_mixed",
        [
            (0, 0), (10, 0), (20, 50), (80, 50), (95, 100), (215, 100),
            (240, 0), (260, 0), (275, 40), (335, 40), (350, 0), (370, 0),
        ],
        dt=0.1,
    )
    result = simulate(vehicle, cycle)
    summary = summarize(result, vehicle)

    print(f"\nCycle: {cycle.name}")
    for key, val in summary.items():
        print(f"  {key:<24} {val:10.3f}")

    result.to_csv(RESULTS / "step1_synthetic_timeseries.csv", index=False)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(result["time_s"], result["speed_kmh"])
    axes[0].set_ylabel("Speed [km/h]")
    axes[1].plot(result["time_s"], result["f_aero_n"], label="Aero")
    axes[1].plot(result["time_s"], result["f_roll_n"], label="Rolling")
    axes[1].plot(result["time_s"], result["f_inertia_n"], label="Inertia")
    axes[1].set_ylabel("Force [N]")
    axes[1].legend()
    axes[2].plot(result["time_s"], result["p_batt_w"] / 1000, label="Battery")
    axes[2].plot(result["time_s"], result["p_wheel_w"] / 1000, label="Wheel", alpha=0.7)
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set_ylabel("Power [kW]")
    axes[2].set_xlabel("Time [s]")
    axes[2].legend()
    fig.suptitle(f"{vehicle.name} - {cycle.name}")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "step1_synthetic_cycle.png", dpi=150)
    print(f"\nSaved plot and CSV to {RESULTS}")


if __name__ == "__main__":
    main()
