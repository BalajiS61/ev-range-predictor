"""Command-line range prediction (physics model, plus ML if trained).

Example:
  python scripts/predict_range.py --cycle udds --ambient -5 --payload 150
  python scripts/predict_range.py --cycle constant --speed 120 --ambient 30
"""

from __future__ import annotations

import argparse

from evrange.app_logic import (
    CONSTANT, CYCLE_KEYS, TripInputs, available_cycles, load_base, load_predictor, ml_range,
    ml_warnings, run_trip,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cycle", default="udds", choices=CYCLE_KEYS + [CONSTANT])
    p.add_argument("--speed", type=float, default=100.0, help="km/h, for --cycle constant")
    p.add_argument("--ambient", type=float, default=20.0, help="ambient temperature [C]")
    p.add_argument("--no-hvac", action="store_true", help="cabin heating/cooling off")
    p.add_argument("--heat-pump", action="store_true", help="heat pump (COP 2.5) instead of PTC heater")
    p.add_argument("--payload", type=float, default=0.0, help="extra payload [kg]")
    p.add_argument("--hills", type=float, default=0.0, help="hill grade amplitude [%%]")
    p.add_argument("--soh", type=float, default=1.0, help="battery state of health (0.7-1.0)")
    p.add_argument("--aux", type=float, default=0.0, help="extra electrical load [W]")
    a = p.parse_args()

    inputs = TripInputs(
        cycle_key=a.cycle, constant_speed_kmh=a.speed, ambient_c=a.ambient, hvac=not a.no_hvac,
        heater_cop=2.5 if a.heat_pump else 1.0, extra_payload_kg=a.payload,
        hill_amplitude_pct=a.hills, soh=a.soh, extra_aux_w=a.aux,
    )
    base, env = load_base()
    cycles = available_cycles()
    res = run_trip(base, env, cycles, inputs)
    ml = ml_range(load_predictor(), base, env, cycles, inputs, res["usable_kwh"])

    print(f"Trip: {a.cycle}{f' at {a.speed:g} km/h' if a.cycle == CONSTANT else ''}, {a.ambient:g} C, "
          f"HVAC {'off' if a.no_hvac else ('heat pump' if a.heat_pump else 'on')}, "
          f"+{a.payload:g} kg, hills {a.hills:g}%, SOH {100 * a.soh:.0f}%")
    print(f"  Physics range      {res['range_km']:7.1f} km")
    print(f"  Consumption        {res['consumption_wh_per_km']:7.1f} Wh/km")
    print(f"  HVAC power         {res['hvac_w']:7.0f} W")
    print(f"  Usable energy      {res['usable_kwh']:7.1f} kWh")
    if ml is None:
        print("  ML range           (no trained model - run scripts/step5_ml.py)")
    else:
        print(f"  ML range           {ml:7.1f} km  ({ml - res['range_km']:+.1f} km vs physics)")
        for note in ml_warnings(inputs, env):
            print(f"  ML caution: {note}")


if __name__ == "__main__":
    main()
