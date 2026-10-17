from pathlib import Path

import numpy as np
import pytest

from evrange import EVModel, piecewise_cycle
from evrange.environment import Environment, air_density
from evrange.scenarios import (
    Scenario, build_model, constant_speed_cycle, epa_label_range, evaluate, hilly, sweep,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def setup():
    model = EVModel.from_yaml(ROOT / "configs" / "compact_ev.yaml", ROOT / "configs" / "compact_ev_powertrain.yaml")
    env = Environment.from_yaml(ROOT / "configs" / "environment.yaml")
    cycle = piecewise_cycle(
        "mix", [(0, 0), (10, 0), (20, 50), (80, 50), (95, 100), (215, 100),
                (240, 0), (260, 0), (275, 40), (335, 40), (350, 0), (370, 0)])
    return model, env, cycle


def test_air_density_standard_atmosphere():
    assert air_density(15.0) == pytest.approx(1.225, abs=1e-3)
    assert air_density(-10.0) > air_density(35.0)


def test_hvac_power_shape(setup):
    _, env, _ = setup
    cabin = env.cabin
    assert cabin.electrical_power_w(cabin.setpoint_c) == 0.0
    assert cabin.electrical_power_w(-10.0) > cabin.electrical_power_w(5.0) > 0
    assert cabin.electrical_power_w(38.0) > 0
    assert cabin.electrical_power_w(-60.0) == pytest.approx(cabin.max_electrical_w)


def test_battery_thermal_factors(setup):
    _, env, _ = setup
    bt = env.battery_thermal
    assert bt.resistance_factor(25.0) == pytest.approx(1.0)
    assert bt.resistance_factor(-10.0) > 2.0
    assert bt.capacity_factor_at(25.0) == pytest.approx(1.0)
    assert bt.capacity_factor_at(-20.0) < bt.capacity_factor_at(0.0) < 1.0


def test_baseline_scenario_reproduces_base_model(setup):
    model, env, cycle = setup
    base = model.range_test(cycle)["range_km"]
    assert evaluate(model, env, cycle, Scenario())["range_km"] == pytest.approx(base, rel=1e-9)


def test_payload_reduces_range(setup):
    model, env, cycle = setup
    df = sweep(model, env, cycle, "extra_payload_kg", [0, 300])
    assert df["range_km"].iloc[1] < df["range_km"].iloc[0]


def test_cold_with_heater_worse_than_cold_without(setup):
    model, env, cycle = setup
    mild = evaluate(model, env, cycle, Scenario(ambient_c=21.0))["range_km"]
    cold = evaluate(model, env, cycle, Scenario(ambient_c=-10.0))["range_km"]
    cold_hvac = evaluate(model, env, cycle, Scenario(ambient_c=-10.0, hvac=True))["range_km"]
    assert cold_hvac < cold < mild


def test_hills_have_zero_net_elevation_and_cost_energy(setup):
    model, env, cycle = setup
    h = hilly(cycle, 5.0)
    v, t = h.speed_ms, h.time_s
    climb = np.sum(0.5 * (v[1:] * np.sin(h.grade_rad[1:]) + v[:-1] * np.sin(h.grade_rad[:-1])) * np.diff(t))
    assert abs(climb) < 0.5                                   # metres of net elevation
    flat = evaluate(model, env, cycle, Scenario())["range_km"]
    hills = evaluate(model, env, cycle, Scenario(hill_amplitude_pct=5.0))["range_km"]
    assert hills < flat


def test_aged_battery_scales_range(setup):
    model, env, cycle = setup
    new = evaluate(model, env, cycle, Scenario())["range_km"]
    aged = evaluate(model, env, cycle, Scenario(soh=0.8))["range_km"]
    assert aged / new == pytest.approx(0.8, rel=0.02)


def test_range_falls_at_high_constant_speed(setup):
    model, env, _ = setup
    r = [evaluate(model, env, constant_speed_cycle(v), Scenario())["range_km"] for v in (60, 100, 140)]
    assert r[0] > r[1] > r[2]


def test_scenario_validation():
    with pytest.raises(ValueError):
        Scenario(soh=1.2)
    with pytest.raises(ValueError):
        Scenario(hvac=True)


def test_build_model_does_not_modify_base(setup):
    model, env, _ = setup
    before = (model.vehicle.payload_kg, model.battery.cell_capacity_ah)
    build_model(model, env, Scenario(extra_payload_kg=200, soh=0.7, ambient_c=-5, hvac=True))
    assert (model.vehicle.payload_kg, model.battery.cell_capacity_ah) == before


def test_epa_label_is_70_percent_of_unadjusted(setup):
    model, env, cycle = setup
    lab = epa_label_range(model, env, cycle, constant_speed_cycle(80))
    assert lab["label_range_km"] == pytest.approx(0.7 * lab["unadjusted_range_km"])
