from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from evrange import BatteryPack, EVModel, Motor, RegenStrategy, piecewise_cycle
from evrange.ev_model import _regen_torque_for_power

ROOT = Path(__file__).resolve().parents[1]
VEHICLE_YAML = ROOT / "configs" / "compact_ev.yaml"
POWERTRAIN_YAML = ROOT / "configs" / "compact_ev_powertrain.yaml"


@pytest.fixture
def model() -> EVModel:
    return EVModel.from_yaml(VEHICLE_YAML, POWERTRAIN_YAML)


def _mixed_cycle():
    return piecewise_cycle(
        "mix",
        [(0, 0), (10, 0), (20, 50), (80, 50), (95, 100), (215, 100),
         (240, 0), (260, 0), (275, 40), (335, 40), (350, 0), (370, 0)],
    )


def _small_pack(**changes) -> BatteryPack:
    base = BatteryPack(
        cells_series=1, cells_parallel=1, cell_capacity_ah=10.0, cell_r0_ohm=0.01,
        cell_r1_ohm=0.0, cell_tau_s=10.0, cell_v_min=2.8, cell_v_max=4.2,
        soc_min=0.0, soc_max=1.0, ocv_soc=(0.0, 1.0), ocv_v=(3.6, 3.6),
    )
    return replace(base, **changes)


# ---- motor --------------------------------------------------------------------
def test_motor_torque_envelope(model):
    m = model.motor
    w_base = m.peak_power_w / m.peak_torque_nm
    assert m.max_torque(np.array([0.5 * w_base]))[0] == pytest.approx(m.peak_torque_nm)
    assert m.max_torque(np.array([2 * w_base]))[0] == pytest.approx(m.peak_power_w / (2 * w_base))
    assert m.max_torque(np.array([1.1 * m.max_speed_rads]))[0] == 0.0


def test_motor_efficiency_realistic(model):
    _, _, eff = model.motor.efficiency_map()
    peak = np.nanmax(eff)
    assert 0.92 < peak < 0.99
    assert np.nanmin(eff) >= 0.0


def test_motor_idle_has_no_losses(model):
    assert model.motor.losses(np.array([0.0]), np.array([0.0]))[0] == 0.0


def test_regen_torque_hits_power_limit(model):
    m = model.motor
    w = np.array([300.0, 800.0])
    t = _regen_torque_for_power(m, w, 50_000.0)
    assert np.all(t < 0)
    assert np.allclose(m.electrical_power(t, w), -50_000.0, rtol=1e-6)


def test_invalid_motor_rejected():
    with pytest.raises(ValueError):
        Motor(0, 1, 1, 0, 0, 0, 0)


# ---- battery ------------------------------------------------------------------
def test_constant_current_discharge_soc():
    pack = _small_pack()
    t = np.arange(0, 1801.0)
    i_target = 10.0                                   # 1C
    # terminal power for 1C with flat OCV and R0: P = V I - R0 I^2
    p = np.full_like(t, 3.6 * i_target - 0.01 * i_target**2)
    res = pack.run(p, t, soc0=1.0)
    assert np.allclose(res["current_a"], i_target, rtol=1e-9)
    assert res["soc"][-1] == pytest.approx(0.5, rel=1e-9)
    assert np.allclose(res["p_loss_w"], 0.01 * i_target**2)


def test_zero_power_keeps_state():
    pack = _small_pack()
    t = np.arange(0, 100.0)
    res = pack.run(np.zeros_like(t), t, soc0=0.6)
    assert np.allclose(res["soc"], 0.6)


def test_voltage_limits_cap_current():
    pack = _small_pack()
    t = np.arange(0, 3.0)
    big = pack.run(np.full_like(t, 1e6), t, soc0=0.9)       # impossible discharge
    assert big["limited"].all()
    assert np.all(big["v_terminal"] >= pack.v_min - 1e-9)
    charge = pack.run(np.full_like(t, -1e6), t, soc0=0.9)   # impossible charge
    assert charge["limited"].all()
    assert np.all(charge["v_terminal"] <= pack.v_max + 1e-9)


def test_usable_energy(model):
    b = model.battery
    assert 50 < b.usable_energy_kwh < b.nominal_energy_kwh < 70
    flat = _small_pack()
    assert flat.energy_kwh(0, 1) == pytest.approx(3.6 * 10 / 1000)


def test_invalid_ocv_rejected():
    with pytest.raises(ValueError):
        _small_pack(ocv_soc=(0.0, 0.5), ocv_v=(3.6, 3.5))


# ---- full model -----------------------------------------------------------------
def test_regen_fade():
    r = RegenStrategy(max_power_w=1e4, fade_start_kmh=10, fade_end_kmh=5)
    assert np.allclose(r.fade(np.array([0, 5, 7.5, 10, 50]) / 3.6), [0, 0, 0.5, 1, 1])


def test_energy_balance_closes(model):
    df = model.run_cycle(_mixed_cycle(), soc0=0.5)
    e = model.energy_breakdown(df)
    sinks = (e["battery_loss_kwh"] + e["aux_kwh"] + e["motor_loss_kwh"] + e["gear_loss_kwh"]
             + e["aero_kwh"] + e["rolling_kwh"] + e["grade_kwh"] + e["inertia_net_kwh"]
             + e["friction_brake_kwh"] + e["regen_curtailed_kwh"]
             - e["unmet_traction_kwh"] - e["battery_shortfall_kwh"])
    assert e["battery_chemical_kwh"] == pytest.approx(sinks, rel=1e-6)


def test_detailed_model_costs_more_than_wheel_energy(model):
    s = model.summarize(model.run_cycle(_mixed_cycle(), soc0=0.5))
    wheel = s["aero_kwh"] + s["rolling_kwh"]
    assert s["battery_terminal_kwh"] > wheel
    assert 80 < s["motor_eff_motoring_pct"] < 97
    assert s["torque_limited_samples"] == 0


def test_weak_motor_cannot_follow_trace(model):
    weak = EVModel(model.vehicle, model.drivetrain,
                   replace(model.motor, peak_power_w=10_000.0), model.battery, model.regen)
    s = weak.summarize(weak.run_cycle(_mixed_cycle(), soc0=0.5))
    assert s["unmet_traction_kwh"] > 0
    assert s["torque_limited_samples"] > 0


def test_range_test_consistent_with_energy(model):
    r = model.range_test(_mixed_cycle())
    usable = model.battery.usable_energy_kwh
    assert r["energy_used_kwh"] == pytest.approx(usable, rel=0.03)
    assert r["range_km"] == pytest.approx(1000 * r["energy_used_kwh"] / r["consumption_wh_per_km"])
    assert r["trace"]["soc"].iloc[-1] == pytest.approx(model.battery.soc_min)


def test_more_aux_load_less_range(model):
    heavy = EVModel(model.vehicle.with_updates(aux_power_w=3000.0), model.drivetrain,
                    model.motor, model.battery, model.regen)
    assert heavy.range_test(_mixed_cycle())["range_km"] < model.range_test(_mixed_cycle())["range_km"]
