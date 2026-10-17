import numpy as np
import pytest

from evrange import DriveCycle, Vehicle, piecewise_cycle, simulate, summarize
from evrange.vehicle import GRAVITY


@pytest.fixture
def vehicle() -> Vehicle:
    return Vehicle(
        name="test", mass_kg=1500, payload_kg=0, drag_coefficient=0.3,
        frontal_area_m2=2.2, rolling_resistance=0.01, rotational_mass_factor=0.05,
        drivetrain_efficiency=0.9, regen_efficiency=0.8, regen_fraction=1.0,
        aux_power_w=300, battery_usable_kwh=50,
    )


def test_constant_speed_forces(vehicle):
    v = 25.0
    cyc = DriveCycle("c", np.arange(0, 100.0), np.full(100, v))
    res = simulate(vehicle, cyc)
    assert np.allclose(res["f_aero_n"], 0.5 * 1.225 * 0.3 * 2.2 * v**2)
    assert np.allclose(res["f_roll_n"], 1500 * GRAVITY * 0.01)
    assert np.allclose(res["f_inertia_n"], 0.0)


def test_standstill_only_aux(vehicle):
    cyc = DriveCycle("idle", np.arange(0, 60.0), np.zeros(60))
    res = simulate(vehicle, cyc)
    assert np.allclose(res["p_batt_w"], vehicle.aux_power_w)
    assert summarize(res, vehicle)["distance_km"] == 0.0


def test_kinetic_energy_round_trip():
    lossless = Vehicle(
        name="ideal", mass_kg=1000, payload_kg=0, drag_coefficient=0.0,
        frontal_area_m2=1.0, rolling_resistance=0.0, rotational_mass_factor=0.0,
        drivetrain_efficiency=1.0, regen_efficiency=1.0, regen_fraction=1.0,
        aux_power_w=0.0, battery_usable_kwh=10,
    )
    cyc = piecewise_cycle("updown", [(0, 0), (20, 72), (40, 0)], dt=0.01)
    res = simulate(lossless, cyc)
    peak_ke_wh = 0.5 * 1000 * 20.0**2 / 3600
    assert abs(res["energy_batt_wh"].iloc[-1]) < 0.01 * peak_ke_wh


def test_uphill_costs_more(vehicle):
    t = np.arange(0, 300.0)
    v = np.full_like(t, 15.0)
    flat = summarize(simulate(vehicle, DriveCycle("f", t, v)), vehicle)
    hill = summarize(simulate(vehicle, DriveCycle("h", t, v, np.full_like(t, 0.03))), vehicle)
    assert hill["energy_kwh"] > flat["energy_kwh"]


def test_invalid_parameters_rejected(vehicle):
    with pytest.raises(ValueError):
        vehicle.with_updates(regen_fraction=1.5)
    with pytest.raises(ValueError):
        vehicle.with_updates(mass_kg=-10)
