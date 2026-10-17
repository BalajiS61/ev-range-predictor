import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from evrange.app_logic import (  # noqa: E402
    CONSTANT, TripInputs, fig_breakdown, fig_efficiency_map, fig_range_trace, fig_temperature,
    fig_trip_signals, load_base, make_cycle, ml_range, ml_warnings, run_trip, temperature_curve,
)
from evrange.cycles import piecewise_cycle  # noqa: E402


@pytest.fixture
def ctx():
    base, env = load_base()
    cycles = {"udds": piecewise_cycle("udds", [(0, 0), (10, 0), (25, 45), (90, 45), (105, 0), (125, 0),
                                               (140, 30), (200, 30), (215, 0), (230, 0)])}
    return base, env, cycles


def test_constant_speed_trip(ctx):
    base, env, cycles = ctx
    res = run_trip(base, env, cycles, TripInputs(cycle_key=CONSTANT, constant_speed_kmh=100.0))
    assert 150 < res["range_km"] < 600
    assert res["breakdown_wh_per_km"].sum() > 0
    assert {"run", "range_trace", "model", "cycle"} <= set(res)


def test_heat_pump_helps_in_winter(ctx):
    base, env, cycles = ctx
    ptc = run_trip(base, env, cycles, TripInputs(ambient_c=-10.0, heater_cop=1.0))
    hp = run_trip(base, env, cycles, TripInputs(ambient_c=-10.0, heater_cop=2.5))
    assert hp["hvac_w"] < ptc["hvac_w"]
    assert hp["range_km"] > ptc["range_km"]


def test_breakdown_matches_consumption(ctx):
    base, env, cycles = ctx
    res = run_trip(base, env, cycles, TripInputs(ambient_c=20.0))
    # cell energy per km = sum of sinks; terminal consumption is slightly lower (battery loss)
    assert res["breakdown_wh_per_km"].sum() == pytest.approx(res["consumption_wh_per_km"], rel=0.05)


def test_missing_cycle_raises(ctx):
    _, _, cycles = ctx
    with pytest.raises(KeyError):
        make_cycle(TripInputs(cycle_key="hwfet"), cycles)


def test_ml_range_none_without_model(ctx):
    base, env, cycles = ctx
    assert ml_range(None, base, env, cycles, TripInputs(), 57.0) is None


def test_ml_warnings():
    assert ml_warnings(TripInputs()) == []
    notes = ml_warnings(TripInputs(ambient_c=-5.0, extra_aux_w=900.0, heater_cop=2.5))
    assert len(notes) == 2
    _, env = load_base()
    assert ml_warnings(TripInputs(ambient_c=30.0, heater_cop=2.5), env) == []   # no heating needed
    assert ml_warnings(TripInputs(cycle_key=CONSTANT, constant_speed_kmh=150.0))
    with pytest.raises(ValueError):
        TripInputs(heater_cop=0.0)


def test_temperature_curve_and_figures(ctx):
    base, env, cycles = ctx
    curve = temperature_curve(base, env, cycles, TripInputs(), temps=(-10, 20, 35))
    assert list(curve["ambient_c"]) == [-10.0, 20.0, 35.0]
    assert curve["range_km"].iloc[1] > curve["range_km"].iloc[0]
    res = run_trip(base, env, cycles, TripInputs())
    figs = [fig_trip_signals(res["run"], "t"), fig_breakdown(res["breakdown_wh_per_km"], 10.0),
            fig_range_trace(res["range_trace"]), fig_temperature(curve, 20.0),
            fig_efficiency_map(res["model"], res["run"])]
    assert all(hasattr(f, "savefig") for f in figs)
    assert isinstance(curve, pd.DataFrame)
