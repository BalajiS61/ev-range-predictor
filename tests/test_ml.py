from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evrange import EVModel, piecewise_cycle
from evrange.environment import Environment
from evrange.ml import (
    ALL_FEATURES, RAW_FEATURES, TARGET, RangePredictor, compare_models, generate_dataset,
    leave_one_family_out, make_models, scale_cycle, trip_features,
)
from evrange.scenarios import Scenario

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def small_data():
    model = EVModel.from_yaml(ROOT / "configs" / "compact_ev.yaml", ROOT / "configs" / "compact_ev_powertrain.yaml")
    env = Environment.from_yaml(ROOT / "configs" / "environment.yaml")
    cycles = {
        "city": piecewise_cycle("city", [(0, 0), (10, 0), (25, 40), (60, 40), (75, 0), (95, 0),
                                         (110, 30), (150, 30), (165, 0), (180, 0)]),
        "road": piecewise_cycle("road", [(0, 0), (30, 90), (300, 90), (330, 60), (500, 60), (540, 0)]),
    }
    df = generate_dataset(model, env, cycles, n_samples=160, seed=1)
    return model, env, cycles, df


def test_scale_cycle_scales_distance():
    c = piecewise_cycle("c", [(0, 0), (10, 36), (20, 0)])
    assert scale_cycle(c, 1.5).distance_km == pytest.approx(1.5 * c.distance_km)


def test_trip_features_complete(small_data):
    model, env, cycles, _ = small_data
    f = trip_features(cycles["city"], Scenario(ambient_c=-5.0, hvac=True), env, model.vehicle.aux_power_w)
    assert set(ALL_FEATURES) <= set(f)
    assert f["hvac_wh_per_km"] > 0
    assert all(np.isfinite(v) for v in f.values())


def test_dataset_is_valid_and_reproducible(small_data):
    model, env, cycles, df = small_data
    assert len(df) == 160
    assert df[ALL_FEATURES + [TARGET, "range_km"]].notna().all().all()
    assert (df[TARGET] > 50).all() and (df[TARGET] < 600).all()
    assert (df[TARGET] >= df["terminal_consumption_wh_per_km"]).all()   # cell energy includes battery losses
    again = generate_dataset(model, env, cycles, n_samples=5, seed=1)
    assert np.allclose(again[TARGET], df[TARGET].iloc[:5])


def test_physics_features_help_linear_model(small_data):
    *_, df = small_data
    specs = [s for s in make_models() if "regression" in s.name]
    metrics, _, _ = compare_models(df, specs, cv_folds=3)
    raw = metrics.loc["Linear regression (raw)", "whkm_mae"]
    phys = metrics.loc["Ridge regression (physics features)", "whkm_mae"]
    assert phys < raw


def test_tree_model_learns_something(small_data):
    *_, df = small_data
    spec = [s for s in make_models() if s.name == "Gradient boosting (physics features)"]
    metrics, _, _ = compare_models(df, spec, cv_folds=3)
    assert metrics["whkm_r2"].iloc[0] > 0.8


def test_leave_one_family_out_shape(small_data):
    *_, df = small_data
    spec = make_models()[1]
    out = leave_one_family_out(df, spec)
    assert set(out.index) == set(df["cycle_family"])


def test_predictor_save_load_roundtrip(small_data, tmp_path):
    *_, df = small_data
    spec = make_models()[1]
    est = spec.estimator.fit(df[spec.features], df[TARGET])
    p = RangePredictor(est, spec.features, {"model": spec.name})
    path = tmp_path / "m.joblib"
    p.save(path)
    q = RangePredictor.load(path)
    X = df.iloc[:10]
    assert np.allclose(p.predict_consumption(X), q.predict_consumption(X))
    assert np.allclose(q.predict_range(X, X["usable_kwh"]), 1000 * X["usable_kwh"] / q.predict_consumption(X))
    assert path.with_suffix(".json").exists()
    with pytest.raises(KeyError):
        q.predict_consumption(pd.DataFrame({"ambient_c": [1.0]}))
    assert set(RAW_FEATURES) < set(ALL_FEATURES)
