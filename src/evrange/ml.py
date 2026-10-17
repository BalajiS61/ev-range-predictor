"""ML surrogate for the physics model: synthetic trip dataset, model comparison, predictor.

The physics model (Step 3/4) labels each synthetic trip with its cell-level
consumption [Wh/km]: energy drawn from the cells per km, including battery
I^2R losses, averaged over passes at high, mid and low SOC. ML models learn it
from features known before a trip (weather, load, route/cycle descriptors,
battery health); range = usable cell energy / predicted consumption, which is
exactly how the full-depletion range test ends (SOC window used up).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from evrange.cycle_stats import cycle_statistics
from evrange.cycles import DriveCycle
from evrange.environment import Environment
from evrange.ev_model import EVModel
from evrange.scenarios import Scenario, build_model, constant_speed_cycle, hilly, hvac_power_w

TARGET = "cell_consumption_wh_per_km"
LABEL_SOCS = (0.85, 0.50, 0.15)

SCENARIO_FEATURES = [
    "ambient_c", "hvac_on", "extra_payload_kg", "hill_amplitude_pct", "soh", "extra_aux_w",
]
CYCLE_FEATURES = [
    "avg_speed_kmh", "avg_moving_speed_kmh", "max_speed_kmh", "idle_pct",
    "stops_per_km", "pke_ms2", "mean_pos_accel_ms2", "cruise_pct",
]
RAW_FEATURES = SCENARIO_FEATURES + CYCLE_FEATURES
PHYSICS_FEATURES = ["aero_v2_ms2", "hvac_wh_per_km", "aux_wh_per_km"]
ALL_FEATURES = RAW_FEATURES + PHYSICS_FEATURES


# ---- features ----------------------------------------------------------------------
def scale_cycle(cycle: DriveCycle, factor: float) -> DriveCycle:
    """Scale speeds (and therefore accelerations) by a factor, same duration."""
    return DriveCycle(f"{cycle.name}_x{factor:.2f}", cycle.time_s, cycle.speed_ms * factor, cycle.grade_rad)


def trip_features(cycle: DriveCycle, scenario: Scenario, env: Environment, base_aux_w: float) -> dict:
    """Features available before the trip: scenario settings + cycle descriptors."""
    s = cycle_statistics(cycle)
    v, t = cycle.speed_ms, cycle.time_s
    v_mid, dt = 0.5 * (v[1:] + v[:-1]), np.diff(t)
    aero_v2 = float(np.sum(v_mid**3 * dt) / np.sum(v_mid * dt))      # distance-weighted mean v^2
    hvac_w = hvac_power_w(env, scenario)
    feats = {
        "ambient_c": 25.0 if scenario.ambient_c is None else scenario.ambient_c,
        "hvac_on": float(scenario.hvac),
        "extra_payload_kg": scenario.extra_payload_kg,
        "hill_amplitude_pct": scenario.hill_amplitude_pct,
        "soh": scenario.soh,
        "extra_aux_w": scenario.extra_aux_w,
        **{k: s[k] for k in CYCLE_FEATURES},
        "aero_v2_ms2": aero_v2,
        "hvac_wh_per_km": hvac_w / s["avg_speed_kmh"],
        "aux_wh_per_km": (base_aux_w + scenario.extra_aux_w) / s["avg_speed_kmh"],
    }
    return feats


# ---- synthetic dataset ---------------------------------------------------------------
def sample_scenario(rng: np.random.Generator) -> Scenario:
    soh = float(rng.uniform(0.7, 1.0))
    return Scenario(
        name="random",
        ambient_c=float(rng.uniform(-20.0, 40.0)),
        hvac=bool(rng.random() < 0.85),
        extra_payload_kg=float(rng.uniform(0.0, 450.0)),
        hill_amplitude_pct=0.0 if rng.random() < 0.5 else float(rng.uniform(0.0, 8.0)),
        soh=soh,
        resistance_growth=1.0 + 2.5 * (1.0 - soh),   # resistance rises as capacity fades
        extra_aux_w=float(rng.uniform(0.0, 400.0)),
    )


def label_trip(base: EVModel, env: Environment, cycle: DriveCycle, sc: Scenario, family: str = "") -> dict:
    """Features + physics labels for one trip (mean of passes at LABEL_SOCS)."""
    model = build_model(base, env, sc)
    trip = hilly(cycle, sc.hill_amplitude_pct, sc.hill_wavelength_km)
    cell, terminal, misses = [], [], 0
    for soc0 in LABEL_SOCS:
        summary = model.summarize(model.run_cycle(trip, soc0=soc0))
        cell.append(1000 * summary["battery_chemical_kwh"] / summary["distance_km"])
        terminal.append(summary["consumption_wh_per_km"])
        misses = max(misses, summary["torque_limited_samples"])
    usable = model.battery.usable_energy_kwh
    row = trip_features(cycle, sc, env, base.vehicle.aux_power_w)
    row.update({
        "cycle_family": family or cycle.name,
        "usable_kwh": usable,
        TARGET: float(np.mean(cell)),
        "terminal_consumption_wh_per_km": float(np.mean(terminal)),
        "range_km": 1000.0 * usable / float(np.mean(cell)),
        "trace_misses": misses,
    })
    return row


def generate_dataset(
    base: EVModel,
    env: Environment,
    base_cycles: dict[str, DriveCycle],
    n_samples: int,
    seed: int = 42,
    constant_speed_share: float = 0.25,
    progress_every: int = 0,
) -> pd.DataFrame:
    """Label random trips with the physics model."""
    rng = np.random.default_rng(seed)
    keys = list(base_cycles)
    rows = []
    for i in range(n_samples):
        if rng.random() < constant_speed_share:
            family = "constant"
            cycle = constant_speed_cycle(float(rng.uniform(30.0, 140.0)), duration_s=900.0)
        else:
            family = keys[int(rng.integers(len(keys)))]
            cycle = scale_cycle(base_cycles[family], float(rng.uniform(0.7, 1.2)))
        sc = sample_scenario(rng)
        rows.append(label_trip(base, env, cycle, sc, family))
        rows[-1].update({"seed": seed, "sample": i})
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  ... generated {i + 1}/{n_samples} trips", flush=True)
    return pd.DataFrame(rows)


# ---- models -----------------------------------------------------------------------
@dataclass
class ModelSpec:
    name: str
    estimator: Pipeline
    features: list[str] = field(default_factory=list)


def make_models(seed: int = 42) -> list[ModelSpec]:
    return [
        ModelSpec("Linear regression (raw)", make_pipeline(StandardScaler(), LinearRegression()), RAW_FEATURES),
        ModelSpec("Ridge regression (physics features)",
                  make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))), ALL_FEATURES),
        ModelSpec("Random forest (raw)",
                  make_pipeline(RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                                      n_jobs=-1, random_state=seed)), RAW_FEATURES),
        ModelSpec("Gradient boosting (raw)",
                  make_pipeline(HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05,
                                                              random_state=seed)), RAW_FEATURES),
        ModelSpec("Gradient boosting (physics features)",
                  make_pipeline(HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05,
                                                              random_state=seed)), ALL_FEATURES),
    ]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        f"{prefix}mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}mape_pct": float(100 * np.mean(np.abs((y_pred - y_true) / y_true))),
        f"{prefix}r2": float(r2_score(y_true, y_pred)),
    }


def compare_models(df: pd.DataFrame, specs: list[ModelSpec], test_size: float = 0.2,
                   seed: int = 42, cv_folds: int = 5) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Hold-out test metrics (consumption and derived range) + k-fold CV R^2."""
    train, test = train_test_split(df, test_size=test_size, random_state=seed)
    rows, fitted = [], {}
    preds = test[["usable_kwh", TARGET, "range_km", "ambient_c", "cycle_family"]].copy()
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    for spec in specs:
        est = clone(spec.estimator).fit(train[spec.features], train[TARGET])
        y_hat = est.predict(test[spec.features])
        range_hat = 1000.0 * test["usable_kwh"].to_numpy() / y_hat
        cv = cross_val_score(clone(spec.estimator), train[spec.features], train[TARGET], cv=kfold, scoring="r2")
        rows.append({
            "model": spec.name,
            **regression_metrics(test[TARGET], y_hat, "whkm_"),
            **regression_metrics(test["range_km"], range_hat, "range_"),
            "cv_r2_mean": float(cv.mean()),
            "cv_r2_std": float(cv.std()),
        })
        fitted[spec.name] = (est, spec.features)
        preds[f"pred_{spec.name}"] = y_hat
    return pd.DataFrame(rows).set_index("model"), fitted, preds


def leave_one_family_out(df: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    """Train without one cycle family, test on it: how well does the model extrapolate?"""
    rows = []
    for fam in sorted(df["cycle_family"].unique()):
        train, test = df[df["cycle_family"] != fam], df[df["cycle_family"] == fam]
        est = clone(spec.estimator).fit(train[spec.features], train[TARGET])
        y_hat = est.predict(test[spec.features])
        rows.append({"held_out_family": fam, "n_test": len(test),
                     **regression_metrics(test[TARGET], y_hat, "whkm_")})
    return pd.DataFrame(rows).set_index("held_out_family")


# ---- deployable predictor ---------------------------------------------------------
class RangePredictor:
    """Trained consumption model + feature list; predicts Wh/km and range."""

    def __init__(self, estimator, features: list[str], metadata: dict | None = None) -> None:
        self.estimator = estimator
        self.features = list(features)
        self.metadata = metadata or {}

    def predict_consumption(self, X: pd.DataFrame) -> np.ndarray:
        missing = [f for f in self.features if f not in X.columns]
        if missing:
            raise KeyError(f"Missing features: {missing}")
        return self.estimator.predict(X[self.features])

    def predict_range(self, X: pd.DataFrame, usable_kwh) -> np.ndarray:
        return 1000.0 * np.asarray(usable_kwh, dtype=float) / self.predict_consumption(X)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"estimator": self.estimator, "features": self.features, "metadata": self.metadata}, path)
        path.with_suffix(".json").write_text(json.dumps({"features": self.features, **self.metadata}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "RangePredictor":
        obj = joblib.load(path)
        return cls(obj["estimator"], obj["features"], obj.get("metadata"))
