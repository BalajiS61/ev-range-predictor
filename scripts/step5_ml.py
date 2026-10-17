"""Step 5: ML range predictor trained on physics-model trips."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

from evrange import EVModel, get_cycle  # noqa: E402
from evrange.environment import Environment  # noqa: E402
from evrange.ml import (  # noqa: E402
    TARGET, RangePredictor, compare_models, generate_dataset, label_trip, leave_one_family_out,
    make_models, sample_scenario, scale_cycle, trip_features,
)
from evrange.scenarios import Scenario, build_model, hilly  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "cycles"
RESULTS = ROOT / "results"
MODEL_PATH = RESULTS / "models" / "range_model.joblib"
BASE_CYCLES = ["udds", "hwfet", "nedc", "midc", "wltc3b"]


def load_cycles() -> dict:
    cycles = {}
    for key in BASE_CYCLES:
        try:
            cycles[key] = get_cycle(key, DATA_DIR)
        except FileNotFoundError:
            pass
    return cycles


def label_check(model: EVModel, env: Environment, cycles: dict, n: int = 12, seed: int = 7) -> pd.DataFrame:
    """Fast training label vs the full-depletion range test on random trips."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        key = list(cycles)[int(rng.integers(len(cycles)))]
        cyc = scale_cycle(cycles[key], float(rng.uniform(0.7, 1.2)))
        sc = sample_scenario(rng)
        label = label_trip(model, env, cyc, sc, key)
        full = build_model(model, env, sc).range_test(hilly(cyc, sc.hill_amplitude_pct, sc.hill_wavelength_km))
        rows.append({"cycle": key, "ambient_c": sc.ambient_c, "label_range_km": label["range_km"],
                     "range_test_km": full["range_km"],
                     "diff_pct": 100 * (label["range_km"] - full["range_km"]) / full["range_km"]})
    return pd.DataFrame(rows)


def plot_parity(preds: pd.DataFrame, name: str, path: Path) -> None:
    y, y_hat = preds[TARGET], preds[f"pred_{name}"]
    r, r_hat = preds["range_km"], 1000 * preds["usable_kwh"] / y_hat
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (a, b, unit) in zip(axes, [(y, y_hat, "Wh/km"), (r, r_hat, "km")]):
        sc = ax.scatter(a, b, c=preds["ambient_c"], cmap="coolwarm", s=8, alpha=0.7)
        lo, hi = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel(f"Physics model [{unit}]")
        ax.set_ylabel(f"ML prediction [{unit}]")
        ax.grid(alpha=0.3)
    axes[0].set_title("Consumption")
    axes[1].set_title("Range")
    fig.colorbar(sc, ax=axes, label="Ambient temperature [C]")
    fig.suptitle(f"{name}: test set")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(metrics: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    names = [n.replace(" (", "\n(") for n in metrics.index]
    axes[0].barh(names, metrics["whkm_mae"])
    axes[0].set_xlabel("MAE [Wh/km]")
    axes[1].barh(names, metrics["range_mape_pct"], color="tab:green")
    axes[1].set_xlabel("Range MAPE [%]")
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Model comparison on held-out trips")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(imp: pd.Series, name: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.sort_values().plot.barh(ax=ax)
    ax.set_xlabel("Permutation importance (increase in MAE, Wh/km)")
    ax.set_title(f"What drives consumption? ({name})")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_vs_temperature(preds: pd.DataFrame, names: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.arange(-20, 45, 5)
    centers = 0.5 * (bins[1:] + bins[:-1])
    for name in names:
        err = np.abs(preds[f"pred_{name}"] - preds[TARGET])
        mae = err.groupby(pd.cut(preds["ambient_c"], bins), observed=False).mean()
        ax.plot(centers, mae.to_numpy(), "o-", label=name)
    ax.set_xlabel("Ambient temperature [C]")
    ax.set_ylabel("MAE [Wh/km]")
    ax.set_title("Prediction error vs temperature")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse", action="store_true", help="reuse results/step5_dataset.csv if it exists")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    pd.set_option("display.width", 170)
    pd.set_option("display.max_columns", 20)
    model = EVModel.from_yaml(ROOT / "configs" / "compact_ev.yaml", ROOT / "configs" / "compact_ev_powertrain.yaml")
    env = Environment.from_yaml(ROOT / "configs" / "environment.yaml")
    cycles = load_cycles()
    if not cycles:
        print("No cycles found. Run: python scripts/download_cycles.py")
        return
    print(f"Base cycles: {', '.join(k.upper() for k in cycles)} (+ constant-speed trips)")

    print("\n=== 1. Label check: fast training label vs full range test ===")
    chk = label_check(model, env, cycles)
    print(chk.round(1).to_string(index=False))
    print(f"Mean |difference|: {chk['diff_pct'].abs().mean():.2f}%")

    data_path = RESULTS / "step5_dataset.csv"
    if args.reuse and data_path.exists():
        df = pd.read_csv(data_path)
        print(f"\nReusing {len(df)} trips from {data_path.name}")
    else:
        print(f"\n=== 2. Generating {args.n_samples} labelled trips ===")
        t0 = time.perf_counter()
        df = generate_dataset(model, env, cycles, args.n_samples, seed=args.seed, progress_every=500)
        gen_s = time.perf_counter() - t0
        df.to_csv(data_path, index=False)
        print(f"Done in {gen_s:.0f} s ({1000 * gen_s / len(df):.1f} ms per trip)")
    print(df[[TARGET, "terminal_consumption_wh_per_km", "range_km", "ambient_c", "extra_payload_kg", "avg_speed_kmh"]].describe().round(1).to_string())

    print("\n=== 3. Model comparison (80/20 split, 5-fold CV on training set) ===")
    specs = make_models(args.seed)
    metrics, fitted, preds = compare_models(df, specs, seed=args.seed)
    cols = ["whkm_mae", "whkm_rmse", "whkm_r2", "range_mae", "range_mape_pct", "cv_r2_mean", "cv_r2_std"]
    print(metrics[cols].round(3).to_string())
    best = metrics["whkm_mae"].idxmin()
    print(f"\nBest model: {best}")

    print("\n=== 4. Extrapolation: leave one cycle family out ===")
    spec_by_name = {s.name: s for s in specs}
    lofo_models = list(dict.fromkeys([best, "Ridge regression (physics features)",
                                      "Gradient boosting (physics features)", "Random forest (raw)"]))
    lofo = {name: leave_one_family_out(df, spec_by_name[name])["whkm_mape_pct"] for name in lofo_models}
    lofo_df = pd.DataFrame(lofo)
    lofo_df.columns = [c.replace(" regression", "").replace("Gradient boosting", "GBoost") for c in lofo_df.columns]
    print("MAPE [%] of consumption on the held-out family:")
    print(lofo_df.round(2).to_string())

    est, feats = fitted[best]
    train_mask = ~df.index.isin(preds.index)
    test_X = df.loc[preds.index, feats]
    imp = permutation_importance(est, test_X, preds[TARGET], n_repeats=5, random_state=args.seed,
                                 scoring="neg_mean_absolute_error")
    importance = pd.Series(imp.importances_mean, index=feats)
    print("\n=== 5. Top drivers of consumption (permutation importance, Wh/km) ===")
    print(importance.sort_values(ascending=False).head(8).round(2).to_string())

    print("\n=== 6. Speed: physics vs ML ===")
    example = df.loc[preds.index[:1], feats]
    t0 = time.perf_counter()
    for _ in range(200):
        est.predict(example)
    ml_ms = 1000 * (time.perf_counter() - t0) / 200
    t0 = time.perf_counter()
    sc = Scenario(ambient_c=-5.0, hvac=True, extra_payload_kg=150.0)
    winter = build_model(model, env, sc)
    full = winter.range_test(cycles["udds"] if "udds" in cycles else next(iter(cycles.values())))
    phys_ms = 1000 * (time.perf_counter() - t0)
    print(f"Full physics range test: {phys_ms:.0f} ms   ML prediction: {ml_ms:.2f} ms per trip")

    key = "udds" if "udds" in cycles else next(iter(cycles))
    feats_row = pd.DataFrame([trip_features(cycles[key], sc, env, model.vehicle.aux_power_w)])
    predictor = RangePredictor(est, feats, {"model": best, "target": TARGET, "n_train": int(train_mask.sum()),
                                            "test_mae_wh_per_km": float(metrics.loc[best, "whkm_mae"])})
    ml_range = float(predictor.predict_range(feats_row, winter.battery.usable_energy_kwh)[0])
    print(f"Example - {key.upper()}, -5 C, heater on, +150 kg: physics {full['range_km']:.0f} km, ML {ml_range:.0f} km")

    predictor.save(MODEL_PATH)
    metrics.to_csv(RESULTS / "step5_model_metrics.csv")
    lofo_df.to_csv(RESULTS / "step5_leave_one_family_out.csv")
    importance.to_csv(RESULTS / "step5_feature_importance.csv", header=["importance_wh_per_km"])
    chk.to_csv(RESULTS / "step5_label_check.csv", index=False)
    plot_parity(preds, best, RESULTS / "step5_parity_best_model.png")
    plot_model_comparison(metrics, RESULTS / "step5_model_comparison.png")
    plot_importance(importance, best, RESULTS / "step5_feature_importance.png")
    plot_error_vs_temperature(preds, [best, "Linear regression (raw)", "Random forest (raw)"],
                              RESULTS / "step5_error_vs_temperature.png")
    print(f"\nSaved model to {MODEL_PATH} and CSVs/plots to {RESULTS}")


if __name__ == "__main__":
    main()
