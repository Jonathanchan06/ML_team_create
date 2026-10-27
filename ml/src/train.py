"""Run the model ladder: baseline -> linear -> XGBoost.

1. Hold out ~20% of songs as the final test set.
2. Score each model with song-grouped CV on the remaining songs (XGBoost tuned with Optuna).
3. Fit each model on all training songs and score it once on the held-out songs.
4. Explain XGBoost with SHAP, save the model bundle for recommend.py, log runs to MLflow.

Usage:
    python ml/src/train.py [--data-dir ml/data/fake] [--version v0.0] [--trials 30] [--no-log]
"""

import argparse
from pathlib import Path

import joblib
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate import (LABEL, ML_DIR, cross_validate, key_recommendation_hit_rate, load_dataset,
                      log_run, score, split_holdout)
from models import xgb
from models.baseline import SingerMeanBaseline
from models.linear import LinearModel

# Planted in make_fake_data.py; if XGBoost doesn't rank this near the top, something is broken
EXPECTED_TOP_FEATURE = "max_above_range"


def plot_shap_importance(importance, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    importance[::-1].plot.barh(ax=ax, color="#4a7ab8")
    ax.set_xlabel("mean |SHAP value| (log-odds)")
    ax.set_title("XGBoost feature importance (held-out songs)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ML_DIR / "data" / "fake")
    parser.add_argument("--version", default="v0.0")
    parser.add_argument("--trials", type=int, default=30, help="Optuna trials for XGBoost")
    parser.add_argument("--no-log", action="store_true", help="skip MLflow logging")
    args = parser.parse_args()

    table, singers, songs = load_dataset(args.data_dir, args.version)
    train, test = split_holdout(table)
    print(f"{len(table)} phrases; train {train['song_id'].nunique()} songs, "
          f"holdout {test['song_id'].nunique()} songs")

    print(f"Tuning XGBoost ({args.trials} Optuna trials)...")
    best_params, _ = xgb.tune(train, n_trials=args.trials)
    ladder = {
        "baseline": SingerMeanBaseline,
        "linear": LinearModel,
        "xgboost": lambda: xgb.XGBModel(**best_params),
    }

    results = {}
    fitted = {}
    for name, make_model in ladder.items():
        cv = cross_validate(make_model, train)
        model = make_model().fit(train)
        holdout = score(test[LABEL], model.predict(test))
        holdout["key_hit_rate"], holdout["key_pairs"] = key_recommendation_hit_rate(model, test)
        results[name] = {**{f"cv_{k}": v for k, v in cv.items()},
                         **{f"holdout_{k}": v for k, v in holdout.items()}}
        fitted[name] = model

    report = pd.DataFrame(results).T
    print("\n" + report.round(4).to_string())

    names = list(ladder)
    for prev, cur in zip(names, names[1:]):
        if report.loc[cur, "cv_mae"] >= report.loc[prev, "cv_mae"]:
            print(f"WARNING: {cur} does not beat {prev} on CV MAE")

    print("\nLinear coefficients (per 1 sd of feature):")
    print(fitted["linear"].coefficients().round(4).to_string())

    importance = fitted["xgboost"].shap_importance(test)
    print("\nXGBoost mean |SHAP| (held-out songs):")
    print(importance.round(4).to_string())
    if EXPECTED_TOP_FEATURE not in importance.index[:3]:
        print(f"WARNING: {EXPECTED_TOP_FEATURE} is not in the top 3 SHAP features; check for a bug")

    reports_dir = ML_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)
    shap_path = reports_dir / "shap_importance.png"
    plot_shap_importance(importance, shap_path)

    artifacts_dir = ML_DIR / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    bundle_path = artifacts_dir / f"model_{args.version}.joblib"
    # Fit the shipped model on every song (train + holdout) now that evaluation is done
    final_model = xgb.XGBModel(**best_params).fit(table)
    joblib.dump({
        "model": final_model,
        "singers": singers,
        "default_tempo": float(songs["tempo_bpm"].median()),
        "dataset_version": args.version,
        "params": best_params,
    }, bundle_path)
    print(f"\nSaved {bundle_path} and {shap_path}")

    if not args.no_log:
        for name, metrics in results.items():
            params = {"dataset_version": args.version, "model": name, "holdout_frac": 0.2}
            if name == "xgboost":
                params.update(best_params)
            log_run(name, params, metrics, artifacts=[shap_path] if name == "xgboost" else ())
        print("Logged runs to MLflow (ml/mlruns.db)")


if __name__ == "__main__":
    main()
