"""Learning curve: test MAE vs number of training songs (tells the data team how much to record).

Uses only the training songs (the holdout stays untouched) and the XGBoost params saved by
train.py, so run train.py first.

Usage:
    python ml/src/learning_curve.py [--data-dir ml/data/fake] [--version v0.0]
"""

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate import ML_DIR, learning_curve, load_dataset, split_holdout
from models.baseline import SingerMeanBaseline
from models.linear import LinearModel
from models.xgb import XGBModel

SONG_COUNTS = [3, 5, 8, 12, 16, 20, 25]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ML_DIR / "data" / "fake")
    parser.add_argument("--version", default="v0.0")
    args = parser.parse_args()

    table, _, _ = load_dataset(args.data_dir, args.version)
    train, _ = split_holdout(table)
    xgb_params = joblib.load(ML_DIR / "artifacts" / f"model_{args.version}.joblib")["params"]

    models = {
        "baseline": SingerMeanBaseline,
        "linear": LinearModel,
        "xgboost": lambda: XGBModel(**xgb_params),
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, make_model in models.items():
        curve = learning_curve(make_model, train, SONG_COUNTS)
        mae = curve["mae"]
        print(f"\n{name}\n{mae.round(4).to_string()}")
        ax.errorbar(mae.index, mae["mean"], yerr=mae["std"], marker="o", capsize=3, label=name)

    ax.set_xlabel("number of training songs")
    ax.set_ylabel("test MAE (song-grouped CV)")
    ax.set_title(f"Learning curve ({args.version})")
    ax.legend()
    fig.tight_layout()
    out = ML_DIR / "reports" / f"learning_curve_{args.version}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
