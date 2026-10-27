"""Splits, metrics and run logging. Every split is grouped by song_id, so no song is ever in
both train and test."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from features import build_feature_table

ML_DIR = Path(__file__).resolve().parents[1]
LABEL = "accuracy"
EXPERIMENT = "karaoke-accuracy"


def load_dataset(data_dir, version):
    """Read phrases, singers and songs, and return (feature table, singers, songs)."""
    data_dir = Path(data_dir)
    phrases = pd.read_parquet(data_dir / f"phrases_{version}.parquet")
    singers = pd.read_csv(data_dir / "singers.csv")
    songs = pd.read_csv(data_dir / "songs.csv")
    return build_feature_table(phrases, singers, songs), singers, songs


def split_holdout(table, test_frac=0.2, seed=0):
    """Hold out a share of songs as the final test set (touch it once, at the end)."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    train_idx, test_idx = next(splitter.split(table, groups=table["song_id"]))
    return table.iloc[train_idx], table.iloc[test_idx]


def song_folds(table, n_splits=5):
    """GroupKFold by song: yields (train_rows, test_rows) DataFrames."""
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(table, groups=table["song_id"]):
        yield table.iloc[train_idx], table.iloc[test_idx]


def score(y_true, y_pred):
    """Main metrics: MAE (lower is better) and Spearman rank correlation (higher is better)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    spearman = spearmanr(y_true, y_pred).statistic if np.std(y_pred) > 0 else 0.0
    return {"mae": float(np.mean(np.abs(y_true - y_pred))), "spearman": float(spearman)}


def cross_validate(make_model, table, n_splits=5):
    """Mean metrics over song-grouped folds. `make_model()` must return an unfitted model."""
    fold_scores = []
    for train, test in song_folds(table, n_splits):
        model = make_model().fit(train)
        fold_scores.append(score(test[LABEL], model.predict(test)))
    return {k: float(np.mean([s[k] for s in fold_scores])) for k in fold_scores[0]}


def key_recommendation_hit_rate(model, test, tolerance=1):
    """Share of (singer, song) pairs with several transposed takes where the key with the best
    predicted mean accuracy is within `tolerance` semitones of the key with the best actual one.
    A model that predicts the same for every key (e.g. the baseline) counts as a miss.
    Returns (hit_rate, number_of_pairs)."""
    test = test.assign(pred=model.predict(test))
    hits = []
    for _, pair in test.groupby(["singer_id", "song_id"], observed=True):
        by_key = pair.groupby("transpose")[[LABEL, "pred"]].mean()
        if len(by_key) < 2:
            continue
        if by_key["pred"].nunique() == 1:
            hits.append(False)
            continue
        hits.append(abs(by_key[LABEL].idxmax() - by_key["pred"].idxmax()) <= tolerance)
    return (float(np.mean(hits)) if hits else float("nan")), len(hits)


def learning_curve(make_model, table, song_counts, n_splits=5, repeats=3, seed=0):
    """Test MAE vs number of training songs. For each fold, train on random subsets of that
    fold's training songs and score on its test songs. Returns a DataFrame per song count."""
    rng = np.random.default_rng(seed)
    rows = []
    for train, test in song_folds(table, n_splits):
        train_songs = train["song_id"].unique()
        for n in song_counts:
            if n > len(train_songs):
                continue
            for _ in range(repeats):
                chosen = rng.choice(train_songs, size=n, replace=False)
                model = make_model().fit(train[train["song_id"].isin(chosen)])
                rows.append({"num_songs": n, **score(test[LABEL], model.predict(test))})
    return pd.DataFrame(rows).groupby("num_songs").agg(["mean", "std"])


def log_run(run_name, params, metrics, artifacts=()):
    """Log one run to MLflow (local SQLite store in ml/mlruns.db; view with `mlflow ui`)."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{(ML_DIR / 'mlruns.db').as_posix()}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        # Pin artifacts to ml/mlruns/ instead of wherever the script happens to be run from
        mlflow.create_experiment(EXPERIMENT, artifact_location=(ML_DIR / "mlruns").as_uri())
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        for path in artifacts:
            mlflow.log_artifact(str(path))
