import numpy as np
import pandas as pd
import pytest

from evaluate import score, song_folds, split_holdout


@pytest.fixture
def table():
    rng = np.random.default_rng(0)
    songs = np.repeat([f"song_{i:02d}" for i in range(20)], 15)
    return pd.DataFrame({"song_id": songs, "accuracy": rng.uniform(0, 1, len(songs))})


def test_holdout_has_no_shared_songs(table):
    train, test = split_holdout(table, test_frac=0.2)
    assert not set(train["song_id"]) & set(test["song_id"])
    assert len(train) + len(test) == len(table)
    assert test["song_id"].nunique() == 4


def test_folds_have_no_shared_songs_and_cover_every_song(table):
    tested = []
    for train, test in song_folds(table, n_splits=5):
        assert not set(train["song_id"]) & set(test["song_id"])
        tested += list(test["song_id"].unique())
    assert sorted(tested) == sorted(table["song_id"].unique())


def test_score():
    s = score([0.2, 0.5, 0.9], [0.3, 0.5, 0.7])
    assert s["mae"] == pytest.approx(0.1)
    assert s["spearman"] == pytest.approx(1.0)


def test_score_constant_prediction_has_zero_spearman():
    assert score([0.2, 0.5, 0.9], [0.5, 0.5, 0.5])["spearman"] == 0.0
