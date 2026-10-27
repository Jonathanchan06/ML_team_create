import numpy as np
import pytest

import make_fake_data as fake
from features import build_feature_table
from models.xgb import XGBModel
from recommend import KEY_RANGE, load_song, predict, recommend_key, recommend_key_from_notes


@pytest.fixture(scope="module")
def bundle():
    """A small model trained on freshly generated fake data (no files needed)."""
    rng = np.random.default_rng(0)
    singers, songs = fake.make_singers(rng), fake.make_songs(rng)
    song_phrases = fake.make_all_song_phrases(rng, songs)
    phrases = fake.build_phrase_rows(rng, singers, songs, song_phrases, with_timing=False)
    singers = singers.drop(columns="skill_offset")
    model = XGBModel(n_estimators=200).fit(build_feature_table(phrases, singers, songs))
    return {"model": model, "singers": singers, "default_tempo": 120.0}


def high_song(bundle, singer_id):
    """Three slow phrases sitting at the top of (and above) the singer's range."""
    top = int(bundle["singers"].set_index("singer_id").loc[singer_id, "range_high_midi"])
    return [[(top - 2 + i, 2.0 * p + 0.5 * i, 0.5) for i in range(4)] for p in range(3)]


def test_predict_returns_one_accuracy_per_phrase(bundle):
    preds = predict(high_song(bundle, "s1"), "s1", transpose=0, bundle=bundle)
    assert len(preds) == 3
    assert all(0 <= p <= 1 for p in preds)


def test_transposing_up_lowers_accuracy_on_a_high_song(bundle):
    song = high_song(bundle, "s1")
    up = np.mean(predict(song, "s1", transpose=4, bundle=bundle))
    down = np.mean(predict(song, "s1", transpose=-3, bundle=bundle))
    assert up < down


def test_recommend_key_moves_a_high_song_down(bundle):
    result = recommend_key_from_notes(high_song(bundle, "s1"), "s1", bundle=bundle)
    assert result["best_transpose"] < 0
    assert set(result["predicted_accuracy"]) == set(KEY_RANGE)
    assert len(result["phrase_predictions"]) == 3
    assert sorted(result["hardest_phrases"]) == [0, 1, 2]


def test_unknown_singer_raises(bundle):
    with pytest.raises(ValueError):
        predict(high_song(bundle, "s1"), "nobody", transpose=0, bundle=bundle)


@pytest.fixture(scope="module")
def melody_dir(tmp_path_factory):
    """Melody files for two unrecorded songs: one low, one written high."""
    rng = np.random.default_rng(5)
    songs = fake.make_songs(rng, count=2, first_index=90)
    songs["centre_midi"] = [55, 75]
    out = tmp_path_factory.mktemp("melodies")
    fake.write_melodies(songs, fake.make_all_song_phrases(rng, songs), out)
    return out


def test_load_song_reads_notes_and_tempo(melody_dir):
    song = load_song("song_90", melody_dir)
    assert song["song_id"] == "song_90"
    assert song["tempo_bpm"] > 0
    assert all(len(note) == 3 for phrase in song["phrases"] for note in phrase)


def test_load_unknown_song_raises(melody_dir):
    with pytest.raises(ValueError):
        load_song("no_such_song", melody_dir)


def test_recommend_key_by_id_for_unrecorded_song(bundle, melody_dir):
    song = load_song("song_91", melody_dir)
    by_id = recommend_key("song_91", "s1", bundle=bundle, melody_dir=melody_dir)
    by_notes = recommend_key_from_notes(song["phrases"], "s1", song["tempo_bpm"], bundle=bundle)
    assert by_id == {"song_id": "song_91", **by_notes}
    assert len(by_id["phrase_predictions"]) == len(song["phrases"])


def test_high_song_gets_a_lower_key_than_low_song(bundle, melody_dir):
    low = recommend_key("song_90", "s1", bundle=bundle, melody_dir=melody_dir)
    high = recommend_key("song_91", "s1", bundle=bundle, melody_dir=melody_dir)
    assert high["best_transpose"] < low["best_transpose"] or high["best_transpose"] == -5
