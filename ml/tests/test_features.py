import pytest

from features import FEATURE_COLUMNS, phrase_features

# Hand-checked phrase: pitches 60, 67, 65, 72 -> intervals +7, -2, +7
NOTES = [(60, 10.0, 0.5), (67, 10.5, 0.5), (65, 11.0, 1.0), (72, 12.0, 0.5)]
RANGE_LOW, RANGE_HIGH = 55, 70


def features(notes=NOTES, low=RANGE_LOW, high=RANGE_HIGH):
    return phrase_features(notes, low, high, tempo=120, phrase_idx=3, num_phrases=12)


def test_range_features():
    f = features()
    assert f["max_above_range"] == 2          # 72 - 70
    assert f["max_below_range"] == 0
    assert f["pct_outside_range"] == 0.25     # only 72 is outside 55-70
    assert f["mean_pitch_rel"] == 3.5         # mean 66 vs range middle 62.5
    assert f["pitch_span"] == 12


def test_below_range():
    f = features(low=63, high=80)
    assert f["max_below_range"] == 3          # 63 - 60
    assert f["max_above_range"] == 0


def test_interval_features():
    f = features()
    assert f["biggest_leap_up"] == 7
    assert f["biggest_leap_down"] == 2
    assert f["mean_abs_interval"] == pytest.approx(16 / 3)
    assert f["num_leaps"] == 2                # two intervals of 7 (> 5)


def test_rhythm_and_context_features():
    f = features()
    assert f["phrase_duration"] == 2.5        # 10.0 -> 12.5
    assert f["notes_per_sec"] == pytest.approx(4 / 2.5)
    assert f["shortest_note"] == 0.5
    assert f["longest_note"] == 1.0
    assert f["num_notes"] == 4
    assert f["tempo"] == 120
    assert f["phrase_position"] == 0.25       # 3 / 12


def test_single_note_has_no_intervals():
    f = features(notes=[(60, 0.0, 1.0)])
    assert f["biggest_leap_up"] == 0 and f["biggest_leap_down"] == 0
    assert f["mean_abs_interval"] == 0 and f["num_leaps"] == 0


def test_dict_notes_from_parquet_match_tuples():
    dict_notes = [{"midi_pitch": p, "start_s": s, "duration_s": d} for p, s, d in NOTES]
    assert features(dict_notes) == features()


def test_empty_phrase_raises():
    with pytest.raises(ValueError):
        features(notes=[])


def test_all_numeric_features_present():
    assert set(features()) | {"singer_id"} == set(FEATURE_COLUMNS)


def test_matches_fake_data_generator():
    """features.py and the generator's planted rule must agree, or planted patterns can't be recovered."""
    import numpy as np
    import make_fake_data as fake

    rng = np.random.default_rng(1)
    for _ in range(50):
        notes = fake.make_phrase_notes(rng, centre_midi=66, tempo_bpm=140, start_time=5.0)
        above, below, leap_up, nps = fake.planted_quantities(notes, 55, 68)
        f = phrase_features(notes, 55, 68, tempo=140, phrase_idx=0, num_phrases=1)
        assert (f["max_above_range"], f["max_below_range"], f["biggest_leap_up"]) == (above, below, leap_up)
        assert f["notes_per_sec"] == pytest.approx(nps)
