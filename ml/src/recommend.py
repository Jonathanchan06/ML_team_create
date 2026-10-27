"""Interface for the app team: predict per-phrase accuracy and recommend a key.

    from recommend import predict, recommend_key
    recommend_key("song_41", "s1")                # -> {"best_transpose": -2, ...}
    predict(song_notes, "s1", transpose=-2)       # -> [0.81, 0.64, ...] one per phrase

recommend_key looks the song up by id in the melody folder (melodies/<song_id>.json), so the
app only passes ids. song_notes: one list of (midi_pitch, start_s, duration_s) per phrase, in
the original key. The trained model bundle is written by train.py (ml/artifacts/model_<version>.joblib).
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import phrase_features

ML_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ML_DIR / "artifacts" / "model_v0.0.joblib"
DEFAULT_MELODY_DIR = ML_DIR / "data" / "fake" / "melodies"
KEY_RANGE = range(-5, 6)

_bundle_cache = {}


def load_bundle(path=DEFAULT_MODEL_PATH):
    """The saved dict: {"model", "singers", "default_tempo", "dataset_version"}."""
    path = Path(path)
    if path not in _bundle_cache:
        _bundle_cache[path] = joblib.load(path)
    return _bundle_cache[path]


def load_song(song_id, melody_dir=DEFAULT_MELODY_DIR):
    """A song's melody: {"song_id", "tempo_bpm", "phrases"}, where phrases is one list of
    (midi_pitch, start_s, duration_s) tuples per phrase, in the original key."""
    path = Path(melody_dir) / f"{song_id}.json"
    if not path.exists():
        raise ValueError(f"unknown song {song_id!r}: no melody file at {path}")
    melody = json.loads(path.read_text())
    phrases = [[(n["midi_pitch"], n["start_s"], n["duration_s"]) for n in notes]
               for notes in melody["phrases"]]
    return {"song_id": melody["song_id"], "tempo_bpm": melody["tempo_bpm"], "phrases": phrases}


def _feature_table(song_notes, singer_id, transpose, tempo_bpm, bundle):
    singers = bundle["singers"].set_index("singer_id")
    if singer_id not in singers.index:
        raise ValueError(f"unknown singer {singer_id!r}; known: {list(singers.index)}")
    singer = singers.loc[singer_id]
    tempo = tempo_bpm if tempo_bpm is not None else bundle["default_tempo"]

    rows = []
    for phrase_idx, notes in enumerate(song_notes):
        shifted = [(pitch + transpose, start, dur) for pitch, start, dur in notes]
        rows.append(phrase_features(shifted, singer["range_low_midi"], singer["range_high_midi"],
                                    tempo, phrase_idx, len(song_notes)))
    table = pd.DataFrame(rows)
    table["singer_id"] = singer_id
    return table


def predict(song_notes, singer_id, transpose, tempo_bpm=None, bundle=None):
    """Predicted accuracy (0-1) for each phrase of the song at this transpose.
    tempo_bpm is optional; without it the training median tempo is used."""
    bundle = bundle or load_bundle()
    table = _feature_table(song_notes, singer_id, transpose, tempo_bpm, bundle)
    return [float(p) for p in bundle["model"].predict(table)]


def recommend_key(song_id, singer_id, worst_phrase_weight=0.0, num_hardest=3,
                  bundle=None, melody_dir=DEFAULT_MELODY_DIR):
    """Best key for a singer on a song, given only ids. The song's notes and tempo come from
    melodies/<song_id>.json, so it works for songs nobody has recorded yet.
    Returns the same dict as recommend_key_from_notes, plus "song_id"."""
    song = load_song(song_id, melody_dir)
    result = recommend_key_from_notes(song["phrases"], singer_id, song["tempo_bpm"],
                                      worst_phrase_weight, num_hardest, bundle)
    return {"song_id": song_id, **result}


def recommend_key_from_notes(song_notes, singer_id, tempo_bpm=None, worst_phrase_weight=0.0,
                             num_hardest=3, bundle=None):
    """Try every transpose in -5..+5 and pick the one with the best score, where
    score = mean phrase accuracy + worst_phrase_weight * worst phrase accuracy.

    Returns {"best_transpose", "predicted_accuracy" (mean per key), "phrase_predictions"
    (per phrase, best key), "hardest_phrases" (phrase indices, hardest first, best key)}.
    """
    bundle = bundle or load_bundle()
    per_key = {k: np.array(predict(song_notes, singer_id, k, tempo_bpm, bundle)) for k in KEY_RANGE}
    scores = {k: p.mean() + worst_phrase_weight * p.min() for k, p in per_key.items()}
    # Ties go to the key closest to the original
    best = max(KEY_RANGE, key=lambda k: (round(scores[k], 6), -abs(k)))
    best_preds = per_key[best]
    return {
        "best_transpose": best,
        "predicted_accuracy": {k: float(p.mean()) for k, p in per_key.items()},
        "phrase_predictions": [float(p) for p in best_preds],
        "hardest_phrases": [int(i) for i in np.argsort(best_preds)[:num_hardest]],
    }
