"""Turn a phrase's target notes (plus singer range and song metadata) into a feature row.

Features come only from the target notes, never from the recording, so the model can
score songs nobody has sung yet.
"""

import numpy as np
import pandas as pd

LEAP_SEMITONES = 5  # an interval bigger than this counts as a leap

NUMERIC_FEATURES = [
    # Range, relative to the singer's comfortable range
    "max_above_range", "max_below_range", "pct_outside_range", "mean_pitch_rel", "pitch_span",
    # Intervals
    "biggest_leap_up", "biggest_leap_down", "mean_abs_interval", "num_leaps",
    # Rhythm
    "notes_per_sec", "shortest_note", "longest_note", "tempo", "phrase_duration", "num_notes",
    # Context
    "phrase_position",
]
CATEGORICAL_FEATURES = ["singer_id"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def notes_to_arrays(notes):
    """Accept notes as (midi, start_s, duration_s) tuples or as dicts read back from parquet."""
    if len(notes) and isinstance(notes[0], dict):
        notes = [(n["midi_pitch"], n["start_s"], n["duration_s"]) for n in notes]
    pitches = np.array([n[0] for n in notes], dtype=float)
    starts = np.array([n[1] for n in notes], dtype=float)
    durations = np.array([n[2] for n in notes], dtype=float)
    return pitches, starts, durations


def phrase_features(notes, range_low, range_high, tempo, phrase_idx, num_phrases):
    """Feature dict for one phrase. `notes` must already be shifted by the transpose."""
    pitches, starts, durations = notes_to_arrays(notes)
    if len(pitches) == 0:
        raise ValueError("phrase has no notes")

    intervals = np.diff(pitches)
    phrase_duration = (starts[-1] + durations[-1]) - starts[0]
    outside = (pitches > range_high) | (pitches < range_low)

    return {
        "max_above_range": max(0.0, pitches.max() - range_high),
        "max_below_range": max(0.0, range_low - pitches.min()),
        "pct_outside_range": outside.mean(),
        "mean_pitch_rel": pitches.mean() - (range_low + range_high) / 2,
        "pitch_span": pitches.max() - pitches.min(),
        "biggest_leap_up": max(0.0, intervals.max()) if len(intervals) else 0.0,
        "biggest_leap_down": max(0.0, -intervals.min()) if len(intervals) else 0.0,
        "mean_abs_interval": np.abs(intervals).mean() if len(intervals) else 0.0,
        "num_leaps": int((np.abs(intervals) > LEAP_SEMITONES).sum()),
        "notes_per_sec": len(pitches) / phrase_duration,
        "shortest_note": durations.min(),
        "longest_note": durations.max(),
        "tempo": float(tempo),
        "phrase_duration": phrase_duration,
        "num_notes": len(pitches),
        "phrase_position": phrase_idx / num_phrases,
    }


def build_feature_table(phrases, singers, songs):
    """One feature row per phrase row, keeping ids and any label columns alongside.

    phrases: the phrases parquet as a DataFrame
    singers: singers.csv (singer_id, range_low_midi, range_high_midi)
    songs:   songs.csv (song_id, tempo_bpm, ...)
    """
    singer_by_id = singers.set_index("singer_id")
    tempo_by_song = songs.set_index("song_id")["tempo_bpm"]
    num_phrases_by_song = phrases.groupby("song_id")["phrase_idx"].max() + 1

    rows = []
    for row in phrases.itertuples(index=False):
        singer = singer_by_id.loc[row.singer_id]
        rows.append(phrase_features(
            row.target_notes,
            singer["range_low_midi"], singer["range_high_midi"],
            tempo_by_song[row.song_id],
            row.phrase_idx, num_phrases_by_song[row.song_id],
        ))

    features = pd.DataFrame(rows, index=phrases.index)
    table = pd.concat([phrases.drop(columns="target_notes"), features], axis=1)
    table["singer_id"] = pd.Categorical(table["singer_id"], categories=sorted(singers["singer_id"]))
    return table
