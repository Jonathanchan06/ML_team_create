"""Generate fake karaoke phrase data in the real data-team format, with planted patterns.

Outputs (in ml/data/fake/ by default):
    phrases_v0.0.parquet  one row per phrase, per take, per singer
    singers.csv           singer_id, range_low_midi, range_high_midi
    songs.csv             song_id, tempo_bpm, original_key, language, genre
    planted_truth.json    the hidden rules used to make the labels (for checking models later)
    melodies/<song_id>.json  every song's target notes in the original key, including songs
                          nobody has recorded (what the app looks up by song_id)

Usage:
    python ml/src/make_fake_data.py [--seed 0] [--out-dir ml/data/fake] [--no-timing]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATASET_VERSION = "v0.0"

# Dataset size
NUM_SINGERS = 6
NUM_SONGS = 40                   # songs with recordings
NUM_UNSUNG_SONGS = 4             # extra songs with a melody but no recordings (new songs for the app)
PHRASES_PER_SONG = (15, 20)      # inclusive range
NOTES_PER_PHRASE = (4, 10)       # inclusive range
MIDI_MIN, MIDI_MAX = 50, 80

# Who sings what
SHARE_SONGS_SUNG_BY_ALL = 0.25   # other songs are sung by 1 or 2 singers
SHARE_PAIRS_TRANSPOSED = 0.30    # these singer-song pairs also get takes at -2 and +2
EXTRA_TRANSPOSES = [-2, 2]

# Planted label rule (accuracy is a share of frames in tune, 0-1)
BASE_ACCURACY = 0.85
SKILL_OFFSET_MAX = 0.10          # each singer's offset is uniform in [-max, +max]
COEF_ABOVE_RANGE = -0.04         # per semitone above the singer's comfortable range
COEF_BELOW_RANGE = -0.02         # per semitone below it
LEAP_FREE_SEMITONES = 4          # upward leaps up to this size cost nothing
COEF_LEAP_UP = -0.015            # per semitone of biggest upward leap beyond LEAP_FREE_SEMITONES
NPS_FREE = 3.0                   # notes per second up to this cost nothing
COEF_NPS = -0.05                 # per note/sec beyond NPS_FREE
ACCURACY_NOISE_SD = 0.05
CENTS_NOISE_SD = 5.0

KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
LANGUAGES = ["en", "zh", "ko", "ja", "fr"]
GENRES = ["pop", "rock", "ballad", "rnb", "musical"]
NOTE_BEATS = [0.25, 0.5, 1.0, 2.0]
NOTE_BEAT_PROBS = [0.15, 0.40, 0.30, 0.15]


def make_singers(rng):
    """Each singer gets a comfortable range and a hidden skill offset."""
    rows = []
    for i in range(NUM_SINGERS):
        low = int(rng.integers(48, 58))
        high = low + int(rng.integers(12, 17))
        skill = float(rng.uniform(-SKILL_OFFSET_MAX, SKILL_OFFSET_MAX))
        rows.append({"singer_id": f"s{i + 1}", "range_low_midi": low,
                     "range_high_midi": high, "skill_offset": skill})
    return pd.DataFrame(rows)


def make_songs(rng, count=NUM_SONGS, first_index=0):
    """Song metadata plus a hidden melodic centre used to place the notes."""
    rows = []
    for i in range(first_index, first_index + count):
        rows.append({
            "song_id": f"song_{i:02d}",
            "tempo_bpm": int(rng.integers(70, 161)),
            "original_key": str(rng.choice(KEYS)),
            "language": str(rng.choice(LANGUAGES)),
            "genre": str(rng.choice(GENRES)),
            "centre_midi": int(rng.integers(58, 73)),
        })
    return pd.DataFrame(rows)


def make_phrase_notes(rng, centre_midi, tempo_bpm, start_time):
    """One phrase as a list of (midi_pitch, start_s, duration_s), in the original key."""
    num_notes = int(rng.integers(NOTES_PER_PHRASE[0], NOTES_PER_PHRASE[1] + 1))
    seconds_per_beat = 60.0 / tempo_bpm

    pitch = centre_midi + int(rng.integers(-4, 5))
    t = start_time
    notes = []
    for _ in range(num_notes):
        pitch = int(np.clip(pitch, MIDI_MIN, MIDI_MAX))
        duration = float(rng.choice(NOTE_BEATS, p=NOTE_BEAT_PROBS)) * seconds_per_beat
        notes.append((pitch, round(t, 4), round(duration, 4)))
        t += duration

        # Mostly small steps, sometimes a big leap
        if rng.random() < 0.15:
            step = int(rng.integers(5, 10))
        else:
            step = int(rng.integers(0, 4))
        direction = 1 if rng.random() < 0.5 else -1
        pitch += direction * step
    return notes


def make_song_phrases(rng, song):
    """All phrases of a song, with a short rest between phrases."""
    num_phrases = int(rng.integers(PHRASES_PER_SONG[0], PHRASES_PER_SONG[1] + 1))
    phrases = []
    t = float(rng.uniform(5, 15))  # intro before the first phrase
    for _ in range(num_phrases):
        notes = make_phrase_notes(rng, song["centre_midi"], song["tempo_bpm"], t)
        phrases.append(notes)
        last_pitch, last_start, last_dur = notes[-1]
        t = last_start + last_dur + float(rng.uniform(0.5, 3.0))
    return phrases


def make_takes(rng, singers, songs):
    """List of (song_id, singer_id, take_id, transpose) for every recording."""
    takes = []
    for song_id in songs["song_id"]:
        if rng.random() < SHARE_SONGS_SUNG_BY_ALL:
            singer_ids = list(singers["singer_id"])
        else:
            count = int(rng.integers(1, 3))
            singer_ids = list(rng.choice(singers["singer_id"], size=count, replace=False))

        for singer_id in singer_ids:
            transposes = [0]
            if rng.random() < SHARE_PAIRS_TRANSPOSED:
                transposes += EXTRA_TRANSPOSES
            for take_id, transpose in enumerate(transposes, start=1):
                takes.append((song_id, str(singer_id), take_id, transpose))
    return takes


def planted_quantities(notes, range_low, range_high):
    """The four phrase properties that drive the planted accuracy rule."""
    pitches = np.array([n[0] for n in notes])
    max_above_range = max(0, int(pitches.max()) - range_high)
    max_below_range = max(0, range_low - int(pitches.min()))

    intervals = np.diff(pitches)
    biggest_leap_up = max(0, int(intervals.max())) if len(intervals) else 0

    phrase_start = notes[0][1]
    phrase_end = notes[-1][1] + notes[-1][2]
    notes_per_sec = len(notes) / (phrase_end - phrase_start)
    return max_above_range, max_below_range, biggest_leap_up, notes_per_sec


def planted_accuracy(rng, notes, singer):
    """Accuracy label from the hidden rule, plus noise, clipped to [0, 1]."""
    above, below, leap_up, nps = planted_quantities(
        notes, singer["range_low_midi"], singer["range_high_midi"])
    acc = (BASE_ACCURACY + singer["skill_offset"]
           + COEF_ABOVE_RANGE * above
           + COEF_BELOW_RANGE * below
           + COEF_LEAP_UP * max(0, leap_up - LEAP_FREE_SEMITONES)
           + COEF_NPS * max(0.0, nps - NPS_FREE)
           + rng.normal(0, ACCURACY_NOISE_SD))
    return float(np.clip(acc, 0, 1)), nps


def timing_labels(rng, nps, onset_bias_ms):
    """Optional timing labels: fast phrases are harder to time."""
    timing_accuracy = np.clip(0.9 - 0.04 * max(0.0, nps - 2) + rng.normal(0, 0.05), 0, 1)
    mean_onset_error_ms = onset_bias_ms + rng.normal(0, 40)
    missed_note_rate = np.clip(0.02 + 0.02 * max(0.0, nps - 3) + rng.normal(0, 0.02), 0, 1)
    return float(timing_accuracy), float(mean_onset_error_ms), float(missed_note_rate)


def make_all_song_phrases(rng, songs):
    """{song_id: list of phrases} for every song, in the original key."""
    return {song["song_id"]: make_song_phrases(rng, song) for _, song in songs.iterrows()}


def build_phrase_rows(rng, singers, songs, song_phrases, with_timing):
    singer_by_id = singers.set_index("singer_id")
    onset_bias_ms = {s: float(rng.normal(0, 30)) for s in singers["singer_id"]}

    rows = []
    for song_id, singer_id, take_id, transpose in make_takes(rng, singers, songs):
        singer = singer_by_id.loc[singer_id]
        for phrase_idx, notes in enumerate(song_phrases[song_id]):
            shifted = [(pitch + transpose, start, dur) for pitch, start, dur in notes]
            accuracy, nps = planted_accuracy(rng, shifted, singer)
            row = {
                "song_id": song_id,
                "singer_id": singer_id,
                "take_id": take_id,
                "transpose": transpose,
                "phrase_idx": phrase_idx,
                "target_notes": [{"midi_pitch": p, "start_s": s, "duration_s": d}
                                 for p, s, d in shifted],
                "accuracy": accuracy,
                "mean_cents_error": max(0.0, (1 - accuracy) * 100 + rng.normal(0, CENTS_NOISE_SD)),
            }
            if with_timing:
                (row["timing_accuracy"], row["mean_onset_error_ms"],
                 row["missed_note_rate"]) = timing_labels(rng, nps, onset_bias_ms[singer_id])
            rows.append(row)
    return pd.DataFrame(rows)


def write_phrases(phrases, path):
    """Write the parquet with an explicit schema so target_notes keeps typed, named fields."""
    note_type = pa.struct([("midi_pitch", pa.int32()),
                           ("start_s", pa.float64()),
                           ("duration_s", pa.float64())])
    fields = [
        ("song_id", pa.string()),
        ("singer_id", pa.string()),
        ("take_id", pa.int64()),
        ("transpose", pa.int64()),
        ("phrase_idx", pa.int64()),
        ("target_notes", pa.list_(note_type)),
        ("accuracy", pa.float64()),
        ("mean_cents_error", pa.float64()),
    ]
    for col in ["timing_accuracy", "mean_onset_error_ms", "missed_note_rate"]:
        if col in phrases.columns:
            fields.append((col, pa.float64()))
    table = pa.Table.from_pandas(phrases, schema=pa.schema(fields), preserve_index=False)
    pq.write_table(table, path)


def write_melodies(songs, song_phrases, melody_dir):
    """One JSON per song: {"song_id", "tempo_bpm", "phrases": [[{midi_pitch, start_s, duration_s}, ...], ...]}."""
    melody_dir.mkdir(parents=True, exist_ok=True)
    for song in songs.itertuples(index=False):
        melody = {
            "song_id": song.song_id,
            "tempo_bpm": song.tempo_bpm,
            "phrases": [[{"midi_pitch": p, "start_s": s, "duration_s": d} for p, s, d in notes]
                        for notes in song_phrases[song.song_id]],
        }
        (melody_dir / f"{song.song_id}.json").write_text(json.dumps(melody, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "fake")
    parser.add_argument("--no-timing", action="store_true", help="leave out the optional timing label columns")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    singers = make_singers(rng)
    songs = make_songs(rng)
    song_phrases = make_all_song_phrases(rng, songs)
    phrases = build_phrase_rows(rng, singers, songs, song_phrases, with_timing=not args.no_timing)

    # Songs nobody has recorded, made last so the recorded data above doesn't change
    unsung = make_songs(rng, NUM_UNSUNG_SONGS, first_index=NUM_SONGS)
    song_phrases.update(make_all_song_phrases(rng, unsung))
    all_songs = pd.concat([songs, unsung], ignore_index=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_phrases(phrases, args.out_dir / f"phrases_{DATASET_VERSION}.parquet")
    singers.drop(columns="skill_offset").to_csv(args.out_dir / "singers.csv", index=False)
    all_songs.drop(columns="centre_midi").to_csv(args.out_dir / "songs.csv", index=False)
    write_melodies(all_songs, song_phrases, args.out_dir / "melodies")

    truth = {
        "dataset_version": DATASET_VERSION,
        "seed": args.seed,
        "base_accuracy": BASE_ACCURACY,
        "coef_above_range": COEF_ABOVE_RANGE,
        "coef_below_range": COEF_BELOW_RANGE,
        "leap_free_semitones": LEAP_FREE_SEMITONES,
        "coef_leap_up": COEF_LEAP_UP,
        "nps_free": NPS_FREE,
        "coef_nps": COEF_NPS,
        "accuracy_noise_sd": ACCURACY_NOISE_SD,
        "skill_offset": dict(zip(singers["singer_id"], singers["skill_offset"].round(4))),
    }
    (args.out_dir / "planted_truth.json").write_text(json.dumps(truth, indent=2))

    print(f"Wrote {len(phrases)} phrases from {phrases['song_id'].nunique()} songs, "
          f"{phrases['singer_id'].nunique()} singers, "
          f"{phrases.groupby(['song_id', 'singer_id', 'take_id']).ngroups} takes")
    print(f"Wrote {len(all_songs)} melodies ({len(unsung)} songs with no recordings: "
          f"{', '.join(unsung['song_id'])})")
    print(f"accuracy mean {phrases['accuracy'].mean():.3f}, sd {phrases['accuracy'].std():.3f}")
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
