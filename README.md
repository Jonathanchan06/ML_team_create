# Karaoke ML Coach

Predicts how accurately a singer will sing each phrase of a song **before they sing it**,
then recommends the best key and flags the hardest phrases.

## Pipeline

| Step | What it does | Status |
|---|---|---|
| 1. Song → target notes | Demucs separates the original vocals; basic-pitch turns them into notes (pitch, start, duration). This is the answer key. | Demucs only (see below) |
| 2. Recording → pitch | PESTO tracks the pitch of a team recording every 10 ms (`program.py`). | Done |
| 3. Pitch + target notes → labels | A scoring script compares the two and gives per-phrase scores (`accuracy`, `mean_cents_error`, timing scores). | Missing |
| 4. Labels → model → predictions | Features from the target notes, model ladder (baseline → linear → XGBoost), `recommend_key` for the app (`ml/`). | Done, on fake data |

Until step 3 exists there are no real labels, so the model in `ml/` is trained on
**fake data** with planted patterns. That tests the code end to end; its predictions
for real singers mean nothing until it is retrained on real scored recordings.

## Setup

From the project folder, create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

Demucs also requires FFmpeg to be installed and available on `PATH`. On Windows,
if you see a TorchCodec / `libtorchcodec` DLL loading error when reading audio,
install an FFmpeg **full-shared** build and make sure its `bin` directory is on
`PATH`.
1. Open gyan.dev’s FFmpeg builds page. https://www.gyan.dev/ffmpeg/builds/
2. Under Release builds, download ffmpeg-release-full-shared.7z. Don’t choose ffmpeg-release-full.7z.
3. Extract it somewhere convenient, such as C:\ffmpeg-shared. The bin folder inside should contain ffmpeg.exe and DLLs such as avcodec-*.dll.
4. Add that extracted bin folder to your Windows Path in Edit System Environment, placing it above the existing FFmpeg entry. You can keep both installations.

## Steps 1-2: separate the audio and track pitch

### Separate the audio

Run Demucs with the path to the audio file:

```powershell
demucs "D:\path\to\audio.mp3"
```

By default, Demucs writes the separated tracks under
`separated\htdemucs\<track name>\`. For example, the vocals are usually saved
as `separated\htdemucs\<track name>\vocals.wav`.

### Configure and run pitch tracking

Open `program.py` and set `audio_path` to the separated vocals file, and
`output_directory` to the folder where you want the CSV files saved. Use the
actual path and extension produced by Demucs (typically `vocals.wav`, not
`vocals.mp3`):

```python
audio_path = r"D:\path\to\separated\htdemucs\track name\vocals.wav"
output_directory = r"D:\path\to\pitch-results"
```

Run the program from the project folder with the virtual environment activated:

```powershell
python program.py
```

The output directory is created if needed. The program writes
`my_file_ORIGINAL.csv` with all pitch estimates and `my_file_FILTERED.csv`
with estimates whose confidence is at least `0.70`.

## Step 4: ML model (`ml/`)

Install the ML dependencies (into the same virtual environment):

```powershell
python -m pip install -r ml/requirements.txt
```

Run from the project folder:

```powershell
python ml/src/make_fake_data.py    # fake data -> ml/data/fake/
python ml/src/train.py             # train and compare models, save ml/artifacts/model_v0.0.joblib
python ml/src/learning_curve.py    # error vs number of training songs -> ml/reports/
python -m pytest ml                # tests
```

`train.py` prints each model's error on songs it never saw, and saves a feature
importance chart to `ml/reports/shap_importance.png`. Runs are logged to MLflow;
to browse them: `cd ml` then `mlflow ui --backend-store-uri sqlite:///mlruns.db`.

### What's in `ml/src/`

| File | Purpose |
|---|---|
| `make_fake_data.py` | Fake songs, singers and labels in the real data format, plus `melodies/<song_id>.json` for every song (some with no recordings) |
| `features.py` | Target notes + singer range → features (e.g. how far above the singer's range) |
| `evaluate.py` | Train/test split by song, error metrics, learning curve, MLflow logging |
| `models/` | `baseline.py` (singer's average), `linear.py`, `xgb.py` |
| `train.py` | Runs the model ladder and saves the trained model |
| `learning_curve.py` | Shows how much data is needed |
| `recommend.py` | Interface for the app team (below) |

### Using the model (app team)

```python
from recommend import recommend_key, load_song, predict

recommend_key("song_41", "s1")
# {"song_id": "song_41", "best_transpose": -2,
#  "predicted_accuracy": {-5: 0.61, ..., 5: 0.32},   # mean per key
#  "phrase_predictions": [...], "hardest_phrases": [2, 9, 5]}
```

The song's notes and tempo come from `ml/data/fake/melodies/<song_id>.json`, so the
app only passes ids. Run it with `ml/src` as the working directory (or on `PYTHONPATH`).

### Data format

One row per phrase, per recording (`phrases_vX.Y.parquet`): `song_id`, `singer_id`,
`take_id`, `transpose`, `phrase_idx`, `target_notes` (list of `midi_pitch`, `start_s`,
`duration_s`), `accuracy` (main label, 0-1), `mean_cents_error`, and optional timing
labels. Supporting files: `singers.csv` (comfortable range) and `songs.csv` (tempo, key,
language, genre). The scoring script in step 3 needs to produce exactly this format;
see `CLAUDE.md` for full details.
