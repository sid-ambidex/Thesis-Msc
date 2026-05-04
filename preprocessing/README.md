# preprocessing/

This folder converts raw interview recordings and ELAN annotation files into a structured, windowed dataset ready for model training. It must be run **before** any of the modality-specific folders.

---

## Files

| File | Purpose |
|------|---------|
| `audio_features.ipynb` | Speaker diarisation + Whisper ASR + initial openSMILE extraction |
| `alignment.ipynb` | Parse ELAN `.eaf` files and align annotations with diarised speech |
| `align_check.py` | Validate alignment quality; produces an overlap analysis CSV |
| `whisper_transcribe.ipynb` | word-level ASR transcription over a diarisation index |
| `windows_labels_and_splits.ipynb` | Generate sliding windows, assign soft labels, create train/val/test splits |
| `MediaPipe.ipynb` | Extract facial blendshapes and upper-body pose keypoints from video frames |
| `feature_aggregation.ipynb` | Summarise per-frame features into per-window statistics and join modalities |

---

## Notebook Details

### `audio_features.ipynb`

Runs the full audio preprocessing pipeline on each recording:

1. **Speaker diarisation** — uses `pyannote/speaker-diarization-3.1` to identify speaker turns
2. **Role assignment** — heuristically assigns *Interviewer* / *Participant* roles (first speaker = interviewer)
3. **ASR transcription** — transcribes each speaker segment with `faster-whisper` (model: `medium`, language: `de`)
4. **openSMILE extraction** — extracts eGeMAPS v02 Low-Level Descriptors per segment

**Key functions:**

- `filter_short_segments(result, min_duration=1.1)` — removes diarisation segments shorter than `min_duration` seconds to reduce spurious turns
- `assign_speaker_roles(segments)` — assigns Interviewer/Participant labels; falls back to raw speaker IDs if the expected number of unique speakers is not found
- `align_transcription_with_speakers(whisper_segments, speaker_segments)` — matches Whisper transcript spans to speaker turns via temporal overlap

**Outputs:** One `_diarization.csv` and `_aligned_segments.csv` per recording in `OUTPUT_DIR`.

---

### `alignment.ipynb`

Parses ELAN `.eaf` XML files and aligns labelled annotation intervals with speaker diarisation to produce the master annotation table.

**Key configuration constants:**

| Constant | Default | Description |
|----------|---------|-------------|
| `ELAN_DIR` | `data/annotations/ELAN` | Directory containing `.eaf` files |
| `DIAR_DIR` | `data/processed/diarization` | Directory of diarisation CSVs |
| `ACTIVE_CONSTRUCTS` | `['CH','IS','MF','OI','SJ','SK']` | Six constructs |
| `MAX_GAP_SEC` | `0.8` | Max gap (s) between annotations to merge |
| `MIN_SEG_SEC` | `1.0` | Minimum segment duration retained |
| `TAU_KEEP` | `0.20` | Min overlap fraction to keep a segment |

**Tier → label mapping:**

```python
TIER_TO_LABEL = {
    "self_kindness":   "SK",
    "self_judgement":  "SJ",
    "common_humanity": "CH",
    "isolation":       "IS",
    "mindfulness":     "MF",
    "over_identified": "OI",
}
```

**Key functions:**

- `canonical_label_from_tier(tier_id)` — maps raw ELAN tier names to short codes via `TIER_TO_LABEL`; case-insensitive substring match
- `parse_eaf_intervals(eaf_path)` — reads `TIME_SLOT` elements and `ALIGNABLE_ANNOTATION` entries; converts milliseconds to seconds
- `overlap(a, b)` — scalar overlap duration between two `Interval` objects
- `merge_with_gaps(intervals, max_gap)` — merges adjacent intervals within `MAX_GAP_SEC`, preserving member list in `.meta`
- `merge_union(intervals, max_gap)` — lightweight union-merge for coverage calculations
- `intersection_length(a, b)` — O(n+m) sweep over two sorted interval lists

**Output:** `elan_master.csv` — one row per annotation interval with columns: `video_id`, `tier`, `label`, `t_start`, `t_end`, `speaker`, `role`.

---

### `align_check.py`

Standalone validation script. Loads an `.eaf` file alongside an `aligned_segments.csv` (produced by `alignment.ipynb`) and checks how many annotated intervals overlap with transcribed speech.

**Usage:**
```bash
python align_check.py
```

```python
eaf_file       = "Pro.eaf"
alignment_csv  = "aligned_segments.csv"
output_csv     = "self_compassion_overlap_analysis.csv"
```

**Key function:**
- `check_overlap(eaf_start, eaf_end, seg_start, seg_end)` — returns `True` if the two intervals overlap (i.e., `not (seg_end < eaf_start or seg_start > eaf_end)`)

**Output:** `self_compassion_overlap_analysis.csv` with columns `tier`, `start`, `end`, `combined_transcript`, `who_said_what`.

---

### `whisper_transcribe.ipynb`

Standalone ASR notebook that (re-)transcribes recordings at word level, reading from a `diarization_index.csv`. Useful when re-transcribing is needed independently of the full `audio_features.ipynb` pipeline.

**Key configuration constants:**

| Constant | Default | Description |
|----------|---------|-------------|
| `OUTPUT_DIR` | `../data/OutputDiar` | Directory with diarisation index and WAV files |
| `ASR_MODEL` | `"medium"` | Overridable via `SC_ASR_MODEL` environment variable |
| `DEVICE` | `"cuda"` | Inference device |
| `DTYPE` | `"float16"` | Compute precision |
| `MIN_SIL` | `1.5` | Minimum silence (s) to mark a speaker turn boundary |

**Key functions:**

- `get_asr()` — lazy singleton: loads the `WhisperModel` once and reuses it across cells
- `transcribe_words_df(wav_path)` — runs `model.transcribe()` with `beam_size=5`, VAD filtering, and `word_timestamps=True`; returns a DataFrame with columns `start`, `end`, `word`
- `infer_roles_from_aligned(diar_csv)` — infers speaker roles from an existing aligned CSV via majority-vote per speaker label
- `words_in_window(words_df, start, end)` — clips a word DataFrame to a time window, adjusting boundary words
- `continuous_blocks_from_diar_exact(diar_df, words_df, min_sil, roles_map)` — merges diarisation segments separated by silence < `MIN_SIL` into continuous blocks, attaching transcribed words to each

---

### `windows_labels_and_splits.ipynb`

Generates the training dataset by sliding a fixed window across each recording and computing multi-label targets.

**Key configuration constants:**

| Constant | Default | Description |
|----------|---------|-------------|
| `ELAN_MASTER` | `"elan_master.csv"` | Master annotation file |
| `OUT_DIR` | `"window/"` | Output directory |
| `WIN` | `4.0` | Window duration in seconds |
| `HOP` | `1.0` | Stride between windows in seconds |
| `TRAIN_FRAC` | `0.8` | Fraction of videos for training |
| `VAL_FRAC` | `0.1` | Fraction of videos for validation |
| `BG_RATIO` | `2.0` | Max background-to-positive window ratio per video |
| `SEED` | `42` | Random seed |

**Label map:**
```python
LABEL_MAP = {
    "MF": ["mindfulness"],
    "SK": ["self_kindness", "common_humanity"],
    "NEG": ["self_judgement", "over_identified"],
}
```

**Key functions:**

- `overlap_fraction(a_start, a_end, b_start, b_end)` — fraction of window `[b_start, b_end]` covered by annotation `[a_start, a_end]`; used to compute the continuous label score `y ∈ [0, 1]`
- `build_windows(t_max, win, hop)` — generates a DataFrame of `(w_start, w_end, w_center)` rows for a recording of length `t_max`
- `label_windows_for_video(df_vid, video_id, win, hop, label_map)` — applies `overlap_fraction` for each label and window; produces columns `y_{label}` (soft), `mask_{label}` (binary present/absent), `mask_any`
- `make_video_strat_table(elan_filtered, label_map)` — creates a stratification key per video based on binned annotation durations (bins: 0, 0–2, 2–5, 5–10, 10+ minutes); used for stratified splitting to maintain label balance

**Outputs:**
- `window/windows_train.csv`, `windows_val.csv`, `windows_test.csv`
- `window/splits_freeze.json` — frozen video ID lists for reproducibility

**Window CSV schema:**

| Column | Type | Description |
|--------|------|-------------|
| `video_id` | str | Recording identifier |
| `w_start` | float | Window start time (s) |
| `w_end` | float | Window end time (s) |
| `w_center` | float | Window midpoint (s) |
| `y_MF`, `y_SK`, `y_SJ` | float | Overlap fraction ∈ [0, 1] |
| `mask_MF`, `mask_SK`, `mask_SJ` | int | 1 if y > 0, else 0 |
| `mask_any` | int | 1 if any label present |
| `is_bg_negative` | int | 1 if sampled background window |

---

### `MediaPipe.ipynb`

Extracts per-frame visual features from raw video files using Google MediaPipe.

**Key configuration constants:**

| Constant | Default | Description |
|----------|---------|-------------|
| `ROOT` | `../data/MP_640/` | Directory of raw video files |
| `FACE_TASK_MODEL` | `model/face_landmarker.task` | Path to MediaPipe FaceLandmarker model |
| `TARGET_FPS` | `10` | Frames extracted per second |
| `FACE_SELECTION_STRATEGY` | `"max_presence"` | How to choose among multiple detected faces |

**Feature dimensions:**

| Feature group | Count | Description |
|--------------|-------|-------------|
| Blendshapes | 52 | Facial action unit scores (e.g. `eyeBlinkLeft`, `mouthSmileLeft`, `browDownRight`); values ∈ [0, 1] |
| Pose keypoints | 24 | 6 upper-body landmarks (leftShoulder, rightShoulder, leftElbow, rightElbow, leftWrist, rightWrist) × 4 coordinates (x, y, z, visibility) |


**Key functions:**
- `create_face_landmarker(model_path)` — initialises a MediaPipe `FaceLandmarker` task; raises `FileNotFoundError` with a clear message if the model file is missing
- `find_videos(root)` — recursively finds all video files with extensions `{.mp4, .mov, .mkv, .avi}`

**Output:** One CSV per video saved to `data/processed/mediapipe/{video_id}{BACKBONE_SUFFIX}` with columns for `timestamp_sec` plus all 76 features.

---

### `feature_aggregation.ipynb`

Joins all modality feature tables and aggregates per-frame values into per-window summary statistics.

**Key configuration constants:**

| Constant | Default | Description |
|----------|---------|-------------|
| `MP_DIR` | `data/processed/mediapipe` | MediaPipe feature CSVs |
| `OS_DIR` | `data/processed/opensmile` | openSMILE feature CSVs |
| `WH_DIR` | `data/processed/whisper` | Whisper transcript CSVs |
| `SEGMENTS_WITH_SPLIT` | `data/out/dataset/segments_with_split.csv` | Output segments file |
| `FLOAT32_OUT` | `True` | Cast all numeric output to float32 |

**Key functions:**

- `summarize_vector(x, prefix)` — computes 8 statistics for a 1D array and returns them as a flat dict with `{prefix}_{stat}` keys: `mean`, `median`, `std`, `iqr`, `min`, `max`, `n`, `valid_pct`
- `interval_overlap(a0, a1, b0, b1)` — scalar overlap between two intervals
- `rowwise_overlap_fraction(rows_start, rows_end, seg_start, seg_end)` — vectorised overlap computation for a batch of rows against one segment; used to assign frames to windows efficiently
- `midpoint_in_segment(times, seg_start, seg_end)` — boolean mask for frames whose timestamp falls inside a window
- `rising_edge_rate(x, t, thr)` — rate of threshold crossings per second (e.g. blink rate, gesture peaks)
- `ensure_float32(df)` — casts all numeric columns to `float32` to reduce memory footprint

**Output:** Per-split CSVs with one row per window and columns for all aggregated features plus label columns from `windows_{split}.csv`.
