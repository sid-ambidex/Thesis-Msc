# audio/

This folder contains the audio-only modality pipeline: acoustic feature extraction, classifier training and evaluation, and feature ablation. All three notebooks operate on the windowed dataset produced by `preprocessing/`.

---

## Files

| File | Purpose |
|------|---------|
| `opensmile.ipynb` | Extract eGeMAPS LLD features from WAV files for each time window |
| `audio_train_eval.ipynb` | Train and evaluate MLP and logistic regression classifiers; save predictions |
| `audio_feature_ablation.ipynb` | Leave-one-group-out feature ablation study |

---

## Run Order

```
1. opensmile.ipynb            ← must run first; produces audio feature CSVs
2. audio_train_eval.ipynb     ← trains models, saves prediction arrays
3. audio_feature_ablation.ipynb  ← (optional) ablation analysis
```

`opensmile.ipynb` depends on the window CSVs from `preprocessing/windows_labels_and_splits.ipynb`.

---

## Notebook Details

### `opensmile.ipynb`

Extracts [eGeMAPS v02](https://audeering.github.io/opensmile/about.html) Low-Level Descriptor (LLD) acoustic features for every window in the dataset.

For each window, the notebook:
1. Locates the corresponding WAV file
2. Slices the audio to the window boundaries
3. Runs openSMILE to extract LLD features
4. Rebases the feature timestamps to absolute recording time
5. Joins the features with the window table

**Key configuration constants:**

| Constant | Default | Description |
|----------|---------|-------------|
| `WIN_DIR` | `"window/"` | Directory with `windows_{split}.csv` files |
| `AUDIO_DIR` | `"audio/"` | Directory with per-recording WAV files |
| `OUT_DIR` | `"out/audio/"` | Output directory |
| `WIN` | `4.0` | Window duration in seconds (must match preprocessing) |
| `HOP` | `1.0` | Hop size in seconds (must match preprocessing) |

**Key functions:**

- `make_smile_lld()` — creates an `opensmile.Smile` instance configured for `eGeMAPSv02` at `LowLevelDescriptors` level (frame-level features)
- `to_seconds_index(df)` — normalises the mixed index types that openSMILE can return (`TimedeltaIndex`, `MultiIndex`, or plain numeric) to a uniform float-seconds index
- `get_wav_duration_seconds(wav_path)` — calls `ffprobe` to retrieve audio duration without loading the file; parses `Duration: HH:MM:SS.ss` from stderr
- `lld_chunk_to_seconds_index(df, chunk_start_s)` — offsets chunk-relative LLD timestamps by `chunk_start_s` to obtain absolute recording timestamps

**Feature set:** eGeMAPS v02 — 88 features covering pitch (F0), energy/loudness, spectral (MFCC, spectral flux, etc.), and voice quality (jitter, shimmer, HNR) descriptors.

**Output:** `out/audio/windows_{split}_with_audio.csv` — the window table augmented with per-window audio feature columns plus a `has_any_audio_feature` flag (1 if at least one frame of valid audio was found in the window).

---

### `audio_train_eval.ipynb`

Trains and evaluates audio-only classifiers for the three labels (`MF`, `SK`, `SJ`) and exports prediction arrays for downstream fusion.

**Key configuration constants:**

| Constant | Default | Description |
|----------|---------|-------------|
| `LABELS` | `["MF","SK","SJ"]` | Target labels |
| `TAU` | `0.2` | Overlap fraction threshold to binarise soft labels |
| `USE_ONLY_MASK_ANY` | `False` | If `True`, train only on windows with at least one annotated label |
| `PRESENT_SOURCE` | `"mask_any"` | Column used as the validity mask |
| `VAR_THRESH` | `1e-6` | Drop features with training variance below this |
| `BATCH` | `512` | MLP mini-batch size |
| `EPOCHS` | `20` | Maximum training epochs |
| `LR` | `1e-3` | AdamW learning rate |
| `WEIGHT_DECAY` | `1e-4` | AdamW weight decay |
| `PATIENCE` | `3` | Early stopping patience (validation loss) |

**Feature pipeline:**

```
Load windows_{split}_with_audio.csv
  → drop metadata columns (video_id, w_start, w_end, label cols, mask cols)
  → z-score normalise using training mean/std
  → fill NaN → 0
  → drop zero-variance features (var < VAR_THRESH)
```

The normalisation statistics (`mu`, `sd`) and feature column lists are saved to `out/audio/audio_models/audio_normalization.pkl` and `audio_feature_meta.pkl`.

**Models trained:**

| Model | Notes |
|-------|-------|
| `LogisticRegression` | One per label; `class_weight='balanced'`; `max_iter=1000` |
| MLP | Multi-label; sigmoid output; AdamW with early stopping on val loss |

**Evaluation metric:** Per-label AP (PR-AUC) and ROC-AUC on the test set. Only windows where `present > 0.5` **and** `mask_{label} > 0.5` are included in each label's evaluation.

**Output files:**

```
out/audio/audio_models/
  audio_normalization.pkl        ← {mu, sd} for inference
  audio_feature_meta.pkl         ← feature column lists
  logreg_{label}.pkl             ← logistic regression per label
  mlp_model.pt                   ← MLP weights

preds_unimodal/{split}/audio/
  probs_{split}.npy              ← [N, 3] sigmoid probabilities
  logits_{split}.npy             ← [N, 3] raw logits
  conf_{split}.npy               ← [N] confidence scores
  present_{split}.npy            ← [N] validity mask
```

---

### `audio_feature_ablation.ipynb`

Runs a **leave-one-feature-group-out** ablation to identify which acoustic feature groups contribute most to each label's AP score.

Uses the same data loading and z-score normalisation pipeline as `audio_train_eval.ipynb`. For each ablation run, one group of features is removed and a logistic regression is retrained. Results are printed and plotted as AP/AUC per group.

Feature groups correspond to eGeMAPS sub-categories (pitch, energy, spectral, cepstral, voice quality). This notebook does not produce files required by downstream steps; it is purely analytical.
