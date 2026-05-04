# video/

This folder contains the video-only modality pipeline: a neural classifier trained on MediaPipe-derived facial and pose features, plus a standalone evaluation and visualisation script.

---

## Files

| File | Purpose |
|------|---------|
| `video_train_eval.ipynb` | Train and evaluate a multi-label MLP on MediaPipe features; save prediction arrays |
| `plot.py` | Load saved prediction arrays and plot AP / ROC-AUC bar charts |

---

## Dependencies

Both files depend on outputs from `preprocessing/`:

- `window/windows_{split}.csv` — window metadata and labels
- `window/splits_freeze.json` — frozen video ID splits
- `processed/mediapipe/{video_id}.csv` — per-frame MediaPipe features

---

## Notebook Details

### `video_train_eval.ipynb`

Trains and evaluates a video-only multi-label classifier using per-frame MediaPipe blendshape and pose features aggregated over 4-second windows.

**Key configuration constants:**

| Constant | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `~/Thesis/project/data/processed` | Root data directory |
| `SPLITS_JSON` | `DATA_DIR/window/splits_freeze.json` | Frozen split file |
| `WIN_TRAIN_CSV` | `DATA_DIR/window/windows_train.csv` | Training window table |
| `WIN_VAL_CSV` | `DATA_DIR/window/windows_val.csv` | Validation window table |
| `WIN_TEST_CSV` | `DATA_DIR/window/windows_test.csv` | Test window table |
| `MEDIAPIPE_DIR` | `DATA_DIR/mediapipe` | Per-video MediaPipe CSVs |
| `VIDEO_DIR` | `~/Thesis/project/data/raw/video` | Raw `.mp4` files |
| `LABELS` | `["MF","SK","SJ"]` | Target labels |
| `WINDOW_SEC` / `HOP_SEC` | read from `splits_freeze.json` | Must match preprocessing |

**Key functions:**

- `standardize_windows(df)` — robustly renames columns to canonical names (`video_id`, `w_start`, `w_end`, `mask`) regardless of source naming variants; validates label columns; drops rows with invalid windows
- `standardize_mediapipe(df)` — normalises the timestamp column, accepting any of `timestamp_sec`, `time_sec`, `t_sec`, `timestamp`, or `time`

**Feature pipeline:**

```
For each window:
  1. Load the MediaPipe CSV for the corresponding video
  2. Select frames where timestamp ∈ [w_start, w_end]
  3. Aggregate per-frame features using summarize_vector (mean, median, std, iqr, min, max, n, valid_pct)
  4. Result: one feature vector per window
```

The total per-window feature dimension is `76 features × 8 statistics = 608 dimensions`, plus derived rate features (e.g. blink rate).

**Model architecture:** Fully connected MLP with sigmoid multi-label output. Architecture details (hidden sizes, dropout, batch norm) are configurable via constants in the notebook.

**Evaluation:** Per-label AP (PR-AUC) and ROC-AUC. Only windows where `present > 0.5` and `mask_{label} > 0.5` are included in each label's evaluation. The notebook also computes **gradient saliency** and **permutation importance** over the test set to identify the most informative features and time-steps within the 4-second window.

**Output files:**

```
preds_unimodal/{split}/video/
  probs_{split}.npy        ← [N, 3] sigmoid probabilities
  logits_{split}.npy       ← [N, 3] raw logits
  conf_{split}.npy         ← [N] or [N, 3] confidence scores
  present_{split}.npy      ← [N] validity mask
```

---

## Script Details

### `plot.py`

Standalone evaluation and visualisation script. Loads pre-saved prediction arrays and the test window CSV to compute and visualise per-label AP and ROC-AUC.

**Usage:**
```bash
python plot.py
```

All paths are configured via constants at the top of the file.

**Key configuration constants:**

| Constant | Default | Description |
|----------|---------|-------------|
| `ROOT` | `Path("preds_unimodal")` | Root of saved prediction arrays |
| `WINDOW_DIR` | `Path("window")` | Directory containing window CSVs |
| `SPLIT` | `"test"` | Which split to evaluate |
| `MODALITY` | `"video"` | Sub-directory name within `ROOT/{SPLIT}/` |
| `TAU` | `0.2` | Label binarisation threshold |
| `LABELS_DATA` | `["MF","SK","SJ"]` | Internal column order |
| `LABELS_DISPLAY` | `["MF","SK","Neg"]` | Display aliases (`SJ` shown as `"Neg"`) |

**Evaluation logic:**

For each label `L`:
```python
keep = (present > 0.5) & (mask_L > 0.5)
y_true  = (y_L[keep] >= TAU).astype(int)
y_score = probs[keep, i]
AP  = average_precision_score(y_true, y_score)
AUC = roc_auc_score(y_true, y_score)
```

**Output:** Prints a per-label summary table and displays a grouped bar chart (PR-AUC vs ROC-AUC for MF, SK, NEG).
