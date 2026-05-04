# fusion/

This folder implements **late fusion** of the three unimodal prediction streams — audio, video, and text — into a single multi-label prediction. It must be run after all three unimodal pipelines have saved their prediction arrays.

---

## Files

| File | Purpose |
|------|---------|
| `fusion.ipynb` | Load unimodal predictions, fuse by confidence-weighted averaging, evaluate all combinations, and save final fused arrays |

---

## Dependencies

The notebook requires the following outputs from upstream steps:

| Source | Path | Required |
|--------|------|---------|
| `audio_train_eval.ipynb` | `preds_unimodal/{split}/audio/` | probs, conf, present `.npy` |
| `video_train_eval.ipynb` | `preds_unimodal/{split}/video/` | probs, conf, present `.npy` |
| `LLM_Unimodal.ipynb` or `llm-qlora.ipynb` | CSV path set in `TEXT_PRED_PATH` | `prob_{label}`, `conf_{label}` per window |
| `windows_labels_and_splits.ipynb` | `window/windows_val.csv` | ground truth labels and window IDs |

---

## Notebook Details

### `fusion.ipynb`

**Key configuration constants:**

| Constant | Default | Description |
|----------|---------|-------------|
| `LABELS` | `["MF","SK","SJ"]` | Target labels |
| `WIN_PATH` | `"windows_val.csv"` | Window ground-truth table |
| `AUDIO_DIR` | `"preds_unimodal/test/audio"` | Audio prediction arrays |
| `VIDEO_DIR` | `"preds_unimodal/test/video"` | Video prediction arrays |
| `TEXT_PRED_PATH` | path to LLaMA SFT CSV | Text predictions (CSV with per-window `prob_{label}` and `conf_{label}`) |
| `TEXT_VARIANT_NAME` | `"text_with_context"` | Label used in summary tables for this text variant |
| `OUT_DIR` | `"fusion_{timestamp}/"` | Output directory (timestamped) |

---

## Fusion Strategy

### Loading unimodal outputs

```python
def load_modality_outputs(mod_dir, split):
    probs = np.load(mod_dir / f"probs_{split}.npy")   # [N, 3]
    conf  = np.load(mod_dir / f"conf_{split}.npy")    # [N] or [N, 3]
    pres  = np.load(mod_dir / f"present_{split}.npy") # [N]
    return probs, conf, pres
```

### Text alignment

Text predictions are stored per-transcript-span (start_ms / end_ms) rather than per-window. The notebook aligns them to windows by matching `video_id` and millisecond timestamps. When multiple text spans overlap a single window, they are combined using confidence-weighted averaging over those spans.

### Confidence-weighted late fusion

```python
def fuse_conf_weighted(probs_list, confs_list, base_weights=None, eps=1e-8):
    """
    Weighted average of probabilities, where each modality's weight is scaled
    by its own confidence at each window and label:

        fused = Σ(w_i * p_i * c_i) / Σ(w_i * c_i)

    base_weights: per-modality scalar weights (default: 1.0 for all)
    """
```

Scalar confidence arrays (shape `[N]`) are broadcast to `[N, 3]` via `broadcast_conf` before fusion, so each label can be weighted independently if label-specific confidences are available.

### Fusion combinations evaluated

The notebook evaluates all seven combinations:

| Name | Modalities |
|------|-----------|
| `audio` | Audio only (baseline) |
| `video` | Video only (baseline) |
| `text_with_context` | Text only (baseline) |
| `fused_text+audio` | Text + Audio |
| `fused_text+video` | Text + Video |
| `fused_audio+video` | Audio + Video |
| `fused_text+audio+video` | All three |

### Validity masking

Two masks are used:
- `mask_av` — windows valid for audio-video evaluation: `mask==1 AND audio_present==1 AND video_present==1`
- `mask_text_eval` — additionally requires `text_present==1` for text-inclusive combinations

---

## Evaluation

For each combination, the notebook computes:

| Metric | Description |
|--------|-------------|
| Macro F1 | Unweighted average F1 across labels |
| Micro F1 | F1 computed globally across all label instances |
| Micro Precision / Recall | Overall precision and recall |
| Per-class Precision / Recall / F1 | Per-label breakdown |

Hard predictions use a threshold of `prob ≥ 0.5`.

Results are printed to the console and saved to:
- `fusion_{timestamp}/fusion_summary_{run_id}.csv` — flat summary table (one row per combination)
- `fusion_{timestamp}/fusion_metrics_{run_id}.json` — full nested metrics including per-class breakdowns

---

## Output Files

```
fusion_{timestamp}/
  fusion_summary_{run_id}.csv       ← Macro/micro F1, precision, recall per combination
  fusion_metrics_{run_id}.json      ← Full per-class metrics in JSON
  plots/
    {name}_prf.png                  ← Precision/Recall/F1 bar chart per combination
    {name}_confusion_{label}.png    ← Confusion matrix per combination per label
  fused_probs_test.npy              ← [N, 3] final fused probabilities (TAV fusion)
  fused_conf_test.npy               ← [N, 3] confidence of fused predictions
  fused_present_test.npy            ← [N] validity mask for fused predictions
```

The final fused arrays (`fused_probs_test.npy`, etc.) use the **text+audio+video** combination (`F_TAV`) by default. These can be used directly for any downstream task requiring final predictions.

---

## Diagnostic Plots

For each fusion combination, two plot types are saved:

**PRF bar chart (`{name}_prf.png`):**
Three grouped bars per label — Precision, Recall, F1 — for a quick visual comparison of the precision/recall trade-off per construct.

**Confusion matrix (`{name}_confusion_{label}.png`):**
2×2 confusion matrix (TN, FP, FN, TP) per label, with counts annotated in each cell.
