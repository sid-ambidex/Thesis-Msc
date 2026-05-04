# LLM/

This folder contains the text-based modality pipeline using large language models (LLaMA). Two complementary approaches are provided:

| Approach | Notebook | Description |
|----------|---------|-------------|
| Zero-shot inference | `LLM_Unimodal.ipynb` | Prompts a pre-trained LLaMA model in German without any task-specific training |
| Supervised fine-tuning | `llm-qlora.ipynb` | Fine-tunes LLaMA using QLoRA (4-bit quantisation + LoRA adapters) on labelled samples |

Both produce prediction arrays in the same format for downstream fusion.

---

## Files

| File | Purpose |
|------|---------|
| `LLM_Unimodal.ipynb` | Zero-shot/few-shot multi-label classification + diagnostic visualisation |
| `llm-qlora.ipynb` | QLoRA supervised fine-tuning of a LLaMA causal language model |

---

## Dependencies

- `window/windows_{split}.csv` — window metadata and labels (from `preprocessing/`)
- `processed/whisper/{video_id}.csv` — word-level transcripts with timestamps (from `preprocessing/`)
- For `llm-qlora.ipynb`: `llm/all_samples.csv` — pre-built sample table with `context_text`, `target_text`, label columns

---

## Model

Both notebooks use **`meta-llama/Llama-3.2-3B-Instruct`**. Download via HuggingFace:

```bash
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct
```

Requires a HuggingFace account and accepted model terms. Set `MODEL_DIR` to the local snapshot path.

---

## Notebook Details

### `LLM_Unimodal.ipynb`

Performs window-level multi-label classification by prompting LLaMA in German with a structured [CONTEXT] + [TARGET] format.

**Key configuration constants:**

| Constant | Default | Description |
|----------|---------|-------------|
| `WIN_TRAIN/VAL/TEST` | see notebook | Paths to window CSV files |
| `WHISPER_DIR` | `data/processed/whisper` | Directory with per-video transcript CSVs |
| `PRED_OUT_ROOT` | `data/processed/preds_unimodal` | Output root for prediction arrays |
| `MODEL_DIR` | HuggingFace cache path | Local LLaMA snapshot directory |
| `MAX_LEN` | `1800` | Maximum prompt token length |
| `CONTEXT_LOOKBACK_S` | `15.0` | How far back (s) to gather prior conversation context |
| `TARGET_PAD_S` | `2.0` | Expand window by ±2 s when collecting speech |
| `MIN_TEXT_CHARS` | `20` | Minimum target text length; shorter → `present=0` |
| `MIN_OVERLAP_MS` | `250` | Minimum ms overlap to match a transcript turn to a window |
| `MIN_OVERLAP_FRAC` | `0.25` | Minimum fraction overlap for turn-window matching |
| `GEN_KW` | `{max_new_tokens: 40, do_sample: False}` | Generation parameters |

**Prompt structure:**

Each window is formatted as:
```
[CONTEXT]
<prior conversation turns within CONTEXT_LOOKBACK_S seconds>

[TARGET]
<speech within the 4s window ± TARGET_PAD_S>
```

**System prompt language:** German. The model is instructed to classify only the `[TARGET]` segment (not the context) and respond exclusively in JSON format.

**Expected JSON output:**
```json
{"MF": 0, "SK": 0, "NEG": 0, "conf": 0.5}
```

**Confidence levels:**

| Value | Meaning |
|-------|---------|
| `0.8` | Clear, explicit evidence in `[TARGET]` |
| `0.5` | Partial evidence, not strong |
| `0.2` | Uncertain / text too short / unclear |

**Key rules from prompt:**
- No label from events/actions alone; only explicit linguistic evidence
- Empty or filler text (e.g. *"äh"*, *"hm"*, *"ja"*) → all labels = 0, `conf` = 0.2
- Pauses and disfluencies are ignored unless co-occurring with clear self-evaluative language

**Diagnostic utilities:**

The notebook includes:

- `plot_text_presence_conf_and_probs_from_disk(split, video_id, ...)` — plots presence, LLM confidence, and per-label probabilities over time for a given recording; saves a PNG to `llm_window_aligned/figs/`
- `top_lexical_contrast(dbg_df, label, topk)` — computes log-frequency-ratio word scores between positive and negative windows using the `target_text` column; helps identify discriminative vocabulary
- `error_buckets(win_df, dbg_df, probs, label, topn)` — returns the top false positives and false negatives sorted by predicted probability, for qualitative error inspection

**Output files:**

```
preds_unimodal/{split}/text/
  probs_{split}.npy        ← [N, 3] probabilities for MF, SK, SJ
  logits_{split}.npy       ← [N, 3] (conf broadcast to match shape)
  conf_{split}.npy         ← [N] scalar confidence per window
  present_{split}.npy      ← [N] 1 if text is long enough, else 0
```

---

### `llm-qlora.ipynb`

Fine-tunes LLaMA using [QLoRA]: 4-bit NF4 quantisation (bitsandbytes) with LoRA adapters (PEFT). This enables fine-tuning a 3B-parameter model on a single consumer GPU.

**Key configuration constants:**

| Constant | Default | Description |
|----------|---------|-------------|
| `DATA_CSV` | `"llm/all_samples.csv"` | Training sample table |
| `MODEL_DIR` | HuggingFace cache | Base LLaMA model directory |
| `OUT_DIR` | `"llm/finetune_runs"` | Output directory for checkpoints |
| `MAX_LEN` | `1800` | Max sequence length |
| `BATCH_SIZE` | `1` | Per-device batch size |
| `GRAD_ACCUM` | `16` | Gradient accumulation steps (effective batch = 16) |
| `LR` | `2e-4` | Learning rate for LoRA parameters |
| `EPOCHS` | `3` | Training epochs |
| `WARMUP_RATIO` | `0.05` | Linear warmup fraction |
| `HOLDOUT_FRAC` | `0.10` | Internal validation holdout from training data |
| `SEED` | `42` | Random seed |

**Input CSV schema (`llm/all_samples.csv`):**

| Column | Type | Description |
|--------|------|-------------|
| `context_text` | str | Prior conversation turns |
| `target_text` | str | Speech in the target window |
| `target_speaker` | str | Speaker role of the target |
| `context_speaker` | str | Speaker role of the context (optional) |
| `start_ms`, `end_ms` | int | Window boundaries in milliseconds |
| `MF`, `SK`, `NEG` | int | Binary ground-truth labels (0 or 1) |
| `file_id` / `video_id` | str | Recording identifier |

**Context construction (`build_model_input_ctx`):**

Two policies are applied depending on whether the speaker changed between context and target:

| Case | Context used | Rationale |
|------|-------------|-----------|
| Speaker changed (interviewer question → participant reply) | Up to `CTX_DIFF_MAX_CHARS` (600) chars of prior context | Interviewer's full question helps the model understand the topic |
| Same speaker continues | Last `CTX_SAME_N_WORDS` (40) words | Only recent prior speech is relevant |

**Speaker change detection (`_is_question_like`):**

A heuristic for German text checks for:
- A `?` in the last 120 characters, **or**
- A leading interrogative word: *warum, wieso, wie, was, wann, wo, welche, können, kannst, würdest, hast*

**Training format:** Each sample is formatted as a prompt with the JSON label as the completion target. The model is trained to produce the same `{"MF":…,"SK":…,"SJ":…,"conf":…}` JSON format as the zero-shot notebook, enabling the same decoding logic at inference time.

**Output:** LoRA adapter checkpoints saved under `OUT_DIR/`. Load with:
```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, adapter_path)
```
