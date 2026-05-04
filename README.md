# Multimodal Machine Learning for Higher-Order Complex Behavioral Constructs 
## Self-Compassion in Structured Reflective Teacher Training

This repository contains the codebase for an MSc thesis on the **automatic detection of self-compassion constructs** in video-recorded reflective interviews with trainee teachers. The project implements an end-to-end multimodal machine learning pipeline — from raw recordings and expert annotations through to late-fusion predictions covering **audio**, **video**, and **text** modalities.

---

## Background

Self-compassion (Neff, 2003) describes how individuals relate to themselves during difficult experiences. This work operationalizes it as a **multi-label temporal classification problem**: given a 4-second window of a German-language interview, predict which of three primary constructs are present:

| Label | Construct(s) | Description |
|-------|-------------|-------------|
| `MF` | Mindfulness | Non-judgmental awareness of one's own thoughts and feelings |
| `SK` | Self-Kindness + Common Humanity | Treating oneself with warmth; recognizing shared human struggle |
| `NEG` | Self-Judgment + Over-Identification | Harsh self-criticism, rumination, or self-blame |

Human expert annotators labeled recordings using [ELAN]. The code processes raw `.mp4` recordings end-to-end, from speaker diarization and transcription through feature extraction, model training, and multimodal fusion.

---

## Repository Structure

```
Thesis-Msc/
│
├── preprocessing/          # Raw data → windowed, labelled feature dataset
│   ├── audio_features.ipynb          # Diarisation + Whisper ASR + openSMILE
│   ├── MediaPipe.ipynb               # Facial blendshape + pose extraction
│   ├── whisper_transcribe.ipynb      # Standalone ASR with word timestamps
│   ├── alignment.ipynb               # ELAN annotation parsing and alignment
│   ├── align_check.py                # Overlap validation script
│   ├── feature_aggregation.ipynb     # Per-window feature summarisation
│   └── windows_labels_and_splits.ipynb  # Sliding windows, labels, train/val/test splits
│
├── audio/                  # Audio-only model
│   ├── opensmile.ipynb               # eGeMAPS feature extraction per window
│   ├── audio_train_eval.ipynb        # MLP + logistic regression training & evaluation
│   └── audio_feature_ablation.ipynb  # Feature group ablation study
│
├── video/                  # Video-only model
│   ├── video_train_eval.ipynb        # MediaPipe-based model training & evaluation
│   └── plot.py                       # AP/AUC bar chart plotter
│
├── LLM/                    # Text-only model (LLaMA)
│   ├── LLM_Unimodal.ipynb            # Zero-shot inference + timeline visualisation
│   └── llm-qlora.ipynb               # QLoRA supervised fine-tuning
│
└── fusion/                 # Multimodal late fusion
    └── fusion.ipynb                  # Confidence-weighted fusion + full evaluation
```

---

## Pipeline Overview

```
Raw .mp4 recordings + ELAN .eaf annotations
         │
         ▼
 [preprocessing]
  1. Speaker diarisation (pyannote) + ASR (faster-whisper)
  2. Parse ELAN annotations → align with speech segments
  3. Extract features: MediaPipe (video), eGeMAPS (audio), Whisper (text)
  4. Slide 4s/1s-hop windows → compute soft labels → stratified splits
  5. Aggregate per-frame features into per-window summary vectors
         │
         ├──────────────────────────────────────────┐
         ▼                    ▼                     ▼
   [audio]               [video]                 [LLM]
  MLP / LogReg        MLP on MediaPipe       LLaMA-3.2-3B
  on eGeMAPS          blendshapes+pose       (zero-shot or
  features            features               QLoRA fine-tuned)
         │                    │                     │
         └──────────────────────────────────────────┘
                              │
                              ▼
                         [fusion]
              Confidence-weighted late fusion
              of audio + video + text probs
                              │
                              ▼
             Final per-window MF / SK / NEG predictions
```

---

## Data Format

### Inputs
- **Videos**: `.mp4` files, one per interview session
- **Annotations**: ELAN `.eaf` XML files with six tiers: `self_kindness`, `self_judgement`, `common_humanity`, `isolation`, `mindfulness`, `over_identified`
- **Language**: German

### Key Intermediate Files

| File | Description |
|------|-------------|
| `elan_master.csv` | All ELAN annotations aligned with diarised speech |
| `window/windows_{split}.csv` | Per-window labels and metadata (one row = one 4s window) |
| `window/splits_freeze.json` | Frozen train/val/test video ID lists |
| `processed/mediapipe/{id}.csv` | Per-frame facial and pose features |
| `processed/opensmile/{id}.csv` | Per-frame eGeMAPS acoustic features |
| `processed/whisper/{id}.csv` | Word-level transcripts with timestamps |

### Model Outputs (for fusion)
Each unimodal model saves three NumPy arrays per split under `preds_unimodal/{split}/{modality}/`:

| File | Shape | Description |
|------|-------|-------------|
| `probs_{split}.npy` | `[N, 3]` | Sigmoid probabilities for MF, SK, SJ |
| `logits_{split}.npy` | `[N, 3]` | Raw model logits |
| `conf_{split}.npy` | `[N]` or `[N, 3]` | Prediction confidence |
| `present_{split}.npy` | `[N]` | Binary validity mask (1 = window has usable features) |

---

## Environment Setup

### Requirements

```bash
pip install torch torchvision torchaudio          # PyTorch (CUDA build recommended)
pip install transformers peft bitsandbytes        # HuggingFace + QLoRA
pip install pyannote.audio                        # Speaker diarisation
pip install faster-whisper                        # ASR
pip install opensmile                             # Acoustic features
pip install mediapipe opencv-python               # Video features
pip install scikit-learn pandas numpy matplotlib
pip install pympi-mpi                             # ELAN file parsing
pip install ffmpeg-python tqdm
```

> **HuggingFace token**: `pyannote/speaker-diarization-3.1`. Set `HF_TOKEN` in `preprocessing/audio_features.ipynb`.

> **LLaMA model**: `meta-llama/Llama-3.2-3B-Instruct` requires a HuggingFace account.


## Expected Directory Layout

```
data/
  annotations/ELAN/         ← .eaf annotation files
  raw/video/                ← raw .mp4 recordings
  processed/
    diarization/            ← pyannote diarisation CSVs
    mediapipe/              ← per-video MediaPipe CSVs
    opensmile/              ← per-video openSMILE CSVs
    whisper/                ← per-video Whisper transcript CSVs
    window/                 ← windows_{train,val,test}.csv, splits_freeze.json
  out/
    dataset/                ← segments_with_split.csv, targets_multi.csv
    reports/                ← diagnostic reports

preds_unimodal/
  {split}/{modality}/       ← probs, logits, conf, present .npy arrays
```

---

## Running Order

Run notebooks in this order:

```
1. preprocessing/MediaPipe.ipynb
2. preprocessing/audio_features.ipynb
3. preprocessing/whisper_transcribe.ipynb
4. audio/opensmile.ipynb
5. preprocessing/alignment.ipynb 
6. preprocessing/align_check.py (optional)
7. preprocessing/feature_aggregation.ipynb
8. preprocessing/windows_labels_and_splits.ipynb
   ───────────────────────────
9a. audio/audio_train_eval.ipynb
9b. video/video_train_eval.ipynb
9c. LLM/llm-qlora.ipynb
9d. LLM/LLM_Unimodal.ipynb
   ───────────────────────────
10. fusion/fusion.ipynb
```

---

## Windowing Strategy

Windows are generated with a **4-second duration** and **1-second hop**. Labels are computed as the **overlap fraction** between each window and each annotation interval: a window is positive for label `L` if ≥ 20% of it overlaps with an annotation of construct `L`. Background negatives are included at up to 2× the positive count per video.

Splits are **stratified by video** (not by window) using a binned representation of annotation duration per label, ensuring label distribution is preserved across train/val/test.

---

## Label Mapping

| Short code | ELAN tier(s) |
|-----------|-------------|
| `MF` | `mindfulness` |
| `SK` | `self_kindness`, `common_humanity` |
| `NEG` | `self_judgement`, `over_identified` |

---

## License

See [LICENSE](LICENSE) for terms.
