import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path("preds_unimodal")
WINDOW_DIR = Path("window")
SPLIT = "test"
MODALITY = "video"
TAU = 0.2

LABELS_DATA = ["MF", "SK", "SJ"]
LABELS_DISPLAY = ["MF", "SK", "Neg"]


probs  = np.load(ROOT / SPLIT / MODALITY / f"probs_{SPLIT}.npy").astype(np.float32)
logits = np.load(ROOT / SPLIT / MODALITY / f"logits_{SPLIT}.npy").astype(np.float32)

present = np.load(ROOT / SPLIT / MODALITY / f"present_{SPLIT}.npy").astype(np.float32)


win_path = WINDOW_DIR / f"windows_{SPLIT}.csv"
win_df = pd.read_csv(win_path)

print("Loaded:", win_path)
print("Columns:", list(win_df.columns))

y_cols = {"MF": "y_MF", "SK": "y_SK", "SJ": "y_SJ"}
m_cols = {"MF": "mask_MF", "SK": "mask_SK", "SJ": "mask_SJ"}

for lab in LABELS_DATA:
    assert y_cols[lab] in win_df.columns, f"Missing {y_cols[lab]}"
    assert m_cols[lab] in win_df.columns, f"Missing {m_cols[lab]}"

Y_cont = win_df[[y_cols["MF"], y_cols["SK"], y_cols["SJ"]]].values.astype(np.float32)
M_lab  = win_df[[m_cols["MF"], m_cols["SK"], m_cols["SJ"]]].values.astype(np.float32)


Y = (Y_cont >= TAU).astype(np.int32)

AP, AUC, SUP = {}, {}, {}

for i, lab in enumerate(LABELS_DATA):
    keep = (present > 0.5) & (M_lab[:, i] > 0.5)

    y_true = Y[keep, i].astype(np.int32)
    y_score = probs[keep, i].astype(np.float32)

    SUP[lab] = int(keep.sum())

    if SUP[lab] == 0 or len(np.unique(y_true)) < 2:
        AP[lab] = np.nan
        AUC[lab] = np.nan
    else:
        AP[lab]  = float(average_precision_score(y_true, y_score))
        AUC[lab] = float(roc_auc_score(y_true, y_score))

for lab in LABELS_DATA:
    print(f" {lab}: AP={AP[lab]:.3f} | AUC={AUC[lab]:.3f} | N={SUP[lab]}")


ap_vals  = [AP["MF"], AP["SK"], AP["SJ"]]
auc_vals = [AUC["MF"], AUC["SK"], AUC["SJ"]]

x = np.arange(len(LABELS_DISPLAY))
w = 0.35

plt.figure(figsize=(6,4))
plt.bar(x - w/2, ap_vals,  width=w, label="PR-AUC (AP)")
plt.bar(x + w/2, auc_vals, width=w, label="ROC-AUC")

plt.xticks(x, LABELS_DISPLAY)
plt.ylim(0, 1.0)
plt.ylabel("Score")
plt.title(f"Video-only Baseline ({SPLIT})  τ={TAU}")
plt.legend(frameon=False)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
