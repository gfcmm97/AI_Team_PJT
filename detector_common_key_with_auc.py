import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.metrics import precision_recall_curve, auc

# =========================
# 1. Paths
# =========================
PRED_FILES = {
    "convnet": "shared_results/convnet_128_preds.csv",
    "resnet": "shared_results/resnet_128_preds.csv",
    "sit_512": "shared_results/sit_stem_512_preds.csv",
    "sit_relu_768": "shared_results/sit_stem_ReLU_768_preds.csv",
    "rocket": "shared_results/rocket_128_preds.csv",
    "knn": "shared_results/knn_1024_preds.csv",
}

GT_PATH = "/home/ygang/Documents/anomaly/MSAD/data/TSB/metrics/final_labeled_dataset_converted_v2.csv"
SAVE_DIR = Path("shared_results")
SAVE_DIR.mkdir(exist_ok=True)

# =========================
# 2. Load predictions
# =========================
pred_dfs = {}
key_sets = []

for name, path in PRED_FILES.items():
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str).str.replace("\\", "/", regex=False)
    pred_dfs[name] = df
    key_sets.append(set(df.index))

# =========================
# 3. Common keys
# =========================
common_keys = sorted(set.intersection(*key_sets))
print(f"[INFO] Common keys = {len(common_keys)}")

# =========================
# 4. Detector (majority vote)
# =========================
detector_rows = []

for fname in common_keys:
    votes = []
    for name, df in pred_dfs.items():
        cls_col = [c for c in df.columns if c.endswith("_class")][0]
        votes.append(df.loc[fname, cls_col])

    counter = Counter(votes)
    final_label, vote_count = counter.most_common(1)[0]

    detector_rows.append({
        "fname": fname,
        "detector": final_label,
        "vote_count": vote_count,
        "total_models": len(votes),
        "vote_ratio": vote_count / len(votes)
    })

detector_df = pd.DataFrame(detector_rows).set_index("fname")
detector_df.to_csv(SAVE_DIR / "final_detector_common_key.csv")
print("[INFO] Detector saved")

# =========================
# 5. Load GT
# =========================
gt = pd.read_csv(GT_PATH)
gt["fname"] = gt["dataset"] + "/" + gt["filename"]
gt = gt.set_index("fname")

merged = detector_df.join(gt, how="inner")
print(f"[INFO] Matched with GT = {len(merged)}")

# =========================
# 6. AUC-PR
# =========================
# binary: anomaly = 1, normal = 0
y_true = (merged["label"] != "NORMA").astype(int)
y_score = merged["vote_ratio"]

precision, recall, _ = precision_recall_curve(y_true, y_score)
auc_pr = auc(recall, precision)

auc_df = pd.DataFrame({
    "AUC_PR": [auc_pr],
    "num_samples": [len(merged)]
})

auc_df.to_csv(SAVE_DIR / "final_auc_pr_common_key.csv", index=False)
print(f"[RESULT] Final AUC-PR = {auc_pr:.6f}")