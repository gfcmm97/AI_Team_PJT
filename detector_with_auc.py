import os
import pandas as pd
from collections import Counter
from sklearn.metrics import average_precision_score

# -------------------------
# 경로 설정
# -------------------------
model_pred_paths = [
    "/home/ygang/Documents/anomaly/MSAD/shared_results/convnet_128_preds.csv",
    "/home/ygang/Documents/anomaly/MSAD/shared_results/knn_1024_preds.csv",
    "/home/ygang/Documents/anomaly/MSAD/shared_results/resnet_128_preds.csv",
    "/home/ygang/Documents/anomaly/MSAD/shared_results/rocket_128_preds.csv",
    "/home/ygang/Documents/anomaly/MSAD/shared_results/sit_stem_512_preds.csv",
    "/home/ygang/Documents/anomaly/MSAD/shared_results/sit_stem_ReLU_768_preds.csv",
]

label_path = "/home/ygang/Documents/anomaly/MSAD/final_labeled_dataset.csv"

detector_output_csv = "/home/ygang/Documents/anomaly/MSAD/shared_results/final_detector_results.csv"
auc_output_csv = "/home/ygang/Documents/anomaly/MSAD/shared_results/final_auc_pr_results.csv"

# -------------------------
# 1) CSV 로딩 + index 정규화
# -------------------------
dfs = []
for p in model_pred_paths:
    df = pd.read_csv(p, index_col=0)
    df.index = df.index.map(str)
    df.index = df.index.str.replace("\\", "/", regex=False)
    dfs.append(df)

# -------------------------
# 2) 공통된 fnames만 사용
# -------------------------
common_fnames = set(dfs[0].index)
for df in dfs[1:]:
    common_fnames &= set(df.index)

common_fnames = sorted(list(common_fnames))
print(f"[INFO] Total common fnames across all models = {len(common_fnames)}")

# -------------------------
# 3) 모델 이름 추출
# -------------------------
model_names = [
    os.path.basename(p).replace("_preds.csv", "")
    for p in model_pred_paths
]

# -------------------------
# 4) 다수결 Detector 생성
# -------------------------
rows = []
for fname in common_fnames:
    votes = []
    for df in dfs:
        cls_col = [c for c in df.columns if c.endswith("_class")][0]
        try:
            votes.append(df.loc[fname, cls_col])
        except KeyError:
            votes.append("MISSING")

    clean_votes = [v for v in votes if v != "MISSING"]
    if len(clean_votes) == 0:
        continue

    final_label = Counter(clean_votes).most_common(1)[0][0]
    rows.append([fname, *votes, final_label])

detector_cols = ["fname"] + model_names + ["final_detector"]
detector_df = pd.DataFrame(rows, columns=detector_cols).set_index("fname")

detector_df.to_csv(detector_output_csv)
print(f"[INFO] Detector CSV saved → {detector_output_csv}")

# -------------------------
# 5) GT label 로딩 + fname 생성
# -------------------------
labels = pd.read_csv(label_path)

# GT 파일 형태 검증 후 변환
if set(labels.columns) == {"dataset", "filename", "label"}:
    labels["fname"] = labels["dataset"].astype(str) + "/" + labels["filename"].astype(str)
else:
    raise ValueError("GT file must contain dataset, filename, label columns")

labels["fname"] = labels["fname"].str.replace("\\", "/", regex=False)
labels.set_index("fname", inplace=True)

print("[INFO] GT labels converted & indexed")

# -------------------------
# 6) Detector + Label 매칭
# -------------------------
merged = detector_df.join(labels, how="inner")
print(f"[INFO] Total matched rows with GT labels = {len(merged)}")

# anomaly label: 정상=0, 나머지=1
y_true = (merged["label"] != "NORMA").astype(int)

# -------------------------
# 7) detector class → anomaly score 변환
# -------------------------
anomaly_scores = merged["final_detector"].apply(lambda x: 0 if x == "NORMA" else 1)

# -------------------------
# 8) AUC-PR 계산
# -------------------------
auc_pr = average_precision_score(y_true, anomaly_scores)

pd.DataFrame({"metric": ["AUC_PR"], "value": [auc_pr]}).to_csv(auc_output_csv, index=False)

print(f"[INFO] AUC-PR saved → {auc_output_csv}")
print(f"[RESULT] Final AUC-PR = {auc_pr:.6f}")