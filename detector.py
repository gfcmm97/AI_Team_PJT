import os
import pandas as pd
from collections import Counter

# -------------------------------
# 설정: 사용자가 제공한 정확한 파일 경로
# -------------------------------
csv_paths = [
    "/home/ygang/Documents/anomaly/MSAD/shared_results/convnet_128_preds.csv",
    "/home/ygang/Documents/anomaly/MSAD/shared_results/knn_1024_preds.csv",
    "/home/ygang/Documents/anomaly/MSAD/shared_results/resnet_128_preds.csv",
    "/home/ygang/Documents/anomaly/MSAD/shared_results/rocket_128_preds.csv",
    "/home/ygang/Documents/anomaly/MSAD/shared_results/sit_stem_512_preds.csv",
    "/home/ygang/Documents/anomaly/MSAD/shared_results/sit_stem_ReLU_768_preds.csv",
]

output_path = "/home/ygang/Documents/anomaly/MSAD/shared_results/final_detector_results.csv"

# -------------------------------
# Load prediction CSVs
# -------------------------------
dfs = []
for path in csv_paths:
    df = pd.read_csv(path, index_col=0)
    dfs.append(df)

# -------------------------------
# 모든 CSV의 공통 파일명(FNAME) 추출
# -------------------------------
fnames = dfs[0].index.tolist()

# -------------------------------
# 각 파일에 대해 6개 모델 예측을 담아 다수결 투표
# -------------------------------
final_preds = []

for fname in fnames:
    votes = []
    for df in dfs:
        # 각 모델의 class column 찾기
        class_col = [c for c in df.columns if c.endswith("_class")][0]
        votes.append(df.loc[fname, class_col])

    final_label = Counter(votes).most_common(1)[0][0]

    final_preds.append([fname, *votes, final_label])

# -------------------------------
# 결과 저장
# -------------------------------
cols = ["fname",
        "convnet_128", "knn_1024", "resnet_128",
        "rocket_128", "sit_stem_512", "sit_stem_ReLU_768",
        "final_detector"]

result_df = pd.DataFrame(final_preds, columns=cols)
result_df.set_index("fname", inplace=True)

result_df.to_csv(output_path)
print(f"[INFO] Saved detector results → {output_path}")