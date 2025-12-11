from utils.data_loader import DataLoader
from utils.config import TSB_data_path
from label_nabkaggle_with_meta import extract_features
import numpy as np

print("=== TSB 전체 데이터 feature 검사 시작 ===")

dataloader = DataLoader(TSB_data_path)
datasets = dataloader.get_dataset_names()

bad_ts = []

for ds in datasets:
    print(f"\n### Checking dataset: {ds}")

    x_list, y_dummy, fnames = dataloader.load([ds])

    for ts, fname in zip(x_list, fnames):
        try:
            # ts는 시계열 (1D array)이어야 함
            if not isinstance(ts, np.ndarray):
                bad_ts.append((fname, "not numpy array"))
                continue

            # 길이 0 체크
            if len(ts) == 0:
                bad_ts.append((fname, "empty timeseries"))
                continue

            # 숫자 변환 체크
            try:
                ts_float = ts.astype(float)
            except Exception as e:
                bad_ts.append((fname, f"non-numeric values: {e}"))
                continue

            # feature 추출 체크
            feats = extract_features(ts_float)

            if feats.shape != (11,):
                bad_ts.append((fname, f"wrong feature shape: {feats.shape}"))

        except Exception as e:
            bad_ts.append((fname, f"error: {e}"))

print("\n=== 문제 있는 시계열 목록 ===")
for fname, reason in bad_ts:
    print(f"- {fname} → {reason}")

print(f"\n총 문제 TS 개수: {len(bad_ts)}")