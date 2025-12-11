########################################################################
# Pseudo-labeling NAB_KAGGLE using meta-features + RandomForest
#
# - final_labeled_dataset_converted.csv 기반으로 메타 모델 학습
# - NAB_KAGGLE은 anomaly 없는 파일도 강제로 로드 (방법 B)
# - 예측된 라벨을 결합 → final_labeled_dataset_converted_v2.csv 생성
########################################################################

import os
import numpy as np
import pandas as pd

from utils.data_loader import DataLoader
from utils.config import TSB_data_path, TSB_metrics_path

from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier


########################################################################
# 1) Feature extractor
########################################################################
def extract_features(ts: np.ndarray) -> np.ndarray:
    """NaN 방지 기능이 포함된 robust meta-feature extractor"""
    ts = np.asarray(ts, dtype=float).flatten()

    # 기본 값 계산
    length = len(ts)
    mean = np.mean(ts)
    std = np.std(ts)
    vmin = np.min(ts)
    vmax = np.max(ts)

    # skew, kurtosis는 identical value TS에서 NaN 발생 가능
    try:
        s = skew(ts, nan_policy="omit")
    except:
        s = 0.0

    try:
        k = kurtosis(ts, nan_policy="omit")
    except:
        k = 0.0

    # percentile 계산도 NaN 발생 가능
    try:
        q25 = np.percentile(ts, 25)
        q50 = np.percentile(ts, 50)
        q75 = np.percentile(ts, 75)
    except:
        q25 = q50 = q75 = 0.0

    # energy
    try:
        energy = np.mean(ts ** 2)
    except:
        energy = 0.0

    feats = np.array([
        length, mean, std, vmin, vmax,
        s, k, q25, q50, q75, energy
    ], dtype=float)

    # 🔥 마지막 방어: NaN 또는 inf가 있으면 모두 0으로 치환
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    return feats


########################################################################
# 2) NAB_KAGGLE 전용 raw loader (anomaly 여부와 상관없이 모든 파일 로드)
########################################################################
def load_timeseries_raw(dir_path):
    ts_list = []
    fname_list = []

    for root, dirs, files in os.walk(dir_path):
        for f in files:
            if f.endswith(".out"):
                full = os.path.join(root, f)
                try:
                    curr = np.loadtxt(full, delimiter=",")
                except:
                    continue

                # 반드시 2컬럼 형태여야 함
                if curr.ndim != 2 or curr.shape[1] != 2:
                    continue

                # 첫 번째 컬럼만 사용 (TS 값)
                ts_list.append(curr[:, 0])

                # dataset/file 형태의 relative path 반환
                rel = full.replace(TSB_data_path, "").lstrip("/")
                fname_list.append(rel)

    return ts_list, fname_list


########################################################################
# 3) Main
########################################################################
def main():
    # --------------------------------------------------------------
    # (0) 기존 라벨 로드
    # --------------------------------------------------------------
    label_path = os.path.join(TSB_metrics_path, "final_labeled_dataset_converted.csv")
    print(f"[INFO] Loading labels from: {label_path}")
    df = pd.read_csv(label_path)

    # key 생성
    df["key"] = df["dataset"].astype(str) + "/" + df["filename"].astype(str)

    # 중복 제거
    before = len(df)
    df = df.drop_duplicates(subset=["key"], keep="first")
    removed = before - len(df)
    print(f"[INFO] Removed duplicated rows: {removed}")

    df = df.set_index("key")

    # --------------------------------------------------------------
    # (1) TSB 전체 데이터 중 NAB_KAGGLE 제외하고 로딩
    # --------------------------------------------------------------
    dataloader = DataLoader(TSB_data_path)
    datasets_all = dataloader.get_dataset_names()

    datasets_train = [d for d in datasets_all if d != "NAB_KAGGLE"]
    datasets_nab = ["NAB_KAGGLE"]

    print("[INFO] Train datasets:", datasets_train)
    print("[INFO] NAB_KAGGLE dataset:", datasets_nab)

    # Train dataset 로드
    x_train_list, y_dummy, fnames_train = dataloader.load(datasets_train)

    # --------------------------------------------------------------
    # (2) NAB_KAGGLE 전용 raw 로딩 (방법 B 핵심)
    # --------------------------------------------------------------
    nab_path = os.path.join(TSB_data_path, "NAB_KAGGLE")
    x_nab_list, fnames_nab = load_timeseries_raw(nab_path)

    # --------------------------------------------------------------
    # (3) Feature + Label 정리
    # --------------------------------------------------------------
    X_train = []
    y_train = []

    # Train 데이터 → feature + label
    for ts, fname in zip(x_train_list, fnames_train):
        dataset_name = fname.split("/")[0]
        filename = fname.split("/", 1)[1]
        key = f"{dataset_name}/{filename}"

        if key not in df.index:
            continue

        label = df.loc[key, "label"]
        feats = extract_features(ts)

        X_train.append(feats)
        y_train.append(label)

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=object)

    print(f"[INFO] Training samples: {X_train.shape[0]}")

    # NAB_KAGGLE feature
    X_nab = []
    nab_keys = []

    for ts, fname in zip(x_nab_list, fnames_nab):
        feats = extract_features(ts)
        X_nab.append(feats)
        nab_keys.append(fname)

    X_nab = np.asarray(X_nab, dtype=float)
    nab_keys = np.asarray(nab_keys, dtype=object)

    print(f"[INFO] NAB_KAGGLE samples to label: {len(X_nab)}")

    if len(X_nab) == 0:
        print("[WARN] No NAB_KAGGLE files found for labeling.")
        return

    # --------------------------------------------------------------
    # (4) RandomForest 메타 모델 학습
    # --------------------------------------------------------------
    print("[INFO] Training RandomForest model...")
    clf = RandomForestClassifier(
        n_estimators=300,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42
    )
    clf.fit(X_train, y_train)

    # --------------------------------------------------------------
    # (5) NAB_KAGGLE 예측
    # --------------------------------------------------------------
    print("[INFO] Predicting labels for NAB_KAGGLE...")
    nab_pred = clf.predict(X_nab)

    # 결과 dataframe
    nab_rows = []
    for key, lab in zip(nab_keys, nab_pred):
        dataset_name = key.split("/")[0]
        filename = key.split("/", 1)[1]
        nab_rows.append({
            "dataset": dataset_name,
            "filename": filename,
            "label": lab
        })

    nab_df = pd.DataFrame(nab_rows)

    print("[INFO] NAB_KAGGLE predicted labels (head):")
    print(nab_df.head())

    # --------------------------------------------------------------
    # (6) 기존 라벨 + NAB_KAGGLE 라벨 merge
    # --------------------------------------------------------------
    original = pd.read_csv(label_path)
    merged = pd.concat([original, nab_df], ignore_index=True)

    merged["key"] = merged["dataset"].astype(str) + "/" + merged["filename"].astype(str)
    merged = merged.drop_duplicates(subset=["key"], keep="first").drop(columns=["key"])

    out_path = os.path.join(TSB_metrics_path, "final_labeled_dataset_converted_v2.csv")
    merged.to_csv(out_path, index=False)

    print(f"[INFO] Saved merged label file → {out_path}")


########################################################################
if __name__ == "__main__":
    main()