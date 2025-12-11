########################################################################
# Modified for: YG’s MSAD + (TSB + PTBXL + NAB) unified dataset
# Using final_labeled_dataset_converted.csv → direct label mapping
########################################################################

import sys
import os
from tqdm import tqdm
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import math

from utils.data_loader import DataLoader

########################################################################
# Utility functions
########################################################################

def z_normalization(ts, decimals=5):
    # Z-normalization (all windows with the same value go to 0)
    if len(set(ts)) == 1:
        ts = ts - np.mean(ts)
    else:
        ts = (ts - np.mean(ts)) / np.std(ts)
    ts = np.around(ts, decimals=decimals)

    return ts

def split_ts(data, window_size):
    """
    Split a time series into windows.
    If the series length is not divisible, first window overlaps the second.
    """
    modulo = data.shape[0] % window_size
    k = data[modulo:].shape[0] / window_size
    assert math.ceil(k) == k

    data_split = np.split(data[modulo:], k)
    if modulo != 0:
        data_split.insert(0, list(data[:window_size]))

    return np.asarray(data_split)

########################################################################
# Load LABELS from final_labeled_dataset_converted.csv
########################################################################

def load_label_csv(metric_path, metric):
    """
    Load the USER-PROVIDED label file.
    Expected format:
        dataset, filename, label
    """
    files = [f for f in os.listdir(metric_path) if f.endswith(".csv")]
    assert len(files) == 1, "metrics folder must contain exactly ONE label CSV file."

    df = pd.read_csv(os.path.join(metric_path, files[0]))

    # Clean filename mapping
    df["filename"] = df["filename"].apply(lambda x: os.path.basename(x))

    # Create index as (dataset/filename)
    df.index = df["dataset"] + "/" + df["filename"]

    # label column must exist
    assert metric in df.columns, f"label column '{metric}' not found."

    return df[[metric]]


########################################################################
# Create dataset with user labels
########################################################################

def create_tmp_dataset(
    name,
    save_dir,
    data_path,
    metric_path,
    window_size,
    metric,
):
    """
    Create MSAD window dataset using user's final labels (not metrics).
    """

    name = f"{name}_{window_size}"

    # ----------------------------------------------------
    # 1) Load ALL time series (TSB + PTBXL + NAB_KAGGLE)
    # ----------------------------------------------------
    dataloader = DataLoader(data_path)
    datasets = dataloader.get_dataset_names()
    x, y, fnames = dataloader.load(datasets)

    # fnames format: dataset/file.csv
    fnames = [f.replace("\\", "/") for f in fnames]

    # ----------------------------------------------------
    # 2) Load USER LABELS
    # ----------------------------------------------------
    label_df = load_label_csv(metric_path, metric)

    # Remove TS that do not exist in label file
    idx_to_delete = [i for i, f in enumerate(fnames) if f not in label_df.index]
    idx_short = [i for i, ts in enumerate(x) if ts.shape[0] < window_size]
    idx_to_delete.extend(idx_short)

    if len(idx_to_delete) > 0:
        print(f">>> Removing {len(idx_to_delete)} time series due to missing label or being too short")

        # 🔥 중복 제거 + 정렬 + index 검증
        idx_to_delete = sorted(list(set(idx_to_delete)), reverse=True)
        idx_to_delete = [i for i in idx_to_delete if i < len(x)]

        for idx in idx_to_delete:
            del x[idx]
            del y[idx]
            del fnames[idx]

    # After cleaning, align label_df
    label_df = label_df.loc[fnames]

    # ----------------------------------------------------
    # 3) Build label index mapping
    # ----------------------------------------------------
    unique_labels = sorted(label_df[metric].unique())
    label_to_id = {lab: i for i, lab in enumerate(unique_labels)}
    print("Detected labels:", label_to_id)

    # ----------------------------------------------------
    # 4) Split TS + assign labels
    # ----------------------------------------------------
    ts_list = []
    label_list = []

    for ts, lab in tqdm(zip(x, label_df[metric]), total=len(x), desc="Create dataset"):

        ts = z_normalization(ts)
        ts_split = split_ts(ts, window_size)

        ts_list.append(ts_split)
        label_list.append(np.ones(len(ts_split)) * label_to_id[lab])

    # ----------------------------------------------------
    # 5) Save dataset
    # ----------------------------------------------------
    for d in datasets:
        Path(os.path.join(save_dir, name, d)).mkdir(parents=True, exist_ok=True)

    for ts_splits, labels, fname in tqdm(zip(ts_list, label_list, fnames),
                                         total=len(ts_list), desc="Save dataset"):
        dataset_name = fname.split("/")[0]
        file_name = fname.split("/")[-1]

        new_indices = [file_name + f".{i}" for i in range(len(ts_splits))]

        data = np.concatenate((labels[:, None], ts_splits), axis=1)

        col_names = ["label"] + [f"val_{i}" for i in range(window_size)]
        df = pd.DataFrame(data, index=new_indices, columns=col_names)

        df.to_csv(os.path.join(save_dir, name, dataset_name, f"{file_name}.csv"))

    print(f"=== Window dataset created: {name} ===")


########################################################################
# Main
########################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default="TSB")
    parser.add_argument('-s', '--save_dir', type=str, required=True)
    parser.add_argument('-p', '--path', type=str, required=True)
    parser.add_argument('-mp', '--metric_path', type=str, required=True)
    parser.add_argument('-w', '--window_size', type=str, required=True)
    parser.add_argument('-m', '--metric', type=str, default='label')

    args = parser.parse_args()

    if args.window_size == "all":
        window_sizes = [32, 64, 128, 256, 512, 768, 1024, 1536]
        for size in window_sizes:
            create_tmp_dataset(
                name=args.name,
                save_dir=args.save_dir,
                data_path=args.path,
                metric_path=args.metric_path,
                window_size=size,
                metric=args.metric,
            )
    else:
        create_tmp_dataset(
            name=args.name,
            save_dir=args.save_dir,
            data_path=args.path,
            metric_path=args.metric_path,
            window_size=int(args.window_size),
            metric=args.metric,
        )