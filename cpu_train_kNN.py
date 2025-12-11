########################################################################
# CPU 전용 kNN 학습 스크립트
# 기존 train_feature_based.py 중 kNN만 추출하여 단일 모델 훈련용으로 재작성
#
# 파일명: cpu_train_kNN.py
########################################################################

import argparse
import os
import re
from time import perf_counter
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.neighbors import KNeighborsClassifier

from utils.timeseries_dataset import create_splits
from eval_feature_based import eval_feature_based
from utils.evaluator import save_classifier
from utils.config import *


def train_kNN(
        data_path,
        split_per=0.7,
        seed=None,
        read_from_file=None,
        eval_model=False,
        path_save=None
    ):
    """
    Train only kNN-1024 model on feature-based dataset (.csv)
    """

    # Detect window size from path name
    window_size = int(re.search(r'\d+', data_path).group())
    training_stats = {}

    # Extract original dataset path (e.g., "data/TSB/TSB_128" → "data/TSB")
    original_dataset = "/".join(data_path.split("/")[:-1])

    # ----------------------------------------------------
    # Load train/val/test splits
    # ----------------------------------------------------
    train_set, val_set, test_set = create_splits(
        original_dataset,
        split_per=split_per,
        seed=seed,
        read_from_file=read_from_file,
    )
    train_names = [x[:-4] for x in train_set]
    val_names = [x[:-4] for x in val_set]
    test_names = [x[:-4] for x in test_set]

    # ----------------------------------------------------
    # Load tabular feature-based dataset
    # ----------------------------------------------------
    data = pd.read_csv(data_path, index_col=0)

    # Convert index: "filename.window" → MultiIndex(name, window_num)
    new_index = [tuple(x.rsplit('.', 1)) for x in data.index]
    data.index = pd.MultiIndex.from_tuples(new_index, names=["name", "n_window"])

    # Subset by split
    train_df = data.loc[data.index.get_level_values("name").isin(train_names)]
    val_df = data.loc[data.index.get_level_values("name").isin(val_names)]
    test_df = data.loc[data.index.get_level_values("name").isin(test_names)]

    # ----------------------------------------------------
    # Split X, y
    # ----------------------------------------------------
    y_train, X_train = train_df['label'], train_df.drop(columns=['label'])
    y_val, X_val = val_df['label'], val_df.drop(columns=['label'])
    y_test, X_test = test_df['label'], test_df.drop(columns=['label'])

    print(f"\n[INFO] Feature-based dataset loaded.")
    print(f"[INFO] Train size = {len(y_train)}, Val size = {len(y_val)}, Test size = {len(y_test)}")

    # ----------------------------------------------------
    # Define kNN model (1024 features assumption)
    # ----------------------------------------------------
    print("\n----------------------------------")
    print("Training kNN Classifier (k=5)...")

    classifier = KNeighborsClassifier(
        n_neighbors=5,
        n_jobs=-1
    )

    tic = perf_counter()
    classifier.fit(X_train, y_train)
    toc = perf_counter()

    training_stats["training_time"] = toc - tic
    print(f"[INFO] Training time: {training_stats['training_time']:.3f} sec")

    # ----------------------------------------------------
    # Validation Accuracy
    # ----------------------------------------------------
    tic = perf_counter()
    val_acc = classifier.score(X_val, y_val)
    toc = perf_counter()

    training_stats["val_acc"] = val_acc
    training_stats["avg_inf_time"] = ((toc - tic) / X_val.shape[0]) * 1000

    print(f"[INFO] Validation accuracy: {val_acc:.3%}")
    print(f"[INFO] Inference time: {training_stats['avg_inf_time']:.3f} ms per sample")

    # ----------------------------------------------------
    # Save training stats
    # ----------------------------------------------------
    clf_name = f"knn_{window_size}"

    timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
    df_stats = pd.DataFrame.from_dict(training_stats, orient="index", columns=["training_stats"])
    df_stats.to_csv(os.path.join(save_done_training, f"{clf_name}_{timestamp}.csv"))

    # ----------------------------------------------------
    # Save model
    # ----------------------------------------------------
    saving_dir = os.path.join(path_save, clf_name)
    saved_model_path = save_classifier(classifier, saving_dir, fname=None)
    print(f"[INFO] Saved trained kNN model → {saved_model_path}")

    # ----------------------------------------------------
    # Optional evaluation
    # ----------------------------------------------------
    if eval_model:
        eval_set = test_names if len(test_names) > 0 else val_names
        print("\n[INFO] Running evaluation on test/val set...")
        eval_feature_based(
            data_path=data_path,
            model_name=clf_name,
            model_path=saved_model_path,
            path_save=path_save_results,
            fnames=eval_set,
        )
        print("[INFO] Evaluation completed.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="cpu_train_kNN",
        description="Train only kNN classifier on feature-based dataset"
    )
    parser.add_argument("-p", "--path", type=str, required=True, help="Path to feature CSV dataset")
    parser.add_argument("-sp", "--split_per", type=float, default=0.7)
    parser.add_argument("-s", "--seed", type=int, default=None)
    parser.add_argument("-f", "--file", type=str, default=None)
    parser.add_argument("-e", "--eval_true", action="store_true")
    parser.add_argument("-ps", "--path_save", type=str, default="results/weights")

    args = parser.parse_args()

    train_kNN(
        data_path=args.path,
        split_per=args.split_per,
        seed=args.seed,
        read_from_file=args.file,
        eval_model=args.eval_true,
        path_save=args.path_save,
    )