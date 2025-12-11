########################################################################
# CPU-only MiniRocket Training Script
#
# GPU 사용 불가능 환경에서도 안정적으로 학습할 수 있도록
# GPU 관련 코드를 모두 제거한 CPU 전용 버전
#
# 실행 예:
#   python cpu_train_rocket.py --path data/TSB/TSB_128 \
#        --split_per 0.7 --seed 42 --eval_true \
#        --path_save results/weights
########################################################################

import argparse
import os
import re
from time import perf_counter
from datetime import datetime
from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle
from sktime.transformations.panel.rocket import MiniRocket
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

from utils.timeseries_dataset import create_splits, TimeseriesDataset
from utils.evaluator import save_classifier
from utils.config import *
from eval_rocket import eval_rocket


def run_rocket_cpu(
    data_path,
    split_per=0.7,
    seed=None,
    read_from_file=None,
    eval_model=False,
    path_save="results/weights"
):
    # Load window size
    window_size = int(re.search(r"\d+", data_path).group())
    classifier_name = f"rocket_cpu_{window_size}"

    # Load splits
    train_set, val_set, test_set = create_splits(
        data_path,
        split_per=split_per,
        seed=seed,
        read_from_file=read_from_file,
    )

    # Load actual data
    train_data = TimeseriesDataset(data_path, fnames=train_set)
    val_data = TimeseriesDataset(data_path, fnames=val_set)
    test_data = TimeseriesDataset(data_path, fnames=test_set)

    X_train, y_train = train_data.__getallsamples__().astype("float32"), train_data.__getalllabels__()
    X_val,   y_val   = val_data.__getallsamples__().astype("float32"), val_data.__getalllabels__()

    print(f"[INFO] CPU Rocket Train size = {len(y_train)}, Val size = {len(y_val)}")

    # MiniRocket feature extractor (CPU only)
    minirocket = MiniRocket(
        num_kernels=10000,
        n_jobs=-1       # 사용 가능한 CPU 스레드 모두 사용
    )

    scaler = StandardScaler(with_mean=False, copy=False)
    clf = SGDClassifier(loss="log_loss", n_jobs=-1)

    # Fit MiniRocket
    tic = perf_counter()
    X_train = minirocket.fit_transform(X_train).to_numpy()
    print(f"MiniRocket fitted in {perf_counter() - tic:.3f} sec")

    # Shuffle indices
    batch_size = 32768
    indexes = np.arange(X_train.shape[0])
    idx_shuffled = shuffle(indexes)

    # Fit scaler batch-by-batch
    for i in tqdm(range(0, X_train.shape[0], batch_size), desc="fitting-scaler"):
        batch = idx_shuffled[i:i+batch_size]
        scaler.partial_fit(X_train[batch])

    # Transform X_train batch-by-batch
    for i in tqdm(range(0, X_train.shape[0], batch_size), desc="transforming"):
        batch = idx_shuffled[i:i+batch_size]
        X_train[batch] = scaler.transform(X_train[batch])

    # Train classifier batch-by-batch
    for i in tqdm(range(0, X_train.shape[0], batch_size), desc="training"):
        batch = idx_shuffled[i:i+batch_size]
        clf.partial_fit(X_train[batch], y_train[batch], classes=np.arange(12))

    toc = perf_counter()

    # Create pipeline
    classifier = make_pipeline(minirocket, scaler, clf)

    # Save training stats
    stats = {
        "training_time": toc - tic,
        "val_acc": clf.score(minirocket.transform(X_val).to_numpy(), y_val),
    }

    print(f"[INFO] Training time: {stats['training_time']:.3f} sec")
    print(f"[INFO] Validation accuracy: {stats['val_acc']:.3%}")

    # Save stats
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    out_stats = pd.DataFrame.from_dict(stats, orient="index")
    out_stats.to_csv(os.path.join(save_done_training, f"{classifier_name}_{timestamp}.csv"))

    # Save model
    save_dir = os.path.join(path_save, classifier_name)
    saved_path = save_classifier(classifier, save_dir, fname=None)

    # Evaluate if requested
    if eval_model:
        eval_set = test_set if len(test_set) > 0 else val_set
        eval_rocket(
            data_path=data_path,
            model_path=saved_path,
            path_save=path_save_results,
            fnames=eval_set,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU-only MiniRocket training")
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--split_per", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--eval_true", action="store_true")
    parser.add_argument("--path_save", type=str, default="results/weights")

    args = parser.parse_args()

    run_rocket_cpu(
        data_path=args.path,
        split_per=args.split_per,
        seed=args.seed,
        read_from_file=args.file,
        eval_model=args.eval_true,
        path_save=args.path_save,
    )