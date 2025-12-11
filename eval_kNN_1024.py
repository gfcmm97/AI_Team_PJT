########################################################################
#
# Custom evaluation script for kNN-1024 (Feature-based)
# Simplified version of eval_feature_based.py
#
########################################################################

import argparse
import re
import os
from tqdm import tqdm
from time import perf_counter
from collections import Counter
import pandas as pd

from utils.evaluator import Evaluator, load_classifier
from utils.config import detector_names


def eval_kNN_1024(
    data_path,
    model_path,
    path_save=None,
    fnames=None,
):
    """
    Predict labels using the kNN-1024 feature-based model.

    :param data_path: Path to TSFresh feature CSV (TSFRESH_TSB_1024.csv)
    :param model_path: Path to trained kNN model
    :param path_save: Where to save prediction results
    :param fnames: Optional list of series names. If None → predict all.
    """
    window_size = int(re.search(r'\d+', str(data_path)).group())
    classifier_name = f"knn_{window_size}"

    print(f"[INFO] Evaluating model: {classifier_name}")
    print(f"[INFO] Loading data: {data_path}")
    print(f"[INFO] Loading model: {model_path}")

    # Load model
    model = load_classifier(model_path)

    # Load dataset (TSFresh CSV)
    df = pd.read_csv(data_path, index_col=0)
    labels, data = df['label'], df.drop('label', axis=1)

    # If fnames not provided → infer from index
    if fnames is None:
        data_index = list(data.index)
        fnames = list(set([tuple(x.rsplit('.', 1))[0] for x in data_index]))

    all_preds = []
    inf_time = []

    print(f"[INFO] Total files to evaluate: {len(fnames)}")

    for fname in tqdm(fnames, desc='Computing', unit='files'):
        # Extract rows belonging to this series
        x = data.filter(like=fname, axis=0)

        # Predict all windows of the time series
        tic = perf_counter()
        preds = model.predict(x)
        toc = perf_counter()

        # Majority vote
        counter = Counter(preds)
        most_voted = counter.most_common(1)[0][0]

        # Map class index → detector name
        pred_label = detector_names[int(most_voted)]

        # Store
        all_preds.append(pred_label)
        inf_time.append(toc - tic)

    # Create output dataframe
    results = pd.DataFrame(
        data=zip(all_preds, inf_time),
        columns=[f"{classifier_name}_class", f"{classifier_name}_inf"],
        index=fnames
    )

    print(results)

    # Save results
    if path_save is not None:
        save_path = os.path.join(path_save, f"{classifier_name}_preds.csv")
        results.to_csv(save_path)
        print(f"[INFO] Saved predictions → {save_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='eval_kNN_1024',
        description='Evaluate kNN-1024 model on feature-based data.'
    )

    parser.add_argument(
        '-d', '--data',
        type=str,
        help='Path to TSFRESH_TSB_1024.csv',
        required=True
    )
    parser.add_argument(
        '-mp', '--model_path',
        type=str,
        help='Path to trained kNN_1024 model (.pkl)',
        required=True
    )
    parser.add_argument(
        '-ps', '--path_save',
        type=str,
        help='Where to save predictions',
        default="results/raw_predictions"
    )

    args = parser.parse_args()
    eval_kNN_1024(
        data_path=args.data,
        model_path=args.model_path,
        path_save=args.path_save,
    )