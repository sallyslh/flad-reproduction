# evaluate_test.py
# Evaluates a trained global model on every client's TEST set (paper, Table 6)
# and adds the metrics needed for the pitfall audit (precision, FPR, MCC).
#
# Usage:
#   python evaluate_test.py --clients Dataset/CSV-DDoS2019 --model log/flad_seed0/1t-1n-mlp-global-model.h5 --out results/flad_seed0_test.csv

import argparse
import glob
import os

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf  # noqa: E402


def load_h5(path):
    with h5py.File(path, "r") as h:
        X = np.array(h["set_x"][:])
        y = np.array(h["set_y"][:])
    return X.reshape(X.shape[0], X.shape[1], X.shape[2], 1), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.model, compile=False)
    rows = []
    for folder in sorted(glob.glob(os.path.join(args.clients, "*", ""))):
        files = glob.glob(os.path.join(folder, "*-test.hdf5"))
        if not files:
            continue
        X, y = load_h5(files[0])
        p = np.squeeze(model.predict(X, batch_size=2048, verbose=0))
        pred = (p > args.threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append({
            "client": os.path.basename(os.path.normpath(folder)),
            "samples": len(y),
            "TPR": tp / (tp + fn) if tp + fn else np.nan,
            "FPR": fp / (fp + tn) if fp + tn else np.nan,
            "Precision": tp / (tp + fp) if tp + fp else np.nan,
            "F1": f1_score(y, pred, zero_division=0),
            "MCC": matthews_corrcoef(y, pred),
        })
    df = pd.DataFrame(rows)
    avg = df.drop(columns=["client", "samples"]).mean()
    df.loc[len(df)] = {"client": "Average", "samples": df["samples"].sum(), **avg.to_dict()}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
