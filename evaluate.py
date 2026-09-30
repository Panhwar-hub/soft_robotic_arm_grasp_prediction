#!/usr/bin/env python3
"""
Standalone inference script for the Soft_Robotic_Arm object classifier.

Takes a RAW capacitive-signal recording (time + 3 channels) as input, runs
the same preprocessing / feature extraction used at training time, and
returns the predicted object name (a string: e.g. "Ball", "Square box",
"Triangular").

Usage (command line):
    python3 evaluate.py path/to/recording.csv [more.csv ...]
    python3 evaluate.py path/to/recording.csv --proba   # also show class probabilities

Usage (import from your own robot-control code):
    from evaluate import predict_from_raw, predict_from_csv
    label = predict_from_csv("Trial_20260930_120000.csv")
    label = predict_from_raw(t_seconds, ch0_pF, ch1_pF, ch2_pF)   # raw arrays/lists
"""
import os, sys, json, argparse
import joblib
import numpy as np
import pandas as pd

from feature_extraction import extract_features, extract_features_from_df

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(HERE, "model_artifacts")

_pipe = None
_metadata = None


def load_model(artifacts_dir=ARTIFACTS_DIR):
    """Load (and cache) the trained pipeline + metadata. Call once; both
    predict_from_raw and predict_from_csv call this for you automatically."""
    global _pipe, _metadata
    if _pipe is None:
        _pipe = joblib.load(os.path.join(artifacts_dir, "model.joblib"))
        with open(os.path.join(artifacts_dir, "metadata.json")) as fh:
            _metadata = json.load(fh)
    return _pipe, _metadata


def _feature_vector(feat_dict, metadata):
    cols = metadata["feature_columns_in_order"]
    missing = [c for c in cols if c not in feat_dict]
    if missing:
        raise ValueError(
            f"Feature extraction did not produce {len(missing)} column(s) the "
            f"model expects (e.g. {missing[:5]}). feature_extraction.py may have "
            f"changed since the model was trained -- retrain with train_model.py."
        )
    return np.array([[feat_dict[c] for c in cols]])


def predict_from_raw(t, ch0, ch1, ch2, return_proba=False, artifacts_dir=ARTIFACTS_DIR):
    """
    t, ch0, ch1, ch2 : 1-D array-like of equal length -- raw trial time (s)
    and the three raw channel readings (pF) for ONE grasp recording.

    Returns the predicted class name (str), or (str, dict[class -> prob])
    if return_proba=True.
    """
    pipe, metadata = load_model(artifacts_dir)
    feats = extract_features(t, ch0, ch1, ch2)
    X = _feature_vector(feats, metadata)
    pred = pipe.predict(X)[0]
    if not return_proba:
        return pred
    proba = pipe.predict_proba(X)[0]
    return pred, dict(zip(pipe.classes_, proba))


def predict_from_csv(path, return_proba=False, artifacts_dir=ARTIFACTS_DIR):
    """path: a raw recording CSV in the same format as the training data
    (must have Trial_Time_s + one of the known channel-column sets; see
    CHANNEL_COLUMN_ALIASES in feature_extraction.py). Class/Recording_ID
    columns are not required -- this only reads the signal."""
    pipe, metadata = load_model(artifacts_dir)
    df = pd.read_csv(path)
    feats = extract_features_from_df(df)
    X = _feature_vector(feats, metadata)
    pred = pipe.predict(X)[0]
    if not return_proba:
        return pred
    proba = pipe.predict_proba(X)[0]
    return pred, dict(zip(pipe.classes_, proba))


def main():
    ap = argparse.ArgumentParser(description="Classify a raw capacitive-signal grasp recording.")
    ap.add_argument("csv_files", nargs="+", help="one or more raw recording CSV files")
    ap.add_argument("--proba", action="store_true", help="also print class probabilities")
    args = ap.parse_args()

    for path in args.csv_files:
        try:
            if args.proba:
                label, proba = predict_from_csv(path, return_proba=True)
                proba_str = ", ".join(f"{c}={p:.3f}" for c, p in sorted(proba.items(), key=lambda kv: -kv[1]))
                print(f"{os.path.basename(path)}: {label}   ({proba_str})")
            else:
                label = predict_from_csv(path)
                print(f"{os.path.basename(path)}: {label}")
        except Exception as e:
            print(f"{os.path.basename(path)}: ERROR - {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
