"""
Train the object-classifier on the capacitive-signal features and save it
for later use.

Model choice: hand-crafted-feature Logistic Regression (StandardScaler ->
SelectKBest(top 20, ANOVA F) -> LogisticRegression). In head-to-head testing
on this data this matched or beat a MiniRocket raw-signal baseline while
being far cheaper to run on whatever computer/microcontroller drives the arm,
and it stays interpretable (you can point at which physical feature drove a
decision). SelectKBest is INSIDE the pipeline so feature selection is refit
on the 90% training split only -- no leakage from the held-out 10%.

Usage: python3 train_model.py
Outputs (in ../model_artifacts/):
  model.joblib       - the fitted sklearn Pipeline
  metadata.json       - class names, ordered feature-column list, split info
  holdout_report.txt  - accuracy / confusion matrix / classification report on the 10% test set
"""
import os, json, joblib
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix, classification_report

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FEATURES_CSV = os.path.join(ROOT, "analysis", "features_clean.csv")
OUT_DIR = os.path.join(HERE, "model_artifacts")
RANDOM_STATE = 0
TEST_SIZE = 0.10
K_FEATURES = 20


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    F = pd.read_csv(FEATURES_CSV)
    meta_cols = ["Class", "Recording_ID"]
    feature_cols = [c for c in F.columns if c not in meta_cols]
    X = F[feature_cols].values
    y = F["Class"].values

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, F["Recording_ID"].values,
        test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=K_FEATURES)),
        ("clf", LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced")),
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    classes = sorted(F["Class"].unique())
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    report = classification_report(y_test, y_pred, labels=classes)

    selected_mask = pipe.named_steps["select"].get_support()
    selected_features = [c for c, keep in zip(feature_cols, selected_mask) if keep]

    lines = []
    lines.append(f"Train/test split: {len(X_train)} train / {len(X_test)} test "
                  f"(stratified {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}, random_state={RANDOM_STATE})")
    lines.append(f"Train class counts: {pd.Series(y_train).value_counts().to_dict()}")
    lines.append(f"Test  class counts: {pd.Series(y_test).value_counts().to_dict()}")
    lines.append(f"\nHeld-out test accuracy:          {acc:.4f}")
    lines.append(f"Held-out test balanced accuracy: {bal_acc:.4f}")
    lines.append(f"\nConfusion matrix (rows=true, cols=predicted), labels={classes}")
    lines.append(str(cm))
    lines.append(f"\n{report}")
    lines.append(f"\nSelected {len(selected_features)}/{len(feature_cols)} features (top-{K_FEATURES} by ANOVA F, fit on train split only):")
    lines.append(", ".join(selected_features))
    report_text = "\n".join(lines)
    print(report_text)

    with open(os.path.join(OUT_DIR, "holdout_report.txt"), "w") as fh:
        fh.write(report_text + "\n")

    # Refit on ALL data (train+test) for the artifact you actually deploy --
    # standard practice once the held-out score above has told you how well
    # this pipeline generalises. The held-out numbers above remain the
    # honest estimate of real-world accuracy; this final refit just makes
    # use of every labeled recording you have.
    final_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=K_FEATURES)),
        ("clf", LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced")),
    ])
    final_pipe.fit(X, y)
    joblib.dump(final_pipe, os.path.join(OUT_DIR, "model.joblib"))

    metadata = {
        "classes": classes,
        "feature_columns_in_order": feature_cols,  # exact order the pipeline expects
        "k_features_selected": K_FEATURES,
        "held_out_test_accuracy": acc,
        "held_out_test_balanced_accuracy": bal_acc,
        "n_recordings_total": len(F),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "held_out_test_recording_ids": list(id_test),
    }
    with open(os.path.join(OUT_DIR, "metadata.json"), "w") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"\nSaved model.joblib, metadata.json, holdout_report.txt -> {OUT_DIR}")


if __name__ == "__main__":
    main()
