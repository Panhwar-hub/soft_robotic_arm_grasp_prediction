"""
Train the object-classifier on the capacitive-signal features and save it
for later use.

Model choice: hand-crafted-feature Logistic Regression over ALL 97 features
(StandardScaler -> LogisticRegression, C=2, class_weight="balanced"). This
replaces an earlier version that first ran SelectKBest down to 20 features --
that made sense when the dataset was small (84 recordings, ~1 feature per
sample). With 486 recordings now (~5 per feature) the extra features no
longer overfit, and a search (search_model.py / robust_compare.py in this
folder) over feature-count x classifier family x hyperparameters, validated
across 15 independent random 90/10 splits (not just one -- a single split's
"best" model turned out NOT to be the most robust one; see robust_compare.py),
found plain LogReg on all 97 features consistently ~1.4 points more accurate
and noticeably lower-variance than the old k=20 pipeline, and than every
tree-based / SVM / KNN alternative tried. See model_search_results.json and
robust_compare_results.json for the full comparison.

Usage: python3 train_model.py
Outputs (in model_artifacts/):
  model.joblib         - the fitted sklearn Pipeline
  metadata.json        - class names, ordered feature-column list, split info
  holdout_report.txt   - accuracy / confusion matrix / classification report on the 10% test set
  confusion_matrix.png - confusion matrix heatmap with accuracy in the title
  feature_space_2d.png - UMAP (falls back to t-SNE) 2D view of the feature space, colored by class
"""
import os, json, joblib
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix, classification_report

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FEATURES_CSV = os.path.join(ROOT, "analysis", "features_clean.csv")
OUT_DIR = os.path.join(HERE, "model_artifacts")
RANDOM_STATE = 0
TEST_SIZE = 0.10
LOGREG_C = 2.0


def build_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=LOGREG_C, max_iter=3000, class_weight="balanced")),
    ])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    F = pd.read_csv(FEATURES_CSV)
    meta_cols = ["Class", "Recording_ID"]
    feature_cols = [c for c in F.columns if c not in meta_cols]
    X = F[feature_cols].values
    y = F["Class"].values
    classes = sorted(F["Class"].unique())

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, F["Recording_ID"].values,
        test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    report = classification_report(y_test, y_pred, labels=classes)

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
    lines.append(f"\nModel: LogisticRegression(C={LOGREG_C}) on all {len(feature_cols)} features (no feature selection -- see module docstring).")
    report_text = "\n".join(lines)
    print(report_text)

    with open(os.path.join(OUT_DIR, "holdout_report.txt"), "w") as fh:
        fh.write(report_text + "\n")

    # ---- confusion matrix + accuracy plot ----
    from plot_diagnostics import plot_confusion_matrix, plot_feature_space_2d
    plot_confusion_matrix(cm, classes, acc, bal_acc, os.path.join(OUT_DIR, "confusion_matrix.png"))
    plot_feature_space_2d(X, y, classes, os.path.join(OUT_DIR, "feature_space_2d.png"))

    # Refit on ALL data (train+test) for the artifact you actually deploy --
    # standard practice once the held-out score above has told you how well
    # this pipeline generalises. The held-out numbers above remain the
    # honest estimate of real-world accuracy; this final refit just makes
    # use of every labeled recording you have.
    final_pipe = build_pipeline()
    final_pipe.fit(X, y)
    joblib.dump(final_pipe, os.path.join(OUT_DIR, "model.joblib"))

    metadata = {
        "classes": classes,
        "feature_columns_in_order": feature_cols,  # exact order the pipeline expects
        "model": f"LogisticRegression(C={LOGREG_C}, class_weight='balanced') on all {len(feature_cols)} features",
        "held_out_test_accuracy": acc,
        "held_out_test_balanced_accuracy": bal_acc,
        "n_recordings_total": len(F),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "held_out_test_recording_ids": list(id_test),
    }
    with open(os.path.join(OUT_DIR, "metadata.json"), "w") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"\nSaved model.joblib, metadata.json, holdout_report.txt, confusion_matrix.png, feature_space_2d.png -> {OUT_DIR}")


if __name__ == "__main__":
    main()
