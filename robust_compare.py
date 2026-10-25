import os, json
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
F = pd.read_csv(os.path.join(ROOT, "analysis", "features_clean.csv"))
feature_cols = [c for c in F.columns if c not in ("Class", "Recording_ID")]
X = F[feature_cols].values
y = F["Class"].values

def make(k, clf):
    return Pipeline([("scaler", StandardScaler()), ("select", SelectKBest(f_classif, k=k)), ("clf", clf)])

candidates = {
    "A: current (k=20, LogReg C=0.5)": make(20, LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced")),
    "B: LogReg k=97, C=1":             make(97, LogisticRegression(C=1, max_iter=2000, class_weight="balanced")),
    "C: SVM-rbf k=20, C=30, g=0.01":   make(20, SVC(C=30, gamma=0.01, class_weight="balanced", probability=True)),
    "D: RandomForest k=30":            make(30, RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_leaf=1, class_weight="balanced", random_state=0, n_jobs=1)),
    "E: GradientBoosting k=30":        make(30, GradientBoostingClassifier(n_estimators=120, max_depth=2, learning_rate=0.1, random_state=0)),
}

N_REPEATS = 15
scores = {name: {"acc": [], "bal_acc": []} for name in candidates}
for seed in range(N_REPEATS):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.10, stratify=y, random_state=seed)
    for name, pipe in candidates.items():
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        scores[name]["acc"].append(accuracy_score(y_te, pred))
        scores[name]["bal_acc"].append(balanced_accuracy_score(y_te, pred))

print(f"Mean +/- std over {N_REPEATS} random 90/10 stratified splits (held-out each time):\n")
rows = []
for name, s in scores.items():
    acc = np.array(s["acc"]); bacc = np.array(s["bal_acc"])
    print(f"{name:35s} acc={acc.mean():.4f}+/-{acc.std():.4f}   bal_acc={bacc.mean():.4f}+/-{bacc.std():.4f}")
    rows.append({"name": name, "acc_mean": acc.mean(), "acc_std": acc.std(), "bal_acc_mean": bacc.mean(), "bal_acc_std": bacc.std()})

with open(os.path.join(HERE, "robust_compare_results.json"), "w") as fh:
    json.dump(rows, fh, indent=2)
