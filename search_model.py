import os, json, time, sys
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
F = pd.read_csv(os.path.join(ROOT, "analysis", "features_clean.csv"))
feature_cols = [c for c in F.columns if c not in ("Class", "Recording_ID")]
X = F[feature_cols].values
y = F["Class"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, stratify=y, random_state=0)
print(f"train={len(X_train)} test={len(X_test)} n_features={len(feature_cols)}", flush=True)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
K_GRID = [10, 15, 20, 30, 50, 97]

candidates = {
    "LogReg": (LogisticRegression(max_iter=2000, class_weight="balanced"),
               {"select__k": K_GRID, "clf__C": [0.1, 0.3, 1, 3, 10]}),
    "SVM-rbf": (SVC(class_weight="balanced"),
                {"select__k": K_GRID, "clf__C": [1, 3, 10, 30], "clf__gamma": ["scale", 0.01, 0.1]}),
    "KNN": (KNeighborsClassifier(),
            {"select__k": K_GRID, "clf__n_neighbors": [3, 5, 9, 15], "clf__weights": ["uniform", "distance"]}),
    "RandomForest": (RandomForestClassifier(class_weight="balanced", random_state=0, n_jobs=1, n_estimators=200),
                      {"select__k": [30, 50, 97], "clf__max_depth": [None, 15], "clf__min_samples_leaf": [1, 3]}),
    "ExtraTrees": (ExtraTreesClassifier(class_weight="balanced", random_state=0, n_jobs=1, n_estimators=200),
                   {"select__k": [30, 50, 97], "clf__max_depth": [None, 15], "clf__min_samples_leaf": [1, 3]}),
    "GradientBoosting": (GradientBoostingClassifier(random_state=0, n_estimators=120),
                         {"select__k": [30, 50], "clf__max_depth": [2, 3], "clf__learning_rate": [0.1]}),
}

results = []
best_overall = None
for name, (clf, grid) in candidates.items():
    t0 = time.time()
    pipe = Pipeline([("scaler", StandardScaler()), ("select", SelectKBest(score_func=f_classif)), ("clf", clf)])
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="balanced_accuracy", n_jobs=-1, refit=True)
    gs.fit(X_train, y_train)
    dt = time.time() - t0
    results.append({"classifier": name, "best_cv_balanced_acc": gs.best_score_, "best_params": gs.best_params_, "seconds": dt})
    print(f"{name:18s} best CV bal-acc={gs.best_score_:.4f}  params={gs.best_params_}  ({dt:.0f}s)", flush=True)
    if best_overall is None or gs.best_score_ > best_overall[0]:
        best_overall = (gs.best_score_, name, gs.best_estimator_, gs.best_params_)

results_sorted = sorted(results, key=lambda r: -r["best_cv_balanced_acc"])
with open(os.path.join(HERE, "model_search_results.json"), "w") as fh:
    json.dump(results_sorted, fh, indent=2, default=str)

best_score, best_name, best_est, best_params = best_overall
print(f"\n=== BEST ON TRAIN-CV: {best_name}  cv_bal_acc={best_score:.4f}  params={best_params} ===", flush=True)

import joblib
current_retrain = Pipeline([("scaler", StandardScaler()), ("select", SelectKBest(score_func=f_classif, k=20)),
                             ("clf", LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced"))])
current_retrain.fit(X_train, y_train)
pred_current = current_retrain.predict(X_test)
pred_best = best_est.predict(X_test)

print(f"\nCurrent production pipeline (k=20, LogReg C=0.5) on held-out test:")
print(f"  accuracy={accuracy_score(y_test, pred_current):.4f}  balanced_accuracy={balanced_accuracy_score(y_test, pred_current):.4f}")
print(f"\nBest found pipeline ({best_name}, {best_params}) on held-out test:")
print(f"  accuracy={accuracy_score(y_test, pred_best):.4f}  balanced_accuracy={balanced_accuracy_score(y_test, pred_best):.4f}")

with open(os.path.join(HERE, "search_summary.json"), "w") as fh:
    json.dump({
        "best_name": best_name, "best_params": best_params, "best_cv_balanced_acc": best_score,
        "current_test_acc": accuracy_score(y_test, pred_current), "current_test_bal_acc": balanced_accuracy_score(y_test, pred_current),
        "best_test_acc": accuracy_score(y_test, pred_best), "best_test_bal_acc": balanced_accuracy_score(y_test, pred_best),
    }, fh, indent=2)
