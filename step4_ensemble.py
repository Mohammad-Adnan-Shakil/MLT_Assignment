# step4_ensemble.py
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

SEED = 42
X_train, X_test, y_train, y_test = joblib.load("models/split_dedup.pkl")
rf = joblib.load("models/randomforest.pkl")
xgb = joblib.load("models/xgboost.pkl")

voting = VotingClassifier(
    estimators=[("rf", clone(rf)), ("xgb", clone(xgb))],
    voting="soft", n_jobs=-1)

stacking = StackingClassifier(
    estimators=[("rf", clone(rf)), ("xgb", clone(xgb))],
    final_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
    stack_method="predict_proba", n_jobs=-1)

models = {
    "RandomForest": clone(rf),
    "XGBoost": clone(xgb),
    "SoftVoting": voting,
    "Stacking": stacking,
}

# ---- Same 5 CV folds for every model (fair comparison) ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
rows = []
for name, model in models.items():
    print(f"CV: {name} ...")
    s = cross_validate(model, X_train, y_train, cv=cv,
                       scoring=scoring, n_jobs=-1)
    row = {"model": name}
    for m in scoring:
        row[f"cv_{m}"] = f"{s['test_'+m].mean():.4f} ± {s['test_'+m].std():.4f}"
    row["_cv_f1_folds"] = s["test_f1"].round(4).tolist()
    rows.append(row)

# ---- Fit on full train, evaluate once on held-out test ----
for row in rows:
    name = row["model"]
    model = models[name].fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    row.update({
        "test_accuracy": round(accuracy_score(y_test, pred), 4),
        "test_precision": round(precision_score(y_test, pred), 4),
        "test_recall": round(recall_score(y_test, pred), 4),
        "test_f1": round(f1_score(y_test, pred), 4),
        "test_roc_auc": round(roc_auc_score(y_test, proba), 4),
    })
    joblib.dump(model, f"models/final_{name.lower()}.pkl")

res = pd.DataFrame(rows).set_index("model")
pd.set_option("display.width", 200)
print("\n=== Cross-validation (train, 5-fold) ===")
print(res[[c for c in res.columns if c.startswith("cv_")]])
print("\n=== Held-out test ===")
print(res[[c for c in res.columns if c.startswith("test_")]])

if hasattr(models["Stacking"], "final_estimator_"):
    w = models["Stacking"].final_estimator_.coef_[0]
    print(f"\nStacking meta-learner weights -> RF: {w[0]:.3f}, XGB: {w[1]:.3f}")

res.to_csv("models/model_comparison.csv")
print("\nSaved: models/final_*.pkl, models/model_comparison.csv")