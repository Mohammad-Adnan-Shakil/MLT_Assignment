# step3_train.py
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     RandomizedSearchCV)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from xgboost import XGBClassifier

SEED = 42
os.makedirs("models", exist_ok=True)

df = pd.read_csv("phishing_processed.csv")
print(f"Raw rows: {len(df)}")
df = df.drop_duplicates().reset_index(drop=True)
print(f"After dedup: {len(df)}")
print("Class split:", df["label"].value_counts().to_dict())

X = df.drop(columns="label")
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"Train: {X_train.shape}  Test: {X_test.shape}")

joblib.dump((X_train, X_test, y_train, y_test), "models/split_dedup.pkl")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# ---- Random Forest ----
rf_space = {
    "n_estimators": [200, 300, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5],
}
rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=SEED, n_jobs=-1),
    rf_space, n_iter=30, scoring="f1", cv=cv,
    random_state=SEED, n_jobs=-1, verbose=1)
rf_search.fit(X_train, y_train)

# ---- XGBoost ----
xgb_space = {
    "n_estimators": [200, 300, 500],
    "max_depth": [3, 4, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.1, 0.3],
}
xgb_search = RandomizedSearchCV(
    XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=-1),
    xgb_space, n_iter=30, scoring="f1", cv=cv,
    random_state=SEED, n_jobs=-1, verbose=1)
xgb_search.fit(X_train, y_train)

# ---- Results ----
results = {}
for name, search in [("RandomForest", rf_search), ("XGBoost", xgb_search)]:
    model = search.best_estimator_
    idx = search.best_index_
    cv_mean = search.cv_results_["mean_test_score"][idx]
    cv_std = search.cv_results_["std_test_score"][idx]

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    results[name] = {
        "best_params": search.best_params_,
        "cv_f1_mean": round(cv_mean, 4),
        "cv_f1_std": round(cv_std, 4),
        "test_accuracy": round(accuracy_score(y_test, pred), 4),
        "test_precision": round(precision_score(y_test, pred), 4),
        "test_recall": round(recall_score(y_test, pred), 4),
        "test_f1": round(f1_score(y_test, pred), 4),
        "test_roc_auc": round(roc_auc_score(y_test, proba), 4),
    }
    joblib.dump(model, f"models/{name.lower()}.pkl")

    print(f"\n=== {name} ===")
    print("Best params:", search.best_params_)
    print(f"CV F1: {cv_mean:.4f} ± {cv_std:.4f}")
    for k in ["test_accuracy", "test_precision", "test_recall",
              "test_f1", "test_roc_auc"]:
        print(f"{k:16s}: {results[name][k]}")

with open("models/individual_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nSaved: models/randomforest.pkl, models/xgboost.pkl, "
      "models/split_dedup.pkl, models/individual_results.json")