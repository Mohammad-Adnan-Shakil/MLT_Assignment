# step5b_freq_check.py
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

SEED = 42
raw = pd.read_csv("phishing_processed.csv")
features = [c for c in raw.columns if c != "label"]

# How many times each feature vector appears in the full raw dataset
raw["freq"] = raw.groupby(features)["label"].transform("size")

def bucket(f):
    return pd.cut(f, bins=[0, 1, 3, 10, np.inf],
                  labels=["1 (unique)", "2-3", "4-10", ">10"])

xgb = joblib.load("models/xgboost.pkl")

# ---- A) Raw split: accuracy by vector frequency ----
tr, te = train_test_split(raw, test_size=0.2, stratify=raw["label"],
                          random_state=SEED)
m = clone(xgb).fit(tr[features], tr["label"])
te = te.assign(correct=(m.predict(te[features]) == te["label"]).astype(int),
               bucket=bucket(te["freq"]))
print("=== A) Raw test set: accuracy by how often the vector occurs ===")
print(te.groupby("bucket", observed=True)
        .agg(rows=("correct", "size"), accuracy=("correct", "mean"))
        .round(4))
print(f"Overall raw-test accuracy (duplicates counted): {te['correct'].mean():.4f}")
uniq = te.drop_duplicates(subset=features + ["label"])
print(f"Raw-test accuracy, unique vectors only:         {uniq['correct'].mean():.4f}"
      f"  (n={len(uniq)})")

# ---- B) Dedup test set: where do the errors come from? ----
X_train, X_test, y_train, y_test = joblib.load("models/split_dedup.pkl")
final = joblib.load("models/final_softvoting.pkl")
freq_map = raw.drop_duplicates(subset=features + ["label"]) \
              .set_index(features)["freq"]
freq_map = freq_map[~freq_map.index.duplicated()]
dt = X_test.copy()
dt["label"] = y_test.values
dt["freq"] = freq_map.reindex(pd.MultiIndex.from_frame(X_test)).values
dt["correct"] = (final.predict(X_test) == y_test.values).astype(int)
dt["bucket"] = bucket(dt["freq"])
print("\n=== B) Dedup test set (Soft Voting): accuracy by original frequency ===")
print(dt.groupby("bucket", observed=True)
        .agg(rows=("correct", "size"), accuracy=("correct", "mean"))
        .round(4))