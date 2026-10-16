# step5c_conflicts.py
import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

SEED = 42
raw = pd.read_csv("phishing_processed.csv")
features = [c for c in raw.columns if c != "label"]

# Feature vectors that appear with BOTH labels
nlab = raw.groupby(features)["label"].nunique()
conflict_keys = set(nlab[nlab > 1].index)
print(f"Conflicting feature vectors: {len(conflict_keys)}")

def is_conflict(X):
    return pd.Series([tuple(r) in conflict_keys for r in X[features].values],
                     index=X.index)

def report(title, X, y, pred):
    c = is_conflict(X)
    ok = pred == y.values
    print(f"\n=== {title} ===")
    print(f"All rows:          acc {ok.mean():.4f}  (n={len(y)}, errors={(~ok).sum()})")
    print(f"Conflict rows:     acc {ok[c.values].mean():.4f}  "
          f"(n={c.sum()}, errors={(~ok[c.values]).sum()})")
    print(f"Non-conflict rows: acc {ok[~c.values].mean():.4f}  "
          f"(n={(~c).sum()}, errors={(~ok[~c.values]).sum()})")

# Dedup test set, final model
X_train, X_test, y_train, y_test = joblib.load("models/split_dedup.pkl")
final = joblib.load("models/final_softvoting.pkl")
report("Dedup test (Soft Voting)", X_test, y_test, final.predict(X_test))

# Raw split, same model config
tr, te = train_test_split(raw, test_size=0.2, stratify=raw["label"],
                          random_state=SEED)
m = clone(final).fit(tr[features], tr["label"])
report("Raw test (Soft Voting)", te[features], te["label"],
       m.predict(te[features]))