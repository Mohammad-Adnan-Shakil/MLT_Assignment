# step5_evaluate.py
import itertools
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc, accuracy_score, f1_score,
                             classification_report)
from sklearn.model_selection import train_test_split

SEED = 42
plt.rcParams.update({"font.family": "serif", "font.size": 11})
X_train, X_test, y_train, y_test = joblib.load("models/split_dedup.pkl")

names = ["RandomForest", "XGBoost", "SoftVoting", "Stacking"]
models = {n: joblib.load(f"models/final_{n.lower()}.pkl") for n in names}
preds = {n: m.predict(X_test) for n, m in models.items()}
probas = {n: m.predict_proba(X_test)[:, 1] for n, m in models.items()}

# ---- Classification report for final model ----
print("=== Soft Voting: classification report (test) ===")
print(classification_report(y_test, preds["SoftVoting"],
                            target_names=["Legitimate", "Phishing"],
                            digits=4))

# ---- Fig 4: Confusion matrices ----
fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
for ax, n in zip(axes, names):
    cm = confusion_matrix(y_test, preds[n])
    ConfusionMatrixDisplay(cm, display_labels=["Legit", "Phish"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    fn, fp = cm[1, 0], cm[0, 1]
    ax.set_title(f"{n}\nFP={fp}  FN={fn}")
fig.tight_layout()
fig.savefig("figures/fig4_confusion_matrices.png", dpi=300)
plt.close(fig)

# Standalone CM for final model (paper often wants one large figure)
fig, ax = plt.subplots(figsize=(4.8, 4.2))
ConfusionMatrixDisplay(confusion_matrix(y_test, preds["SoftVoting"]),
                       display_labels=["Legitimate", "Phishing"]).plot(
    ax=ax, cmap="Blues", colorbar=False, values_format="d")
ax.set_title("Soft Voting Ensemble — Test Set")
fig.tight_layout()
fig.savefig("figures/fig4b_confusion_softvoting.png", dpi=300)
plt.close(fig)

# ---- Fig 5: ROC curves ----
fig, ax = plt.subplots(figsize=(5.5, 5))
styles = {"RandomForest": "--", "XGBoost": "-.",
          "SoftVoting": "-", "Stacking": ":"}
for n in names:
    fpr, tpr, _ = roc_curve(y_test, probas[n])
    ax.plot(fpr, tpr, styles[n], lw=1.8,
            label=f"{n} (AUC = {auc(fpr, tpr):.4f})")
ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls=":")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Test Set")
ax.legend(loc="lower right", fontsize=9)
# inset zoom on top-left where curves differ
ins = ax.inset_axes([0.35, 0.3, 0.38, 0.38])
for n in names:
    fpr, tpr, _ = roc_curve(y_test, probas[n])
    ins.plot(fpr, tpr, styles[n], lw=1.4)
ins.set_xlim(0, 0.15); ins.set_ylim(0.85, 1.0)
ins.tick_params(labelsize=7)
ax.indicate_inset_zoom(ins, edgecolor="grey")
fig.tight_layout()
fig.savefig("figures/fig5_roc_curves.png", dpi=300)
plt.close(fig)

# ---- McNemar's exact test (pairwise, test set) ----
print("=== McNemar exact test (test set) ===")
mc_rows = []
for a, b in itertools.combinations(names, 2):
    ca, cb = preds[a] == y_test.values, preds[b] == y_test.values
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if (n01 + n10) else 1.0
    mc_rows.append({"model_A": a, "model_B": b,
                    "A_right_B_wrong": n01, "A_wrong_B_right": n10,
                    "p_value": round(p, 4), "significant_0.05": p < 0.05})
mc = pd.DataFrame(mc_rows)
print(mc.to_string(index=False))
mc.to_csv("models/mcnemar.csv", index=False)

# ---- Leakage ablation: raw (duplicated) data, random split ----
print("\n=== Leakage ablation ===")
raw = pd.read_csv("phishing_processed.csv")
Xr, yr = raw.drop(columns="label"), raw["label"]
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
    Xr, yr, test_size=0.2, stratify=yr, random_state=SEED)

train_keys = set(map(tuple, Xr_tr.values))
seen = np.array([tuple(r) in train_keys for r in Xr_te.values])
print(f"Raw test rows: {len(Xr_te)} | feature vector also in train: "
      f"{seen.sum()} ({seen.mean()*100:.1f}%)")

base_rf = joblib.load("models/randomforest.pkl")
base_xgb = joblib.load("models/xgboost.pkl")
abl_models = {
    "XGBoost": clone(base_xgb),
    "SoftVoting": VotingClassifier(
        [("rf", clone(base_rf)), ("xgb", clone(base_xgb))],
        voting="soft", n_jobs=-1),
}
abl_rows = []
for n, m in abl_models.items():
    m.fit(Xr_tr, yr_tr)
    p = m.predict(Xr_te)
    abl_rows.append({
        "model": n,
        "raw_acc": accuracy_score(yr_te, p),
        "raw_seen_acc": accuracy_score(yr_te[seen], p[seen]),
        "raw_unseen_acc": accuracy_score(yr_te[~seen], p[~seen]),
        "dedup_acc": accuracy_score(y_test, preds[n]),
        "raw_f1": f1_score(yr_te, p),
        "dedup_f1": f1_score(y_test, preds[n]),
    })
abl = pd.DataFrame(abl_rows).set_index("model").round(4)
print(abl.to_string())
abl.to_csv("models/leakage_ablation.csv")

# ---- Fig 6: Leakage ablation ----
fig, ax = plt.subplots(figsize=(7, 4.2))
cols = ["raw_acc", "raw_seen_acc", "raw_unseen_acc", "dedup_acc"]
labels = ["Raw split\n(all)", "Raw split\n(seen in train)",
          "Raw split\n(unseen)", "Deduplicated\n(ours)"]
x = np.arange(len(cols)); w = 0.36
for i, (n, c) in enumerate(zip(abl.index, ["#4C72B0", "#C44E52"])):
    vals = abl.loc[n, cols].values
    bars = ax.bar(x + (i - 0.5) * w, vals, w, label=n, color=c)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.002, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Test accuracy")
ax.set_ylim(0.85, 1.01)
ax.set_title("Effect of Duplicate Leakage on Reported Accuracy")
ax.legend(loc="lower left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("figures/fig6_leakage_ablation.png", dpi=300)
plt.close(fig)

print("\nSaved: figures/fig4, fig4b, fig5, fig6; "
      "models/mcnemar.csv, models/leakage_ablation.csv")