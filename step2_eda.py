# step2_eda.py
import os
import matplotlib
matplotlib.use("Agg")  # save only, no popup windows
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 11})

df = pd.read_csv("phishing_clean.csv")
TARGET = "result"

# Phishing = positive class (1). Required by XGBoost (labels in [0, n)),
# and makes precision/recall refer to phishing detection.
df["label"] = df[TARGET].map({-1: 1, 1: 0})
df = df.drop(columns=TARGET)
features = [c for c in df.columns if c != "label"]

# ---- Duplicate check ----
n_dup = df.duplicated().sum()
n_dup_feat = df.duplicated(subset=features).sum()
conflict = (df.groupby(features)["label"].nunique() > 1).sum()
print(f"Exact duplicate rows:            {n_dup} ({n_dup/len(df)*100:.1f}%)")
print(f"Duplicate feature vectors:       {n_dup_feat}")
print(f"Feature vectors w/ both labels:  {conflict}")
print(f"Unique rows after dedup:         {len(df.drop_duplicates())}")

# ---- Fig 1: Class distribution ----
counts = df["label"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(["Legitimate", "Phishing"], counts.values,
              color=["#4C72B0", "#C44E52"], width=0.55)
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width()/2, v + 80,
            f"{v}\n({v/len(df)*100:.1f}%)", ha="center", va="bottom")
ax.set_ylabel("Number of samples")
ax.set_title("Class Distribution")
ax.set_ylim(0, counts.max() * 1.18)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("figures/fig1_class_distribution.png", dpi=300)
plt.close(fig)

# ---- Fig 2: Feature correlation heatmap ----
corr = df[features].corr(method="spearman")  # ordinal {-1,0,1} features
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.3, cbar_kws={"shrink": 0.7}, ax=ax)
ax.set_title("Spearman Correlation Between Features")
fig.tight_layout()
fig.savefig("figures/fig2_correlation_heatmap.png", dpi=300)
plt.close(fig)

# ---- Fig 3: Feature correlation with target ----
tcorr = df[features].corrwith(df["label"], method="spearman").sort_values()
fig, ax = plt.subplots(figsize=(7, 9))
ax.barh(tcorr.index, tcorr.values,
        color=["#C44E52" if v > 0 else "#4C72B0" for v in tcorr.values])
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Spearman correlation with phishing label")
ax.set_title("Feature–Target Correlation")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("figures/fig3_target_correlation.png", dpi=300)
plt.close(fig)

# Highly correlated feature pairs (multicollinearity note for paper)
pairs = corr.where(~mask).stack()
high = pairs[pairs.abs() > 0.6].sort_values(key=abs, ascending=False)
print("\nFeature pairs with |rho| > 0.6:")
print(high if len(high) else "None")

print("\nTop 5 features by |correlation| with phishing:")
print(tcorr.reindex(tcorr.abs().sort_values(ascending=False).index).head())

df.to_csv("phishing_processed.csv", index=False)
print("\nSaved: figures/fig1-3, phishing_processed.csv")