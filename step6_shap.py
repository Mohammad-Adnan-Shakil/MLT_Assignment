# step6_shap.py
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

SEED = 42
plt.rcParams.update({"font.family": "serif", "font.size": 11})
X_train, X_test, y_train, y_test = joblib.load("models/split_dedup.pkl")
final = joblib.load("models/final_softvoting.pkl")
rf = final.named_estimators_["rf"]
xgb = final.named_estimators_["xgb"]
features = list(X_test.columns)

# ---- SHAP in probability space, same background for both models ----
background = shap.sample(X_train, 100, random_state=SEED)

def explain(model):
    ex = shap.TreeExplainer(model, data=background,
                            model_output="probability",
                            feature_perturbation="interventional")
    sv = ex.shap_values(X_test, check_additivity=False)
    ev = ex.expected_value
    if isinstance(sv, list):          # older shap: list per class
        sv, ev = sv[1], ev[1]
    elif np.ndim(sv) == 3:            # newer shap: (n, f, classes)
        sv, ev = sv[:, :, 1], np.atleast_1d(ev)[-1]
    return np.asarray(sv), float(np.ravel(ev)[-1])

print("Explaining RF ...");  sv_rf, ev_rf = explain(rf)
print("Explaining XGB ..."); sv_xgb, ev_xgb = explain(xgb)

sv_ens = 0.5 * (sv_rf + sv_xgb)
ev_ens = 0.5 * (ev_rf + ev_xgb)

# Additivity check: base + sum(SHAP) must equal ensemble P(phishing)
proba = final.predict_proba(X_test)[:, 1]
recon = ev_ens + sv_ens.sum(axis=1)
print(f"Additivity check — max |recon - proba|: "
      f"{np.abs(recon - proba).max():.2e}")

expl = shap.Explanation(values=sv_ens, base_values=np.full(len(X_test), ev_ens),
                        data=X_test.values, feature_names=features)
joblib.dump({"sv_rf": sv_rf, "sv_xgb": sv_xgb, "sv_ens": sv_ens,
             "ev_ens": ev_ens}, "models/shap_values.pkl")

def save(path):
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ---- Fig 7: Global beeswarm ----
shap.plots.beeswarm(expl, max_display=15, show=False)
plt.title("SHAP Summary — Soft Voting Ensemble")
save("figures/fig7_shap_beeswarm.png")

# ---- Fig 8: Global importance bar ----
shap.plots.bar(expl, max_display=15, show=False)
plt.title("Mean |SHAP| — Global Feature Importance")
save("figures/fig8_shap_importance_bar.png")

imp = pd.DataFrame({
    "ensemble": np.abs(sv_ens).mean(0),
    "rf": np.abs(sv_rf).mean(0),
    "xgb": np.abs(sv_xgb).mean(0),
}, index=features).sort_values("ensemble", ascending=False)
imp.round(5).to_csv("models/shap_importance.csv")
print("\nTop 10 features (mean |SHAP|, probability units):")
print(imp.head(10).round(4))
rho, p = spearmanr(imp["rf"], imp["xgb"])
print(f"\nRF vs XGB importance rank agreement: Spearman rho={rho:.3f} (p={p:.1e})")

# ---- Grouped SHAP for the correlated page-behaviour cluster ----
cluster = ["popupwindow", "favicon", "port", "submitting_to_email",
           "on_mouseover", "iframe", "rightclick"]
idx = [features.index(c) for c in cluster]
grp = np.abs(sv_ens[:, idx].sum(axis=1)).mean()
print(f"\nPage-behaviour cluster ({len(cluster)} features, |rho| 0.6-0.94):")
print(f"  grouped mean |SHAP| = {grp:.4f}  "
      f"vs largest member alone = {imp.loc[cluster, 'ensemble'].max():.4f}")

# ---- Fig 9: Dependence plot for the top feature ----
top = imp.index[0]
shap.plots.scatter(expl[:, top], color=expl, show=False)
plt.title(f"SHAP Dependence — {top}")
save("figures/fig9_shap_dependence_top.png")

# ---- Fig 10-12: Per-instance explanations ----
raw = pd.read_csv("phishing_processed.csv")
nlab = raw.groupby(features)["label"].nunique()
conflict = set(nlab[nlab > 1].index)
not_conf = np.array([tuple(r) not in conflict for r in X_test.values])

pred = (proba >= 0.5).astype(int)
yt = y_test.values
cases = {
    "fig10_waterfall_true_positive":  (yt == 1) & (pred == 1),
    "fig11_waterfall_true_negative":  (yt == 0) & (pred == 0),
    "fig12_waterfall_false_negative": (yt == 1) & (pred == 0) & not_conf,
}
for fname, mask in cases.items():
    cand = np.where(mask)[0]
    if len(cand) == 0:
        print(f"No instance for {fname}"); continue
    # TP: most confident; TN: most confident; FN: most confidently missed
    i = cand[np.argmax(proba[cand])] if "positive" in fname \
        else cand[np.argmin(proba[cand])]
    shap.plots.waterfall(expl[i], max_display=12, show=False)
    plt.title(f"{fname.split('_', 2)[2].replace('_', ' ').title()} "
              f"— P(phishing)={proba[i]:.3f}, true={'Phish' if yt[i] else 'Legit'}")
    save(f"figures/{fname}.png")
    print(f"{fname}: test row {i}, P(phishing)={proba[i]:.3f}")

print("\nSaved: figures/fig7-fig12, models/shap_values.pkl, models/shap_importance.csv")