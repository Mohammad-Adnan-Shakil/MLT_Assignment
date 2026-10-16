# Explainable Phishing Website Detection with a Random Forest + XGBoost Ensemble and SHAP

An end-to-end, reproducible machine-learning pipeline for detecting phishing websites on the UCI Phishing Websites dataset. It combines Random Forest and XGBoost in soft-voting and stacking ensembles and explains predictions with exact SHAP decompositions.

The pipeline also audits the dataset itself. **47.1% of its rows are exact duplicates**, and evaluating on the raw data reports roughly 2.2 percentage points higher accuracy than evaluating on deduplicated data. All primary results here are reported on deduplicated data.

> CA3 research assignment: Mohammad Adnan Shakil | Dileep MK | Sultan Salauddin Ansari , B.Tech CSE, Presidency University, Bengaluru.

---

## Key Findings

| | |
|---|---|
| **Final model** | Soft Voting (RF + XGBoost) |
| **Test accuracy / F1 (phishing)** | 0.9530 / 0.9541 |
| **Test ROC-AUC** | 0.9923 |
| **Variance reduction** | CV F1 std 0.0034 vs 0.0059 for either base model (~42% lower) |
| **Duplicate rows in dataset** | 5,206 of 11,055 (47.1%) |
| **Accuracy inflation on raw data** | 97.5% raw vs 95.3% deduplicated |
| **Irreducible label noise** | 64 feature vectors carry both labels; they cause 31% of test errors while making up 2.4% of the test set |
| **Top SHAP features** | `sslfinal_state`, `url_of_anchor` (~3× the weight of any other feature) |
| **SHAP exactness** | max additivity error 1.18e-07 |

1. **The ensembles do not significantly beat XGBoost on accuracy.** McNemar's test gives p = 1.0 for XGBoost vs Soft Voting. What the ensemble adds is **stability across data splits**, not a higher score.
2. **Random Forest alone is significantly worse** than XGBoost and both ensembles (McNemar p < 0.01).
3. **Duplicate rows inflate reported accuracy.** The gap persists after removing label-conflicting samples and is not explained by train–test memorisation or frequency weighting. All three mechanisms were tested; see [Leakage Analysis](#leakage-analysis).

---

## Dataset

**UCI Phishing Websites** (Mohammad, Thabtah & McCluskey, 2012). [UCI repository, id 327](https://archive.ics.uci.edu/dataset/327/phishing+websites)

- 11,055 samples, 30 features: 22 binary and 8 ternary, encoded as −1 / 0 / 1
- Binary target `result`: −1 = phishing, 1 = legitimate. This repo remaps it to **phishing = 1, legitimate = 0**, so precision, recall and F1 refer to phishing detection.
- No missing values

| | Raw | Deduplicated |
|---|---|---|
| Rows | 11,055 | 5,849 |
| Phishing | 4,898 (44.3%) | 3,019 (51.6%) |
| Legitimate | 6,157 (55.7%) | 2,830 (48.4%) |

The duplicates are mostly legitimate sites, so the raw dataset's class imbalance is largely an artefact of that duplication.

The dataset is fetched automatically through `ucimlrepo`, so no manual download is needed. If `Training Dataset.arff` is placed in the repo root, it is loaded instead.

---

## Pipeline

```
step1_load.py        Load data (ARFF or ucimlrepo), shape, class distribution
step2_eda.py         Label remap, duplicate audit, class plot, Spearman heatmaps
step3_train.py       Deduplicate, 80/20 stratified split, tune RF and XGBoost
step4_ensemble.py    Soft voting + stacking, unified 5-fold CV comparison
step5_evaluate.py    Confusion matrices, ROC, McNemar tests, leakage ablation
step5b_freq_check.py Leakage diagnostic: accuracy vs feature-vector frequency
step5c_conflicts.py  Leakage diagnostic: label-conflicting duplicates
step6_shap.py        Exact ensemble SHAP: global, grouped, per-instance
```

### Methodology

- **Deduplication before splitting.** Exact duplicate rows are removed before the train/test split. The 64 label-conflicting feature vectors are kept, because they are real label noise and removing them would make the task artificially clean.
- **Split:** 80/20 stratified, fixed seed 42. Train 4,679 rows, test 1,170.
- **Tuning:** `RandomizedSearchCV`, 30 iterations per model, 5-fold stratified CV on the **training set only**, optimising F1 on the phishing class. The test set is used exactly once.
- **Ensembles:**
  - Soft Voting: unweighted mean of the RF and XGBoost probabilities
  - Stacking: logistic-regression meta-learner on 5-fold out-of-fold probabilities
- **Fair comparison:** all four models are evaluated on the **same 5 CV folds**, reported as mean ± std, plus one held-out test evaluation.
- **Significance testing:** pairwise McNemar exact test on the test set.
- **Correlation:** Spearman, because the features are ordinal (−1/0/1).
- **Explainability:** SHAP `TreeExplainer` in probability space with interventional perturbation, using one shared 100-sample background for both models. Because Soft Voting averages the two probabilities, the ensemble's SHAP values are exactly the mean of the RF and XGBoost SHAP values. This is verified by an additivity check.

### Tuned hyperparameters

| Model | Parameters |
|---|---|
| Random Forest | `n_estimators=500, max_depth=None, min_samples_split=5, min_samples_leaf=1, max_features='sqrt'` |
| XGBoost | `n_estimators=200, max_depth=4, learning_rate=0.2, subsample=0.7, colsample_bytree=0.6, min_child_weight=1, gamma=0.3` |

---

## Results

### Cross-validation (training set, same 5 folds, mean ± std)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.9500 ± 0.0062 | 0.9570 ± 0.0103 | 0.9458 ± 0.0099 | 0.9513 ± 0.0059 | 0.9899 ± 0.0027 |
| XGBoost | 0.9517 ± 0.0061 | 0.9549 ± 0.0095 | 0.9516 ± 0.0115 | 0.9531 ± 0.0059 | 0.9917 ± 0.0023 |
| **Soft Voting** | **0.9523 ± 0.0036** | 0.9545 ± 0.0088 | 0.9532 ± 0.0091 | **0.9538 ± 0.0034** | 0.9915 ± 0.0025 |
| Stacking | 0.9515 ± 0.0036 | 0.9530 ± 0.0089 | 0.9532 ± 0.0104 | 0.9530 ± 0.0036 | 0.9914 ± 0.0026 |

### Held-out test set (n = 1,170)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.9368 | 0.9538 | 0.9222 | 0.9377 | 0.9886 |
| XGBoost | 0.9530 | 0.9613 | 0.9470 | 0.9541 | 0.9933 |
| **Soft Voting** | **0.9530** | **0.9629** | 0.9454 | **0.9541** | 0.9923 |
| Stacking | 0.9513 | 0.9612 | 0.9437 | 0.9524 | 0.9921 |

On the test set, Soft Voting makes **33 false negatives** (phishing sites missed) and **22 false positives**.

### McNemar exact test (test set)

| Comparison | A right / B wrong | A wrong / B right | p-value |
|---|---|---|---|
| RF vs XGBoost | 14 | 33 | **0.0079** |
| RF vs Soft Voting | 7 | 26 | **0.0013** |
| RF vs Stacking | 7 | 24 | **0.0033** |
| XGBoost vs Soft Voting | 12 | 12 | 1.0000 |
| XGBoost vs Stacking | 13 | 11 | 0.8388 |
| Soft Voting vs Stacking | 2 | 0 | 0.5000 |

The stacking meta-learner weights RF and XGBoost almost equally (4.63 vs 4.02), consistent with the two models making highly correlated errors.

---

## Leakage Analysis

The usual practice on this dataset is a random split of the raw data. That puts **64.6% of test rows** in the position of sharing an exact feature vector with the training set.

| Model | Raw split accuracy | Deduplicated accuracy | Raw F1 | Deduplicated F1 |
|---|---|---|---|---|
| XGBoost | 0.9706 | 0.9530 | 0.9666 | 0.9541 |
| Soft Voting | 0.9751 | 0.9530 | 0.9718 | 0.9541 |

Three candidate mechanisms for the gap were tested:

| Hypothesis | Test | Result |
|---|---|---|
| Memorisation of seen rows | Raw-test accuracy on rows seen in training vs rows not seen | Ruled out: 0.970 seen vs 0.972 unseen (XGBoost) |
| Frequency weighting | Accuracy by how often a feature vector occurs; raw test with each unique vector counted once | Ruled out: flat across frequency (0.964–0.980); unique-only accuracy is still 0.9685 |
| Label-conflicting duplicates | Accuracy on conflict vs non-conflict rows | Partial only: non-conflict accuracy is still 0.987 raw vs 0.967 deduplicated |

**Conclusion:** raw evaluation inflates accuracy by about 2 pp through a mechanism not identified here, so deduplicated results are reported as primary.

A separate finding: the 28 label-conflicting rows in the deduplicated test set are predicted at 39% accuracy and cause **17 of the 55 errors**. That sets an irreducible error floor for this feature set.

---

## Explainability (SHAP)

### Global importance (mean |SHAP|, probability units)

| Rank | Feature | Ensemble | RF | XGBoost |
|---|---|---|---|---|
| 1 | `sslfinal_state` | 0.1558 | 0.1633 | 0.1484 |
| 2 | `url_of_anchor` | 0.1429 | 0.1442 | 0.1434 |
| 3 | `prefix_suffix` | 0.0518 | 0.0520 | 0.0517 |
| 4 | `having_sub_domain` | 0.0380 | 0.0400 | 0.0361 |
| 5 | `web_traffic` | 0.0369 | 0.0349 | 0.0404 |
| 6 | `links_in_tags` | 0.0352 | 0.0393 | 0.0328 |
| 7 | `sfh` | 0.0308 | 0.0295 | 0.0321 |
| 8 | `having_ip_address` | 0.0244 | 0.0166 | 0.0322 |
| 9 | `links_pointing_to_page` | 0.0230 | 0.0150 | 0.0316 |
| 10 | `request_url` | 0.0185 | 0.0168 | 0.0204 |

- **The two models agree:** their importance rankings match closely (Spearman ρ = 0.939, p = 1.8e-14).
- **SHAP and EDA agree:** the top two SHAP features are also the top two by Spearman correlation with the label (−0.74, −0.70).
- **Multicollinearity:** seven page-behaviour features are inter-correlated (|ρ| 0.6–0.94): `popupwindow`, `favicon`, `port`, `submitting_to_email`, `on_mouseover`, `iframe` and `rightclick`. SHAP splits credit among them. Their grouped mean |SHAP| is 0.0137, against 0.0075 for the largest single member, which is still minor next to the top features.

Per-instance waterfall plots explain a confident true positive, a confident true negative, and a confidently **missed** phishing site (P(phishing) = 0.118). Label-conflicting rows are excluded when selecting the missed site, so it reflects a genuine model error.

---

## Figures

All figures are saved at 300 dpi in `figures/`.

| File | Content |
|---|---|
| `fig1_class_distribution.png` | Class distribution (raw data) |
| `fig2_correlation_heatmap.png` | Spearman feature–feature correlation |
| `fig3_target_correlation.png` | Feature–target correlation |
| `fig4_confusion_matrices.png` | Confusion matrices, all four models |
| `fig4b_confusion_softvoting.png` | Confusion matrix, final model |
| `fig5_roc_curves.png` | ROC curves with zoomed inset |
| `fig6_leakage_ablation.png` | Raw vs deduplicated accuracy |
| `fig7_shap_beeswarm.png` | SHAP summary (beeswarm) |
| `fig8_shap_importance_bar.png` | Global mean \|SHAP\| |
| `fig9_shap_dependence_top.png` | Dependence plot for `sslfinal_state` |
| `fig10_waterfall_true_positive.png` | Explanation: correctly detected phishing site |
| `fig11_waterfall_true_negative.png` | Explanation: correctly identified legitimate site |
| `fig12_waterfall_false_negative.png` | Explanation: missed phishing site |

Tabular outputs are in `models/`: `individual_results.json`, `model_comparison.csv`, `mcnemar.csv`, `leakage_ablation.csv` and `shap_importance.csv`.

---

## Reproducing

Tested with Python 3.14, scikit-learn 1.9.1, XGBoost 3.4.1, SHAP 0.54.1 and pandas 3.0.6.

```bash
git clone https://github.com/Mohammad-Adnan-Shakil/MLT_Assignment.git
cd MLT_Assignment
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

Run the scripts in order. Each one reads the outputs of the previous step.

```bash
python step1_load.py
python step2_eda.py
python step3_train.py        # ~2-4 min (hyperparameter search)
python step4_ensemble.py     # ~3-5 min
python step5_evaluate.py
python step5b_freq_check.py
python step5c_conflicts.py
python step6_shap.py         # ~6 min (interventional SHAP on RF)
```

Trained model files (`models/*.pkl`) are not committed because of their size; they are regenerated by steps 3–4. A fixed seed (42) is used throughout.

---

## Limitations

- **Single train/test split.** The test results come from one split. Cross-validation means and standard deviations are reported to show the variance.
- **Hyperparameters are reused in the ablation.** The leakage ablation applies hyperparameters tuned on deduplicated data and does not re-tune them on raw data.
- **Unexplained inflation.** The mechanism behind the ~2 pp raw-data inflation is not identified.
- **Correlated features.** SHAP attributions within correlated feature groups are not uniquely determined. Grouped importance is reported for the main cluster.
- **Reliance, not causation.** SHAP shows what the model relies on, not what causes phishing.
- **Pre-engineered features.** The 30 features are pre-extracted, discretised indicators from a 2012 dataset. They are not computed from live URLs, so they may not reflect current phishing techniques. For example, phishing sites now commonly use valid SSL certificates.
- **Fixed threshold.** The decision threshold is 0.5. Security deployments, where false negatives are costly, would tune it.

---

## Citation

Dataset:

> Mohammad, R., Thabtah, F., & McCluskey, L. (2012). *Phishing Websites* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C51W2X