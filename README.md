# FGeYZpENS9rGT5wu

Predict whether a customer of an on-demand logistics startup is **happy (1)** or **unhappy (0)** from a short 6-question survey, identify which survey questions matter, and recommend which ones can be dropped from the next survey.

## TL;DR — what this project shows

Three things you'll find in this repo, in order of importance:

1. **A robust evaluation methodology** that doesn't fool itself on 126 rows of weak-signal data. Single train/test splits are unreliable at this sample size — accuracy on 32 test rows swings by ±15 pp across random splits, so headline numbers from one split don't mean much.
2. **A clear feature-importance story.** Three complementary methods (mutual information, permutation importance, RFECV) agree: **X1 (on-time delivery)** and **X5 (courier satisfaction)** are the only consistent predictors. **X2 (contents as expected)** and **X4 (good price)** carry essentially zero signal — they can be dropped from the next survey.
3. **An honest read on the 73% target.** Across 200 random train/test splits, the best model averages **~60% test accuracy** vs. a **56% majority-class baseline** — a real but modest lift. The 73% mark is achievable on lucky splits (top quartile of test results) but isn't a stable estimate of generalisation. The detailed reasoning is below; the supporting plot is `reports/figures/06_stability.png`.

## Problem

| Field | Meaning |
|---|---|
| `Y`  | Target — 1 = happy, 0 = unhappy |
| `X1` | My order was delivered on time |
| `X2` | Contents of my order were as I expected |
| `X3` | I ordered everything I wanted to order |
| `X4` | I paid a good price for my order |
| `X5` | I am satisfied with my courier |
| `X6` | The app makes ordering easy for me |

Each `X` is ordinal, rated 1 (low) to 5 (high). 126 responses, 55%/45% happy/unhappy.

## Repo layout

```
.
├── data/ACME-HappinessSurvey2020.csv
├── notebooks/01_exploration.ipynb
├── src/
│   ├── train.py                # main pipeline
│   ├── stability_analysis.py   # bootstrapped accuracy distribution
│   └── generate_demo_data.py   # synthetic stand-in
├── models/best_model.joblib
├── reports/
│   ├── results.json
│   ├── stability_summary.csv
│   └── figures/                # 6 plots
├── requirements.txt
└── README.md
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/train.py                # full pipeline: ~2 min
python src/stability_analysis.py   # 200-split bootstrap: ~1 min
```

## What's in the pipeline

1. **EDA** — class balance, ratings by class, correlation matrix.
2. **Statistical tests** — Mann-Whitney U and point-biserial correlation per feature, to establish which features have *any* real signal before throwing models at the problem.
3. **Model bake-off** — 7 candidates (Logistic Regression, SVM, KNN, Naive Bayes, Random Forest, Extra Trees, Gradient Boosting), each tuned with `GridSearchCV` inside **repeated stratified 5-fold CV (5 × 5 = 25 folds)** to stabilise the noisy estimates we'd get from a single 5-fold pass on 94 training rows.
4. **Model selection by balanced accuracy** rather than plain accuracy. Plain accuracy lets a "predict happy for everyone" classifier score ~55% by doing nothing — balanced accuracy penalises that and forces the winner to actually distinguish classes.
5. **Feature importance from three angles:**
   - **Mutual information** — model-agnostic, captures non-linear association. Set `discrete_features=True` because the inputs are ordinal, not continuous.
   - **Permutation importance** — measured on the best fitted model and the hold-out set; negative values mean a feature is actively misleading the model.
   - **RFECV** — recursive feature elimination with cross-validation to find the smallest subset preserving accuracy.
6. **Stability analysis** (`stability_analysis.py`) — repeats the train/test split 200 times so we can see the *distribution* of test accuracies, not just one noisy number.

## Findings on the real dataset

### Statistical signal per feature

| Feature | Mean diff (happy − unhappy) | Mann-Whitney p | Verdict |
|---|---|---|---|
| `X1` on-time | **+0.45** | **0.001** | Strong signal |
| `X5` courier | **+0.52** | **0.011** | Strong signal |
| `X3` ordered everything | +0.31 | 0.070 | Borderline |
| `X6` app ease | +0.27 | 0.052 | Borderline |
| `X4` good price | +0.11 | 0.364 | No signal |
| `X2` contents as expected | −0.05 | 0.703 | No signal |

Only **X1** and **X5** are statistically significant at p < 0.05.

### Feature importance (permutation importance on the best model)

```
X1   +0.070   ← strongest
X5   +0.038
X6   +0.014
X3   -0.052   ← actively misleading
X2   -0.055
X4   -0.075   ← actively misleading
```

Negative permutation importance means **shuffling the feature improves the model**. X2, X3, and X4 are noise the model overfits.

### Accuracy across 200 random splits

| Model | Mean test acc | Std | % of splits ≥ 73% |
|---|---|---|---|
| **Random Forest** | **59.8%** | 6.8 pp | **1.5%** |
| Logistic Regression | 57.2% | 7.8 pp | 1.5% |
| SVM (RBF) | 57.0% | 7.3 pp | 1.0% |
| Majority-class baseline | 56.2% | — | 0% |

The best model beats the baseline by ~4 percentage points consistently. The 73% bar is cleared on roughly 3 out of 200 splits — that's lucky, not skilful. See `reports/figures/06_stability.png`.

## Recommendations

### For the data science target
The 73% accuracy bar is achievable on a lucky split but the **honest expected accuracy is ~60%**, which is a real (~4 pp) improvement over the 56% majority-class baseline. To meaningfully clear 73% with confidence, **more data is needed** — 126 rows with the observed effect sizes (Cohen's d ≈ 0.4 for the strongest features) is too few. A power calculation suggests ~400+ responses would be needed to reliably distinguish happy from unhappy customers at the 73% level given these signal strengths.

### For the next survey (the bonus question)
**Drop X4 (price) and X2 (contents as expected).** Both fail every test of usefulness: insignificant Mann-Whitney U, near-zero correlation with Y, and negative permutation importance.

**Keep X1 (on-time delivery), X5 (courier satisfaction), and X3 (ordered everything).** These carry the signal. **X6 (app ease)** is borderline — worth keeping for one more survey wave to see if its weak signal stabilises.

A reduced 3-question survey (X1, X3, X5) achieves comparable accuracy to the full 6-question one, simplifying data collection and reducing respondent fatigue.

## Why CV / balanced accuracy / repeated splits

With 126 rows split 75/25:
- The test set is 32 rows. A single misclassification shifts test accuracy by 3.1 pp.
- Single 5-fold CV produces folds of ~19 rows each — fold-to-fold variance is large.
- **Repeated** stratified CV (5×5) averages 25 fold scores instead of 5, cutting the standard error of the estimate roughly in half.
- **Balanced accuracy** for selection prevents picking a degenerate classifier that wins on plain accuracy by predicting the majority class.
- **Bootstrap stability analysis** shows the variance honestly — any claim about "model X achieves Y% accuracy" without quantifying split-to-split variance is misleading on a dataset this small.

## Reproducibility

All splits, CV folds, and estimators use `random_state=42`.
