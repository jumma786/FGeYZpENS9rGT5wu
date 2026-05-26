"""
Train, tune, and evaluate models on the Apziva happiness survey.

Run:  python src/train.py
Outputs:
  - models/best_model.joblib
  - reports/results.json
  - reports/figures/*.png
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFECV, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "ACME-HappinessSurvey2020.csv"
FIG_DIR = ROOT / "reports" / "figures"
MODEL_DIR = ROOT / "models"
RESULTS_PATH = ROOT / "reports" / "results.json"

FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET_ACCURACY = 0.73
FEATURES = ["X1", "X2", "X3", "X4", "X5", "X6"]

sns.set_theme(style="whitegrid", context="talk")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Either download the real CSV from the "
            "Apziva brief or run `python src/generate_demo_data.py`."
        )
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows × {df.shape[1]} cols from {DATA_PATH.name}")
    return df


def eda_summary(df: pd.DataFrame) -> None:
    print("\n=== EDA SUMMARY ===")
    print(df.describe().round(2))
    print("\nClass balance:")
    print(df["Y"].value_counts(normalize=True).round(3).to_string())
    print(f"\nMissing values: {df.isna().sum().sum()}")

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature correlation matrix", pad=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_correlation.png", dpi=140)
    plt.close(fig)

    # Distribution by class
    melt = df.melt(id_vars="Y", value_vars=FEATURES, var_name="feature", value_name="rating")
    g = sns.catplot(
        data=melt, x="rating", col="feature", hue="Y", kind="count",
        col_wrap=3, height=3.2, aspect=1.1, palette="Set2",
    )
    g.fig.suptitle("Rating distribution by happiness class", y=1.02)
    g.savefig(FIG_DIR / "02_distributions.png", dpi=140)
    plt.close(g.fig)


# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
def build_candidates() -> dict[str, tuple[Pipeline, dict]]:
    """Return {name: (pipeline, param_grid)}."""
    return {
        "logreg": (
            Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            ]),
            {"clf__C": [0.01, 0.1, 1, 10],
             "clf__class_weight": [None, "balanced"]},
        ),
        "svm": (
            Pipeline([
                ("scale", StandardScaler()),
                ("clf", SVC(probability=True, random_state=RANDOM_STATE)),
            ]),
            {"clf__C": [0.1, 1, 10],
             "clf__kernel": ["rbf", "linear"],
             "clf__class_weight": [None, "balanced"]},
        ),
        "knn": (
            Pipeline([
                ("scale", StandardScaler()),
                ("clf", KNeighborsClassifier()),
            ]),
            {"clf__n_neighbors": [5, 7, 11, 15],
             "clf__weights": ["uniform", "distance"]},
        ),
        "nb": (
            Pipeline([("clf", GaussianNB())]),
            {"clf__var_smoothing": [1e-9, 1e-7]},
        ),
        "rf": (
            Pipeline([("clf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE))]),
            {"clf__max_depth": [None, 3, 5],
             "clf__min_samples_split": [2, 5],
             "clf__class_weight": [None, "balanced"]},
        ),
        "extratrees": (
            Pipeline([("clf", ExtraTreesClassifier(n_estimators=300, random_state=RANDOM_STATE))]),
            {"clf__max_depth": [None, 3, 5],
             "clf__class_weight": [None, "balanced"]},
        ),
        "gbm": (
            Pipeline([("clf", GradientBoostingClassifier(random_state=RANDOM_STATE))]),
            {"clf__n_estimators": [100, 200],
             "clf__learning_rate": [0.05, 0.1],
             "clf__max_depth": [2, 3]},
        ),
    }


def evaluate(name: str, model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4) if proba is not None else None,
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }


def train_and_compare(X_train, y_train, X_test, y_test) -> tuple[dict, dict]:
    # Repeated stratified CV: 5 folds × 5 repeats = 25 fits per param combo.
    # On 94 rows a single 5-fold split is noisy; repetition stabilises the estimate.
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)

    # Use balanced accuracy for selection so a degenerate "always predict 1"
    # classifier (which scores ~55% accuracy) doesn't win.
    results, fitted = {}, {}

    for name, (pipe, grid) in build_candidates().items():
        print(f"\n>> Tuning {name} ...")
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="balanced_accuracy",
                          n_jobs=-1, refit=True)
        gs.fit(X_train, y_train)

        # Also compute plain accuracy via CV for reporting
        cv_acc = cross_val_score(gs.best_estimator_, X_train, y_train,
                                 cv=cv, scoring="accuracy", n_jobs=-1).mean()
        cv_bal_acc = gs.best_score_

        eval_ = evaluate(name, gs.best_estimator_, X_test, y_test)
        eval_["best_params"] = gs.best_params_
        eval_["cv_accuracy"] = round(cv_acc, 4)
        eval_["cv_balanced_accuracy"] = round(cv_bal_acc, 4)
        results[name] = eval_
        fitted[name] = gs.best_estimator_
        print(f"   CV acc {cv_acc:.3f} | CV bal-acc {cv_bal_acc:.3f} | "
              f"Test acc {eval_['accuracy']:.3f} | F1 {eval_['f1']:.3f}")
    return results, fitted


# ---------------------------------------------------------------------------
# Feature selection / importance
# ---------------------------------------------------------------------------
def feature_analysis(best_name: str, best_model, X_train, y_train, X_test, y_test) -> dict:
    print("\n=== FEATURE IMPORTANCE & SELECTION ===")
    out: dict = {}

    # 1. Mutual information (model-agnostic, captures non-linearity).
    #    discrete_features=True because the survey responses are ordinal 1-5,
    #    not continuous - otherwise MI estimates are unstable on integer columns.
    mi = mutual_info_classif(X_train, y_train, discrete_features=True,
                             random_state=RANDOM_STATE)
    mi_series = pd.Series(mi, index=FEATURES).sort_values(ascending=False)
    out["mutual_information"] = mi_series.round(4).to_dict()
    print("\nMutual information:")
    print(mi_series.round(4).to_string())

    # 2. Permutation importance on the best fitted model
    perm = permutation_importance(best_model, X_test, y_test,
                                  n_repeats=30, random_state=RANDOM_STATE, n_jobs=-1)
    perm_series = pd.Series(perm.importances_mean, index=FEATURES).sort_values(ascending=False)
    out["permutation_importance"] = perm_series.round(4).to_dict()
    print(f"\nPermutation importance ({best_name}):")
    print(perm_series.round(4).to_string())

    # 3. RFECV to find the minimal feature subset that preserves accuracy.
    #    Wrap with a tree-based estimator since it exposes feature_importances_.
    rfecv = RFECV(
        estimator=RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        step=1,
        cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE),
        scoring="balanced_accuracy",
        min_features_to_select=1,
        n_jobs=-1,
    )
    rfecv.fit(X_train, y_train)
    selected = [f for f, keep in zip(FEATURES, rfecv.support_) if keep]
    out["rfecv_selected"] = selected
    out["rfecv_n_features"] = int(rfecv.n_features_)
    out["rfecv_cv_scores"] = [round(s, 4) for s in rfecv.cv_results_["mean_test_score"]]
    print(f"\nRFECV selected {rfecv.n_features_} features: {selected}")

    # Plot RFECV curve
    fig, ax = plt.subplots(figsize=(8, 5))
    n_feats = range(1, len(rfecv.cv_results_["mean_test_score"]) + 1)
    ax.plot(n_feats, rfecv.cv_results_["mean_test_score"], marker="o", linewidth=2)
    ax.fill_between(
        n_feats,
        np.array(rfecv.cv_results_["mean_test_score"]) - np.array(rfecv.cv_results_["std_test_score"]),
        np.array(rfecv.cv_results_["mean_test_score"]) + np.array(rfecv.cv_results_["std_test_score"]),
        alpha=0.2,
    )
    ax.set_xlabel("Number of features selected")
    ax.set_ylabel("CV accuracy")
    ax.set_title("RFECV: minimal feature subset")
    ax.axhline(TARGET_ACCURACY, color="crimson", linestyle="--", label=f"Target = {TARGET_ACCURACY}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_rfecv.png", dpi=140)
    plt.close(fig)

    # 4. Retrain best model on the RFECV subset to verify it preserves performance
    if selected:
        X_train_sub = X_train[selected]
        X_test_sub = X_test[selected]
        best_model_sub = clone(best_model)
        best_model_sub.fit(X_train_sub, y_train)
        acc_sub = accuracy_score(y_test, best_model_sub.predict(X_test_sub))
        out["accuracy_on_selected_only"] = round(acc_sub, 4)
        print(f"\nAccuracy using only {selected}: {acc_sub:.3f}")

    # Importance bar chart
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    mi_series.plot.bar(ax=axes[0], color="#4c72b0")
    axes[0].set_title("Mutual information")
    axes[0].set_ylabel("MI score")
    perm_series.plot.bar(ax=axes[1], color="#dd8452")
    axes[1].set_title(f"Permutation importance ({best_name})")
    axes[1].set_ylabel("Mean accuracy drop")
    for ax in axes:
        ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_feature_importance.png", dpi=140)
    plt.close(fig)

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    df = load_data()
    eda_summary(df)

    X = df[FEATURES]
    y = df["Y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTrain: {len(X_train)} rows | Test: {len(X_test)} rows")

    results, fitted = train_and_compare(X_train, y_train, X_test, y_test)

    # Pick winner by CV balanced accuracy. This avoids selecting a degenerate
    # "predict majority class" model that happens to score ~55% accuracy on
    # this slightly imbalanced dataset.
    best_name = max(results, key=lambda k: results[k]["cv_balanced_accuracy"])
    best_model = fitted[best_name]
    print(f"\nBest model: {best_name} "
          f"(CV bal-acc = {results[best_name]['cv_balanced_accuracy']:.3f}, "
          f"CV acc = {results[best_name]['cv_accuracy']:.3f})")

    print("\nClassification report on hold-out test set:")
    print(classification_report(y_test, best_model.predict(X_test), digits=3))

    # Confusion matrix figure
    cm = confusion_matrix(y_test, best_model.predict(X_test))
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Unhappy", "Happy"],
                yticklabels=["Unhappy", "Happy"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix — {best_name}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_confusion_matrix.png", dpi=140)
    plt.close(fig)

    feature_info = feature_analysis(best_name, best_model, X_train, y_train, X_test, y_test)

    # Persist
    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")
    payload = {
        "target_accuracy": TARGET_ACCURACY,
        "best_model": best_name,
        "model_results": results,
        "feature_analysis": feature_info,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved model -> {MODEL_DIR / 'best_model.joblib'}")
    print(f"Saved results -> {RESULTS_PATH}")
    print(f"Saved figures -> {FIG_DIR}")

    if results[best_name]["accuracy"] >= TARGET_ACCURACY:
        print(f"\n✅ Target {TARGET_ACCURACY:.0%} reached: {results[best_name]['accuracy']:.1%}")
    else:
        print(f"\n⚠️  Test accuracy {results[best_name]['accuracy']:.1%} below target "
              f"{TARGET_ACCURACY:.0%}. CV accuracy is {results[best_name]['cv_accuracy']:.1%}.")


if __name__ == "__main__":
    main()
