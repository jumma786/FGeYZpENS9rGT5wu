"""
Honest small-data stability analysis.

With only 126 rows, a single train/test split gives a noisy accuracy estimate.
This script runs the model selection across many random splits to show the
*distribution* of test accuracies, which is the right way to evaluate on
such a tiny dataset.

Run:  python src/stability_analysis.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "ACME-HappinessSurvey2020.csv"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")

FEATURES = ["X1", "X2", "X3", "X4", "X5", "X6"]
N_REPEATS = 200
TARGET = 0.73


def build_models() -> dict:
    return {
        "Logistic Reg.": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                                       random_state=42)),
        ]),
        "SVM (RBF)": Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(C=1.0, kernel="rbf", probability=True, random_state=42)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=5, random_state=42),
        "Majority-class": None,  # baseline
    }


def run():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df["Y"]

    models = build_models()
    results = {name: [] for name in models}

    print(f"Running {N_REPEATS} random 75/25 splits to estimate accuracy distribution...")
    for seed in range(N_REPEATS):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=seed
        )
        for name, model in models.items():
            if model is None:
                pred = np.full(len(y_te), y_tr.mode().iloc[0])
            else:
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)
            results[name].append(accuracy_score(y_te, pred))

    summary = pd.DataFrame(results).agg(["mean", "std", "min", "max"]).T
    summary["≥ 73% rate"] = pd.DataFrame(results).apply(
        lambda c: (c >= TARGET).mean()
    )
    summary = summary.round(3)
    print("\n=== Test accuracy across 200 random splits ===")
    print(summary)

    summary.to_csv(ROOT / "reports" / "stability_summary.csv")

    fig, ax = plt.subplots(figsize=(10, 6))
    longform = pd.DataFrame(results).melt(var_name="Model", value_name="Test accuracy")
    sns.violinplot(data=longform, x="Model", y="Test accuracy", inner="quartile",
                   ax=ax, palette="Set2", cut=0)
    ax.axhline(TARGET, color="crimson", linestyle="--", linewidth=2,
               label=f"Target = {TARGET:.0%}")
    ax.axhline(y.mean(), color="grey", linestyle=":", linewidth=2,
               label=f"Majority-class baseline = {y.mean():.1%}")
    ax.set_title(f"Accuracy distribution across {N_REPEATS} random train/test splits")
    ax.set_ylim(0.3, 1.0)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_stability.png", dpi=140)
    plt.close(fig)

    print("\n=== Honest take ===")
    print(f"Majority-class baseline:          {summary.loc['Majority-class', 'mean']:.1%}")
    print(f"Best model mean accuracy:         {summary['mean'].drop('Majority-class').max():.1%}")
    print(f"Best model 73%-clearance rate:    {summary['≥ 73% rate'].drop('Majority-class').max():.1%}")
    print(f"\nSee {FIG_DIR}/06_stability.png")


if __name__ == "__main__":
    run()
