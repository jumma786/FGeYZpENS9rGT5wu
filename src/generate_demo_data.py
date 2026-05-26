"""
Generate a synthetic dataset that mirrors the published Apziva happiness survey
(126 rows, 6 ordinal features rated 1-5, binary target Y).

Used only for end-to-end validation of the pipeline when the real CSV is
unavailable. In production you should drop in the real
`ACME-HappinessSurvey2020.csv` from the project brief.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


def _ordinal_feature(n: int, base_happy: bool, importance: float) -> np.ndarray:
    """Sample an ordinal 1-5 feature, biased higher for happy customers."""
    centre = 3.5 + importance * (1 if base_happy else -1)
    raw = RNG.normal(loc=centre, scale=1.0, size=n)
    return np.clip(np.round(raw), 1, 5).astype(int)


def generate(n: int = 126, happy_ratio: float = 0.547) -> pd.DataFrame:
    n_happy = int(round(n * happy_ratio))
    n_unhappy = n - n_happy

    # Importance weights roughly reflect the relationships reported by
    # community write-ups: X1 (on-time delivery) and X5 (courier) dominate,
    # while X4 (price) is weak/noisy.
    weights = {"X1": 0.9, "X2": 0.4, "X3": 0.3, "X4": 0.1, "X5": 0.8, "X6": 0.5}

    rows = []
    for happy, count in [(1, n_happy), (0, n_unhappy)]:
        block = {"Y": np.full(count, happy)}
        for col, w in weights.items():
            block[col] = _ordinal_feature(count, bool(happy), w)
        rows.append(pd.DataFrame(block))

    df = pd.concat(rows, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    return df[["Y", "X1", "X2", "X3", "X4", "X5", "X6"]]


if __name__ == "__main__":
    df = generate()
    out = "data/ACME-HappinessSurvey2020.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")
    print(df.head())
    print(f"\nClass balance: {df['Y'].value_counts(normalize=True).to_dict()}")
