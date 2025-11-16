from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.utils import Bunch


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TITANIC_CSV = DATA_DIR / "titanic_raw.csv"
PENGUINS_CSV = DATA_DIR / "penguins_raw.csv"


def load_titanic_dataset() -> Bunch:
    """
    Return a cleaned Titanic passenger dataset suitable for downstream experiments.

    The raw CSV (vendored from OpenML) contains many textual columns that are not
    directly useful for our tree models. We keep a compact subset, impute missing
    numeric values with medians, and one-hot encode the categorical fields.
    """

    if not TITANIC_CSV.exists():
        raise FileNotFoundError(
            f"Missing Titanic dataset at {TITANIC_CSV}. Please place the CSV there "
            "or regenerate it via the README instructions."
        )

    df = pd.read_csv(TITANIC_CSV)
    df = df.dropna(subset=["survived"]).copy()

    feature_columns = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    features = df[feature_columns].copy()
    numeric_cols = ["pclass", "age", "sibsp", "parch", "fare"]
    categorical_cols = ["sex", "embarked"]

    for col in numeric_cols:
        features[col] = pd.to_numeric(features[col], errors="coerce")
        median = features[col].median()
        features[col] = features[col].fillna(median)

    for col in categorical_cols:
        features[col] = features[col].fillna("missing")

    processed = pd.get_dummies(features, columns=categorical_cols, prefix=categorical_cols, drop_first=False)

    target = df["survived"].astype(int)
    return Bunch(
        data=processed.to_numpy(),
        target=target.to_numpy(),
        feature_names=list(processed.columns),
    )


def load_penguins_dataset() -> Bunch:
    """
    Load a cleaned Palmer Penguins dataset.

    We keep the standard numeric measurements and encode the categorical columns.
    """

    if not PENGUINS_CSV.exists():
        raise FileNotFoundError(
            f"Missing Penguins dataset at {PENGUINS_CSV}. Please place the CSV there "
            "or regenerate it via the README instructions."
        )

    df = pd.read_csv(PENGUINS_CSV)
    df = df.dropna(subset=["species"]).copy()

    feature_columns = [
        "island",
        "sex",
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]
    features = df[feature_columns].copy()
    numeric_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    categorical_cols = ["island", "sex"]

    for col in numeric_cols:
        features[col] = pd.to_numeric(features[col], errors="coerce")
        median = features[col].median()
        features[col] = features[col].fillna(median)

    for col in categorical_cols:
        features[col] = features[col].fillna("missing")

    processed = pd.get_dummies(features, columns=categorical_cols, drop_first=False)
    target = df["species"].astype(str)
    return Bunch(
        data=processed.to_numpy(),
        target=target.to_numpy(),
        feature_names=list(processed.columns),
    )
