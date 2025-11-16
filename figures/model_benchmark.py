from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    import xgboost as xgb
except ImportError as exc:  # pragma: no cover - xgboost is required for the experiments
    raise RuntimeError(
        "xgboost is required to reproduce the Figures 5-6 experiments."
    ) from exc

from .cache_utils import load_or_build
from .common import compute_tree_shap


@dataclass
class BenchmarkArtifacts:
    feature_names: List[str]
    X_full: pd.DataFrame
    y_full: pd.Series
    importance_df: pd.DataFrame
    accuracy: Dict[str, Tuple[float, float]]


def prepare_model_benchmarks(
    *,
    random_state: int = 42,
    kernel_sample_size: int = 120,
    kernel_background_size: int = 60,
    cache_dir: Path | None = None,
) -> BenchmarkArtifacts:
    if cache_dir is not None:
        cache_path = cache_dir / f"benchmarks_rs{random_state}.pkl"
        return load_or_build(
            cache_path,
            lambda: _build_benchmark_artifacts(random_state, kernel_sample_size, kernel_background_size),
        )
    return _build_benchmark_artifacts(random_state, kernel_sample_size, kernel_background_size)


def _build_benchmark_artifacts(random_state: int, kernel_sample_size: int, kernel_background_size: int) -> BenchmarkArtifacts:
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=random_state,
        stratify=y,
    )

    tree = DecisionTreeClassifier(random_state=random_state)
    rf = RandomForestClassifier(n_estimators=500, random_state=random_state)
    svm = SVC(
        kernel="rbf",
        C=1e5,
        probability=True,
        random_state=random_state,
    )
    xgb_clf = xgb.XGBClassifier(
        n_estimators=500,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric="logloss",
    )

    models = {
        "Tree": tree,
        "Random_Forest": rf,
        "SVM": svm,
        "XGBoost": xgb_clf,
    }

    for model in models.values():
        model.fit(X_train, y_train)

    accuracy = {
        name: (
            accuracy_score(y_train, model.predict(X_train)),
            accuracy_score(y_test, model.predict(X_test)),
        )
        for name, model in models.items()
    }

    n_features = len(data.feature_names)
    tree_importances = ensure_feature_vector(
        normalize(models["Tree"].feature_importances_), n_features, "Tree importances"
    )
    rf_importances = ensure_feature_vector(
        normalize(models["Random_Forest"].feature_importances_), n_features, "Random Forest importances"
    )
    xgb_importances = ensure_feature_vector(
        normalize(models["XGBoost"].feature_importances_), n_features, "XGBoost importances"
    )

    tree_shap = ensure_feature_vector(
        normalize(compute_tree_shap(models["Tree"], X_train)[0]), n_features, "Tree SHAP"
    )
    rf_shap = ensure_feature_vector(
        normalize(compute_tree_shap(models["Random_Forest"], X_train)[0]), n_features, "Random Forest SHAP"
    )
    xgb_shap = ensure_feature_vector(
        normalize(compute_tree_shap(models["XGBoost"], X_train)[0]), n_features, "XGBoost SHAP"
    )
    svm_shap = ensure_feature_vector(
        normalize(
            compute_kernel_shap(
                models["SVM"],
                X_train,
                random_state,
                kernel_sample_size,
                kernel_background_size,
                n_features,
            )
        ),
        n_features,
        "SVM SHAP",
    )

    importance_df = pd.DataFrame(
        {
            "Feature": data.feature_names,
            "Tree": tree_importances,
            "Tree_SHAP": tree_shap,
            "Random_Forest": rf_importances,
            "Random_Forest_SHAP": rf_shap,
            "SVM": svm_shap,
            "XGBoost": xgb_importances,
            "XGBoost_SHAP": xgb_shap,
        }
    )

    return BenchmarkArtifacts(
        feature_names=list(data.feature_names),
        X_full=X,
        y_full=y,
        importance_df=importance_df,
        accuracy=accuracy,
    )


def compute_kernel_shap(
    model: SVC,
    X_train: pd.DataFrame,
    random_state: int,
    sample_size: int,
    background_size: int,
    n_features: int,
) -> np.ndarray:
    background = X_train.sample(
        n=min(background_size, len(X_train)),
        random_state=random_state,
    ).to_numpy()
    explainer = shap.KernelExplainer(model.predict_proba, background)
    eval_points = X_train.sample(
        n=min(sample_size, len(X_train)),
        random_state=random_state + 7,
    ).to_numpy()

    shap_values = explainer.shap_values(eval_points)
    return aggregate_kernel_shap(shap_values, n_features=n_features)


def aggregate_kernel_shap(values, *, n_features: int):
    """
    KernelExplainer returns either:
      * A list (per class) of explanation arrays (n_samples, n_features)
      * A single array (n_samples, n_features) for binary/regression.
    We always return a per-feature vector aggregated across samples/classes.
    """

    def to_array(v):
        if hasattr(v, "values"):
            v = v.values
        return np.asarray(v)

    if isinstance(values, list):
        per_class = [aggregate_kernel_shap(v, n_features=n_features) for v in values]
        stacked = np.vstack(per_class)
        return stacked.mean(axis=0)

    arr = np.abs(to_array(values))

    if arr.ndim == 1:
        if arr.shape[0] == n_features:
            return arr
        raise ValueError(f"Kernel SHAP vector has length {arr.shape[0]}, expected {n_features}")

    flat = arr.reshape(-1)
    if flat.size % n_features != 0:
        raise ValueError(f"Kernel SHAP output of shape {arr.shape} incompatible with {n_features} features")

    reshaped = flat.reshape(-1, n_features)
    return reshaped.mean(axis=0)


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    max_val = np.max(np.abs(values))
    if max_val == 0:
        return values
    return values / max_val


def ensure_feature_vector(values: np.ndarray, n_features: int, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.shape[0] != n_features:
        raise ValueError(f"{label} must be length {n_features}, got shape {arr.shape}")
    return arr
