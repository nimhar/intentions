from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.model_selection import train_test_split

from mci.evaluators.sklearn_evaluator import SklearnEvaluator
from mci.estimators.permutation_samplling import PermutationSampling

try:
    import shap
except ImportError as exc:  # pragma: no cover - shap is an optional heavy dependency
    raise RuntimeError(
        "shap is required to reproduce the figures. Please install the project "
        "using `pip install -r requirements.txt`."
    ) from exc


TaskType = Literal["classification", "regression"]


@dataclass(frozen=True)
class DatasetSpec:
    """Definition for a dataset that should be evaluated."""

    name: str
    loader: Callable[[], object]
    drop_features: Sequence[str] = ()
    task_type: TaskType = "classification"


@dataclass
class ImportanceResult:
    """Container for the importance values used across multiple figures."""

    dataset_name: str
    feature_names: List[str]
    mci_values: np.ndarray
    shap_predicted: np.ndarray
    shap_all_classes: np.ndarray
    shap_std: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Feature": self.feature_names,
                "MCI Value": self.mci_values,
                "SHAP Value": self.shap_predicted,
                "SHAP All": self.shap_all_classes,
                "SHAP Std": self.shap_std,
            }
        )

    def kendall_tau(self) -> Tuple[float, float]:
        return kendall_tau_from_values(self.mci_values, self.shap_predicted)


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("'", "")
    )


def load_dataset(spec: DatasetSpec) -> Tuple[pd.DataFrame, pd.Series]:
    data = spec.loader()
    feature_names: Optional[Sequence[str]] = getattr(data, "feature_names", None)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(data.data.shape[1])]

    X = pd.DataFrame(data.data, columns=list(feature_names))
    y = pd.Series(data.target, name="target")

    if spec.drop_features:
        X = X.drop(columns=list(spec.drop_features), errors="ignore")

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    task_type: TaskType,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    stratify = y if task_type == "classification" else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def compute_mci_vs_shap(
    spec: DatasetSpec,
    model_factory: Callable[[], object],
    *,
    n_permutations: int = 2**5,
    random_state: int = 42,
) -> ImportanceResult:
    X, y = load_dataset(spec)
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        task_type=spec.task_type,
        random_state=random_state,
    )

    model = model_factory()
    if hasattr(model, "random_state"):
        setattr(model, "random_state", random_state)
    model.fit(X_train, y_train)

    evaluator = SklearnEvaluator(model)
    estimator = PermutationSampling(evaluator, n_permutations=n_permutations)
    mci_scores = estimator.mci_values(X_test, y_test)
    mci_values = np.asarray([float(v) for v in mci_scores.mci_values])

    shap_predicted, shap_all_classes, shap_std = compute_tree_shap(model, X)

    return ImportanceResult(
        dataset_name=spec.name,
        feature_names=list(X.columns),
        mci_values=mci_values,
        shap_predicted=shap_predicted,
        shap_all_classes=shap_all_classes,
        shap_std=shap_std,
    )


def compute_tree_shap(model, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    explainer = shap.TreeExplainer(model)
    raw_values = explainer.shap_values(X)
    shap_values = normalize_shap_values(raw_values, n_samples=len(X), n_features=X.shape[1])

    if shap_values.ndim == 3 and hasattr(model, "classes_"):
        class_lookup = {cls: idx for idx, cls in enumerate(model.classes_)}
        predictions = model.predict(X)
        class_indices = np.vectorize(class_lookup.get)(predictions)
        sample_indices = np.arange(len(predictions))
        shap_predicted = shap_values[sample_indices, class_indices]
        shap_all_classes = np.mean(np.abs(shap_values), axis=1)
    else:
        shap_predicted = shap_values
        shap_all_classes = shap_values

    shap_abs_pred = np.abs(shap_predicted)
    shap_mean_pred = shap_abs_pred.mean(axis=0)
    shap_std_pred = shap_abs_pred.std(axis=0)
    shap_mean_all = np.abs(shap_all_classes).mean(axis=0)

    return shap_mean_pred, shap_mean_all, shap_std_pred


def normalize_shap_values(values, *, n_samples: int, n_features: int) -> np.ndarray:
    """
    SHAP returns different container types depending on model + version.
    This helper coerces them into a consistent ndarray layout:
        - (n_samples, n_features) for single-output tasks
        - (n_samples, n_classes, n_features) for multi-class.
    """

    if isinstance(values, (list, tuple)):
        class_arrays = [
            ensure_2d_array(_raw_shap_array(v), n_samples, n_features) for v in values
        ]
        return np.stack(class_arrays, axis=1)

    array = _raw_shap_array(values)
    if array.ndim == 1:
        array = array.reshape(n_samples, -1)

    if array.ndim == 2:
        return ensure_2d_array(array, n_samples, n_features)

    if array.ndim == 3:
        return reorder_axes(array, n_samples, n_features)

    raise RuntimeError(f"Unsupported SHAP output dimensions: {array.shape}")


def _raw_shap_array(values) -> np.ndarray:
    if hasattr(values, "values"):
        return np.asarray(values.values)
    return np.asarray(values)


def ensure_2d_array(array: np.ndarray, n_samples: int, n_features: int) -> np.ndarray:
    if array.shape == (n_samples, n_features):
        return array
    if array.shape == (n_features, n_samples):
        return array.T
    raise RuntimeError(f"Unexpected SHAP shape {array.shape}; expected {(n_samples, n_features)}")


def reorder_axes(array: np.ndarray, n_samples: int, n_features: int) -> np.ndarray:
    shape = array.shape
    axes = list(range(array.ndim))

    feature_axis = _find_axis_with_size(shape, n_features)
    sample_axis = _find_axis_with_size(shape, n_samples)
    if feature_axis is None or sample_axis is None:
        raise RuntimeError(f"Cannot align SHAP output shape {shape}")

    remaining_axes = [ax for ax in axes if ax not in (feature_axis, sample_axis)]
    class_axis = remaining_axes[0] if remaining_axes else None

    if class_axis is None:
        # No class axis: collapse into 2D array.
        transposed = np.transpose(array, (sample_axis, feature_axis))
        return ensure_2d_array(transposed, n_samples, n_features)

    order = [sample_axis, class_axis, feature_axis]
    return np.transpose(array, order)


def _find_axis_with_size(shape: Sequence[int], size: int) -> Optional[int]:
    matches = [idx for idx, dim in enumerate(shape) if dim == size]
    return matches[0] if matches else None


def kendall_tau_from_values(values_a: Sequence[float], values_b: Sequence[float]) -> Tuple[float, float]:
    series_a = pd.Series(values_a).rank(ascending=False)
    series_b = pd.Series(values_b).rank(ascending=False)
    tau, p_value = kendalltau(series_a, series_b)
    return tau, p_value


def top_features(df: pd.DataFrame, column: str, k: int = 8) -> pd.DataFrame:
    return df.sort_values(column, ascending=False).head(k).copy()


def format_feature_labels(features: Iterable[str]) -> List[str]:
    return [str(feature).replace("_", " ") for feature in features]
