from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from scipy.stats import kendalltau
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from .cache_utils import load_or_build
from .common import DatasetSpec, compute_mci_vs_shap, ensure_output_dir, load_dataset
from .model_benchmark import BenchmarkArtifacts, prepare_model_benchmarks, normalize


MODEL_COLUMNS = ["Tree", "Random_Forest", "SVM", "XGBoost"]
MODEL_PAIRS = [
    ("Tree", "Random_Forest"),
    ("Tree", "SVM"),
    ("Random_Forest", "SVM"),
    ("Random_Forest", "XGBoost"),
    ("SVM", "XGBoost"),
]
METHOD_COLUMNS = ["Expected_SHAP", "MCI", "Ablation", "Bivariate"]
METHOD_PAIRS = list(combinations(METHOD_COLUMNS, 2))


def run_separable_set_analysis(
    output_dir: Path,
    artifacts: BenchmarkArtifacts | None = None,
) -> BenchmarkArtifacts:
    output_dir = ensure_output_dir(output_dir / "separable_sets")
    cache_dir = ensure_output_dir(output_dir / "cache")
    artifacts = artifacts or prepare_model_benchmarks(cache_dir=cache_dir)

    corr_matrix = artifacts.X_full.corr().abs()
    linkage = build_linkage(corr_matrix)

    heatmap_path = output_dir / "figure_06_correlation_heatmaps.png"
    plot_correlation_heatmaps(artifacts, corr_matrix, linkage, heatmap_path)

    trend_path = output_dir / "appendix_b5_kendall_tau_vs_threshold.png"
    plot_kendall_tau_trend(artifacts, linkage, trend_path)

    method_df = compute_method_importances(cache_dir)
    method_heatmap_path = output_dir / "figure_06_methods_heatmaps.png"
    plot_method_heatmaps(method_df, linkage, artifacts.feature_names, method_heatmap_path)

    method_trend_path = output_dir / "appendix_b5_methods_kendall.png"
    plot_method_kendall_trend(method_df, linkage, method_trend_path)

    return artifacts


def build_linkage(corr_matrix: pd.DataFrame):
    distance_matrix = 1 - corr_matrix
    condensed = squareform(distance_matrix.values)
    return sch.linkage(condensed, method="average")


def cluster_features(
    linkage_matrix,
    threshold: float,
    feature_names: Sequence[str],
) -> Dict[int, List[str]]:
    labels = sch.fcluster(linkage_matrix, t=threshold or 1e-6, criterion="distance")
    groups: Dict[int, List[str]] = defaultdict(list)
    for feature, label in zip(feature_names, labels):
        groups[label].append(feature)
    return groups


def compute_group_importances(
    importance_df: pd.DataFrame,
    groups: Dict[int, List[str]],
    columns: Sequence[str],
) -> pd.DataFrame:
    group_data = {"Group": list(groups.keys())}
    df = importance_df.set_index("Feature")

    for column in columns:
        values = []
        for features in groups.values():
            score = df.loc[features, column].sum()
            values.append(score)
        group_data[column] = values

    return pd.DataFrame(group_data)


def compute_method_importances(cache_dir: Path) -> pd.DataFrame:
    cache_path = cache_dir / "method_importances.pkl"
    return load_or_build(cache_path, lambda: _build_method_importances(cache_dir))


def _build_method_importances(cache_dir: Path) -> pd.DataFrame:
    spec = DatasetSpec(name="Breast Cancer", loader=load_breast_cancer)
    X, y = load_dataset(spec)

    def mci_builder():
        return compute_mci_vs_shap(
            spec,
            lambda: RandomForestClassifier(n_estimators=500, random_state=42),
            n_permutations=2**5,
        )

    mci_cache = cache_dir / "breast_mci.pkl"
    result = load_or_build(mci_cache, mci_builder)

    shap_values = normalize(result.shap_all_classes)
    mci_values = normalize(result.mci_values)
    ablation = normalize(compute_ablation_scores(X, y))
    bivariate = normalize(compute_bivariate_association(X, y))

    return pd.DataFrame(
        {
            "Feature": list(X.columns),
            "Expected_SHAP": shap_values,
            "MCI": mci_values,
            "Ablation": ablation,
            "Bivariate": bivariate,
        }
    )


def compute_ablation_scores(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> np.ndarray:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )
    base_model = RandomForestClassifier(n_estimators=500, random_state=random_state)
    base_model.fit(X_train, y_train)
    base_score = accuracy_score(y_test, base_model.predict(X_test))

    drops = []
    for feature in X.columns:
        model = RandomForestClassifier(n_estimators=500, random_state=random_state)
        model.fit(X_train.drop(columns=[feature]), y_train)
        score = accuracy_score(y_test, model.predict(X_test.drop(columns=[feature])))
        drops.append(max(base_score - score, 0))
    return np.array(drops)


def compute_bivariate_association(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    y_values = y.to_numpy()
    values = []
    for column in X.columns:
        feature = X[column].to_numpy()
        if np.std(feature) == 0:
            values.append(0.0)
            continue
        corr = np.corrcoef(feature, y_values)[0, 1]
        values.append(abs(corr))
    return np.array(values)


def plot_correlation_heatmaps(
    artifacts: BenchmarkArtifacts,
    corr_matrix: pd.DataFrame,
    linkage_matrix,
    output_path: Path,
    *,
    thresholds: Sequence[float] = (0.0, 0.15, 0.25, 0.5),
) -> None:
    rows, cols = 2, 2
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5.5))
    axes_flat = axes.flatten()

    for ax, threshold in zip(axes_flat, thresholds):
        groups = cluster_features(linkage_matrix, threshold, artifacts.feature_names)
        group_df = compute_group_importances(artifacts.importance_df, groups, MODEL_COLUMNS)
        corr = group_df.drop(columns="Group").corr(method="kendall")
        sns.heatmap(
            corr,
            annot=True,
            vmin=0,
            vmax=1,
            cmap="RdYlGn",
            ax=ax,
            cbar=False,
        )
        ax.set_title(
            f"Threshold {threshold:.2f}\nGroups: {len(groups)}",
            fontsize=12,
        )

    for ax in axes_flat[len(thresholds) :]:
        ax.axis("off")

    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(0, 1))
    fig.colorbar(sm, cax=cbar_ax, label="Kendall Tau")

    fig.suptitle("Kendall Tau correlations across separable-set thresholds", fontsize=16, weight="bold")
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_kendall_tau_trend(
    artifacts: BenchmarkArtifacts,
    linkage_matrix,
    output_path: Path,
    *,
    thresholds: Iterable[float] = np.linspace(0.05, 1.0, 15),
) -> None:
    tau_records = {f"{a} vs {b}": [] for a, b in MODEL_PAIRS}
    thresholds = list(thresholds)
    single_cluster_threshold = None

    for threshold in thresholds:
        groups = cluster_features(linkage_matrix, threshold, artifacts.feature_names)
        group_df = compute_group_importances(artifacts.importance_df, groups, MODEL_COLUMNS)
        standardized = group_df.drop(columns="Group").apply(lambda col: col / col.mean())
        for a, b in MODEL_PAIRS:
            tau, _ = kendalltau(standardized[a], standardized[b])
            tau_records[f"{a} vs {b}"].append(tau)
        if single_cluster_threshold is None and len(groups) == 1:
            single_cluster_threshold = threshold

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("husl", len(tau_records))
    for (pair, values), color in zip(tau_records.items(), palette):
        ax.plot(thresholds, values, marker="o", color=color, label=pair)

    ax.set_xlabel("Correlation Threshold")
    ax.set_ylabel("Kendall Tau")
    ax.set_title("Kendall Tau correlation vs. correlation threshold")
    ax.grid(True, linestyle="--", alpha=0.6)
    if single_cluster_threshold is not None:
        ax.axvline(single_cluster_threshold, color="black", linestyle="--", linewidth=1.2, label=f"Single cluster at {single_cluster_threshold:.2f}")
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_method_heatmaps(
    method_df: pd.DataFrame,
    linkage_matrix,
    feature_names: Sequence[str],
    output_path: Path,
    *,
    thresholds: Sequence[float] = (0.0, 0.2, 0.4, 0.6),
) -> None:
    rows, cols = 2, 2
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5.5))
    axes_flat = axes.flatten()

    for ax, threshold in zip(axes_flat, thresholds):
        groups = cluster_features(linkage_matrix, threshold, feature_names)
        group_df = compute_group_importances(method_df, groups, METHOD_COLUMNS)
        corr = group_df.drop(columns="Group").corr(method="kendall")
        sns.heatmap(
            corr,
            annot=True,
            vmin=0,
            vmax=1,
            cmap="RdYlGn",
            ax=ax,
            cbar=False,
        )
        ax.set_title(f"Threshold {threshold:.2f}\nGroups: {len(groups)}", fontsize=12)

    for ax in axes_flat[len(thresholds) :]:
        ax.axis("off")

    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(0, 1))
    fig.colorbar(sm, cax=cbar_ax, label="Kendall Tau")

    fig.suptitle("Method-level agreement across separable-set thresholds", fontsize=16, weight="bold")
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_method_kendall_trend(
    method_df: pd.DataFrame,
    linkage_matrix,
    output_path: Path,
    *,
    thresholds: Iterable[float] = (0.0, 0.2, 0.4, 0.6),
) -> None:
    tau_records = {f"{a} vs {b}": [] for a, b in METHOD_PAIRS}
    thresholds = list(thresholds)
    single_cluster_threshold = None

    for threshold in thresholds:
        groups = cluster_features(linkage_matrix, threshold, method_df["Feature"])
        group_df = compute_group_importances(method_df, groups, METHOD_COLUMNS)
        standardized = group_df.drop(columns="Group").apply(lambda col: col / col.mean())
        for a, b in METHOD_PAIRS:
            tau, _ = kendalltau(standardized[a], standardized[b])
            tau_records[f"{a} vs {b}"].append(tau)
        if single_cluster_threshold is None and len(groups) == 1:
            single_cluster_threshold = threshold

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("coolwarm", len(tau_records))
    for (pair, values), color in zip(tau_records.items(), palette):
        ax.plot(thresholds, values, marker="o", color=color, label=pair)

    ax.set_xlabel("Correlation Threshold")
    ax.set_ylabel("Kendall Tau")
    ax.set_title("Method agreement vs. threshold", fontsize=16)
    ax.grid(True, linestyle="--", alpha=0.6)
    if single_cluster_threshold is not None:
        ax.axvline(single_cluster_threshold, color="black", linestyle="--", linewidth=1.2, label=f"Single cluster at {single_cluster_threshold:.2f}")
    ax.legend(loc="lower right", ncol=2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
