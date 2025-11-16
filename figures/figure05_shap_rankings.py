from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import ConnectionPatch, Rectangle
from scipy.stats import kendalltau

from .common import ensure_output_dir, format_feature_labels
from .model_benchmark import BenchmarkArtifacts, prepare_model_benchmarks


def run_model_consistency_study(output_dir: Path) -> BenchmarkArtifacts:
    output_dir = ensure_output_dir(output_dir / "model_consistency")
    cache_dir = ensure_output_dir(output_dir / "cache")
    artifacts = prepare_model_benchmarks(cache_dir=cache_dir)

    plot_shap_ranking_bump_chart(
        artifacts,
        "Random_Forest_SHAP",
        "XGBoost_SHAP",
        output_dir / "figure_05_shap_ranking_rf_vs_xgb.png",
    )
    plot_model_bar_comparison(artifacts, output_dir / "appendix_b1_model_importances.png")
    plot_pairwise_spaghetti(
        artifacts,
        [
            ("Random_Forest_SHAP", "XGBoost_SHAP"),
            ("Random_Forest_SHAP", "SVM"),
            ("Tree_SHAP", "Random_Forest_SHAP"),
            ("Tree_SHAP", "XGBoost_SHAP"),
        ],
        output_dir / "appendix_b4_pairwise_spaghetti.png",
    )

    return artifacts


def plot_shap_ranking_bump_chart(
    artifacts: BenchmarkArtifacts,
    column_left: str,
    column_right: str,
    output_path: Path,
    *,
    top_k: int = 20,
) -> None:
    df = artifacts.importance_df[["Feature", column_left, column_right]].copy()
    df = df.sort_values(column_left, ascending=False).head(top_k)

    df_left = df.sort_values(column_left, ascending=True).set_index("Feature")
    df_right = df.sort_values(column_right, ascending=True).set_index("Feature")
    y_left = np.arange(len(df_left))[::-1]
    y_right = np.arange(len(df_right))[::-1]
    left_y = {feat: y for feat, y in zip(df_left.index, y_left)}
    right_y = {feat: y for feat, y in zip(df_right.index, y_right)}

    tau, p_value = kendall_tau(df[column_left], df[column_right])

    fig, ax = plt.subplots(figsize=(8.5, 9))
    fig.subplots_adjust(top=0.75, bottom=0.18)
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(-0.5, len(df_left) - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)

    shade = Rectangle(
        (-0.18, -0.5),
        1.36,
        len(df_left),
        facecolor="#f8f8f8",
        edgecolor="#d4d4d4",
        linewidth=1.2,
        zorder=0,
    )
    ax.add_patch(shade)

    left_labels = [f"#{idx+1:<2}  {label}" for idx, label in enumerate(format_feature_labels(df_left.index))]
    right_labels = [f"{label}  #{idx+1:<2}" for idx, label in enumerate(format_feature_labels(df_right.index))]

    for idx, label in enumerate(left_labels):
        ax.text(
            -0.22,
            y_left[idx],
            label,
            ha="right",
            va="center",
            fontsize=12,
            fontweight="medium",
            color="#184d47",
        )
    for idx, label in enumerate(right_labels):
        ax.text(
            1.22,
            y_right[idx],
            label,
            ha="left",
            va="center",
            fontsize=12,
            fontweight="medium",
            color="#7b112c",
        )

    ax.scatter(
        np.zeros(len(df_left)),
        y_left,
        color="#0f5132",
        s=150,
        marker="o",
        edgecolors="white",
        linewidth=1.5,
        zorder=3,
    )
    ax.scatter(
        np.ones(len(df_right)),
        y_right,
        color="#a3203a",
        s=150,
        marker="s",
        edgecolors="white",
        linewidth=1.5,
        zorder=3,
    )

    cmap = plt.cm.plasma
    for feature in df["Feature"]:
        if feature not in left_y or feature not in right_y:
            continue
        diff = abs(left_y[feature] - right_y[feature])
        norm_diff = diff / max(1, len(df_left) - 1)
        color = cmap(0.2 + 0.6 * norm_diff)
        linewidth = 1.5 + 4 * norm_diff
        ax.plot(
            [0, 1],
            [left_y[feature], right_y[feature]],
            color=color,
            linewidth=linewidth,
            alpha=0.95,
            zorder=2,
        )

    fig.suptitle("SHAP Ranking Comparison", fontsize=18, weight="bold", color="#0b1a33", y=0.9)
    fig.text(
        0.5,
        0.81,
        f"{friendly_label(column_left, artifacts.accuracy)} vs. {friendly_label(column_right, artifacts.accuracy)}",
        ha="center",
        fontsize=12,
        color="#0b1a33",
    )
    fig.text(
        0.5,
        0.775,
        f"Kendall's Tau = {tau:.3f}, p-value = {p_value:.2g}",
        ha="center",
        fontsize=11,
        color="#4a4a4a",
    )
    fig.text(
        0.5,
        0.05,
        "Connectors mirror the main SHAP vs. MCI comparison: thicker, darker strokes highlight ranking disagreements.",
        ha="center",
        fontsize=10,
        color="#4a4a4a",
    )

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_model_bar_comparison(
    artifacts: BenchmarkArtifacts,
    output_path: Path,
    *,
    top_k: int = 10,
) -> None:
    df = artifacts.importance_df.copy()
    value_columns = [col for col in df.columns if col != "Feature"]
    df["variance"] = df[value_columns].var(axis=1)
    subset = df.nlargest(top_k, "variance")
    features = format_feature_labels(subset["Feature"])
    index = np.arange(len(subset))
    bar_width = 0.12

    fig, ax = plt.subplots(figsize=(18, 9))
    ax.set_facecolor("#f9f9f9")
    fig.patch.set_facecolor("white")
    palette = {
        "Tree": "#1f77b4",
        "Tree_SHAP": "#0e3b6f",
        "Random_Forest": "#2ca02c",
        "Random_Forest_SHAP": "#196619",
        "SVM": "#d62728",
        "XGBoost": "#9467bd",
        "XGBoost_SHAP": "#5b2c6f",
    }

    for idx, column in enumerate(value_columns):
        color = palette.get(column, "#7f7f7f")
        ax.bar(
            index + idx * bar_width,
            subset[column],
            bar_width,
            label=friendly_label(column, artifacts.accuracy),
            color=color,
            edgecolor="#333333",
            linewidth=0.6,
        )

    ax.set_xticks(index + (len(value_columns) - 1) * bar_width / 2)
    ax.set_xticklabels(features, rotation=45, ha="right", fontsize=11)
    ax.set_ylabel("Normalized importance", fontsize=12)
    ax.set_title("Feature Importance Comparison Across Models", fontsize=16, weight="bold")
    legend = ax.legend(loc="upper right", ncol=2, frameon=True, fontsize=11)
    legend.get_frame().set_alpha(0.95)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pairwise_spaghetti(
    artifacts: BenchmarkArtifacts,
    pairs: Sequence[Tuple[str, str]],
    output_path: Path,
    *,
    top_k: int = 15,
) -> None:
    df = artifacts.importance_df.copy()
    rows, cols = 2, 2
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.5, rows * 6.5))
    axes_flat = axes.flatten()

    for ax, (col_left, col_right) in zip(axes_flat, pairs):
        subset = df[["Feature", col_left, col_right]].copy()
        subset = subset.sort_values(col_left, ascending=False).head(top_k)
        df_left = subset.sort_values(col_left, ascending=True).set_index("Feature")
        df_right = subset.sort_values(col_right, ascending=True).set_index("Feature")
        y_left = np.arange(len(df_left))[::-1]
        y_right = np.arange(len(df_right))[::-1]
        left_y = {feat: y for feat, y in zip(df_left.index, y_left)}
        right_y = {feat: y for feat, y in zip(df_right.index, y_right)}

        tau, p_value = kendall_tau(subset[col_left], subset[col_right])

        ax.set_xlim(-0.35, 1.35)
        ax.set_ylim(-0.5, len(df_left) - 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_visible(False)

        shade = Rectangle(
            (-0.18, -0.5),
            1.36,
            len(df_left),
            facecolor="#f9f9f9",
            edgecolor="#e0e0e0",
            linewidth=1,
            zorder=0,
        )
        ax.add_patch(shade)

        left_labels = [f"#{idx+1:<2}  {label}" for idx, label in enumerate(format_feature_labels(df_left.index))]
        right_labels = [f"{label}  #{idx+1:<2}" for idx, label in enumerate(format_feature_labels(df_right.index))]

        for idx, label in enumerate(left_labels):
            ax.text(-0.22, y_left[idx], label, ha="right", va="center", fontsize=10, color="#084c61")
        for idx, label in enumerate(right_labels):
            ax.text(1.22, y_right[idx], label, ha="left", va="center", fontsize=10, color="#7b112c")

        ax.scatter(np.zeros(len(df_left)), y_left, color="#0f6890", s=120, edgecolors="white", linewidth=1.2)
        ax.scatter(np.ones(len(df_right)), y_right, color="#a2132f", s=120, edgecolors="white", linewidth=1.2)

        cmap = plt.cm.inferno
        for feature in subset["Feature"]:
            if feature not in left_y or feature not in right_y:
                continue
            diff = abs(left_y[feature] - right_y[feature])
            norm_diff = diff / max(1, len(df_left) - 1)
            ax.plot(
                [0, 1],
                [left_y[feature], right_y[feature]],
                color=cmap(0.2 + 0.6 * norm_diff),
                linewidth=1.2 + 3 * norm_diff,
                alpha=0.9,
            )

        ax.set_title(
            f"{friendly_label(col_left)} vs.\n{friendly_label(col_right)}\nTau={tau:.3f}, p={p_value:.2g}",
            fontsize=11,
        )

    for ax in axes_flat[len(pairs) :]:
        ax.axis("off")

    fig.suptitle("Comparative Analysis of SHAP Rankings Across Models", fontsize=18, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def kendall_tau(values_a: Iterable[float], values_b: Iterable[float]) -> Tuple[float, float]:
    series_a = pd.Series(values_a).rank(ascending=False)
    series_b = pd.Series(values_b).rank(ascending=False)
    return kendalltau(series_a, series_b)


def friendly_label(column: str, accuracy: dict | None = None) -> str:
    label = column.replace("_SHAP", "").replace("_", " ")
    base = column.replace("_SHAP", "")
    if accuracy and base in accuracy:
        train, test = accuracy[base]
        label = f"{label}\n(Train={train:.2f}, Test={test:.2f})"
    return label
