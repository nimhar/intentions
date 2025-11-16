from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import ConnectionPatch, Rectangle
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier

from .common import (
    DatasetSpec,
    ImportanceResult,
    compute_mci_vs_shap,
    ensure_output_dir,
    format_feature_labels,
    kendall_tau_from_values,
    slugify,
    top_features,
)
from .cache_utils import load_or_build

DEFAULT_DATASETS: Sequence[DatasetSpec] = (
    DatasetSpec(name="Breast Cancer", loader=load_breast_cancer),
    DatasetSpec(name="Iris", loader=load_iris),
    DatasetSpec(name="Wine", loader=load_wine),
    DatasetSpec(name="Digits", loader=load_digits),
)


def run_mci_vs_shap_experiments(
    output_dir: Path,
    dataset_specs: Sequence[DatasetSpec] = DEFAULT_DATASETS,
    *,
    n_permutations: int = 2**5,
) -> list[ImportanceResult]:
    output_dir = ensure_output_dir(output_dir / "mci_vs_shap")
    cache_dir = ensure_output_dir(output_dir / "cache")

    results: list[ImportanceResult] = []
    for spec in dataset_specs:
        cache_path = cache_dir / f"{slugify(spec.name)}_np{n_permutations}.pkl"

        def builder(spec=spec):
            return compute_mci_vs_shap(
                spec,
                lambda: RandomForestClassifier(n_estimators=500, random_state=42),
                n_permutations=n_permutations,
            )

        result = load_or_build(cache_path, builder)
        results.append(result)

        dataset_dir = ensure_output_dir(output_dir / slugify(spec.name))
        dual_path = dataset_dir / f"{slugify(spec.name)}_dual_bars.png"
        spaghetti_path = dataset_dir / f"{slugify(spec.name)}_spaghetti.png"

        plot_dual_importance_bars(result, dual_path)
        plot_spaghetti_comparison(result, spaghetti_path, tau_info=result.kendall_tau())

    combined_path = output_dir / "combined_top_features.png"
    plot_combined_rankings(results, combined_path)

    json_path = output_dir / "kendall_tau_results.json"
    write_tau_summary(results, json_path)

    return results


def plot_dual_importance_bars(result: ImportanceResult, output_path: Path, *, top_k: int = 8) -> None:
    df = result.to_frame()
    tau, _ = result.kendall_tau()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=False)
    plot_config = [
        ("MCI Value", "SHAP Value", axes[0], "Top features by MCI"),
        ("SHAP Value", "MCI Value", axes[1], "Top features by SHAP"),
    ]

    for primary, secondary, ax, title in plot_config:
        subset = top_features(df, primary, k=top_k)
        y_pos = np.arange(len(subset))
        ax.barh(y_pos, subset[primary], color="#c51b7d", alpha=0.75, label=primary)
        ax.barh(
            y_pos,
            subset[secondary],
            color="#2c7fb8",
            alpha=0.6,
            label=secondary,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(format_feature_labels(subset["Feature"]))
        ax.invert_yaxis()
        ax.set_xlabel("Importance Score")
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="x", linestyle="--", alpha=0.5)

    fig.suptitle(
        f"MCI vs. SHAP — {result.dataset_name} (Kendall Tau={tau:.2f})",
        fontsize=16,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_spaghetti_comparison(
    result: ImportanceResult,
    output_path: Path,
    *,
    max_features: int = 12,
    tau_info: tuple[float, float] | None = None,
) -> None:
    df = result.to_frame()[["Feature", "MCI Value", "SHAP Value"]].copy()
    half = max_features // 2
    top_by_mci = top_features(df, "MCI Value", k=half)["Feature"]
    top_by_shap = top_features(df, "SHAP Value", k=half)["Feature"]
    focus_features = set(top_by_mci).union(set(top_by_shap))
    df = df[df["Feature"].isin(focus_features)].copy().set_index("Feature")

    shap_rank = df["SHAP Value"].rank(ascending=False, method="min").astype(int)
    mci_rank = df["MCI Value"].rank(ascending=False, method="min").astype(int)
    df["SHAP Rank"] = shap_rank
    df["MCI Rank"] = mci_rank

    df_left = df.sort_values("SHAP Value", ascending=True)
    df_right = df.sort_values("MCI Value", ascending=True)

    n_rows = max(len(df_left), len(df_right))
    y_pos_left = np.arange(len(df_left))[::-1]
    y_pos_right = np.arange(len(df_right))[::-1]
    left_y = {feat: y for feat, y in zip(df_left.index, y_pos_left)}
    right_y = {feat: y for feat, y in zip(df_right.index, y_pos_right)}

    tau, p_value = tau_info if tau_info else kendall_tau_from_values(df["MCI Value"], df["SHAP Value"])

    fig, ax = plt.subplots(figsize=(8.5, 10))
    fig.subplots_adjust(top=0.8, bottom=0.18)
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)

    shade = Rectangle(
        (-0.18, -0.5),
        1.36,
        n_rows,
        facecolor="#f5f5f5",
        edgecolor="#d4d4d4",
        linewidth=1.2,
        zorder=0,
    )
    ax.add_patch(shade)

    left_labels = [f"#{df_left['SHAP Rank'][f]:<2}  {label}" for f, label in zip(df_left.index, format_feature_labels(df_left.index))]
    right_labels = [f"{label}  #{df_right['MCI Rank'][f]:<2}" for f, label in zip(df_right.index, format_feature_labels(df_right.index))]

    for idx, label in enumerate(left_labels):
        ax.text(
            -0.22,
            y_pos_left[idx],
            label,
            ha="right",
            va="center",
            fontsize=12,
            fontweight="medium",
            color="#003049",
        )

    for idx, label in enumerate(right_labels):
        ax.text(
            1.22,
            y_pos_right[idx],
            label,
            ha="left",
            va="center",
            fontsize=12,
            fontweight="medium",
            color="#5c1130",
        )

    ax.scatter(np.zeros(len(df_left)), y_pos_left, color="#1d4e89", s=160, marker="o", edgecolors="white", linewidth=1.5, zorder=3)
    ax.scatter(np.ones(len(df_right)), y_pos_right, color="#a0193c", s=160, marker="s", edgecolors="white", linewidth=1.5, zorder=3)

    cmap = plt.cm.magma
    for feature in df.index:
        if feature not in left_y or feature not in right_y:
            continue
        diff = abs(left_y[feature] - right_y[feature])
        norm_diff = diff / max(1, n_rows - 1)
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

    ax.text(
        -0.32,
        n_rows + 0.1,
        "Features ranked by SHAP",
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="#003049",
    )
    ax.text(
        1.32,
        n_rows + 0.1,
        "Features ranked by MCI",
        ha="right",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="#5c1130",
    )

    subtitle = f"Kendall's Tau = {tau:.3f}, p-value = {p_value:.2g}"
    fig.suptitle("Global Feature Importance Comparison", fontsize=18, weight="bold", y=0.98, color="#0b1a33")
    fig.text(
        0.5,
        0.93,
        f"{result.dataset_name} — Expected value of local SHAPs vs. Global MCI rankings",
        ha="center",
        fontsize=13,
        color="#0b1a33",
    )
    fig.text(
        0.5,
        0.89,
        subtitle,
        ha="center",
        fontsize=12,
        color="#4a4a4a",
    )
    fig.text(
        0.5,
        0.03,
        "Lines connect identical features under two explanatory intentions. Thicker, darker strokes indicate larger rank disagreements between expected SHAP values and global MCI importance.",
        ha="center",
        fontsize=11,
        color="#4a4a4a",
    )

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_combined_rankings(results: Iterable[ImportanceResult], output_path: Path, *, top_k: int = 8) -> None:
    results = list(results)
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 6))

    if len(results) == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        df = result.to_frame()
        subset = top_features(df, "SHAP Value", k=top_k)
        y_pos = np.arange(len(subset))
        ax.barh(y_pos, subset["MCI Value"], color="#c51b7d", alpha=0.75, label="MCI")
        ax.barh(
            y_pos,
            subset["SHAP All"],
            color="#2c7fb8",
            alpha=0.6,
            label="Expected value of local SHAPs",
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(format_feature_labels(subset["Feature"]))
        ax.invert_yaxis()
        ax.set_title(result.dataset_name)
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        if ax is axes[-1]:
            ax.legend()

    fig.suptitle("Top features: Expected SHAP values vs. MCI", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_tau_summary(results: Iterable[ImportanceResult], output_path: Path) -> None:
    summary = {}
    for result in results:
        tau, p_value = result.kendall_tau()
        summary[result.dataset_name] = {"tau": float(tau), "p_value": float(p_value)}

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def plot_global_grid(
    results: Sequence[ImportanceResult],
    output_path: Path,
    *,
    columns: int = 2,
) -> None:
    rows = int(np.ceil(len(results) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(7 * columns, 5 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, result in zip(axes_flat, results):
        df = result.to_frame()
        subset = df.sort_values("SHAP Value", ascending=False).head(10)
        y_pos = np.arange(len(subset))
        shap_vals = subset["SHAP Value"].to_numpy()
        mci_vals = subset["MCI Value"].to_numpy()
        ax.hlines(y_pos, shap_vals, mci_vals, color="#cccccc", linewidth=1.5, alpha=0.8)
        ax.scatter(shap_vals, y_pos, color="#1d4e89", marker="o", s=60, label="Expected SHAP")
        ax.scatter(mci_vals, y_pos, color="#c3512f", marker="s", s=60, label="MCI")
        ax.set_title(result.dataset_name, fontsize=13, weight="bold")
        ax.set_xlabel("Importance score")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(format_feature_labels(subset["Feature"]))
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        tau, p_value = result.kendall_tau()
        ax.text(
            0.02,
            0.02,
            f"Tau={tau:.2f}\np={p_value:.2g}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.7, "boxstyle": "round,pad=0.2"},
        )

    for ax in axes_flat[len(results) :]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=True)
    fig.suptitle("Global feature importance comparison", fontsize=18, weight="bold")
    fig.tight_layout(rect=[0, 0, 0.96, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
