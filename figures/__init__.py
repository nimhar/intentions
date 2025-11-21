from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable

def _build_figure3a(output: Path) -> None:
    from .figure02_mci_vs_shap import run_mci_vs_shap_experiments

    run_mci_vs_shap_experiments(output)


def _build_figure3b(output: Path) -> None:
    from .figure05_shap_rankings import run_model_consistency_study

    run_model_consistency_study(output)


def _build_figure6(output: Path) -> None:
    from .figure06_separable_sets import run_separable_set_analysis

    run_separable_set_analysis(output)


FIGURE_BUILDERS: Dict[str, Callable[[Path], None]] = {
    # Primary keys (Figure 3 subpanels in the manuscript)
    "figure3a": _build_figure3a,
    "figure3b": _build_figure3b,
    "figure6": _build_figure6,
    # Legacy aliases kept for backward compatibility
    "figure2": _build_figure3a,
    "figure5": _build_figure3b,
}

DEFAULT_FIGURE_ORDER = ("figure3a", "figure3b", "figure6")


def build_all_figures(
    output_dir: Path,
    *,
    figure_names: Iterable[str] | None = None,
) -> None:
    selected = list(figure_names) if figure_names else list(DEFAULT_FIGURE_ORDER)
    for figure in selected:
        if figure not in FIGURE_BUILDERS:
            raise ValueError(f"Unknown figure '{figure}'. Valid options: {sorted(FIGURE_BUILDERS)}")
        FIGURE_BUILDERS[figure](output_dir)


def build_appendix_grid(output_dir: Path) -> None:
    from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine

    from .common import DatasetSpec
    from .datasets import load_penguins_dataset, load_titanic_dataset
    from .figure02_mci_vs_shap import plot_global_grid, run_mci_vs_shap_experiments

    dataset_specs = (
        DatasetSpec("Breast Cancer", load_breast_cancer),
        DatasetSpec("Iris", load_iris),
        DatasetSpec("Wine", load_wine),
        DatasetSpec("Digits", load_digits),
        DatasetSpec("Titanic", load_titanic_dataset),
        DatasetSpec("Penguins", load_penguins_dataset),
    )
    results = run_mci_vs_shap_experiments(output_dir, dataset_specs=dataset_specs)
    plot_global_grid(results, Path(output_dir) / "mci_vs_shap" / "appendix_grid.png")


__all__ = [
    "FIGURE_BUILDERS",
    "build_all_figures",
    "build_appendix_grid",
]
