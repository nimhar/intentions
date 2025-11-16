from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable
def _build_figure2(output: Path) -> None:
    from .figure02_mci_vs_shap import run_mci_vs_shap_experiments

    run_mci_vs_shap_experiments(output)


def _build_figure5(output: Path) -> None:
    from .figure05_shap_rankings import run_model_consistency_study

    run_model_consistency_study(output)


def _build_figure6(output: Path) -> None:
    from .figure06_separable_sets import run_separable_set_analysis

    run_separable_set_analysis(output)


FIGURE_BUILDERS: Dict[str, Callable[[Path], None]] = {
    "figure2": _build_figure2,
    "figure5": _build_figure5,
    "figure6": _build_figure6,
}


def build_all_figures(
    output_dir: Path,
    *,
    figure_names: Iterable[str] | None = None,
) -> None:
    selected = list(figure_names) if figure_names else list(FIGURE_BUILDERS.keys())
    for figure in selected:
        if figure not in FIGURE_BUILDERS:
            raise ValueError(f"Unknown figure '{figure}'. Valid options: {sorted(FIGURE_BUILDERS)}")
        FIGURE_BUILDERS[figure](output_dir)


def build_appendix_grid(output_dir: Path) -> None:
    from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine

    from .common import DatasetSpec
    from .figure02_mci_vs_shap import plot_global_grid, run_mci_vs_shap_experiments

    dataset_specs = (
        DatasetSpec("Breast Cancer", load_breast_cancer),
        DatasetSpec("Iris", load_iris),
        DatasetSpec("Wine", load_wine),
        DatasetSpec("Digits", load_digits),
    )
    results = run_mci_vs_shap_experiments(output_dir, dataset_specs=dataset_specs)
    plot_global_grid(results, Path(output_dir) / "mci_vs_shap" / "appendix_grid.png")


__all__ = [
    "FIGURE_BUILDERS",
    "build_all_figures",
    "build_appendix_grid",
]
