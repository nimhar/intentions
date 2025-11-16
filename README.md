# Intentions Figure Reproduction Kit

This repository packages the exact code that was used to produce the figures that appear in the Intentions paper.  
All figures can be regenerated from scratch with a single command, and each script is documented and modular so that
individual experiments can be inspected or extended.

The Marginal Contribution Importance (MCI) implementation under `mci/` is vendored from the
[TAU-MLwell/Marginal-Contribution-Feature-Importance](https://github.com/TAU-MLwell/Marginal-Contribution-Feature-Importance)
project to guarantee identical behaviour.

## Reproducing the figures

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Generate every figure (main paper + appendix)
python main.py --include-appendix-grid

# Generate only a subset (e.g., Figures 2 and 5)
python main.py --figures figure2 figure5
```

Artifacts are written to `outputs/`:

| Figure | Command | Output file |
| --- | --- | --- |
| Figure 2 – Global Feature Importance (MCI vs. SHAP) | `python mci_vs_shap.py` or `python main.py --figures figure2` | `outputs/mci_vs_shap/*_dual_bars.png` (+ spaghetti + combined plot + tau json) |
| Figure 5 – SHAP ranking disagreement (RF vs. XGBoost) | `python twomodels.py` or `python main.py --figures figure5` | `outputs/model_consistency/figure_05_shap_ranking_rf_vs_xgb.png` (+ Appendix B1/B4 panels) |
| Figure 6 – Separable Sets approximation | `python twomodels_grouped.py` or `python main.py --figures figure6` | `outputs/separable_sets/figure_06_correlation_heatmaps.png` (+ Appendix B5 trend) |
| Appendix B2/B3 grid | `python mcishap.py` or `python main.py --include-appendix-grid` | `outputs/mci_vs_shap/appendix_grid.png` |

The appendix figures B1, B4, and B5 are generated together with Figures 5 and 6 to keep the experimental
configuration in sync. JSON files containing Kendall's Tau correlations are saved alongside the figures for
traceability.

## Repository layout

```
intentions/
├── figures/                     # Reusable modules for every figure
│   ├── common.py                # Dataset helpers + MCI/SHAP utilities
│   ├── figure02_mci_vs_shap.py  # Figure 2 + appendix B2/B3
│   ├── figure05_shap_rankings.py# Figure 5 + appendix B1/B4
│   ├── figure06_separable_sets.py# Figure 6 + appendix B5
│   └── model_benchmark.py       # Shared model training + importance summaries
├── mci/                         # MCI reference implementation
├── main.py                      # CLI dispatcher for figure generation
├── mci_vs_shap.py               # Shortcut for Figure 2 assets
├── mcishap.py                   # Appendix grid (B2 + B3 + digits)
├── twomodels.py                 # Shortcut for Figures 5/B1/B4
├── twomodels_grouped.py         # Shortcut for Figures 6/B5
├── requirements.txt             # Exact dependencies
└── outputs/                     # Created by the scripts (contains *.png + *.json)
```

## Implementation notes

- **Deterministic runs** – All estimators use fixed `random_state` arguments. The MCI estimator runs
  `2**5` permutations by default; adjust via the `n_permutations` argument inside `figures/figure02_mci_vs_shap.py`.
- **SHAP summaries** – Tree-based models rely on `shap.TreeExplainer`, while the SVM uses
  `shap.KernelExplainer` with a small evaluation subset to keep runtime manageable.
- **Separable sets** – Figure 6 and Appendix B5 reuse the exact model importances from the benchmark run
  to ensure that the clustering study uses the same feature information as the model disagreement study.
- **Outputs** – Each module writes descriptive filenames (for example,
  `figure_05_shap_ranking_rf_vs_xgb.png`). Delete `outputs/` to start over.

## Extending the experiments

The figure modules are regular Python packages, so you can import and reuse them directly, e.g.:

```python
from pathlib import Path
from figures.common import DatasetSpec
from sklearn.datasets import load_diabetes
from figures.figure02_mci_vs_shap import run_mci_vs_shap_experiments

specs = (DatasetSpec("Diabetes", load_diabetes, task_type="regression"),)
results = run_mci_vs_shap_experiments(Path("outputs"), dataset_specs=specs)
```

Each module returns structured results (for example `ImportanceResult` or `BenchmarkArtifacts`)
so downstream analyses can be scripted inside notebooks or other tooling.

If you add new experiments, please keep the deterministic seeds in place and document the additional
commands inside this README for future readers.

## License

The code inherits the original license of the TAU-MLwell MCI repository. Please ensure that any derivative
works cite both the Intentions paper and the original MCI work.
