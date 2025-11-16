from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from figures import FIGURE_BUILDERS, build_all_figures, build_appendix_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the figures that accompany the Intentions Paper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--figures",
        nargs="*",
        choices=sorted(FIGURE_BUILDERS.keys()),
        help="Subset of figures to generate. Defaults to all.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where artifacts will be saved.",
    )
    parser.add_argument(
        "--include-appendix-grid",
        action="store_true",
        help="Also generate the Appendix (B2–B3) SHAP vs. MCI grid.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    sanitize_matplotlib_stylelib()
    build_all_figures(output_dir, figure_names=args.figures)

    if args.include_appendix_grid:
        build_appendix_grid(output_dir)


def sanitize_matplotlib_stylelib() -> None:
    """
    Some macOS setups create hidden resource-fork files (prefixed with ._)
    inside site-packages. Matplotlib tries to parse every *.mplstyle file,
    so these binary blobs cause decode errors. We proactively delete them.
    """
    spec = importlib.util.find_spec("matplotlib")
    if not spec or not spec.submodule_search_locations:
        return

    try:
        base = Path(spec.submodule_search_locations[0])
        stylelib = base / "mpl-data" / "stylelib"
        if not stylelib.exists():
            return
        for stray in stylelib.glob("._*.mplstyle"):
            try:
                stray.unlink()
            except OSError:
                pass
    except Exception:
        # Silently ignore – this is a best-effort cleanup.
        return


if __name__ == "__main__":
    main()
