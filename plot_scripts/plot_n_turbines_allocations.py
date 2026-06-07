import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plot_scripts.config import PLOT_DIR, FIGWIDTH
from plot_scripts.plot_case_studies import (
    _draw_barycentric,
    _load_coalition_metrics,
    _members,
)
from plot_scripts.plot_n_turbines_operational_fairness import (
    CASE_LABELS,
    CASE_ORDER,
    plot_n_turbines_operational_fairness,
)


def plot_n_turbines_barycentric_allocations(
    input_root="results/case_studies/n_turbines",
    output=None,
):
    output = (
        Path(output)
        if output is not None
        else Path(PLOT_DIR) / "3 WFs" / "n_turbines_barycentric_allocations.svg"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(
        1,
        len(CASE_ORDER),
        figsize=(FIGWIDTH / 2.54, 5.3 / 2.54),
        constrained_layout=False,
    )
    if len(CASE_ORDER) == 1:
        axs = [axs]

    for ax, case_name in zip(axs, CASE_ORDER):
        coalition_path = Path(input_root) / case_name / "coalition_oos.csv"
        if not coalition_path.exists():
            raise FileNotFoundError(f"Missing n_turbines coalition OOS file: {coalition_path}")

        _, cost, savings = _load_coalition_metrics(coalition_path)
        _draw_barycentric(ax, _members("BCD"), savings, cost, CASE_LABELS[case_name])

    handles_by_label = {}
    for ax in axs:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            handles_by_label.setdefault(label, handle)

    legend_order = ["Core", "Core centre", "Minmax allocation"]
    labels = [label for label in legend_order if label in handles_by_label]
    handles = [handles_by_label[label] for label in labels]
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.99),
        )
        fig.subplots_adjust(top=0.78, wspace=0.24)

    fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote {output}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot allocation and operational fairness for n_turbines cases."
    )
    parser.add_argument("--input-root", default="results/case_studies/n_turbines")
    parser.add_argument(
        "--output-dir",
        default=str(Path(PLOT_DIR) / "3 WFs"),
        help="Directory where n_turbines SVG plots are written.",
    )
    return parser


def plot_n_turbines_cases(
    input_root="results/case_studies/n_turbines",
    output_dir=None,
):
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(PLOT_DIR) / "3 WFs"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_n_turbines_operational_fairness(
        input_root=input_root,
        output=output_dir / "n_turbines_operational_fairness.svg",
    )
    plot_n_turbines_barycentric_allocations(
        input_root=input_root,
        output=output_dir / "n_turbines_barycentric_allocations.svg",
    )


def main():
    args = build_parser().parse_args()
    plot_n_turbines_cases(args.input_root, args.output_dir)


if __name__ == "__main__":
    main()
