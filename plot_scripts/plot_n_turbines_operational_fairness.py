import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plot_scripts.config import PLOT_DIR, colors, FIGWIDTH
from plot_scripts.plot_case_studies import _display_member


CASE_ORDER = ["BCD_low", "BCD_mixed", "BCD_high"]
CASE_LABELS = {
    "BCD_low": "Low",
    "BCD_mixed": "Mixed",
    "BCD_high": "High",
}
WINDFARM_COLORS = {
    "B": colors.blue,
    "C": colors.red,
    "D": colors.green,
}


def _load_windfarm_results(root):
    frames = []
    missing = []
    for case in CASE_ORDER:
        path = Path(root) / case / "windfarm_oos.csv"
        if not path.exists():
            missing.append(path)
            continue

        df = pd.read_csv(path)
        df["case"] = case
        frames.append(df)

    if missing:
        formatted = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Missing n_turbines windfarm OOS files:\n" + formatted
        )
    if not frames:
        raise ValueError("No n_turbines windfarm OOS files were loaded.")

    return pd.concat(frames, ignore_index=True)


def plot_n_turbines_operational_fairness(
    input_root="results/case_studies/n_turbines",
    output=None,
):
    df = _load_windfarm_results(input_root)
    df = df[df["coalition"].astype(str) == "BCD"].copy()
    if df.empty:
        raise ValueError("No BCD rows found in n_turbines windfarm OOS results.")

    metrics = [
        ("value_based_availability", "Value-based availability [%]"),
        ("time_based_availability", "Time-based availability [%]"),
    ]

    fig, axs = plt.subplots(
        1,
        2,
        figsize=(FIGWIDTH / 2.54, 5.5 / 2.54),
        constrained_layout=False,
    )
    x_centers = np.arange(len(CASE_ORDER))
    bar_width = 0.18
    shared_legend_handles = {}

    for ax, (metric, ylabel) in zip(axs, metrics):
        metric_df = df[["case", "wind_farm", metric]].copy()
        metric_df[metric] = pd.to_numeric(metric_df[metric], errors="coerce") * 100
        metric_df = metric_df.dropna(subset=[metric])

        for x_idx, case in enumerate(CASE_ORDER):
            group = metric_df[metric_df["case"] == case].sort_values("wind_farm")
            offsets = (np.arange(len(group)) - (len(group) - 1) / 2) * bar_width
            for offset, row in zip(offsets, group.to_dict("records")):
                wind_farm = str(row["wind_farm"])
                bars = ax.bar(
                    x_centers[x_idx] + offset,
                    row[metric],
                    width=bar_width,
                    color=WINDFARM_COLORS.get(wind_farm, colors.blue),
                )
                shared_legend_handles.setdefault(wind_farm, bars[0])

        ax.set_xticks(x_centers)
        ax.set_xticklabels([CASE_LABELS[case] for case in CASE_ORDER])
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_ylim(95, 100)
        ax.grid(axis="y", color="0.90", linewidth=0.6)

    if shared_legend_handles:
        fig.legend(
            [shared_legend_handles[w] for w in sorted(shared_legend_handles)],
            [_display_member(w) for w in sorted(shared_legend_handles)],
            loc="upper center",
            ncol=len(shared_legend_handles),
            frameon=False,
            fontsize=7,
            bbox_to_anchor=(0.5, 0.95),
        )
        fig.subplots_adjust(top=0.76, wspace=0.28)

    output = (
        Path(output)
        if output is not None
        else Path(PLOT_DIR) / "3 WFs" / "n_turbines_operational_fairness.svg"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(f"Wrote {output}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot operational fairness for n_turbines BCD case variants."
    )
    parser.add_argument("--input-root", default="results/case_studies/n_turbines")
    parser.add_argument("--output", default=None)
    return parser


def main():
    args = build_parser().parse_args()
    plot_n_turbines_operational_fairness(args.input_root, args.output)


if __name__ == "__main__":
    main()
