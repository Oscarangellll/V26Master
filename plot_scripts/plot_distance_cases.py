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

from plot_scripts.config import FIGWIDTH, PLOT_DIR, colors
from plot_scripts.plot_case_studies import (
    VESSEL_MIX_CMAP,
    _display_coalition,
    _display_member,
    _draw_barycentric,
    _load_coalition_metrics,
    _members,
)


CASE_ORDER = ["BCD_close", "BCG_cluster_far", "BEG_spread"]
CASE_COALITIONS = {
    "BCD_close": "BCD",
    "BCG_cluster_far": "BCG",
    "BEG_spread": "BEG",
}
CASE_LABELS = {
    "BCD_close": "Close",
    "BCG_cluster_far": "Cluster-far",
    "BEG_spread": "Spread",
}
WINDFARM_COLORS = {
    "B": colors.blue,
    "C": colors.red,
    "D": colors.green,
    "E": colors.orange,
    "G": colors.purple,
}


def _load_distance_case_metrics(input_root):
    rows = []
    case_games = {}
    for case_name in CASE_ORDER:
        coalition_path = Path(input_root) / case_name / "coalition_oos.csv"
        if not coalition_path.exists():
            raise FileNotFoundError(f"Missing distance coalition OOS file: {coalition_path}")

        df, cost, savings = _load_coalition_metrics(coalition_path)
        coalition = CASE_COALITIONS[case_name]
        row = df[df["coalition"] == coalition]
        if row.empty:
            raise ValueError(f"No grand-coalition row {coalition} found in {coalition_path}")

        record = row.iloc[0].to_dict()
        record["case"] = case_name
        record["case_label"] = CASE_LABELS[case_name]
        rows.append(record)
        case_games[case_name] = (cost, savings)

    return pd.DataFrame(rows), case_games


def _stable_core_legend(ax):
    ax.scatter([], [], marker="o", facecolor="white", edgecolor="0.20", s=18, label="Stable core")
    ax.scatter([], [], marker="X", facecolor="white", edgecolor="0.20", s=22, label="No stable core")


def _plot_synergy_panels(df, panels, output):
    fig, axs = plt.subplots(
        1,
        2,
        figsize=(FIGWIDTH / 2.54, 5.3 / 2.54),
        constrained_layout=False,
    )

    scatter_for_colorbar = None
    for ax, (x_col, xlabel) in zip(axs, panels):
        for _, row in df.iterrows():
            has_core = bool(row["has_stable_core"])
            marker = "o" if has_core else "X"
            scatter = ax.scatter(
                row[x_col],
                row["synergy"] * 100,
                c=[row["SOV_share"]],
                cmap=VESSEL_MIX_CMAP,
                vmin=0,
                vmax=1,
                marker=marker,
                alpha=0.82,
                s=24 if has_core else 28,
                edgecolor="0.20",
                linewidth=0.25,
                zorder=3,
            )
            scatter_for_colorbar = scatter
            ax.annotate(
                _display_coalition(row["coalition"]),
                (row[x_col], row["synergy"] * 100),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )

        ax.set_xlabel(xlabel)
        ax.grid(color="0.90", linewidth=0.6)

    axs[0].set_ylabel("")
    axs[1].set_ylabel("")
    fig.supylabel("Synergy [%]", x=0.03, fontsize=8)

    _stable_core_legend(axs[0])
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.42, 0.99),
    )

    if scatter_for_colorbar is not None:
        cbar_ax = fig.add_axes([0.895, 0.25, 0.018, 0.46])
        cbar = fig.colorbar(scatter_for_colorbar, cax=cbar_ax)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(["CTV", "Mixed", "SOV"])

    fig.subplots_adjust(top=0.80, bottom=0.20, left=0.09, right=0.855, wspace=0.35)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(f"Wrote {output}")


def _plot_barycentric(input_root, case_games, output):
    fig, axs = plt.subplots(
        1,
        len(CASE_ORDER),
        figsize=(FIGWIDTH / 2.54, 5.3 / 2.54),
        constrained_layout=False,
    )
    if len(CASE_ORDER) == 1:
        axs = [axs]

    for ax, case_name in zip(axs, CASE_ORDER):
        coalition = CASE_COALITIONS[case_name]
        cost, savings = case_games[case_name]
        title = f"{CASE_LABELS[case_name]}\n({_display_coalition(coalition)})"
        _draw_barycentric(ax, _members(coalition), savings, cost, title)

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
        fig.subplots_adjust(top=0.75, wspace=0.24)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote {output}")


def _load_distance_windfarm_results(input_root):
    frames = []
    for case_name in CASE_ORDER:
        path = Path(input_root) / case_name / "windfarm_oos.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing distance windfarm OOS file: {path}")
        df = pd.read_csv(path)
        df["case"] = case_name
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def _plot_operational_fairness(input_root, output):
    df = _load_distance_windfarm_results(input_root)
    selected = []
    for case_name in CASE_ORDER:
        coalition = CASE_COALITIONS[case_name]
        part = df[(df["case"] == case_name) & (df["coalition"].astype(str) == coalition)].copy()
        part["case_label"] = CASE_LABELS[case_name]
        selected.append(part)
    selected = pd.concat(selected, ignore_index=True)
    if selected.empty:
        raise ValueError("No grand-coalition windfarm rows found for distance cases.")

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
        metric_df = selected[["case", "wind_farm", metric]].copy()
        metric_df[metric] = pd.to_numeric(metric_df[metric], errors="coerce") * 100
        metric_df = metric_df.dropna(subset=[metric])

        for x_idx, case_name in enumerate(CASE_ORDER):
            group = metric_df[metric_df["case"] == case_name].sort_values("wind_farm")
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
        ax.set_xticklabels(
            [f"{CASE_LABELS[c]}\n({_display_coalition(CASE_COALITIONS[c])})" for c in CASE_ORDER]
        )
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

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(f"Wrote {output}")


def plot_distance_cases(
    input_root="results/case_studies/distance",
    output_dir=None,
):
    output_dir = Path(output_dir) if output_dir is not None else Path(PLOT_DIR) / "3 WFs"
    df, case_games = _load_distance_case_metrics(input_root)

    # The distance-vs-synergy plots are based on the base case with all
    # three-wind-farm coalitions. The dedicated distance experiments are only
    # used for allocation and operational fairness diagnostics.
    _plot_barycentric(input_root, case_games, output_dir / "case_barycentric_allocations.svg")
    _plot_operational_fairness(input_root, output_dir / "case_operational_fairness_selected_triads.svg")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot uniform-100 distance case-study results."
    )
    parser.add_argument("--input-root", default="results/case_studies/distance")
    parser.add_argument(
        "--output-dir",
        default=str(Path(PLOT_DIR) / "3 WFs"),
        help="Directory where distance SVG plots are written.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    plot_distance_cases(args.input_root, args.output_dir)


if __name__ == "__main__":
    main()
