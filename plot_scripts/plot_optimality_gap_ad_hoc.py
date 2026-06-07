import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from plot_scripts.config import FIGWIDTH, PLOT_DIR, colors


CASES = ["2W2B", "3W2B"]
MIN_COMPLETE_REPLICATIONS = 19


def _read_gap(path: Path, column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[column] = pd.to_numeric(df[column], errors="coerce")
    out = (
        df.dropna(subset=[column])
        .groupby("tree_size", as_index=False)
        .agg(gap=(column, "mean"), n=(column, "size"))
        .sort_values("tree_size")
    )
    out["gap"] *= 100
    return out


def _plot_line(ax, df: pd.DataFrame, label: str, color: str, linestyle: str = "-") -> None:
    if df.empty:
        return

    complete = df[df["n"] >= MIN_COMPLETE_REPLICATIONS]
    incomplete = df[df["n"] < MIN_COMPLETE_REPLICATIONS]

    ax.plot(
        complete["tree_size"],
        complete["gap"],
        marker="o",
        linewidth=1.8,
        markersize=4,
        color=color,
        linestyle=linestyle,
        label=label,
    )
    if not incomplete.empty:
        ax.plot(
            incomplete["tree_size"],
            incomplete["gap"],
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=color,
            linestyle=":",
        )


def plot_optimality_gap_ad_hoc() -> None:
    root = Path("results") / "stability"
    fig, axes = plt.subplots(1, 2, figsize=(FIGWIDTH, 0.42 * FIGWIDTH), sharey=True)

    for ax, case in zip(axes, CASES):
        mip = _read_gap(root / case / "mip" / "ISS.csv", "MIPGap")
        _plot_line(ax, mip, "Direct MIP", colors.direct_mip, "-")

        ax.axhline(2, color="0.55", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_title(case)
        ax.set_xlabel("Number of scenarios")
        ax.set_ylim(0, 3)
        ax.grid(True, axis="y", alpha=0.25)

    axes[0].set_ylabel("Optimality gap [%]")
    axes[-1].legend(loc="upper left", frameon=False)
    fig.tight_layout()

    out = Path(PLOT_DIR) / "ISS_OSS" / "optimality_gap_2W2B_3W2B.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    plot_optimality_gap_ad_hoc()
