from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler


@dataclass(frozen=True)
class PlotStyle:
    figure_size: tuple[float, float] = (8.0, 4.8)
    figure_size_wide: tuple[float, float] = (12.0, 5.2)
    dpi: int = 200
    font_family: tuple[str, ...] = ("DejaVu Sans",)
    font_size: int = 10
    title_size: int = 12
    label_size: int = 11
    tick_size: int = 9
    legend_size: int = 9
    line_width: float = 2.0
    marker_size: float = 5.5
    grid_alpha: float = 0.25
    palette: tuple[str, ...] = (
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#17becf",
    )

    def apply(self) -> None:
        mpl.rcParams.update(
            {
                "figure.figsize": self.figure_size,
                "figure.dpi": self.dpi,
                "savefig.dpi": self.dpi,
                "font.family": list(self.font_family),
                "font.size": self.font_size,
                "axes.titlesize": self.title_size,
                "axes.labelsize": self.label_size,
                "xtick.labelsize": self.tick_size,
                "ytick.labelsize": self.tick_size,
                "legend.fontsize": self.legend_size,
                "lines.linewidth": self.line_width,
                "lines.markersize": self.marker_size,
                "axes.grid": True,
                "grid.alpha": self.grid_alpha,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.prop_cycle": cycler(color=list(self.palette)),
            }
        )


DEFAULT_STYLE = PlotStyle()


def apply_default_style() -> None:
    DEFAULT_STYLE.apply()


def should_save(action: str) -> bool:
    return action in {"save", "both"}


def should_show(action: str) -> bool:
    return action in {"show", "both"}


def prepare_output_path(output_dir: str | Path, filename: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


def finalize_figure(fig: plt.Figure, action: str, output_path: str | Path | None = None) -> None:
    if should_save(action) and output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved figure to: {output_path}")

    if should_show(action):
        plt.show()
    else:
        plt.close(fig)


def save_table(df, action: str, output_path: str | Path) -> None:
    if not should_save(action):
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved table to: {output_path}")