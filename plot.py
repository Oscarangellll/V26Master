from __future__ import annotations

import argparse

from plot_scripts import PLOT_MODULES
from plot_scripts.style import apply_default_style


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--action",
        choices=["show", "save", "both"],
        default="both",
        help="Choose whether plots are shown, saved, or both.",
    )
    common.add_argument(
        "--output-dir",
        default="plot_scripts/plots",
        help="Directory where plot images are written when --action includes save.",
    )
    common.add_argument(
        "--table-dir",
        default="plot_scripts/tables",
        help="Directory where table outputs are written when --action includes save.",
    )

    parser = argparse.ArgumentParser(
        description="Run thesis plotting scripts with a shared style",
        parents=[common],
    )

    subparsers = parser.add_subparsers(dest="plot", required=True)
    for module in PLOT_MODULES:
        module.register_parser(subparsers, common)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    apply_default_style()
    args.func(args)


if __name__ == "__main__":
    main()