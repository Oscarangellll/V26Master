import argparse
from itertools import combinations
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plot_scripts.plot_case_studies import _coalition_key


def partitions(items):
    items = tuple(items)
    if not items:
        yield ()
        return

    first, rest = items[0], items[1:]
    for partition in partitions(rest):
        yield ((first,),) + partition
        for idx, block in enumerate(partition):
            new_block = tuple(sorted(block + (first,)))
            yield partition[:idx] + (new_block,) + partition[idx + 1 :]


def canonical_partition(partition):
    return tuple(sorted((_coalition_key(block) for block in partition), key=lambda x: (len(x), x)))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find the best coalition partition for a selected actor set."
    )
    parser.add_argument(
        "--coalition-path",
        default="results/case_studies/base/coalition_oos.csv",
        help="Path to coalition OOS result file.",
    )
    parser.add_argument(
        "--actors",
        nargs="+",
        help=(
            "Wind farms to include. Accepts either space-separated labels "
            "(--actors B C D) or a compact string (--actors BCD). "
            "Defaults to all singleton coalitions in the result file."
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of best configurations to print.",
    )
    return parser.parse_args()


def parse_actors(raw_actors, df):
    if not raw_actors:
        return sorted(df.loc[df["coalition_size"] == 1, "coalition"].tolist())

    if len(raw_actors) == 1:
        actors = list(_coalition_key(raw_actors[0]))
    else:
        actors = [_coalition_key(actor) for actor in raw_actors]

    if any(len(actor) != 1 for actor in actors):
        raise ValueError("Each actor must be a single wind-farm label.")

    return sorted(set(actors))


def load_coalition_costs(path):
    df = pd.read_csv(path)
    df["coalition"] = df["coalition"].map(_coalition_key)
    df["objective"] = pd.to_numeric(df["objective"], errors="coerce")
    df = (
        df.sort_values(["coalition", "objective"])
        .drop_duplicates(subset=["coalition"], keep="first")
        .reset_index(drop=True)
    )

    cost = dict(zip(df["coalition"], df["objective"]))
    savings = {}
    for coalition in df["coalition"]:
        members = tuple(coalition)
        if all(member in cost for member in members):
            savings[coalition] = sum(cost[member] for member in members) - cost[coalition]

    return df, cost, savings


def main():
    args = parse_args()
    df, cost, savings = load_coalition_costs(args.coalition_path)
    actors = parse_actors(args.actors, df)

    missing_singletons = [actor for actor in actors if actor not in cost]
    if missing_singletons:
        raise ValueError(
            "Missing singleton result(s) for actor(s): "
            + ", ".join(missing_singletons)
        )

    standalone_total = sum(cost[a] for a in actors)

    seen = set()
    rows = []
    for partition in partitions(actors):
        config = canonical_partition(partition)
        if config in seen:
            continue
        seen.add(config)

        if any(block not in cost for block in config):
            continue

        total_cost = sum(cost[block] for block in config)
        total_savings = standalone_total - total_cost
        synergy = total_savings / total_cost if total_cost else 0.0
        rows.append((config, total_cost, total_savings, synergy))

    rows.sort(key=lambda row: row[2], reverse=True)

    print(f"Actors: {''.join(actors)}")
    print(f"Coalition results: {args.coalition_path}")
    print(f"Number of feasible coalition configurations: {len(rows)}")
    print(f"Standalone total cost: {standalone_total / 1e6:.3f} MEUR")
    print()
    print("Top coalition configurations by total cost savings:")
    for rank, (config, total_cost, total_savings, synergy) in enumerate(rows[:args.top], start=1):
        print(
            f"{rank:>2}. {' + '.join(config):<24} "
            f"savings={total_savings / 1e6:>7.3f} MEUR  "
            f"cost={total_cost / 1e6:>7.3f} MEUR  "
            f"synergy={synergy * 100:>6.2f}%"
        )

    grand = _coalition_key(actors)
    if grand in cost:
        grand_row = next(row for row in rows if row[0] == (grand,))
        grand_rank = rows.index(grand_row) + 1
        print()
        print(
            f"Grand coalition {grand} ranks {grand_rank} of {len(rows)} "
            f"with savings={grand_row[2] / 1e6:.3f} MEUR "
            f"and synergy={grand_row[3] * 100:.2f}%."
        )


if __name__ == "__main__":
    main()
