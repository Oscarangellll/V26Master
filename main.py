import argparse
import csv
import math
from itertools import combinations
from pathlib import Path

import numpy as np

from config.case_config import CaseConfig
from data.fixed_data import data
from optimization_models import ConsensusModelMP


COLUMNS = [
    "coalition",
    "coalition_size",
    "objective",
    "first_stage_cost",
    "second_stage_cost",
    "charter_cost_ST",
    "charter_cost_LT",
    "charter_cost_mob",
    "downtime_cost",
    "travel_cost_S",
    "travel_cost_M",
    "Con_total_runtime",
    "Con_eta_runtime",
    "Con_gamma_LT_runtime",
    "Con_gamma_ST_runtime",
    "Con_Master_runtime",
    "MIPGap",
    "has_solution",
    "status",
    "eta",
    "gamma_LT",
    "gamma_ST",
    "scenarios",
    "seed",
    "node_id",
    "num_nodes",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Solve wind farm coalitions with consensus heuristic"
    )

    parser.add_argument(
        "--actors",
        nargs="+",
        default=None,
        help="Wind farms to include. Default: all wind farms.",
    )

    parser.add_argument(
        "--num-nodes",
        type=int,
        required=True, 
        help="Total number of nodes used.",
    )

    parser.add_argument(
        "--node-id",
        type=int,
        required=True, 
        help="This node's id, from 0 to num_nodes - 1.",
    )

    parser.add_argument(
        "--n-scenarios",
        type=int,
        required=True, 
        help="Number of scenarios in each coalition solve.",
    )

    parser.add_argument(
        "--scenario-start",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--scenario-end",
        type=int,
        default=1500,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Default: results/coalitions/con_mp/node_{node_id}.csv",
    )

    return parser


def coalition_name(coalition: tuple[str, ...]) -> str:
    return "".join(sorted(coalition))


def generate_coalitions(actors: list[str]):
    actors = sorted(actors)

    for r in range(1, len(actors) + 1):
        for coalition in combinations(actors, r):
            yield coalition


def assign_coalitions(coalitions, node_id: int, num_nodes: int):
    sorted_coalitions = sorted(
        coalitions,
        key=lambda c: (-len(c), coalition_name(c)),
    )

    return [
        coalition
        for i, coalition in enumerate(sorted_coalitions)
        if i % num_nodes == node_id
    ]


def sample_scenarios(args, rng):
    scenario_pool = np.arange(args.scenario_start, args.scenario_end + 1)

    scenario_ids = rng.choice(
        scenario_pool,
        size=args.n_scenarios,
        replace=False,
    )

    scenario_ids = [int(s) for s in scenario_ids]
    weights = {s: 1.0 / len(scenario_ids) for s in scenario_ids}

    return scenario_ids, weights


def safe_eval(getter):
    try:
        value = getter()
    except Exception:
        return None

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


def encode_key(key):
    if isinstance(key, tuple):
        return "|".join(map(str, key))
    return str(key)


def encode_solution_group(solution, group):
    items = sorted(
        (
            (encode_key(key), val)
            for (var_group, key), val in solution
            if var_group == group and val > 0
        ),
        key=lambda t: t[0],
    )
    return ";".join(f"{key}:{val}" for key, val in items)


def encode_scenarios(scenario_ids):
    return ";".join(map(str, sorted(int(s) for s in scenario_ids)))


def extract_solution(model):
    var_groups = ["eta", "gamma_LT", "gamma_ST"]

    if getattr(model, "SolCount", 0) <= 0:
        return frozenset()

    return frozenset(
        ((group, idx), int(round(var.X)))
        for group in var_groups
        for idx, var in getattr(model, group).items()
    )

def write_header(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)


def write_row(path: Path, row: list):
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def solve_coalition(args, coalition, rng):
    case = CaseConfig(coalition=coalition)

    scenario_ids, weights = sample_scenarios(args, rng)

    model = ConsensusModelMP(case, scenario_ids, weights)
    model.optimize()

    has_solution = getattr(model, "SolCount", 0) > 0
    solution = extract_solution(model)

    row = [
        coalition_name(coalition),
        len(coalition),
        safe_eval(lambda: model.ObjVal) if has_solution else None,
        safe_eval(lambda: model.first_obj.getValue()) if has_solution else None,
        safe_eval(lambda: model.second_obj.getValue()) if has_solution else None,
        safe_eval(lambda: model.charter_cost_ST.getValue()) if has_solution else None,
        safe_eval(lambda: model.charter_cost_LT.getValue()) if has_solution else None,
        safe_eval(lambda: model.charter_cost_mob.getValue()) if has_solution else None,
        safe_eval(lambda: model.downtime_cost.getValue()) if has_solution else None,
        safe_eval(lambda: model.travel_cost_S.getValue()) if has_solution else None,
        safe_eval(lambda: model.travel_cost_M.getValue()) if has_solution else None,
        safe_eval(lambda: model.total_consensus_time),
        safe_eval(lambda: model.time_to_fix_eta),
        safe_eval(lambda: model.time_to_fix_gamma_LT),
        safe_eval(lambda: model.time_to_tighten_gamma_ST),
        safe_eval(lambda: model.Runtime),
        safe_eval(lambda: model.MIPGap) if has_solution else None,
        has_solution,
        safe_eval(lambda: model.Status),
        encode_solution_group(solution, "eta"),
        encode_solution_group(solution, "gamma_LT"),
        encode_solution_group(solution, "gamma_ST"),
        encode_scenarios(scenario_ids),
        args.seed,
        args.node_id,
        args.num_nodes,
    ]

    return row


def main() -> None:
    import multiprocessing as mp
    mp.set_start_method("spawn")

    args = build_parser().parse_args()

    if not (0 <= args.node_id < args.num_nodes):
        raise ValueError("--node-id must satisfy 0 <= node_id < num_nodes")

    scenario_pool_size = args.scenario_end - args.scenario_start + 1
    if args.n_scenarios > scenario_pool_size:
        raise ValueError("--n-scenarios cannot exceed size of scenario range")

    rng = np.random.default_rng(seed=args.seed)

    actors = args.actors or [w.name for w in data.wind_farms]

    output_path = (
        Path(args.output)
        if args.output is not None
        else Path("results/coalitions/con_mp") / f"node_{args.node_id}.csv"
    )

    write_header(output_path)

    all_coalitions = list(generate_coalitions(actors))

    assigned = assign_coalitions(
        all_coalitions,
        node_id=args.node_id,
        num_nodes=args.num_nodes,

    )   
    print(
        f"Node {args.node_id}/{args.num_nodes}: "
        f"{len(assigned)} of {len(all_coalitions)} coalitions assigned."
    )

    for coalition in assigned:
        name = coalition_name(coalition)

        print(f"[solve] {name} {coalition}")

        try:
            row = solve_coalition(args, coalition, rng)
        except Exception as exc:
            print(f"[error] {name}: {type(exc).__name__}: {exc}")
            row = [
                name,
                len(coalition),
                None, None, None, None, None, None, None, None, None,
                None, None, None, None, None,
                None,
                False,
                None,
                "", "", "",
                "",
                args.seed,
                args.node_id,
                args.num_nodes,
            ]

        write_row(output_path, row)


if __name__ == "__main__":
    main()
