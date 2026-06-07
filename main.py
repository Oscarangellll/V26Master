import argparse
import csv
import math
from itertools import combinations
from pathlib import Path

import numpy as np

from config.case_config import CaseConfig
from config.scenario_config import ScenarioConfig
from data.fixed_data import data
from optimization_models import ConsensusModelMP, OptimizationModel


COLUMNS = [
    "coalition",
    "coalition_size",
    "method",
    "bases",
    "objective",
    "standalone_cost",
    "synergy",
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
    "runtime",
    "MIPGap",
    "has_solution",
    "status",
    "eta",
    "gamma_LT",
    "gamma_ST",
    "alpha",
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
        "--bases",
        nargs="+",
        default=None,
        help="Candidate bases to include. Overrides dynamic base selection when provided.",
    )

    parser.add_argument(
        "--max-multiday-vessels",
        type=int,
        default=None,
        help="Override max number of multiday vessel indices. Default: max(3, coalition size).",
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
        default=500,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--method",
        choices=["con_mp", "mip"],
        default="con_mp",
        help="Solution method. Use mip for small local tests; con_mp is intended for the case-study runs.",
    )

    parser.add_argument(
        "--mip-gap",
        type=float,
        default=0.02,
        help="MIPGap used when --method mip.",
    )

    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Optional Gurobi time limit in seconds.",
    )

    parser.add_argument(
        "--sampling-mode",
        choices=["per-coalition", "shared"],
        default="shared",
        help=(
            "Use per-coalition to draw a new scenario set for each coalition, "
            "or shared to use the same scenario set for every coalition. "
            "Shared is recommended for case-study coalition comparisons."
        ),
    )

    parser.add_argument(
        "--coalition-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Only solve coalitions with these sizes, e.g. --coalition-sizes 5 6.",
    )

    parser.add_argument(
        "--append",
        action="store_true",
        help="Append rows to the output CSV instead of overwriting it.",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip coalitions that already have a row in the output CSV.",
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
        key=lambda c: (len(c), coalition_name(c)),
    )

    return [
        coalition
        for i, coalition in enumerate(sorted_coalitions)
        if i % num_nodes == node_id
    ]


def select_bases(args, coalition):
    if args.bases is not None:
        return [str(b) for b in args.bases]

    return [b.name for b in data.bases]


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


def encode_bases(bases):
    return ";".join(map(str, bases))


def extract_solution(model):
    var_groups = ["eta", "gamma_LT", "gamma_ST", "alpha"]

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


def ensure_header(path: Path):
    if path.exists() and path.stat().st_size > 0:
        return

    write_header(path)


def read_existing_output(path: Path):
    standalone_costs = {}
    existing_coalitions = set()

    if not path.exists() or path.stat().st_size == 0:
        return standalone_costs, existing_coalitions

    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            coalition = row.get("coalition")
            if coalition:
                existing_coalitions.add(coalition)

            if row.get("coalition_size") != "1":
                continue
            if str(row.get("has_solution")).lower() not in {"true", "1"}:
                continue

            objective = row.get("objective")
            if objective in {None, ""}:
                continue

            standalone_costs[coalition] = float(objective)

    return standalone_costs, existing_coalitions


def write_row(path: Path, row: list):
    if len(row) != len(COLUMNS):
        raise ValueError(f"Row has {len(row)} values, expected {len(COLUMNS)}.")

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def set_row_value(row, column, value):
    row[COLUMNS.index(column)] = value


def get_row_value(row, column):
    return row[COLUMNS.index(column)]


def add_synergy_to_row(row, coalition, standalone_costs):
    if str(get_row_value(row, "has_solution")).lower() not in {"true", "1"}:
        return

    objective = get_row_value(row, "objective")
    if objective in {None, ""}:
        return
    objective = float(objective)

    name = coalition_name(coalition)
    if len(coalition) == 1:
        standalone_costs[name] = objective
        set_row_value(row, "standalone_cost", objective)
        set_row_value(row, "synergy", 0.0)
        return

    if not all(w in standalone_costs for w in coalition):
        return

    standalone_cost = sum(standalone_costs[w] for w in coalition)
    cost_savings = standalone_cost - objective
    synergy = cost_savings / objective if objective else None

    set_row_value(row, "standalone_cost", standalone_cost)
    set_row_value(row, "synergy", synergy)


def solve_coalition(args, coalition, rng, shared_sample=None):
    bases = select_bases(args, coalition)
    case = CaseConfig(
        coalition=coalition,
        bases=bases,
        max_multiday_vessels=args.max_multiday_vessels,
    )

    if shared_sample is None:
        scenario_ids, weights = sample_scenarios(args, rng)
    else:
        scenario_ids, weights = shared_sample

    if args.method == "con_mp":
        model = ConsensusModelMP(case, scenario_ids, weights)
    elif args.method == "mip":
        scenario_cfg = ScenarioConfig(case, scenario_ids)
        model = OptimizationModel(case, scenario_cfg, scenario_ids, weights)
        model.Params.OutputFlag = 0
        model.Params.MIPGap = args.mip_gap
        if args.time_limit is not None:
            model.Params.TimeLimit = args.time_limit
    else:
        raise ValueError(f"Unsupported method: {args.method}")

    model.optimize()

    has_solution = getattr(model, "SolCount", 0) > 0
    solution = extract_solution(model)

    row = [
        coalition_name(coalition),
        len(coalition),
        args.method,
        encode_bases(bases),
        safe_eval(lambda: model.ObjVal) if has_solution else None,
        None,
        None,
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
        safe_eval(lambda: model.Runtime),
        safe_eval(lambda: model.MIPGap) if has_solution else None,
        has_solution,
        safe_eval(lambda: model.Status),
        encode_solution_group(solution, "eta"),
        encode_solution_group(solution, "gamma_LT"),
        encode_solution_group(solution, "gamma_ST"),
        encode_solution_group(solution, "alpha"),
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

    all_coalitions = list(generate_coalitions(actors))

    if args.append:
        ensure_header(output_path)
        standalone_costs, existing_coalitions = read_existing_output(output_path)
    else:
        write_header(output_path)
        standalone_costs = {}
        existing_coalitions = set()

    shared_sample = None
    if args.sampling_mode == "shared":
        shared_sample = sample_scenarios(args, rng)
        print(f"Shared scenarios: {encode_scenarios(shared_sample[0])}")

    assigned = assign_coalitions(
        all_coalitions,
        node_id=args.node_id,
        num_nodes=args.num_nodes,

    )
    if args.coalition_sizes is not None:
        allowed_sizes = set(args.coalition_sizes)
        assigned = [
            coalition for coalition in assigned
            if len(coalition) in allowed_sizes
        ]

    print(
        f"Node {args.node_id}/{args.num_nodes}: "
        f"{len(assigned)} of {len(all_coalitions)} coalitions assigned."
    )

    for coalition in assigned:
        name = coalition_name(coalition)

        if args.skip_existing and name in existing_coalitions:
            print(f"[skip] {name} already exists in {output_path}")
            continue

        print(f"[solve] {name} {coalition}")

        try:
            row = solve_coalition(args, coalition, rng, shared_sample=shared_sample)
            add_synergy_to_row(row, coalition, standalone_costs)
        except Exception as exc:
            print(f"[error] {name}: {type(exc).__name__}: {exc}")
            row = [
                name,
                len(coalition),
                args.method,
                "",
                None, None, None,
                None, None, None, None, None, None, None, None,
                None, None, None, None, None, None,
                None,
                False,
                None,
                "", "", "", "",
                "",
                args.seed,
                args.node_id,
                args.num_nodes,
            ]

        write_row(output_path, row)


if __name__ == "__main__":
    main()
