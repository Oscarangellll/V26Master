import argparse
import csv
from collections import defaultdict
import os
from pathlib import Path

import pandas as pd

from config import CaseConfig, ScenarioConfig
from optimization_models import OptimizationModel


COALITION_COLUMNS = [
    "case_id",
    "coalition",
    "coalition_size",
    "bases",
    "count",
    "objective",
    "first_stage_cost",
    "second_stage_cost",
    "charter_cost_ST",
    "charter_cost_LT",
    "charter_cost_mob",
    "downtime_cost",
    "travel_cost_S",
    "travel_cost_M",
    "runtime",
    "MIPGap",
    "eta",
    "gamma_LT",
    "gamma_ST",
    "alpha",
    "iss_scenarios",
    "oos_scenarios",
    "seed",
]


WINDFARM_COLUMNS = [
    "case_id",
    "coalition",
    "coalition_size",
    "bases",
    "wind_farm",
    "n_turbines",
    "count",
    "downtime_cost",
    "potential_revenue",
    "value_based_availability",
    "travel_cost_S",
    "failures",
    "completed_tasks",
    "completion_ratio",
    "avg_total_backlog",
    "max_total_backlog",
    "backlog_task_days",
    "downtime_turbine_hours",
    "potential_turbine_hours",
    "backlog_task_days_per_turbine_day",
    "time_based_availability",
    "time_availability_proxy",
    "value_availability_proxy",
    "iss_scenarios",
    "oos_scenarios",
    "seed",
]


def _parse_solution_group(group_name, encoded):
    if not isinstance(encoded, str) or encoded.strip() == "":
        return []

    entries = []
    for item in encoded.split(";"):
        key_str, val_str = item.rsplit(":", 1)
        val = int(float(val_str))

        if group_name == "eta":
            key = key_str
        elif group_name == "gamma_LT":
            h, b = key_str.split("|", 1)
            key = (h, b)
        elif group_name == "gamma_ST":
            h, b, t = key_str.split("|", 2)
            key = (h, b, t)
        elif group_name == "alpha":
            v, b, t = key_str.split("|", 2)
            key = (v, b, t)
        else:
            raise ValueError(f"Unknown solution group: {group_name}")

        entries.append(((group_name, key), val))

    return entries


def _decode_solution(row):
    solution = []
    for group_name in ["eta", "gamma_LT", "gamma_ST"]:
        solution.extend(_parse_solution_group(group_name, row.get(group_name, "")))

    if bool(row.get("_fix_alpha", False)):
        solution.extend(_parse_solution_group("alpha", row.get("alpha", "")))

    return frozenset(solution)


def _fix_solution(model, solution, fixed_groups):
    for group_name in fixed_groups:
        for _, var in getattr(model, group_name).items():
            var.LB = 0
            var.UB = 0

    for (group_name, key), value in solution:
        group_vars = getattr(model, group_name)
        if key not in group_vars:
            raise KeyError(f"Missing variable key for group={group_name}, key={key}")
        var = group_vars[key]
        var.LB = value
        var.UB = value


def _safe_value(expr):
    try:
        return expr.getValue()
    except Exception:
        return None


def _solution_key(row):
    key = (
        row.get("eta", ""),
        row.get("gamma_LT", ""),
        row.get("gamma_ST", ""),
    )
    if bool(row.get("_fix_alpha", False)):
        key = key + (row.get("alpha", ""),)
    return key


def _coalition_from_name(name):
    return tuple(str(name).strip())


def _parse_bases(encoded):
    if not isinstance(encoded, str) or encoded.strip() == "":
        return None
    return [part.strip() for part in encoded.split(";") if part.strip()]


def _parse_scenarios(encoded):
    if not isinstance(encoded, str) or encoded.strip() == "":
        return []
    return [int(part.strip()) for part in encoded.split(";") if part.strip()]


def _scenario_label(start, end):
    return f"{start}-{end}"


def _validate_scenario_range(start, end):
    scenario_data_dir = Path(os.environ.get("SCENARIO_DATA_DIR", "data/scenario_data"))
    reference_path = scenario_data_dir / "downtime_cost"
    if not reference_path.exists():
        return

    df = pd.read_parquet(reference_path, columns=["s"])
    scenario_ids = pd.to_numeric(df["s"], errors="coerce")
    min_s = int(scenario_ids.min())
    max_s = int(scenario_ids.max())

    if start < min_s or end > max_s:
        raise ValueError(
            f"OOS scenario range {start}-{end} is outside available local scenario data "
            f"{min_s}-{max_s}. Choose a range within this interval or set SCENARIO_DATA_DIR "
            "to a directory containing the requested scenarios."
        )


def _empty_totals():
    return {
        "objective": 0.0,
        "first_stage_cost": 0.0,
        "second_stage_cost": 0.0,
        "charter_cost_ST": 0.0,
        "charter_cost_LT": 0.0,
        "charter_cost_mob": 0.0,
        "downtime_cost": 0.0,
        "travel_cost_S": 0.0,
        "travel_cost_M": 0.0,
        "runtime": 0.0,
        "MIPGap": 0.0,
    }


def _empty_windfarm_totals():
    return defaultdict(lambda: {
        "n_turbines": 0.0,
        "downtime_cost": 0.0,
        "potential_revenue": 0.0,
        "travel_cost_S": 0.0,
        "failures": 0.0,
        "completed_tasks": 0.0,
        "avg_total_backlog": 0.0,
        "max_total_backlog": 0.0,
        "backlog_task_days": 0.0,
        "downtime_turbine_hours": 0.0,
        "potential_turbine_hours": 0.0,
    })


def _collect_windfarm_metrics(model, scenario_id):
    case = model.case
    scenario = model.scenario
    s = int(scenario_id)
    rows = {}
    turbine_counts = {w.name: w.n_turbines for w in case.wind_farms}

    for w in case.W:
        downtime_cost = 0.0
        failures = 0.0
        completed_tasks = 0.0
        backlog_task_days = 0.0
        potential_revenue = 0.0
        daily_backlogs = []

        for d in case.D:
            daily_total_backlog = 0.0
            potential_revenue += scenario.C_D[s][w, d] * turbine_counts[w]

            for m in case.M:
                failures += scenario.F[s][w, m, d]
                completed_tasks += model.z[w, m, d, s].X
                backlog = model.b[w, m, d, s].X
                backlog_task_days += backlog
                daily_total_backlog += backlog
                downtime_cost += scenario.C_D[s][w, d] * backlog

            daily_backlogs.append(daily_total_backlog)

        travel_cost_S = 0.0
        for h in case.H_S:
            for b in case.B:
                for d in case.D:
                    travel_cost_S += case.C_RT[h, b, w] * model.x[h, b, w, d, s].X

        rows[w] = {
            "n_turbines": turbine_counts[w],
            "downtime_cost": downtime_cost,
            "travel_cost_S": travel_cost_S,
            "failures": failures,
            "completed_tasks": completed_tasks,
            "avg_total_backlog": sum(daily_backlogs) / len(daily_backlogs),
            "max_total_backlog": max(daily_backlogs) if daily_backlogs else 0.0,
            "backlog_task_days": backlog_task_days,
            "downtime_turbine_hours": backlog_task_days * 24,
            "potential_turbine_hours": turbine_counts[w] * len(case.D) * 24,
            "potential_revenue": potential_revenue,
        }

    return rows


def _evaluate_solution(case, solution, oos_scenarios, scenario_cfg, fixed_groups):
    totals = _empty_totals()
    windfarm_totals = _empty_windfarm_totals()
    solved_count = 0

    for scenario_id in oos_scenarios:
        scenario_id = int(scenario_id)
        scenario_ids = [scenario_id]
        weights = {scenario_id: 1.0}

        try:
            model = OptimizationModel(case, scenario_cfg, scenario_ids, weights)
            model.Params.OutputFlag = 0
            model.Params.MIPGap = 0.02
            _fix_solution(model, solution, fixed_groups=fixed_groups)
            model.optimize()
        except Exception as exc:
            print(
                f"[case OOS warning] coalition={''.join(case.W)}, scenario={scenario_id}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue

        if getattr(model, "SolCount", 0) <= 0:
            print(
                f"[case OOS warning] no incumbent for coalition={''.join(case.W)}, "
                f"scenario={scenario_id}, status={getattr(model, 'Status', None)}"
            )
            continue

        solved_count += 1
        totals["objective"] += model.ObjVal
        totals["first_stage_cost"] += _safe_value(model.first_obj) or 0.0
        totals["second_stage_cost"] += _safe_value(model.second_obj) or 0.0
        totals["charter_cost_ST"] += _safe_value(model.charter_cost_ST) or 0.0
        totals["charter_cost_LT"] += _safe_value(model.charter_cost_LT) or 0.0
        totals["charter_cost_mob"] += _safe_value(model.charter_cost_mob) or 0.0
        totals["downtime_cost"] += _safe_value(model.downtime_cost) or 0.0
        totals["travel_cost_S"] += _safe_value(model.travel_cost_S) or 0.0
        totals["travel_cost_M"] += _safe_value(model.travel_cost_M) or 0.0
        totals["runtime"] += model.Runtime
        totals["MIPGap"] += model.MIPGap

        for w, metrics in _collect_windfarm_metrics(model, scenario_id).items():
            for key, value in metrics.items():
                windfarm_totals[w][key] += value

    if solved_count == 0:
        return None, None, 0

    for key in totals:
        totals[key] /= solved_count

    windfarm_rows = {}
    for w, metrics in windfarm_totals.items():
        row = {}
        for key, value in metrics.items():
            row[key] = value / solved_count
        potential = row["potential_revenue"]
        row["value_based_availability"] = (
            1.0 - row["downtime_cost"] / potential if potential > 0 else None
        )
        row["value_availability_proxy"] = row["value_based_availability"]
        turbine_days = row["n_turbines"] * len(case.D)
        row["backlog_task_days_per_turbine_day"] = (
            row["backlog_task_days"] / turbine_days if turbine_days > 0 else None
        )
        row["time_based_availability"] = (
            1.0 - row["downtime_turbine_hours"] / row["potential_turbine_hours"]
            if row["potential_turbine_hours"] > 0
            else None
        )
        row["time_availability_proxy"] = row["time_based_availability"]
        row["completion_ratio"] = (
            row["completed_tasks"] / row["failures"] if row["failures"] > 0 else None
        )
        windfarm_rows[w] = row

    return totals, windfarm_rows, solved_count


def _write_header(path, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(columns)


def _append_row(path, row):
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def run(args):
    if not (0 <= args.node_id < args.num_nodes):
        raise ValueError("--node-id must satisfy 0 <= node_id < num_nodes")

    results = pd.read_csv(args.input)
    results = results[results["has_solution"].astype(str).str.lower().isin(["true", "1"])]

    if args.fix_alpha and "alpha" not in results.columns:
        raise ValueError(
            "Input file has no alpha column, but --fix-alpha was requested."
        )
    if "alpha" not in results.columns:
        results["alpha"] = ""

    fixed_groups = ["eta", "gamma_LT", "gamma_ST"]
    if args.fix_alpha:
        fixed_groups.append("alpha")

    coalition_output = Path(args.coalition_output)
    windfarm_output = Path(args.windfarm_output)
    if args.num_nodes > 1:
        if args.coalition_output == build_parser().get_default("coalition_output"):
            coalition_output = coalition_output.with_name(
                f"{coalition_output.stem}_node_{args.node_id}{coalition_output.suffix}"
            )
        if args.windfarm_output == build_parser().get_default("windfarm_output"):
            windfarm_output = windfarm_output.with_name(
                f"{windfarm_output.stem}_node_{args.node_id}{windfarm_output.suffix}"
            )

    _write_header(coalition_output, COALITION_COLUMNS)
    _write_header(windfarm_output, WINDFARM_COLUMNS)

    _validate_scenario_range(args.oos_start, args.oos_end)
    oos_scenarios = list(range(args.oos_start, args.oos_end + 1))
    oos_label = _scenario_label(args.oos_start, args.oos_end)

    unique_rows = []
    evaluated = set()
    selected_coalitions = (
        {"".join(sorted(coalition)) for coalition in args.coalitions}
        if args.coalitions is not None
        else None
    )
    for row in results.to_dict("records"):
        row["_fix_alpha"] = args.fix_alpha
        coalition = str(row["coalition"])
        if selected_coalitions is not None and coalition not in selected_coalitions:
            continue
        key = (coalition, _solution_key(row))
        if key in evaluated:
            continue
        evaluated.add(key)
        unique_rows.append(row)

    unique_rows = sorted(
        unique_rows,
        key=lambda r: (int(r["coalition_size"]), str(r["coalition"])),
    )
    assigned_rows = [
        row for i, row in enumerate(unique_rows) if i % args.num_nodes == args.node_id
    ]

    print(
        f"Node {args.node_id}/{args.num_nodes}: "
        f"{len(assigned_rows)} of {len(unique_rows)} unique coalition solutions assigned."
    )

    for row in assigned_rows:
        coalition = str(row["coalition"])

        row_bases = _parse_bases(row.get("bases", ""))
        case = CaseConfig(
            coalition=_coalition_from_name(coalition),
            bases=args.bases if args.bases is not None else row_bases,
            max_multiday_vessels=args.max_multiday_vessels,
        )
        eval_scenarios = oos_scenarios
        if not eval_scenarios:
            print(f"[case OOS warning] coalition={coalition}: no scenarios to evaluate.")
            continue

        scenario_cfg = ScenarioConfig(case, eval_scenarios)
        solution = _decode_solution(row)

        totals, windfarm_rows, solved_count = _evaluate_solution(
            case=case,
            solution=solution,
            oos_scenarios=eval_scenarios,
            scenario_cfg=scenario_cfg,
            fixed_groups=fixed_groups,
        )
        if totals is None:
            continue

        common = [
            args.case_id,
            coalition,
            int(row["coalition_size"]),
            row.get("bases", ""),
        ]
        meta = [
            row.get("eta", ""),
            row.get("gamma_LT", ""),
            row.get("gamma_ST", ""),
            row.get("alpha", ""),
            row.get("scenarios", ""),
            oos_label,
            row.get("seed", ""),
        ]

        _append_row(
            coalition_output,
            common
            + [
                solved_count,
                totals["objective"],
                totals["first_stage_cost"],
                totals["second_stage_cost"],
                totals["charter_cost_ST"],
                totals["charter_cost_LT"],
                totals["charter_cost_mob"],
                totals["downtime_cost"],
                totals["travel_cost_S"],
                totals["travel_cost_M"],
                totals["runtime"],
                totals["MIPGap"],
            ]
            + meta,
        )

        for w, metrics in windfarm_rows.items():
            _append_row(
                windfarm_output,
                common
                + [
                    w,
                    metrics["n_turbines"],
                    solved_count,
                    metrics["downtime_cost"],
                    metrics["potential_revenue"],
                    metrics["value_based_availability"],
                    metrics["travel_cost_S"],
                    metrics["failures"],
                    metrics["completed_tasks"],
                    metrics["completion_ratio"],
                    metrics["avg_total_backlog"],
                    metrics["max_total_backlog"],
                    metrics["backlog_task_days"],
                    metrics["downtime_turbine_hours"],
                    metrics["potential_turbine_hours"],
                    metrics["backlog_task_days_per_turbine_day"],
                    metrics["time_based_availability"],
                    metrics["time_availability_proxy"],
                    metrics["value_availability_proxy"],
                    row.get("scenarios", ""),
                    oos_label,
                    row.get("seed", ""),
                ],
            )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate case-study coalition solutions on OOS scenarios."
    )
    parser.add_argument("--input", required=True, help="CSV produced by main.py.")
    parser.add_argument("--case-id", default="base")
    parser.add_argument("--bases", nargs="+", default=None)
    parser.add_argument("--max-multiday-vessels", type=int, default=None)
    parser.add_argument("--oos-start", type=int, default=501)
    parser.add_argument("--oos-end", type=int, default=1500)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--node-id", type=int, default=0)
    parser.add_argument(
        "--coalitions",
        nargs="+",
        default=None,
        help="Only evaluate selected coalitions, e.g. --coalitions BCD BCG BEG.",
    )
    parser.add_argument(
        "--coalition-output",
        default="results/case_studies/base/coalition_oos.csv",
    )
    parser.add_argument(
        "--windfarm-output",
        default="results/case_studies/base/windfarm_oos.csv",
    )
    parser.add_argument(
        "--fix-alpha",
        action="store_true",
        help=(
            "Also fix alpha vessel-index decisions during OOS evaluation. "
            "By default only eta, gamma_LT, and gamma_ST are fixed."
        ),
    )
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
