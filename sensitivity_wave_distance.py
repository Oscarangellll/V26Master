import argparse
import copy
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from haversine import haversine, Unit

from config.case_config import CaseConfig
from config.patterns import gen_patterns
from data.fixed_data import data
from optimization_models.optimization_model import OptimizationModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--case", required=True, help="Case name")
    parser.add_argument("-m", "--method", required=True, choices=["mip", "con"], help="Solution method")
    parser.add_argument("--tree_size", type=int, default=9, help="Scenario tree size per instance")
    parser.add_argument("--n_instances", type=int, default=10, help="Number of sampled instances per sensitivity point")
    parser.add_argument("--iss_pool_start", type=int, default=1)
    parser.add_argument("--iss_pool_end", type=int, default=200)
    parser.add_argument("--oos_pool_start", type=int, default=201)
    parser.add_argument("--oos_pool_end", type=int, default=300)
    parser.add_argument("--oos_eval_size", type=int, default=40, help="Number of OOS scenarios used for evaluation")
    parser.add_argument("--seed", type=int, default=400, help="Random seed")
    parser.add_argument("--wave_values", type=float, nargs="+", default=[1.2, 1.4, 1.5, 1.6, 1.8, 2.0])
    parser.add_argument("--oneway_hours_values", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--single_day_vessel", type=str, default="CTV", help="Vessel name for wave sensitivity")
    parser.add_argument("--output_dir", type=str, default="results/sensitivity")
    parser.add_argument("--resume", action="store_true", help="Resume from existing sensitivity_summary.csv and skip completed grid points")
    parser.add_argument("--checkpoint_every", type=int, default=1, help="Refresh heatmap and progress log every N completed grid points")
    parser.add_argument("--solver_time_limit", type=float, default=0.0, help="Optional per-model time limit in seconds (0 = no time limit)")
    parser.add_argument("--show_plot", action="store_true")
    return parser.parse_args()


def copy_case(case_name):
    case = CaseConfig(f"cases/{case_name}.yaml")
    case.vessel_types = [copy.deepcopy(h) for h in case.vessel_types]
    case.bases = [copy.deepcopy(b) for b in case.bases]
    case.wind_farms = [copy.deepcopy(w) for w in case.wind_farms]
    case.maintenance_categories = [copy.deepcopy(m) for m in case.maintenance_categories]
    return case


def max_consecutive_true(values):
    max_len = 0
    cur = 0
    for v in values:
        if v:
            cur += 1
            if cur > max_len:
                max_len = cur
        else:
            cur = 0
    return max_len


def compute_windows(df_working, scenario_ids, max_speed, max_height):
    df = df_working[df_working["s"].isin(scenario_ids)]
    out = {}
    grouped = df.groupby(["s", "wl_id", "d"], sort=False)
    for (s, wl_id, d), g in grouped:
        g = g.sort_values("hour")
        feasible = ((g["speed"] <= max_speed) & (g["height"] <= max_height)).to_numpy()
        out[(wl_id, d, s)] = max_consecutive_true(feasible)
    return out


def set_one_way_hours(case, vessel_name, one_way_hours):
    if len(case.bases) != 1 or len(case.wind_farms) != 1:
        raise ValueError("Sensitivity script currently assumes exactly one base and one wind farm in the case.")

    vessel = next((h for h in case.vessel_types if h.name == vessel_name), None)
    if vessel is None:
        raise ValueError(f"Vessel '{vessel_name}' not found in case vessel types.")

    base = case.bases[0]
    wind_farm = case.wind_farms[0]

    target_km = one_way_hours * vessel.travel_speed
    lat = wind_farm.lat
    km_per_lon = 111.32 * math.cos(math.radians(lat))

    if km_per_lon <= 0:
        raise ValueError("Invalid latitude for longitude-distance conversion.")

    base.lat = lat
    base.lon = wind_farm.lon - target_km / km_per_lon

    realized_km = haversine((base.lat, base.lon), (wind_farm.lat, wind_farm.lon), unit=Unit.KILOMETERS)
    realized_hours = realized_km / vessel.travel_speed
    return realized_km, realized_hours


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


def solution_signature(solution):
    items = sorted((f"{grp}:{encode_key(key)}={val}" for (grp, key), val in solution if val > 0))
    return ";".join(items)


def combo_key(wave, one_way_hours):
    return f"{float(wave):.6f}|{float(one_way_hours):.6f}"


def log_progress(progress_path, message):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with progress_path.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {message}\n")


def build_scenario(case, scenario_ids, weather_windows, failures_by_s, downtime_by_s):
    K_S, K_M, P = gen_patterns(weather_windows, case, scenario_ids)

    F = {}
    for s in scenario_ids:
        for w, m, d, val in failures_by_s[s]:
            F[(w, m, d, s)] = val

    C_D = {}
    for s in scenario_ids:
        for w, d, val in downtime_by_s[s]:
            C_D[(w, d, s)] = val

    return SimpleNamespace(S=scenario_ids, K_S=K_S, K_M=K_M, P=P, F=F, C_D=C_D)


def evaluate_solution_oos(solution, oos_ids, case, scenario_getter, solver_time_limit=0.0):
    results = {
        "objective": 0.0,
        "first_stage_cost": 0.0,
        "second_stage_cost": 0.0,
        "downtime_cost": 0.0,
        "travel_cost_S": 0.0,
        "travel_cost_M": 0.0,
    }

    for sid in oos_ids:
        scenario_cfg = scenario_getter((sid,))
        model = OptimizationModel(case, scenario_cfg)
        model.Params.OutputFlag = 0
        if solver_time_limit and solver_time_limit > 0:
            model.Params.TimeLimit = solver_time_limit

        for (group, key), val in solution:
            var = getattr(model, group)[key]
            var.LB = val
            var.UB = val

        model.optimize()

        results["objective"] += model.ObjVal
        results["first_stage_cost"] += model.first_obj.getValue()
        results["second_stage_cost"] += model.second_obj.getValue()
        results["downtime_cost"] += model.downtime_cost.getValue()
        results["travel_cost_S"] += model.travel_cost_S.getValue()
        results["travel_cost_M"] += model.travel_cost_M.getValue()

    n = len(oos_ids)
    return {k: v / n for k, v in results.items()}


def make_heatmaps(summary_df, output_png, show=False):
    df = summary_df.copy()

    if df.empty:
        return

    wave_vals = sorted(df["wave_ctv"].unique())
    hour_vals = sorted(df["oneway_hours"].unique())

    pivot_obj = df.pivot(index="wave_ctv", columns="oneway_hours", values="oos_mean").loc[wave_vals, hour_vals]
    pivot_ctv = df.pivot(index="wave_ctv", columns="oneway_hours", values="pct_runs_with_ctv").loc[wave_vals, hour_vals]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im1 = axes[0].imshow(pivot_obj.values, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_title("OOS objective mean")
    axes[0].set_xlabel("One-way travel time (hours)")
    axes[0].set_ylabel("CTV max wave")
    axes[0].set_xticks(range(len(hour_vals)))
    axes[0].set_xticklabels([f"{x:.1f}" for x in hour_vals])
    axes[0].set_yticks(range(len(wave_vals)))
    axes[0].set_yticklabels([f"{x:.2f}" for x in wave_vals])
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(pivot_ctv.values, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1)
    axes[1].set_title("Share of runs using CTV")
    axes[1].set_xlabel("One-way travel time (hours)")
    axes[1].set_ylabel("CTV max wave")
    axes[1].set_xticks(range(len(hour_vals)))
    axes[1].set_xticklabels([f"{x:.1f}" for x in hour_vals])
    axes[1].set_yticks(range(len(wave_vals)))
    axes[1].set_yticklabels([f"{x:.2f}" for x in wave_vals])
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    args = parse_args()

    if args.method != "mip":
        raise ValueError("Sensitivity script currently supports only --method mip.")

    iss_pool = np.arange(args.iss_pool_start, args.iss_pool_end + 1)
    oos_pool = np.arange(args.oos_pool_start, args.oos_pool_end + 1)

    if len(set(iss_pool).intersection(set(oos_pool))) > 0:
        raise ValueError("In-sample and out-of-sample pools overlap.")

    if args.tree_size > len(iss_pool):
        raise ValueError("tree_size exceeds ISS pool size.")

    rng = np.random.default_rng(args.seed)

    instance_scenarios = [
        tuple(sorted(rng.choice(iss_pool, size=args.tree_size, replace=False).tolist()))
        for _ in range(args.n_instances)
    ]

    oos_eval_size = min(args.oos_eval_size, len(oos_pool))
    oos_eval_ids = tuple(sorted(rng.choice(oos_pool, size=oos_eval_size, replace=False).tolist()))

    required_scenarios = sorted(set([sid for scen in instance_scenarios for sid in scen] + list(oos_eval_ids)))

    case_template = copy_case(args.case)
    vessel_map = {h.name: h for h in case_template.vessel_types}
    if args.single_day_vessel not in vessel_map:
        raise ValueError(f"single_day_vessel '{args.single_day_vessel}' not present in case vessel types.")

    output_dir = Path(args.output_dir) / args.case / args.method
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_csv = output_dir / "sensitivity_detail.csv"
    summary_csv = output_dir / "sensitivity_summary.csv"
    heatmap_png = output_dir / "sensitivity_heatmaps.png"
    progress_log = output_dir / "sensitivity_progress.log"

    if not args.resume:
        for p in [detail_csv, summary_csv, heatmap_png, progress_log]:
            if p.exists():
                p.unlink()

    completed_keys = set()
    if args.resume and summary_csv.exists():
        existing = pd.read_csv(summary_csv)
        if not existing.empty:
            completed_keys = {
                combo_key(r.wave_ctv, r.oneway_hours)
                for r in existing.itertuples(index=False)
            }
            print(f"Resume mode: found {len(completed_keys)} completed grid point(s).")

    log_progress(progress_log, f"START case={args.case} method={args.method} tree_size={args.tree_size} n_instances={args.n_instances}")

    print("Loading scenario data...")
    df_weather = pd.read_csv("data/scenario_data/weather.csv", usecols=["wl_id", "d", "hour", "s", "speed", "height"])
    df_weather = df_weather[df_weather["s"].isin(required_scenarios)]

    work_day_start = data.work_day_start
    work_day_end = data.work_day_end
    df_working = df_weather[df_weather["hour"].isin(range(work_day_start, work_day_end))]

    df_failures = pd.read_csv("data/scenario_data/failures.csv", usecols=["w", "m", "d", "s", "failures"])
    df_failures = df_failures[df_failures["s"].isin(required_scenarios)]

    df_downtime = pd.read_csv("data/scenario_data/downtime_cost.csv", usecols=["w", "d", "s", "downtime_cost"])
    df_downtime = df_downtime[df_downtime["s"].isin(required_scenarios)]

    failures_by_s = defaultdict(list)
    for r in df_failures.itertuples(index=False):
        failures_by_s[int(r.s)].append((r.w, r.m, int(r.d), float(r.failures)))

    downtime_by_s = defaultdict(list)
    for r in df_downtime.itertuples(index=False):
        downtime_by_s[int(r.s)].append((r.w, int(r.d), float(r.downtime_cost)))

    print("Precomputing weather windows for sensitivity grid...")
    default_windows = {}
    variable_windows = {}

    for h in case_template.vessel_types:
        if h.name == args.single_day_vessel:
            for wave in args.wave_values:
                variable_windows[wave] = compute_windows(df_working, required_scenarios, h.max_wind, wave)
        else:
            default_windows[h.name] = compute_windows(df_working, required_scenarios, h.max_wind, h.max_wave)

    grid_points = [(float(w), float(h)) for w in args.wave_values for h in args.oneway_hours_values]
    grid_total = len(grid_points)
    grid_idx = 0

    for wave, one_way_hours in grid_points:
        grid_idx += 1
        key = combo_key(wave, one_way_hours)

        if key in completed_keys:
            print(f"[{grid_idx}/{grid_total}] wave={wave:.2f}, one_way_hours={one_way_hours:.2f} -> already done, skipping")
            continue

        print(f"[{grid_idx}/{grid_total}] wave={wave:.2f}, one_way_hours={one_way_hours:.2f}")
        log_progress(progress_log, f"GRID_START wave={wave:.2f} one_way_hours={one_way_hours:.2f}")

        case = copy_case(args.case)
        vessel = next(h for h in case.vessel_types if h.name == args.single_day_vessel)
        vessel.max_wave = float(wave)

        realized_km, realized_hours = set_one_way_hours(case, args.single_day_vessel, one_way_hours)

        weather_windows = {}
        for h in case.vessel_types:
            if h.name == args.single_day_vessel:
                src = variable_windows[wave]
            else:
                src = default_windows[h.name]
            for (wl_id, d, s), ww in src.items():
                weather_windows[(h.name, wl_id, d, s)] = ww

        scenario_cache = {}

        def get_scenario(ids_tuple):
            ids = tuple(sorted(ids_tuple))
            if ids not in scenario_cache:
                scenario_cache[ids] = build_scenario(case, list(ids), weather_windows, failures_by_s, downtime_by_s)
            return scenario_cache[ids]

        evaluated_solutions = {}
        combo_rows = []

        for instance_id, scenario_ids in enumerate(instance_scenarios, start=1):
            scenario_cfg = get_scenario(scenario_ids)
            model = OptimizationModel(case, scenario_cfg)
            model.Params.OutputFlag = 0
            if args.solver_time_limit and args.solver_time_limit > 0:
                model.Params.TimeLimit = args.solver_time_limit
            model.optimize()

            solution = frozenset(
                ((group, key), int(var.X))
                for group in ["eta", "gamma_LT", "gamma_ST", "alpha"]
                for key, var in getattr(model, group).items()
            )

            if solution not in evaluated_solutions:
                evaluated_solutions[solution] = evaluate_solution_oos(
                    solution,
                    oos_eval_ids,
                    case,
                    get_scenario,
                    solver_time_limit=args.solver_time_limit,
                )

            ctv_lt = sum(val for ((group, key), val) in solution if group == "gamma_LT" and key[0] == args.single_day_vessel)
            ctv_st = sum(val for ((group, key), val) in solution if group == "gamma_ST" and key[0] == args.single_day_vessel)

            row = {
                "wave_ctv": float(wave),
                "oneway_hours": float(one_way_hours),
                "oneway_hours_realized": float(realized_hours),
                "oneway_km_realized": float(realized_km),
                "instance_id": int(instance_id),
                "iss_objective": float(model.ObjVal),
                "iss_first_stage_cost": float(model.first_obj.getValue()),
                "iss_second_stage_cost": float(model.second_obj.getValue()),
                "iss_downtime_cost": float(model.downtime_cost.getValue()),
                "iss_travel_cost_S": float(model.travel_cost_S.getValue()),
                "iss_travel_cost_M": float(model.travel_cost_M.getValue()),
                "oos_objective": float(evaluated_solutions[solution]["objective"]),
                "ctv_lt_count": int(ctv_lt),
                "ctv_st_count": int(ctv_st),
                "uses_ctv": int((ctv_lt + ctv_st) > 0),
                "gamma_LT": encode_solution_group(solution, "gamma_LT"),
                "gamma_ST": encode_solution_group(solution, "gamma_ST"),
                "solution_signature": solution_signature(solution),
            }

            combo_rows.append(row)
            print(f"    instance {instance_id}/{args.n_instances} done (unique solutions so far: {len(evaluated_solutions)})")

        combo_df = pd.DataFrame(combo_rows)
        summary_row = {
            "wave_ctv": float(wave),
            "oneway_hours": float(one_way_hours),
            "oneway_hours_realized": float(realized_hours),
            "oneway_km_realized": float(realized_km),
            "iss_mean": float(combo_df["iss_objective"].mean()),
            "iss_std": float(combo_df["iss_objective"].std(ddof=1)),
            "oos_mean": float(combo_df["oos_objective"].mean()),
            "oos_std": float(combo_df["oos_objective"].std(ddof=1)),
            "oos_minus_iss_mean": float((combo_df["oos_objective"] - combo_df["iss_objective"]).mean()),
            "ctv_lt_mean": float(combo_df["ctv_lt_count"].mean()),
            "ctv_st_mean": float(combo_df["ctv_st_count"].mean()),
            "pct_runs_with_ctv": float(combo_df["uses_ctv"].mean()),
            "n_unique_solutions": int(combo_df["solution_signature"].nunique()),
            "n_instances": int(len(combo_df)),
        }

        combo_df.to_csv(detail_csv, mode="a", index=False, header=not detail_csv.exists())
        pd.DataFrame([summary_row]).to_csv(summary_csv, mode="a", index=False, header=not summary_csv.exists())
        completed_keys.add(key)

        log_progress(
            progress_log,
            f"GRID_DONE wave={wave:.2f} one_way_hours={one_way_hours:.2f} iss_mean={summary_row['iss_mean']:.2f} oos_mean={summary_row['oos_mean']:.2f} pct_runs_with_ctv={summary_row['pct_runs_with_ctv']:.3f}",
        )

        if len(completed_keys) % max(1, args.checkpoint_every) == 0:
            current_summary = pd.read_csv(summary_csv).sort_values(["wave_ctv", "oneway_hours"]).reset_index(drop=True)
            make_heatmaps(current_summary, heatmap_png, show=False)
            print(f"    checkpoint saved ({len(completed_keys)} completed grid points)")

    if summary_csv.exists():
        summary_df = pd.read_csv(summary_csv).sort_values(["wave_ctv", "oneway_hours"]).reset_index(drop=True)
        make_heatmaps(summary_df, heatmap_png, show=args.show_plot)
        print(f"Saved detail table: {detail_csv}")
        print(f"Saved summary table: {summary_csv}")
        print(f"Saved heatmaps: {heatmap_png}")
        print(f"Saved progress log: {progress_log}")
        log_progress(progress_log, f"END completed={len(summary_df)}")
