"""
Evaluation pipeline for scenario reduction quality using the kmedoids library.

Tests:
    1. Out-of-sample gap: solve on reduced set, evaluate on full set
    2. Sensitivity in K: cost vs number of reduced scenarios
    3. Feature distribution preservation: Wasserstein distance, moments
    4. K-medoids stability across random seeds

Usage:
    python evaluate_reduction.py -c mini -m con -N 100 -K 4 6 8 10 12
    python evaluate_reduction.py -c 1W3B -m con -N 200 -K 6 8 10 12 15 20
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import kmedoids
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler

from config import CaseConfig, ScenarioConfig
from config.scenario_reduction import build_feature_vector, perform_scenario_reduction
from config.weather_windows import find_weather_windows
from optimization_models import OptimizationModel, ConsensusModel
from scenario_models import price_model, weather_model


# ──────────────────────────────────────────────
# Shared data generation
# ──────────────────────────────────────────────

def generate_scenario_data(case, scenario_ids):
    """
    Generate all raw scenario-dependent inputs once:
        - weather
        - prices
        - failures
        - downtime_costs
        - weather_windows
    """
    scenario_ids = list(scenario_ids)

    # Weather
    weather = {}
    for s in scenario_ids:
        rng = np.random.default_rng(seed=s)
        for iso in case.all_wl_ids_for_iso.keys():
            for loc in case.all_wl_ids_for_iso[iso]:
                weather[(s, iso, loc)] = weather_model.simulate(
                    loc, rng, case.periods, case.days_per_period
                )

    # Prices
    prices = {}
    for s in scenario_ids:
        rng = np.random.default_rng(seed=s)
        for iso in case.all_wl_ids_for_iso.keys():
            iso_wind_speeds = np.array(
                [weather[(s, iso, wl_id)][:, 0] for wl_id in sorted(case.all_wl_ids_for_iso[iso])]
            ).T
            iso_wind_speeds = iso_wind_speeds.reshape(-1, 24, iso_wind_speeds.shape[1]).mean(axis=1)
            prices[(s, iso)] = price_model.simulate(
                iso_wind_speeds, iso, rng, case.periods, case.days_per_period
            )

    # Failures
    p_fail = [m.failure_rate / 365 for m in case.maintenance_categories]
    p_fail.append(1 - sum(p_fail))

    failures = {}
    for s in scenario_ids:
        rng = np.random.default_rng(seed=s)
        for w in case.wind_farms:
            draws = rng.multinomial(w.n_turbines, p_fail, size=len(case.D))
            draws = draws[:, :-1]
            for d_idx, d in enumerate(case.D):
                for m_idx, m in enumerate(case.maintenance_categories):
                    failures[(w.name, m.name, d, s)] = draws[d_idx, m_idx]

    # Downtime costs
    downtime_costs = {}
    for w in case.wind_farms:
        for s in scenario_ids:
            sim_speed = weather[(s, w.iso, w.weather_location_id)][:, 0]
            sim_power = case.power_curve(sim_speed)
            n_days = len(sim_power) // 24
            daily_power = sim_power.reshape(n_days, 24).mean(axis=1) * 24
            daily_cost = daily_power * prices[(s, w.iso)]
            for d in case.D:
                downtime_costs[(w.name, d, s)] = daily_cost[d - 1]

    # Weather windows
    weather_windows = find_weather_windows(case, weather, scenario_ids)

    return {
        "weather": weather,
        "prices": prices,
        "failures": failures,
        "downtime_costs": downtime_costs,
        "weather_windows": weather_windows,
    }


def build_feature_matrix(case, scenario_ids, weather_windows, downtime_costs, failures):
    """
    Build raw and standardized feature matrices.
    """
    X = np.array([
        build_feature_vector(case, s, weather_windows, downtime_costs, failures)
        for s in scenario_ids
    ], dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, scaler


def run_kmedoids_from_features(X_scaled, scenario_ids, K, random_state=42):
    """
    Run k-medoids on a standardized feature matrix using a precomputed
    Euclidean distance matrix.

    Returns
    -------
    dict with:
        medoid_indices
        medoid_ids
        labels
        weights
        diss
        model
    """
    diss = squareform(pdist(X_scaled, metric="euclidean"))

    # sklearn-like API from kmedoids package
    km = kmedoids.KMedoids(
        n_clusters=K,
        metric="precomputed",
        method="fasterpam",
        random_state=random_state,
    )
    km.fit(diss)

    medoid_indices = np.asarray(km.medoid_indices_, dtype=int)
    labels = np.asarray(km.labels_, dtype=int)
    medoid_ids = [int(scenario_ids[idx]) for idx in medoid_indices]

    counts = np.bincount(labels, minlength=K)
    weights = {medoid_ids[k]: counts[k] / len(scenario_ids) for k in range(K)}

    return {
        "medoid_indices": medoid_indices,
        "medoid_ids": medoid_ids,
        "labels": labels,
        "weights": weights,
        "diss": diss,
        "model": km,
    }


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def solve_model(case, scenario_cfg, scenario_ids, method, mip_gap=0.01):
    """
    Solve the optimization model and return (model, runtime).

    For consensus method: falls back to MIP if #scenarios > 12.
    """
    effective_method = method
    if method == "con" and len(scenario_ids) > 12:
        print(f"  [NOTE] {len(scenario_ids)} scenarios exceeds consensus worker cap (12), falling back to MIP")
        effective_method = "mip"

    if effective_method == "mip":
        model = OptimizationModel(case, scenario_cfg, scenario_ids)
        model.build_model()
        model.model.setParam("OutputFlag", 1)
        model.model.setParam("MIPGap", mip_gap)
        model.model.setParam("TimeLimit", 43200) # 12 hours solvetime
        model.optimize()
        return model, model.model.Runtime

    elif effective_method == "con":
        judge_seeds = scenario_cfg.S
        master_scenarios = judge_seeds[:]
        cm = ConsensusModel(
            case,
            scenario_cfg,
            judge_seeds_1scenario_each=judge_seeds,
            mip_gap_judges=mip_gap,
            log=False,
        )
        model, runtime = cm.optimize(
            master_scenarios=master_scenarios,
            mip_gap_master=mip_gap,
        )
        return model, runtime

    raise ValueError(f"Unknown method: {method}")


def extract_first_stage(model):
    """
    Extract first-stage decisions as a fixable dict.
    """
    decisions = {}
    for key, var in model.eta.items():
        decisions[("eta", key)] = int(round(var.X))
    for key, var in model.gamma_LT.items():
        decisions[("gamma_LT", key)] = int(round(var.X))
    for key, var in model.gamma_ST.items():
        decisions[("gamma_ST", key)] = int(round(var.X))
    for key, var in model.alpha.items():
        decisions[("alpha", key)] = int(round(var.X))
    return decisions


def evaluate_oos(case, first_stage_decisions, eval_scenario_ids, mip_gap=0.001):
    """
    Evaluate a first-stage solution out-of-sample on a large scenario set.

    Builds ONE model with all eval scenarios and fixes first-stage.
    Returns the objective value.
    """
    eval_cfg = ScenarioConfig(case, eval_scenario_ids, scenario_reduction=False)
    model = OptimizationModel(case, eval_cfg, list(eval_scenario_ids))
    model.build_model()

    for (group, key), val in first_stage_decisions.items():
        var = getattr(model, group)[key]
        var.LB = val
        var.UB = val

    model.model.setParam("OutputFlag", 0)
    model.model.setParam("MIPGap", mip_gap)
    model.model.setParam("TimeLimit", 21600) # 6 hours solvetime
    model.optimize()

    return model.model.ObjVal


def describe_first_stage(decisions):
    eta = [k for (g, k), v in decisions.items() if g == "eta" and v > 0]
    lt = [f"{k}:{v}" for (g, k), v in decisions.items() if g == "gamma_LT" and v > 0]
    st = [f"{k}:{v}" for (g, k), v in decisions.items() if g == "gamma_ST" and v > 0]
    return f"bases={eta}, LT={lt}, ST={st}"


def build_reduced_scenario_config(case, train_seeds, medoid_ids, weights,
                                weather_windows, downtime_costs, failures):
    """
    Build reduced ScenarioConfig manually from selected medoids + weights.
    """
    ww_red = {k: v for k, v in weather_windows.items() if k[3] in medoid_ids}
    cd_red = {k: v for k, v in downtime_costs.items() if k[2] in medoid_ids}
    f_red = {k: v for k, v in failures.items() if k[3] in medoid_ids}

    from config.patterns import gen_patterns
    K_S, K_M, P = gen_patterns(ww_red, case, medoid_ids)

    reduced_cfg = ScenarioConfig.__new__(ScenarioConfig)
    reduced_cfg.case = case
    reduced_cfg.scenarios = train_seeds
    reduced_cfg.scenario_reduction = True
    reduced_cfg.K_S = K_S
    reduced_cfg.K_M = K_M
    reduced_cfg.P = P
    reduced_cfg.C_D = cd_red
    reduced_cfg.F = f_red
    reduced_cfg.S = medoid_ids
    reduced_cfg.scenario_weights = {s: weights[s] for s in medoid_ids}

    return reduced_cfg


# ──────────────────────────────────────────────
# Test 1: Out-of-sample gap
# ──────────────────────────────────────────────

def test_oos_gap(case, method, train_seeds, eval_seeds, K_values, mip_gap=0.01):
    results = []

    print(f"\n{'='*60}")
    print(f"Generating training scenario data for {len(train_seeds)} scenarios...")
    print(f"{'='*60}")
    train_data = generate_scenario_data(case, train_seeds)

    X_full, X_scaled, _ = build_feature_matrix(
        case,
        train_seeds,
        train_data["weather_windows"],
        train_data["downtime_costs"],
        train_data["failures"],
    )

    # Benchmark: solve on full training set
    print(f"\n{'='*60}")
    print(f"BENCHMARK: Solving on full training set ({len(train_seeds)} scenarios)...")
    print(f"{'='*60}")

    bench_cfg = ScenarioConfig(case, train_seeds, scenario_reduction=False)
    bench_model, bench_runtime = solve_model(case, bench_cfg, list(train_seeds), method, mip_gap)
    bench_decisions = extract_first_stage(bench_model)
    bench_iss = bench_model.model.ObjVal

    print(f"  ISS objective: {bench_iss:,.0f}")
    print(f"  Decisions: {describe_first_stage(bench_decisions)}")
    print(f"  Runtime: {bench_runtime:.1f}s")

    print(f"  Evaluating OOS on {len(eval_seeds)} scenarios...")
    bench_oos = evaluate_oos(case, bench_decisions, eval_seeds, mip_gap)
    print(f"  OOS objective: {bench_oos:,.0f}")

    results.append({
        "method": "benchmark",
        "K": len(train_seeds),
        "N_train": len(train_seeds),
        "ISS": bench_iss,
        "MipGAP": bench_model.model.MIPGap,
        "SolverStatus": bench_model.model.Status,
        "OOS": bench_oos,
        "gap_pct": 0.0,
        "runtime": bench_runtime,
        "decisions": describe_first_stage(bench_decisions),
        "same_as_benchmark": True,
    })

    # Reduced cases
    for K in K_values:
        print(f"\n{'='*60}")
        print(f"K={K}: Reducing {len(train_seeds)} → {K} scenarios...")
        print(f"{'='*60}")

        cluster_result = run_kmedoids_from_features(
            X_scaled=X_scaled,
            scenario_ids=train_seeds,
            K=K,
            random_state=42,
        )

        medoid_ids = cluster_result["medoid_ids"]
        weights = cluster_result["weights"]

        print(f"  Medoids: {medoid_ids}")
        print(f"  Weights: {weights}")

        reduced_cfg = build_reduced_scenario_config(
            case=case,
            train_seeds=train_seeds,
            medoid_ids=medoid_ids,
            weights=weights,
            weather_windows=train_data["weather_windows"],
            downtime_costs=train_data["downtime_costs"],
            failures=train_data["failures"],
        )

        red_model, red_runtime = solve_model(case, reduced_cfg, medoid_ids, method, mip_gap)
        red_decisions = extract_first_stage(red_model)
        red_iss = red_model.model.ObjVal

        print(f"  ISS objective (K={K}): {red_iss:,.0f}")
        print(f"  Decisions: {describe_first_stage(red_decisions)}")
        print(f"  Runtime: {red_runtime:.1f}s")

        print(f"  Evaluating OOS on {len(eval_seeds)} scenarios...")
        red_oos = evaluate_oos(case, red_decisions, eval_seeds, mip_gap)
        gap = (red_oos - bench_oos) / bench_oos * 100

        print(f"  OOS objective: {red_oos:,.0f}")
        print(f"  Gap vs benchmark: {gap:+.2f}%")

        same_decisions = (
            describe_first_stage(red_decisions) == describe_first_stage(bench_decisions)
        )

        results.append({
            "method": f"reduced_K{K}",
            "K": K,
            "N_train": len(train_seeds),
            "ISS": red_iss,
            "MipGAP": red_model.model.MIPGap,
            "SolverStatus": red_model.model.Status,
            "OOS": red_oos,
            "gap_pct": gap,
            "runtime": red_runtime,
            "decisions": describe_first_stage(red_decisions),
            "same_as_benchmark": same_decisions,
        })

    return results, bench_oos


# ──────────────────────────────────────────────
# Test 2: Feature distribution preservation
# ──────────────────────────────────────────────

def test_feature_distribution(case, train_seeds, K_values):
    print(f"\n{'='*60}")
    print("FEATURE DISTRIBUTION ANALYSIS")
    print(f"{'='*60}")

    train_data = generate_scenario_data(case, train_seeds)

    X_full, X_scaled, _ = build_feature_matrix(
        case,
        train_seeds,
        train_data["weather_windows"],
        train_data["downtime_costs"],
        train_data["failures"],
    )

    results = {}
    for K in K_values:
        cluster_result = run_kmedoids_from_features(
            X_scaled=X_scaled,
            scenario_ids=train_seeds,
            K=K,
            random_state=42,
        )

        medoid_indices = cluster_result["medoid_indices"]
        cluster_weights = np.array(
            [cluster_result["weights"][int(train_seeds[idx])] for idx in medoid_indices],
            dtype=float,
        )

        X_medoids = X_full[medoid_indices]
        n_features = X_full.shape[1]

        wass_distances = []
        mean_diffs = []
        std_diffs = []

        for f_idx in range(n_features):
            orig_values = X_full[:, f_idx]
            red_values = X_medoids[:, f_idx]

            wass = wasserstein_distance(
                orig_values,
                red_values,
                u_weights=np.ones(len(orig_values)) / len(orig_values),
                v_weights=cluster_weights,
            )
            wass_distances.append(wass)

            orig_mean = np.mean(orig_values)
            red_mean = np.average(red_values, weights=cluster_weights)
            mean_diffs.append(abs(red_mean - orig_mean) / (abs(orig_mean) + 1e-10))

            orig_std = np.std(orig_values)
            red_std = np.sqrt(np.average((red_values - red_mean) ** 2, weights=cluster_weights))
            std_diffs.append(abs(red_std - orig_std) / (abs(orig_std) + 1e-10))

        avg_wass = float(np.mean(wass_distances))
        max_wass = float(np.max(wass_distances))
        avg_mean_diff = float(np.mean(mean_diffs) * 100)
        avg_std_diff = float(np.mean(std_diffs) * 100)

        print(f"\n  K={K}:")
        print(f"    Avg Wasserstein distance:  {avg_wass:.4f}")
        print(f"    Max Wasserstein distance:  {max_wass:.4f}")
        print(f"    Avg mean relative error:   {avg_mean_diff:.2f}%")
        print(f"    Avg std relative error:    {avg_std_diff:.2f}%")

        results[K] = {
            "avg_wasserstein": avg_wass,
            "max_wasserstein": max_wass,
            "avg_mean_error_pct": avg_mean_diff,
            "avg_std_error_pct": avg_std_diff,
        }

    return results


# ──────────────────────────────────────────────
# Test 3: K-medoids stability across seeds
# ──────────────────────────────────────────────

def test_kmedoids_stability(case, train_seeds, K, n_seeds=5):
    print(f"\n{'='*60}")
    print(f"K-MEDOIDS STABILITY (K={K}, {n_seeds} seeds)")
    print(f"{'='*60}")

    train_data = generate_scenario_data(case, train_seeds)

    X_full, X_scaled, _ = build_feature_matrix(
        case,
        train_seeds,
        train_data["weather_windows"],
        train_data["downtime_costs"],
        train_data["failures"],
    )

    all_medoid_sets = []
    all_weight_dicts = []

    for seed in range(n_seeds):
        cluster_result = run_kmedoids_from_features(
            X_scaled=X_scaled,
            scenario_ids=train_seeds,
            K=K,
            random_state=seed,
        )

        medoid_ids = sorted(cluster_result["medoid_ids"])
        weights = {int(k): float(v) for k, v in cluster_result["weights"].items()}

        all_medoid_sets.append(set(medoid_ids))
        all_weight_dicts.append(weights)

        print(f"  Seed {seed}: medoids={medoid_ids}")
        print(f"           weights={[f'{weights[m]:.2f}' for m in medoid_ids]}")

    n = len(all_medoid_sets)
    jaccards = []
    for i in range(n):
        for j in range(i + 1, n):
            intersection = len(all_medoid_sets[i] & all_medoid_sets[j])
            union = len(all_medoid_sets[i] | all_medoid_sets[j])
            jaccards.append(intersection / union if union > 0 else 0.0)

    print(f"\n  Pairwise Jaccard similarity: mean={np.mean(jaccards):.3f}, min={np.min(jaccards):.3f}")

    return {
        "medoid_sets": all_medoid_sets,
        "weights": all_weight_dicts,
        "jaccard_mean": float(np.mean(jaccards)),
        "jaccard_min": float(np.min(jaccards)),
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate scenario reduction quality")
    parser.add_argument("-c", "--case", required=True, help="Case name (e.g., mini, 1W3B)")
    parser.add_argument("-m", "--method", required=True, choices=["mip", "con"], help="Solution method")
    parser.add_argument("-N", "--n_scenarios", type=int, default=100, help="Number of training scenarios to generate")
    parser.add_argument("-K", "--K_values", type=int, nargs="+", default=[4, 6, 8, 10, 12], help="K values to test")
    parser.add_argument("--eval_size", type=int, default=100, help="Number of OOS evaluation scenarios")
    parser.add_argument("--skip_oos", action="store_true", help="Skip OOS gap test (slow)")
    parser.add_argument("--skip_features", action="store_true", help="Skip feature distribution test")
    parser.add_argument("--skip_stability", action="store_true", help="Skip k-medoids stability test")

    args = parser.parse_args()
    case = CaseConfig(f"cases/{args.case}.yaml")

    master_rng = np.random.default_rng(42)
    train_seeds = master_rng.choice(np.arange(1, 1000), size=args.n_scenarios, replace=False)
    eval_seeds = master_rng.choice(np.arange(1001, 10_000), size=args.eval_size, replace=False)

    results_dir = Path("results/evaluation") / args.case / args.method
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Case: {args.case}")
    print(f"Method: {args.method}")
    print(f"Training scenarios: {len(train_seeds)}")
    print(f"Evaluation scenarios: {len(eval_seeds)}")
    print(f"K values to test: {args.K_values}")

    if not args.skip_oos:
        oos_results, bench_oos = test_oos_gap(
            case, args.method, train_seeds, eval_seeds, args.K_values
        )

        oos_path = results_dir / "oos_gap.csv"
        with open(oos_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=oos_results[0].keys())
            writer.writeheader()
            writer.writerows(oos_results)

        print(f"\n{'='*60}")
        print("OOS GAP SUMMARY")
        print(f"{'='*60}")
        print(f"{'Method':<20} {'K':>4} {'ISS':>12} {'OOS':>12} {'Gap':>8} {'Same?':>6}")
        print("-" * 65)
        for r in oos_results:
            same = r.get("same_as_benchmark", "-")
            print(
                f"{r['method']:<20} {r['K']:>4} {r['ISS']:>12,.0f} "
                f"{r['OOS']:>12,.0f} {r['gap_pct']:>+7.2f}% {str(same):>6}"
            )
        print(f"\nSaved to {oos_path}")

    if not args.skip_features:
        feat_results = test_feature_distribution(case, train_seeds, args.K_values)

        feat_path = results_dir / "feature_distribution.csv"
        with open(feat_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["K", "avg_wasserstein", "max_wasserstein", "avg_mean_error_pct", "avg_std_error_pct"])
            for K, r in feat_results.items():
                writer.writerow([
                    K,
                    r["avg_wasserstein"],
                    r["max_wasserstein"],
                    r["avg_mean_error_pct"],
                    r["avg_std_error_pct"],
                ])
        print(f"\nSaved to {feat_path}")

    if not args.skip_stability:
        K_stability = 12
        stab_results = test_kmedoids_stability(case, train_seeds, K_stability, n_seeds=20)

        print(f"\n  Conclusion: Jaccard mean={stab_results['jaccard_mean']:.3f}")
        if stab_results["jaccard_mean"] > 0.7:
            print("  → High stability: medoid selection is robust")
        elif stab_results["jaccard_mean"] > 0.4:
            print("  → Moderate stability: some variation in medoid selection")
        else:
            print("  → Low stability: medoids vary significantly across seeds")

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")