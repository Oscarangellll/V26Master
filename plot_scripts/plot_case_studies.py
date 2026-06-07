import argparse
import sys
from itertools import combinations
from math import factorial, sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from haversine import Unit, haversine
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.fixed_data import data
from plot_scripts.config import PLOT_DIR, TABLE_DIR, colors, FIGWIDTH


plt.rcParams.update({
    "figure.constrained_layout.use": False,
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.markersize": 4,
})


SELECTED_TRIADS = ["BCD", "BCG", "BEG"]
DISPLAY_LABELS = {
    "B": "A",
    "C": "B",
    "D": "C",
    "E": "D",
    "F": "E",
    "G": "F",
}
FALLBACK_WIND_FARMS = {
    "G": {"lat": 53.3, "lon": 1.37, "n_turbines": 60},
    "H": {"lat": 53.99, "lon": 0.48, "n_turbines": 80},
}
MONTH_TO_NUM = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}
VESSEL_MIX_CMAP = LinearSegmentedColormap.from_list(
    "ctv_sov_mix",
    [
        (0.00, "#0072B2"),
        (0.35, "#009E73"),
        (0.65, "#F0E442"),
        (0.88, "#E69F00"),
        (1.00, "#CC0000"),
    ],
)


def _coalition_key(coalition):
    if isinstance(coalition, (list, tuple, set)):
        return "".join(sorted(str(member) for member in coalition))
    return "".join(sorted(str(coalition).strip()))


def _members(coalition):
    return tuple(_coalition_key(coalition))


def _display_member(member):
    return DISPLAY_LABELS.get(str(member), str(member))


def _display_coalition(coalition):
    return "".join(_display_member(member) for member in _members(coalition))


def _all_subsets(players):
    players = tuple(players)
    for r in range(1, len(players) + 1):
        for subset in combinations(players, r):
            yield _coalition_key(subset)


def _parse_solution_group(encoded):
    if not isinstance(encoded, str) or not encoded.strip():
        return []

    entries = []
    for item in encoded.split(";"):
        key, value = item.rsplit(":", 1)
        entries.append((key.split("|"), float(value)))
    return entries


def _charter_summary(row):
    summary = {
        "CTV_ST_vessel_months": 0.0,
        "SOV_ST_vessel_months": 0.0,
        "CTV_LT_vessels": 0.0,
        "SOV_LT_vessels": 0.0,
    }

    for key, value in _parse_solution_group(row.get("gamma_ST", "")):
        vessel_type = key[0]
        summary[f"{vessel_type}_ST_vessel_months"] += value

    for key, value in _parse_solution_group(row.get("gamma_LT", "")):
        vessel_type = key[0]
        summary[f"{vessel_type}_LT_vessels"] += value

    return summary


def _add_vessel_mix_metrics(df):
    df = df.copy()
    df["CTV_LT_vessel_months"] = 12 * df["CTV_LT_vessels"]
    df["SOV_LT_vessel_months"] = 12 * df["SOV_LT_vessels"]
    df["CTV_total_vessel_months"] = (
        df["CTV_ST_vessel_months"] + df["CTV_LT_vessel_months"]
    )
    df["SOV_total_vessel_months"] = (
        df["SOV_ST_vessel_months"] + df["SOV_LT_vessel_months"]
    )
    total = df["CTV_total_vessel_months"] + df["SOV_total_vessel_months"]
    df["SOV_share"] = np.where(total > 0, df["SOV_total_vessel_months"] / total, np.nan)
    return df


def _short_term_monthly_rows(row):
    coalition = _coalition_key(row.get("coalition", ""))
    coalition_size = len(coalition)
    rows = []

    for key, value in _parse_solution_group(row.get("gamma_ST", "")):
        if len(key) < 3:
            continue
        period_label = key[-1]
        if period_label not in MONTH_TO_NUM:
            continue
        rows.append(
            {
                "coalition": coalition,
                "coalition_size": coalition_size,
                "period": MONTH_TO_NUM[period_label],
                "value": value,
            }
        )

    return rows


def _wind_farm_locations():
    locations = {w.name: (w.lat, w.lon) for w in data.wind_farms}
    for name, values in FALLBACK_WIND_FARMS.items():
        locations.setdefault(name, (values["lat"], values["lon"]))
    return locations


def _base_locations():
    return {b.name: (b.lat, b.lon) for b in data.bases}


def _wind_farm_turbines():
    turbines = {w.name: w.n_turbines for w in data.wind_farms}
    for name, values in FALLBACK_WIND_FARMS.items():
        turbines.setdefault(name, values["n_turbines"])
    return turbines


def _distance_features(coalition):
    locations = _wind_farm_locations()
    members = _members(coalition)
    distances = []
    for a, b in combinations(members, 2):
        distances.append(haversine(locations[a], locations[b], unit=Unit.KILOMETERS))

    if not distances:
        return {"avg_pairwise_distance": 0.0, "max_pairwise_distance": 0.0}

    return {
        "avg_pairwise_distance": float(np.mean(distances)),
        "max_pairwise_distance": float(np.max(distances)),
    }


def _turbine_features(coalition):
    turbines = _wind_farm_turbines()
    values = [turbines[member] for member in _members(coalition)]

    return {
        "total_turbines": float(np.sum(values)),
        "turbine_range": float(np.max(values) - np.min(values)),
    }


def _selected_bases(row):
    selected = [
        str(key[0])
        for key, value in _parse_solution_group(row.get("eta", ""))
        if value > 0.5
    ]
    if selected:
        return selected

    bases = row.get("bases", "")
    if isinstance(bases, str) and bases.strip():
        return [base.strip() for base in bases.split(";") if base.strip()]
    return []


def _base_distance_features(row):
    wind_farms = _wind_farm_locations()
    bases = _base_locations()
    selected_bases = [base for base in _selected_bases(row) if base in bases]

    if not selected_bases:
        return {
            "avg_distance_to_selected_base": np.nan,
            "max_distance_to_selected_base": np.nan,
            "range_distance_to_selected_base": np.nan,
        }

    nearest_distances = []
    for member in _members(row.get("coalition", "")):
        if member not in wind_farms:
            continue
        nearest_distances.append(
            min(
                haversine(wind_farms[member], bases[base], unit=Unit.KILOMETERS)
                for base in selected_bases
            )
        )

    if not nearest_distances:
        return {
            "avg_distance_to_selected_base": np.nan,
            "max_distance_to_selected_base": np.nan,
            "range_distance_to_selected_base": np.nan,
        }

    return {
        "avg_distance_to_selected_base": float(np.mean(nearest_distances)),
        "max_distance_to_selected_base": float(np.max(nearest_distances)),
        "range_distance_to_selected_base": float(np.max(nearest_distances) - np.min(nearest_distances)),
    }


def _load_coalition_metrics(path):
    df = pd.read_csv(path)
    df["coalition"] = df["coalition"].map(_coalition_key)

    numeric_cols = [
        "objective",
        "first_stage_cost",
        "second_stage_cost",
        "charter_cost_ST",
        "charter_cost_LT",
        "charter_cost_mob",
        "downtime_cost",
        "travel_cost_S",
        "travel_cost_M",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # If the input contains repeated evaluations of the same coalition, keep the
    # lowest OOS estimate. This avoids double-counting alternative identical rows.
    df = (
        df.sort_values(["coalition", "objective"])
        .drop_duplicates(subset=["coalition"], keep="first")
        .reset_index(drop=True)
    )

    cost = dict(zip(df["coalition"], df["objective"]))
    standalone = {
        coalition: sum(cost[_coalition_key(w)] for w in _members(coalition))
        for coalition in df["coalition"]
        if all(_coalition_key(w) in cost for w in _members(coalition))
    }
    savings = {
        coalition: standalone[coalition] - cost[coalition]
        for coalition in standalone
    }
    synergy = {
        coalition: savings[coalition] / cost[coalition]
        for coalition in savings
        if cost[coalition]
    }

    df["standalone_cost"] = df["coalition"].map(standalone)
    df["savings"] = df["coalition"].map(savings)
    df["synergy"] = df["coalition"].map(synergy)

    charter_rows = []
    for row in df.to_dict("records"):
        charter_rows.append(_charter_summary(row))
    charter_df = pd.DataFrame(charter_rows)
    df = pd.concat([df, charter_df], axis=1)
    df = _add_vessel_mix_metrics(df)

    distance_df = pd.DataFrame([_distance_features(c) for c in df["coalition"]])
    df = pd.concat([df, distance_df], axis=1)
    base_distance_df = pd.DataFrame([_base_distance_features(row) for row in df.to_dict("records")])
    df = pd.concat([df, base_distance_df], axis=1)
    turbine_df = pd.DataFrame([_turbine_features(c) for c in df["coalition"]])
    df = pd.concat([df, turbine_df], axis=1)
    df["coalition_size"] = df["coalition"].str.len()
    df["has_stable_core"] = df["coalition"].map(
        lambda coalition: _has_nonempty_core(_members(coalition), savings)
    )

    return df, cost, savings


def _shapley(players, values):
    players = tuple(players)
    n = len(players)
    result = {p: 0.0 for p in players}

    for p in players:
        others = [q for q in players if q != p]
        for r in range(0, len(others) + 1):
            for subset in combinations(others, r):
                subset_key = _coalition_key(subset)
                with_p_key = _coalition_key(tuple(subset) + (p,))
                weight = factorial(r) * factorial(n - r - 1) / factorial(n)
                result[p] += weight * (values.get(with_p_key, 0.0) - values.get(subset_key, 0.0))

    return result


def _minmax_allocation(players, values, costs):
    players = tuple(players)
    grand = _coalition_key(players)
    total_savings = values.get(grand)
    if total_savings is None:
        return None

    coalitions = [c for c in _all_subsets(players) if c != grand]
    n_players = len(players)
    p_idx = n_players
    slack_start = n_players + 1
    n_vars = n_players + 1 + len(coalitions)

    a_ub = []
    b_ub = []
    for slack_offset, coalition in enumerate(coalitions):
        row = np.zeros(n_vars)
        for idx, player in enumerate(players):
            if player in coalition:
                row[idx] = -1.0
        row[p_idx] = costs.get(coalition, 0.0)
        row[slack_start + slack_offset] = 1.0
        a_ub.append(row)
        b_ub.append(-values.get(coalition, 0.0))

    a_eq = np.zeros((1, n_vars))
    a_eq[0, :n_players] = 1.0
    b_eq = np.array([total_savings])
    bounds = [(None, None)] * (n_players + 1) + [(0.0, None)] * len(coalitions)

    primary_objective = np.zeros(n_vars)
    primary_objective[p_idx] = -1.0
    primary = linprog(
        primary_objective,
        A_ub=np.array(a_ub),
        b_ub=np.array(b_ub),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not primary.success:
        return None

    secondary_objective = np.zeros(n_vars)
    for slack_offset, coalition in enumerate(coalitions):
        coalition_cost = costs.get(coalition, 0.0)
        if coalition_cost:
            secondary_objective[slack_start + slack_offset] = -1.0 / coalition_cost

    fixed_p_row = np.zeros((1, n_vars))
    fixed_p_row[0, p_idx] = 1.0
    secondary_a_eq = np.vstack([a_eq, fixed_p_row])
    secondary_b_eq = np.concatenate([b_eq, np.array([primary.x[p_idx]])])

    secondary = linprog(
        secondary_objective,
        A_ub=np.array(a_ub),
        b_ub=np.array(b_ub),
        A_eq=secondary_a_eq,
        b_eq=secondary_b_eq,
        bounds=bounds,
        method="highs",
    )
    if not secondary.success:
        return None

    return {
        "allocation": {player: secondary.x[idx] for idx, player in enumerate(players)},
        "p": secondary.x[p_idx],
        "normalized_slack": -secondary.fun,
    }


def _has_nonempty_core(players, values, tol=1e-7):
    players = tuple(players)
    if len(players) <= 1:
        return np.nan

    grand = _coalition_key(players)
    total = values.get(grand)
    if total is None:
        return np.nan

    c = np.zeros(len(players))
    a_ub = []
    b_ub = []
    for subset_key in _all_subsets(players):
        if subset_key == grand:
            continue
        row = np.array([-1.0 if player in subset_key else 0.0 for player in players])
        a_ub.append(row)
        b_ub.append(-(values.get(subset_key, 0.0) - tol))

    a_eq = [np.ones(len(players))]
    b_eq = [total]
    result = linprog(
        c,
        A_ub=np.array(a_ub),
        b_ub=np.array(b_ub),
        A_eq=np.array(a_eq),
        b_eq=np.array(b_eq),
        bounds=[(None, None)] * len(players),
        method="highs",
    )
    return bool(result.success)


def _core_vertices(players, values, tol=1e-7):
    players = tuple(players)
    grand = _coalition_key(players)
    total = values.get(grand)
    if total is None or abs(total) < tol:
        return []

    constraints = []
    for subset_key in _all_subsets(players):
        if subset_key == grand:
            continue
        row = np.array([1.0 if p in subset_key else 0.0 for p in players])
        constraints.append((subset_key, row, values.get(subset_key, 0.0)))

    vertices = []
    total_row = np.ones(3)
    for (_, a1, b1), (_, a2, b2) in combinations(constraints, 2):
        A = np.vstack([total_row, a1, a2])
        b = np.array([total, b1, b2])
        if abs(np.linalg.det(A)) < tol:
            continue

        point = np.linalg.solve(A, b)
        if all(np.dot(a, point) >= rhs - tol for _, a, rhs in constraints):
            if not any(np.linalg.norm(point - existing) < 1e-5 for existing in vertices):
                vertices.append(point)

    if not vertices:
        return []

    center = np.mean(vertices, axis=0)
    vertices = sorted(
        vertices,
        key=lambda p: np.arctan2(
            _barycentric_xy(p / total)[1] - _barycentric_xy(center / total)[1],
            _barycentric_xy(p / total)[0] - _barycentric_xy(center / total)[0],
        ),
    )
    return vertices


def _barycentric_xy(weights):
    a, b, c = weights
    return np.array([b + 0.5 * c, (sqrt(3) / 2) * c])


def _xy_to_barycentric(xy):
    x, y = xy
    c = y / (sqrt(3) / 2)
    b = x - 0.5 * c
    a = 1 - b - c
    return np.array([a, b, c])


def _polygon_centroid(points, tol=1e-12):
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return None
    if len(points) < 3:
        return np.mean(points, axis=0)

    x = points[:, 0]
    y = points[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    cross = x * y_next - x_next * y
    area = 0.5 * np.sum(cross)

    if abs(area) < tol:
        return np.mean(points, axis=0)

    cx = np.sum((x + x_next) * cross) / (6 * area)
    cy = np.sum((y + y_next) * cross) / (6 * area)
    return np.array([cx, cy])


def _draw_barycentric(ax, players, values, costs, title):
    players = tuple(players)
    grand = _coalition_key(players)
    total = values.get(grand, 0.0)

    triangle = np.array([
        _barycentric_xy([1, 0, 0]),
        _barycentric_xy([0, 1, 0]),
        _barycentric_xy([0, 0, 1]),
        _barycentric_xy([1, 0, 0]),
    ])
    ax.plot(triangle[:, 0], triangle[:, 1], color="0.25", linewidth=0.8)

    for point, label, ha, va in [
        (triangle[0], _display_member(players[0]), "right", "top"),
        (triangle[1], _display_member(players[1]), "left", "top"),
        (triangle[2], _display_member(players[2]), "center", "bottom"),
    ]:
        ax.text(point[0], point[1], label, ha=ha, va=va)

    if total > 0:
        vertices = _core_vertices(players, values)
        if vertices:
            polygon = np.array([_barycentric_xy(v / total) for v in vertices])
            ax.fill(
                polygon[:, 0],
                polygon[:, 1],
                color=colors.blue,
                alpha=0.20,
                edgecolor=colors.blue,
                linewidth=0.8,
                label="Core",
            )

        if vertices:
            polygon = np.array([_barycentric_xy(v / total) for v in vertices])
            centroid_xy = _polygon_centroid(polygon)
            ax.scatter(
                centroid_xy[0],
                centroid_xy[1],
                color=colors.red,
                marker="x",
                s=24,
                linewidth=1.0,
                label="Core centre",
                zorder=3,
            )

        minmax = _minmax_allocation(players, values, costs)
        if minmax is not None:
            minmax_weights = np.array(
                [minmax["allocation"][p] for p in players]
            ) / total
            minmax_xy = _barycentric_xy(minmax_weights)
            ax.scatter(
                minmax_xy[0],
                minmax_xy[1],
                color=colors.green,
                marker="o",
                s=20,
                label="Minmax allocation",
                zorder=3,
            )

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")


def _write_tables(df, windfarm_df, output_dir):
    Path(TABLE_DIR).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(TABLE_DIR) / "case_study_coalition_metrics.csv", index=False)
    if windfarm_df is not None:
        windfarm_df.to_csv(Path(TABLE_DIR) / "case_study_windfarm_metrics.csv", index=False)


def plot_case_studies(
    coalition_path="results/case_studies/base/coalition_oos.csv",
    windfarm_path="results/case_studies/base/windfarm_oos.csv",
    selected_triads=None,
):
    selected_triads = selected_triads or SELECTED_TRIADS
    plot_root = Path(PLOT_DIR)
    by_size_dir = plot_root / "by size"
    three_wf_dir = plot_root / "3 WFs"
    by_size_dir.mkdir(parents=True, exist_ok=True)
    three_wf_dir.mkdir(parents=True, exist_ok=True)

    df, cost, savings = _load_coalition_metrics(coalition_path)
    if Path(windfarm_path).exists():
        windfarm_df = pd.read_csv(windfarm_path)
    else:
        windfarm_df = None
        print(
            f"[plot warning] {windfarm_path} does not exist. "
            "Operational fairness plot was not regenerated."
        )
    _write_tables(df, windfarm_df, by_size_dir)

    _plot_savings_by_size(df, by_size_dir, normalize_vessel_mix=True)
    _plot_charters_by_size(df, by_size_dir)
    _plot_charters_by_size(df, by_size_dir, normalize_by_wind_farm=True)
    _plot_charters_by_size_total_vs_per_wind_farm(df, by_size_dir)
    _plot_short_term_charters_by_month(df, by_size_dir)
    _plot_distance_synergy(df, selected_triads, three_wf_dir)
    _plot_distance_synergy_all(df, selected_triads, three_wf_dir)
    _plot_base_distance_synergy(df, selected_triads, three_wf_dir)
    _plot_avg_distance_combined_synergy(df, selected_triads, three_wf_dir)
    _plot_turbines_synergy(df, selected_triads, three_wf_dir)
    _plot_turbines_synergy_all(df, selected_triads, three_wf_dir)
    _plot_selected_triad_savings(df, selected_triads, three_wf_dir)
    _plot_barycentric_allocations(selected_triads, savings, cost, three_wf_dir)
    if windfarm_df is not None:
        _plot_operational_fairness(windfarm_df, selected_triads, three_wf_dir)


def _plot_savings_by_size(df, output_dir, normalize_vessel_mix=False):
    fig, axs = plt.subplots(
        1,
        2,
        figsize=(FIGWIDTH / 2.54, 6.4 / 2.54),
        constrained_layout=False,
    )
    scatter_for_colorbar = None
    if normalize_vessel_mix:
        mix_min = float(df["SOV_share"].min())
        mix_max = float(df["SOV_share"].max())
    else:
        mix_min = 0.0
        mix_max = 1.0

    for ax, y_col, ylabel in [
        (axs[0], "synergy", "Synergy [%]"),
        (axs[1], "savings", "Savings [MEUR]"),
    ]:
        plot_df = df.copy()
        y = plot_df[y_col] * 100 if y_col == "synergy" else plot_df[y_col] / 1e6
        jitter = np.linspace(-0.10, 0.10, len(plot_df))
        plot_df["_x"] = plot_df["coalition_size"] + jitter
        plot_df["_y"] = y

        marker_specs = [
            (True, "o", "Stable core"),
            (False, "X", "No stable core"),
        ]
        for has_core, marker, _ in marker_specs:
            group = plot_df[plot_df["has_stable_core"] == has_core]
            if group.empty:
                continue
            scatter = ax.scatter(
                group["_x"],
                group["_y"],
                c=group["SOV_share"],
                cmap=VESSEL_MIX_CMAP,
                vmin=mix_min,
                vmax=mix_max,
                marker=marker,
                alpha=0.95,
                s=18 if has_core else 22,
                edgecolor="0.20",
                linewidth=0.22,
                zorder=3,
            )
            scatter_for_colorbar = scatter

        unknown = plot_df[plot_df["has_stable_core"].isna()]
        if not unknown.empty:
            ax.scatter(
                unknown["_x"],
                unknown["_y"],
                color="0.65",
                alpha=0.50,
                s=11,
                linewidth=0,
                marker=".",
                label="N/A" if ax is axs[0] else None,
            )

        mean = (
            pd.DataFrame({"coalition_size": plot_df["coalition_size"], "value": y})
            .groupby("coalition_size")["value"]
            .mean()
            .reset_index()
        )
        ax.plot(mean["coalition_size"], mean["value"], color=colors.red, linewidth=1.2, label="Mean", zorder=2)
        ax.set_xlabel("Coalition size")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(df["coalition_size"].unique()))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="0.90", linewidth=0.6, zorder=0)

    marker_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="0.20",
            color="0.20",
            markersize=4.2,
            label=label,
        )
        for _, marker, label in marker_specs
    ]
    mean_handle = Line2D([0], [0], color=colors.red, linewidth=1.2, label="Mean")
    fig.legend(
        handles=marker_handles + [mean_handle],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.43, 1.00),
    )
    if scatter_for_colorbar is not None:
        cbar_ax = fig.add_axes([0.895, 0.23, 0.018, 0.50])
        cbar = fig.colorbar(scatter_for_colorbar, cax=cbar_ax)
        cbar.set_label("SOV share", fontsize=7, labelpad=2)
        if normalize_vessel_mix:
            cbar.set_ticks([mix_min, (mix_min + mix_max) / 2, mix_max])
            cbar.set_ticklabels([
                f"{mix_min:.2f}",
                f"{(mix_min + mix_max) / 2:.2f}",
                f"{mix_max:.2f}",
            ])
        else:
            cbar.set_ticks([0, 0.5, 1])
            cbar.set_ticklabels(["CTV", "Mixed", "SOV"])
        cbar.ax.tick_params(labelsize=7, pad=1)

    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.10, right=0.855, wspace=0.34)
    output_name = (
        "case_savings_by_coalition_size_normalized_mix.svg"
        if normalize_vessel_mix
        else "case_savings_by_coalition_size.svg"
    )
    fig.savefig(output_dir / output_name)
    if normalize_vessel_mix:
        fig.savefig(output_dir / "case_savings_by_coalition_size.svg")
    plt.close(fig)


def _plot_charters_by_size(df, output_dir, normalize_by_wind_farm=False):
    plot_df = df.copy()
    plot_df["CTV_LT_vessel_months"] = 12 * plot_df["CTV_LT_vessels"]
    plot_df["SOV_LT_vessel_months"] = 12 * plot_df["SOV_LT_vessels"]

    plot_df["CTV_total_vessel_months"] = (
        plot_df["CTV_ST_vessel_months"] + plot_df["CTV_LT_vessel_months"]
    )
    plot_df["SOV_total_vessel_months"] = (
        plot_df["SOV_ST_vessel_months"] + plot_df["SOV_LT_vessel_months"]
    )
    plot_df["ST_total_vessel_months"] = (
        plot_df["CTV_ST_vessel_months"] + plot_df["SOV_ST_vessel_months"]
    )
    plot_df["LT_total_vessel_months"] = (
        plot_df["CTV_LT_vessel_months"] + plot_df["SOV_LT_vessel_months"]
    )
    plot_df["total_vessel_months_per_wind_farm"] = (
        plot_df["ST_total_vessel_months"] + plot_df["LT_total_vessel_months"]
    ) / plot_df["coalition_size"]

    if normalize_by_wind_farm:
        for column in [
            "CTV_total_vessel_months",
            "SOV_total_vessel_months",
            "ST_total_vessel_months",
            "LT_total_vessel_months",
        ]:
            plot_df[column] = plot_df[column] / plot_df["coalition_size"]

    mean = plot_df.groupby("coalition_size")[
        [
            "CTV_total_vessel_months",
            "SOV_total_vessel_months",
            "ST_total_vessel_months",
            "LT_total_vessel_months",
            "total_vessel_months_per_wind_farm",
        ]
    ].mean()
    x_positions = np.arange(len(mean.index))

    fig, axs = plt.subplots(1, 2, figsize=(FIGWIDTH / 2.54, 7.8 / 2.54), constrained_layout=False)

    def add_per_wind_farm_markers(ax):
        if normalize_by_wind_farm:
            return
        for idx, value in enumerate(mean["total_vessel_months_per_wind_farm"]):
            ax.plot(
                [x_positions[idx] - 0.28, x_positions[idx] + 0.28],
                [value, value],
                color="0.05",
                linewidth=1.2,
                solid_capstyle="butt",
                label="Total per wind farm" if idx == 0 else None,
                zorder=4,
            )

    vessel_mix = mean[["CTV_total_vessel_months", "SOV_total_vessel_months"]]
    vessel_mix.columns = ["CTV", "SOV"]
    vessel_mix.plot(
        kind="bar",
        stacked=True,
        ax=axs[0],
        color=[colors.blue, colors.red],
        width=0.72,
        zorder=2,
    )
    add_per_wind_farm_markers(axs[0])
    axs[0].set_title("Vessel type")
    axs[0].set_xlabel("Coalition size")
    ylabel = "Average vessel-months per wind farm" if normalize_by_wind_farm else "Average vessel-months"
    axs[0].set_ylabel(ylabel)
    axs[0].set_axisbelow(True)
    axs[0].grid(axis="y", color="0.90", linewidth=0.6, zorder=0)
    axs[0].legend(frameon=False)

    charter_type = mean[["ST_total_vessel_months", "LT_total_vessel_months"]]
    charter_type.columns = ["ST", "LT"]
    charter_type.plot(
        kind="bar",
        stacked=True,
        ax=axs[1],
        color=[colors.green, colors.purple],
        width=0.72,
        zorder=2,
    )
    add_per_wind_farm_markers(axs[1])
    axs[1].set_title("Charter duration")
    axs[1].set_xlabel("Coalition size")
    axs[1].set_ylabel(ylabel)
    axs[1].set_axisbelow(True)
    axs[1].grid(axis="y", color="0.90", linewidth=0.6, zorder=0)
    axs[1].legend(frameon=False)

    output_name = (
        "case_charters_by_coalition_size_per_wind_farm.svg"
        if normalize_by_wind_farm
        else "case_charters_by_coalition_size.svg"
    )
    fig.savefig(output_dir / output_name)
    plt.close(fig)


def _plot_charters_by_size_total_vs_per_wind_farm(df, output_dir):
    plot_df = df.copy()
    plot_df["CTV_LT_vessel_months"] = 12 * plot_df["CTV_LT_vessels"]
    plot_df["SOV_LT_vessel_months"] = 12 * plot_df["SOV_LT_vessels"]

    plot_df["CTV_total_vessel_months"] = (
        plot_df["CTV_ST_vessel_months"] + plot_df["CTV_LT_vessel_months"]
    )
    plot_df["SOV_total_vessel_months"] = (
        plot_df["SOV_ST_vessel_months"] + plot_df["SOV_LT_vessel_months"]
    )
    plot_df["ST_total_vessel_months"] = (
        plot_df["CTV_ST_vessel_months"] + plot_df["SOV_ST_vessel_months"]
    )
    plot_df["LT_total_vessel_months"] = (
        plot_df["CTV_LT_vessel_months"] + plot_df["SOV_LT_vessel_months"]
    )

    for column in [
        "CTV_total_vessel_months",
        "SOV_total_vessel_months",
        "ST_total_vessel_months",
        "LT_total_vessel_months",
    ]:
        plot_df[f"{column}_per_wind_farm"] = plot_df[column] / plot_df["coalition_size"]

    mean = plot_df.groupby("coalition_size")[
        [
            "CTV_total_vessel_months",
            "SOV_total_vessel_months",
            "ST_total_vessel_months",
            "LT_total_vessel_months",
            "CTV_total_vessel_months_per_wind_farm",
            "SOV_total_vessel_months_per_wind_farm",
            "ST_total_vessel_months_per_wind_farm",
            "LT_total_vessel_months_per_wind_farm",
        ]
    ].mean()

    fig, axs = plt.subplots(1, 2, figsize=(FIGWIDTH / 2.54, 6 / 2.54), constrained_layout=True)
    x = np.arange(len(mean.index))
    width = 0.34

    panel_specs = [
        (
            axs[0],
            ("CTV_total_vessel_months", "SOV_total_vessel_months"),
            ("CTV_total_vessel_months_per_wind_farm", "SOV_total_vessel_months_per_wind_farm"),
            ("CTV", "SOV"),
            (colors.blue, colors.red),
            "Vessel type",
        ),
        (
            axs[1],
            ("ST_total_vessel_months", "LT_total_vessel_months"),
            ("ST_total_vessel_months_per_wind_farm", "LT_total_vessel_months_per_wind_farm"),
            ("ST", "LT"),
            (colors.green, colors.purple),
            "Charter duration",
        ),
    ]

    for ax, total_cols, normalized_cols, labels, panel_colors, title in panel_specs:
        total_bottom = np.zeros(len(mean))
        normalized_bottom = np.zeros(len(mean))
        for total_col, normalized_col, color in zip(total_cols, normalized_cols, panel_colors):
            ax.bar(
                x - width / 2,
                mean[total_col],
                width,
                bottom=total_bottom,
                color=color,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.4,
                zorder=2,
            )
            ax.bar(
                x + width / 2,
                mean[normalized_col],
                width,
                bottom=normalized_bottom,
                color=color,
                alpha=0.45,
                hatch="//",
                edgecolor="0.25",
                linewidth=0.35,
                zorder=2,
            )
            total_bottom += mean[total_col].to_numpy()
            normalized_bottom += mean[normalized_col].to_numpy()

        ax.set_title(title)
        ax.set_xlabel("Coalition size")
        ax.set_ylabel("Average vessel-months")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(size)) for size in mean.index])
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="0.90", linewidth=0.6, zorder=0)
        ax.legend(
            handles=[
                Patch(facecolor=color, edgecolor="none", label=label)
                for color, label in zip(panel_colors, labels)
            ],
            loc="upper left",
            ncol=2,
            frameon=False,
            handlelength=1.1,
            columnspacing=0.9,
        )

    scale_handles = [
        Patch(facecolor="0.45", edgecolor="white", alpha=0.85, label="Total"),
        Patch(facecolor="0.45", edgecolor="0.25", alpha=0.45, hatch="//", label="Per wind farm"),
    ]
    fig.legend(
        handles=scale_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
        handlelength=1.2,
        columnspacing=1.4,
    )
    fig.set_layout_engine(None)
    fig.subplots_adjust(top=0.82, bottom=0.15, left=0.10, right=0.98, wspace=0.28)

    fig.savefig(output_dir / "case_charters_by_coalition_size_total_vs_per_wind_farm.svg")
    plt.close(fig)


def _plot_short_term_charters_by_month(df, output_dir):
    rows = []
    for row in df.to_dict("records"):
        rows.extend(_short_term_monthly_rows(row))

    if not rows:
        return

    parsed = pd.DataFrame(rows)
    coalition_months = pd.MultiIndex.from_product(
        [df["coalition"], range(1, 13)],
        names=["coalition", "period"],
    ).to_frame(index=False)
    coalition_months = coalition_months.merge(
        df[["coalition", "coalition_size"]],
        on="coalition",
        how="left",
    )
    monthly = (
        parsed.groupby(["coalition", "period"], as_index=False)["value"]
        .sum()
        .merge(coalition_months, on=["coalition", "period"], how="right")
    )
    monthly["value"] = monthly["value"].fillna(0.0)
    monthly["value_per_wind_farm"] = monthly["value"] / monthly["coalition_size"]
    monthly = (
        monthly.groupby(["coalition_size", "period"], as_index=False)["value_per_wind_farm"]
        .mean()
        .rename(columns={"value_per_wind_farm": "value"})
    )

    fig, ax = plt.subplots(figsize=(FIGWIDTH / 2.54, 6.5 / 2.54), constrained_layout=False)
    size_colors = {
        1: colors.cyan,
        2: colors.blue,
        3: colors.red,
        4: colors.green,
        5: colors.orange,
        6: colors.purple,
    }
    for size in sorted(monthly["coalition_size"].unique()):
        group = monthly[monthly["coalition_size"] == size].sort_values("period")
        ax.plot(
            group["period"],
            group["value"],
            color=size_colors.get(int(size), colors.cyan),
            marker="o",
            linewidth=1.4,
            label=f"{int(size)} wind farms",
        )

    ax.set_xlabel("Month")
    ax.set_ylabel("Average ST chartered vessels\nper wind farm")
    ax.set_xticks(range(1, 13))
    ax.grid(color="0.90", linewidth=0.6)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        frameon=False,
        ncol=3,
    )
    fig.subplots_adjust(top=0.78, bottom=0.18, left=0.12, right=0.98)

    fig.savefig(output_dir / "case_ST_charters_per_wind_farm_by_month.svg")
    fig.savefig(output_dir / "case_short_term_charters_per_wind_farm_by_month.svg")
    plt.close(fig)


def _plot_enriched_synergy_panels(
    plot_df,
    panel_specs,
    selected_triads,
    output_path,
    normalize_vessel_mix=False,
):
    plot_df = plot_df.dropna(subset=["synergy", "SOV_share"]).copy()
    if plot_df.empty:
        return

    fig, axs = plt.subplots(
        1,
        len(panel_specs),
        figsize=(FIGWIDTH / 2.54, 6.4 / 2.54),
        constrained_layout=False,
    )
    if len(panel_specs) == 1:
        axs = [axs]

    marker_specs = [
        (True, "o", "Stable core"),
        (False, "X", "No stable core"),
    ]
    scatter_for_colorbar = None
    if normalize_vessel_mix:
        mix_min = float(plot_df["SOV_share"].min())
        mix_max = float(plot_df["SOV_share"].max())
    else:
        mix_min = 0.0
        mix_max = 1.0

    for ax, (x_col, xlabel) in zip(axs, panel_specs):
        for has_core, marker, _ in marker_specs:
            group = plot_df[plot_df["has_stable_core"] == has_core]
            if group.empty:
                continue
            scatter = ax.scatter(
                group[x_col],
                group["synergy"] * 100,
                c=group["SOV_share"],
                cmap=VESSEL_MIX_CMAP,
                vmin=mix_min,
                vmax=mix_max,
                marker=marker,
                s=24 if has_core else 28,
                alpha=0.82,
                edgecolor="0.20",
                linewidth=0.25,
                zorder=3,
            )
            scatter_for_colorbar = scatter

        ax.set_xlabel(xlabel)
        ax.set_ylabel("")
        ax.set_axisbelow(True)
        ax.grid(color="0.90", linewidth=0.6, zorder=0)

    marker_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="0.20",
            color="0.20",
            markersize=4.4,
            label=label,
        )
        for _, marker, label in marker_specs
    ]
    fig.legend(
        handles=marker_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.43, 1.00),
    )
    if scatter_for_colorbar is not None:
        cbar_ax = fig.add_axes([0.895, 0.23, 0.018, 0.50])
        cbar = fig.colorbar(scatter_for_colorbar, cax=cbar_ax)
        cbar.set_label("SOV share", fontsize=7, labelpad=2)
        if normalize_vessel_mix:
            cbar.set_ticks([mix_min, (mix_min + mix_max) / 2, mix_max])
            cbar.set_ticklabels([
                f"{mix_min:.2f}",
                f"{(mix_min + mix_max) / 2:.2f}",
                f"{mix_max:.2f}",
            ])
        else:
            cbar.set_ticks([0, 0.5, 1])
            cbar.set_ticklabels(["CTV", "Mixed", "SOV"])
        cbar.ax.tick_params(labelsize=7, pad=1)

    fig.supylabel("Synergy [%]", x=0.02, fontsize=8)
    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.08, right=0.855, wspace=0.34)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_distance_synergy(df, selected_triads, output_dir):
    triads = df[df["coalition_size"] == 3].copy()
    if triads.empty:
        return

    _plot_enriched_synergy_panels(
        triads,
        [
            ("avg_pairwise_distance", "Average pairwise distance [km]"),
            ("max_pairwise_distance", "Maximum pairwise distance [km]"),
        ],
        selected_triads,
        output_dir / "case_distance_vs_synergy.svg",
        normalize_vessel_mix=True,
    )


def _plot_distance_synergy_all(df, selected_triads, output_dir):
    plot_df = df[df["coalition_size"] >= 2].copy()
    if plot_df.empty:
        return

    with plt.rc_context({"figure.constrained_layout.use": False}):
        fig, axs = plt.subplots(1, 2, figsize=(FIGWIDTH / 2.54, 6 / 2.54))
    size_colors = {
        2: colors.blue,
        3: colors.red,
        4: colors.green,
        5: colors.orange,
        6: colors.purple,
    }
    for ax, x_col, xlabel in [
        (axs[0], "avg_pairwise_distance", "Average pairwise distance [km]"),
        (axs[1], "max_pairwise_distance", "Maximum pairwise distance [km]"),
    ]:
        for size in sorted(plot_df["coalition_size"].unique()):
            group = plot_df[plot_df["coalition_size"] == size]
            color = size_colors.get(int(size), colors.cyan)
            ax.scatter(
                group[x_col],
                group["synergy"] * 100,
                color=color,
                s=22,
                alpha=0.75,
                label=f"{int(size)} wind farms",
            )
            if ax is axs[0] and len(group) >= 2 and group[x_col].nunique() >= 2:
                x = group[x_col].to_numpy()
                y = (group["synergy"] * 100).to_numpy()
                coef = np.polyfit(x, y, 1)
                x_line = np.linspace(x.min(), x.max(), 50)
                ax.plot(x_line, np.polyval(coef, x_line), color=color, linewidth=1.2, alpha=0.9)
        for row in plot_df.to_dict("records"):
            if row["coalition"] in selected_triads:
                ax.annotate(_display_coalition(row["coalition"]), (row[x_col], row["synergy"] * 100), fontsize=7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Synergy [%]")
        ax.grid(color="0.90", linewidth=0.6)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(handles), 5), frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.subplots_adjust(top=0.82)
    fig.savefig(output_dir / "case_distance_vs_synergy_all.svg")
    plt.close(fig)


def _plot_base_distance_synergy(df, selected_triads, output_dir):
    triads = df[df["coalition_size"] == 3].copy()
    triads = triads.dropna(
        subset=[
            "avg_distance_to_selected_base",
            "max_distance_to_selected_base",
            "synergy",
        ]
    )
    if triads.empty:
        return

    _plot_enriched_synergy_panels(
        triads,
        [
            ("avg_distance_to_selected_base", "Average distance to selected base [km]"),
            ("max_distance_to_selected_base", "Maximum distance to selected base [km]"),
        ],
        selected_triads,
        output_dir / "case_base_distance_vs_synergy.svg",
        normalize_vessel_mix=True,
    )


def _plot_avg_distance_combined_synergy(df, selected_triads, output_dir):
    triads = df[df["coalition_size"] == 3].copy()
    triads = triads.dropna(
        subset=[
            "avg_pairwise_distance",
            "avg_distance_to_selected_base",
            "synergy",
            "SOV_share",
        ]
    )
    if triads.empty:
        return

    _plot_enriched_synergy_panels(
        triads,
        [
            ("avg_pairwise_distance", "Average pairwise distance [km]"),
            ("avg_distance_to_selected_base", "Average distance to selected base [km]"),
        ],
        selected_triads,
        output_dir / "case_avg_distance_combined_vs_synergy.svg",
        normalize_vessel_mix=True,
    )


def _plot_turbines_synergy(df, selected_triads, output_dir):
    triads = df[df["coalition_size"] == 3].copy()
    if triads.empty:
        return

    _plot_enriched_synergy_panels(
        triads,
        [
            ("total_turbines", "Total number of turbines"),
            ("turbine_range", "Turbine range within coalition"),
        ],
        selected_triads,
        output_dir / "case_turbines_vs_synergy.svg",
        normalize_vessel_mix=True,
    )


def _plot_turbines_synergy_all(df, selected_triads, output_dir):
    plot_df = df[df["coalition_size"] >= 2].copy()
    if plot_df.empty:
        return

    with plt.rc_context({"figure.constrained_layout.use": False}):
        fig, axs = plt.subplots(1, 2, figsize=(FIGWIDTH / 2.54, 6 / 2.54))
    size_colors = {
        2: colors.blue,
        3: colors.red,
        4: colors.green,
        5: colors.orange,
        6: colors.purple,
    }
    for ax, x_col, xlabel in [
        (axs[0], "total_turbines", "Total number of turbines"),
        (axs[1], "turbine_range", "Turbine range within coalition"),
    ]:
        for size in sorted(plot_df["coalition_size"].unique()):
            group = plot_df[plot_df["coalition_size"] == size]
            color = size_colors.get(int(size), colors.cyan)
            ax.scatter(
                group[x_col],
                group["synergy"] * 100,
                color=color,
                s=22,
                alpha=0.75,
                label=f"{int(size)} wind farms",
            )
            if ax is axs[0] and len(group) >= 2 and group[x_col].nunique() >= 2:
                x = group[x_col].to_numpy()
                y = (group["synergy"] * 100).to_numpy()
                coef = np.polyfit(x, y, 1)
                x_line = np.linspace(x.min(), x.max(), 50)
                ax.plot(x_line, np.polyval(coef, x_line), color=color, linewidth=1.2, alpha=0.9)
        for row in plot_df.to_dict("records"):
            if row["coalition"] in selected_triads:
                ax.annotate(_display_coalition(row["coalition"]), (row[x_col], row["synergy"] * 100), fontsize=7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Synergy [%]")
        ax.grid(color="0.90", linewidth=0.6)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(handles), 5), frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.subplots_adjust(top=0.82)
    fig.savefig(output_dir / "case_turbines_vs_synergy_all.svg")
    plt.close(fig)


def _plot_selected_triad_savings(df, selected_triads, output_dir):
    plot_df = df[df["coalition"].isin(selected_triads)].copy()
    if plot_df.empty:
        return

    plot_df = plot_df.set_index("coalition").reindex(selected_triads).dropna(subset=["objective"])
    fig, ax = plt.subplots(figsize=(FIGWIDTH / 2.54, 5 / 2.54), constrained_layout=True)
    ax.bar([_display_coalition(c) for c in plot_df.index], plot_df["synergy"] * 100, color=colors.blue, alpha=0.85)
    ax.set_ylabel("Synergy [%]")
    ax.set_xlabel("Selected 3-player coalition")
    ax.grid(axis="y", color="0.90", linewidth=0.6)

    fig.savefig(output_dir / "case_selected_triad_savings.svg")
    plt.close(fig)


def _plot_barycentric_allocations(selected_triads, savings, cost, output_dir):
    triads = [t for t in selected_triads if len(t) == 3 and t in savings]
    if not triads:
        return

    fig, axs = plt.subplots(
        1,
        len(triads),
        figsize=(FIGWIDTH / 2.54, 5.3 / 2.54),
        constrained_layout=False,
    )
    if len(triads) == 1:
        axs = [axs]

    for ax, triad in zip(axs, triads):
        _draw_barycentric(ax, _members(triad), savings, cost, _display_coalition(triad))

    handles, labels = axs[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.99),
        )
        fig.subplots_adjust(top=0.78, wspace=0.24)

    fig.savefig(
        output_dir / "case_barycentric_allocations.svg",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


def _plot_operational_fairness(windfarm_df, selected_triads, output_dir):
    windfarm_df["coalition"] = windfarm_df["coalition"].map(_coalition_key)
    selected = windfarm_df[windfarm_df["coalition"].isin(selected_triads)].copy()
    if selected.empty:
        return

    metrics = [
        ("value_based_availability", "Value-based availability [%]"),
        ("time_based_availability", "Time-based availability [%]"),
    ]
    fig, axs = plt.subplots(1, 2, figsize=(FIGWIDTH / 2.54, 5.5 / 2.54), constrained_layout=False)
    shared_legend_handles = {}

    for ax, (metric, ylabel) in zip(axs, metrics):
        if metric not in selected.columns:
            continue

        metric_df = selected[["coalition", "wind_farm", metric]].copy()
        metric_df[metric] = pd.to_numeric(metric_df[metric], errors="coerce") * 100
        metric_df = metric_df.dropna(subset=[metric])

        x_centers = np.arange(len(selected_triads))
        bar_width = 0.18
        legend_handles = {}
        windfarm_colors = {
            "B": colors.blue,
            "C": colors.red,
            "D": colors.green,
            "E": colors.orange,
            "F": colors.cyan,
            "G": colors.purple,
        }
        for x_idx, coalition in enumerate(selected_triads):
            group = metric_df[metric_df["coalition"] == coalition].sort_values("wind_farm")
            if group.empty:
                continue

            offsets = (np.arange(len(group)) - (len(group) - 1) / 2) * bar_width
            for offset, row in zip(offsets, group.to_dict("records")):
                wind_farm = row["wind_farm"]
                color = windfarm_colors.get(str(wind_farm), colors.blue)
                bars = ax.bar(
                    x_centers[x_idx] + offset,
                    row[metric],
                    width=bar_width,
                    color=color,
                    label=_display_member(wind_farm) if wind_farm not in legend_handles else None,
                )
                legend_handles.setdefault(wind_farm, bars[0])
                shared_legend_handles.setdefault(wind_farm, bars[0])

        ax.set_xticks(x_centers)
        ax.set_xticklabels([_display_coalition(c) for c in selected_triads])
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

    fig.savefig(output_dir / "case_operational_fairness_selected_triads.svg")
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(description="Plot case-study results.")
    parser.add_argument(
        "--coalition-path",
        default="results/case_studies/base/coalition_oos.csv",
    )
    parser.add_argument(
        "--windfarm-path",
        default="results/case_studies/base/windfarm_oos.csv",
    )
    parser.add_argument("--triads", "--coalitions", nargs="+", default=SELECTED_TRIADS)
    return parser


def main():
    args = build_parser().parse_args()
    plot_case_studies(args.coalition_path, args.windfarm_path, args.triads)


if __name__ == "__main__":
    main()
