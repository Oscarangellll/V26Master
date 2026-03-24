"""
Scenario reduction via k-medoids clustering on monthly-aggregated features.

Features extracted from:
    - weather_windows: operable hours per (vessel_type, park, day, scenario)
    - downtime_costs: cost per (park, day, scenario)
    - failures: failures per (park, maintenance_type, day, scenario)

Aggregated to monthly level, standardized, and clustered with k-medoids.
Returns medoid scenario indices and their weights.
"""

import numpy as np
import kmedoids


def build_feature_vector(case, scenario_id, weather_windows, downtime_costs, failures, features_setting):
    """
    Build a single feature vector for one scenario.
    
    Features per period (month):
        - weather windows per vessel type per park: mean, std, q10  (|H| × |W| × |T| × 3) 2* 5 * 12 * 3 = 360
        - downtime cost per park: mean, std                         (|W| × |T| × 2) 5 * 12 * 2 = 120
        - failures per park per maintenance type: mean              (|W| × |M| × |T|) 5 * 5 * 12 = 300
        
        360 + 120 + 300 = 780 features total 780 / 12 months = 65 features per month (for interpretability)
    
    Returns: 1D array of concatenated features
    """
    features = []
    
    # Use case.D_t to get {period_name: [days_in_period]}
    periods = case.D_t
    
    if features_setting >= 1:
        #1. Weather windows: amount for each category (short <4h, medium 4-8h, long >8h) per vessel type per park
        for h in case.vessel_types:
            for w in case.wind_farms:
                for period_name, days_in_period in periods.items():
                    short_windows, medium_windows, long_windows = 0, 0, 0
                    for d in days_in_period:
                        hours = weather_windows[scenario_id][(h.name, w.weather_location_id, d)]
                        if hours < 4:
                            short_windows += 1
                        elif hours < 8:
                            medium_windows += 1
                        else:
                            long_windows += 1
                    features.append(short_windows)
                    features.append(medium_windows)
                    features.append(long_windows)
    if features_setting >= 2:
        # 2. Downtime costs: mean, std per period per park
        for w in case.wind_farms:
            for period_name, days_in_period in periods.items():
                costs_this_period = [
                    downtime_costs[scenario_id][(w.name, d)]
                    for d in days_in_period
                ]
                features.append(np.mean(costs_this_period))
                features.append(np.std(costs_this_period))
    if features_setting >= 3:
        # 3. Failures: mean per period per park per maintenance type
        for w in case.wind_farms:
            for m in case.maintenance_categories:
                for period_name, days_in_period in periods.items():
                    failures_this_period = [
                        failures[scenario_id][(w.name, m.name, d)]
                        for d in days_in_period
                    ]
                    # features.append(np.mean(failures_this_period)) #mean
                    features.append(np.sum(failures_this_period)) #sum
    
    return np.array(features)


def perform_scenario_reduction(
    case, 
    scenario_ids, 
    weather_windows, 
    downtime_costs, 
    failures, 
    n_reduced_scenarios,
    features_setting):
    """
    Reduce a large scenario set to a smaller representative set via k-medoids.
    
    Parameters:
    -----------
    case : CaseConfig
        Case configuration with vessel types, parks, maintenance categories, periods
    scenario_ids : list[int]
        All scenario seeds (typically many, e.g., 1000)
    weather_windows : dict
        (vessel_type, park, day, scenario) -> max operable hours
    downtime_costs : dict
        (park, day, scenario) -> cost per unit downtime
    failures : dict
        (park, maintenance_type, day, scenario) -> number of failures
    n_reduced_scenarios : int
        Target number of reduced scenarios (e.g., 12)
    
    Returns:
    --------
    medoid_ids : list[int]
        Scenario IDs of the selected medoids (representatives)
    weights : dict[int, float]
        Weight of each medoid (fraction of original scenarios it represents)
    feature_matrix : ndarray
        Feature matrix used for clustering (for inspection)
    """
    
    # Build feature matrix: one row per scenario
    print(f"Building feature vectors for {len(scenario_ids)} scenarios...")
    features_list = []
    for s_id in scenario_ids:
        feat = build_feature_vector(case, s_id, weather_windows, downtime_costs, failures, features_setting)
        features_list.append(feat)
    
    X = np.array(features_list)  # shape: (n_scenarios, n_features)
    print(f"Feature matrix shape: {X.shape}")
    
    # Standardize features (mean=0, std=1)
    print("Standardizing features...")
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    # Avoid division by zero
    X_std = np.where(X_std == 0, 1, X_std)
    X_scaled = (X - X_mean) / X_std
    
    # K-medoids clustering using pure NumPy/SciPy
    print(f"Running k-medoids with k={n_reduced_scenarios}...")
    # medoid_indices = _kmedoids(X_scaled, n_clusters=n_reduced_scenarios, random_state=42)
    from scipy.spatial.distance import pdist, squareform 
    diss = squareform(pdist(X_scaled, metric='euclidean')) 
    result = kmedoids.fasterpam(diss, medoids=n_reduced_scenarios)
    
    # Result object from kmedoids
    medoid_indices = np.asarray(result.medoids, dtype=int)
    labels = np.asarray(result.labels, dtype=int) 

    medoid_ids = [scenario_ids[idx] for idx in medoid_indices]
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    weights = {}
    for label in range(n_reduced_scenarios):
        if label in unique_labels:
            idx_in_medoids = label
            medoid_id = medoid_ids[idx_in_medoids]
            count = counts[list(unique_labels).index(label)]
            weights[medoid_id] = count / len(scenario_ids)
        else:
            # Medoid with no assigned points (rare)
            weights[medoid_ids[label]] = 0.0
    
    print(f"Clustering complete:")
    print(f"  Medoid IDs: {medoid_ids}")
    print(f"  Weights: {weights}")
    
    return medoid_ids, weights