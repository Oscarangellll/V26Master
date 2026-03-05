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
from sklearn.preprocessing import StandardScaler


def build_feature_vector(case, scenario_id, weather_windows, downtime_costs, failures):
    """
    Build a single feature vector for one scenario.
    
    Features: period aggregates of:
        - operable hours per vessel type per park (mean, std, q10)
        - downtime cost per park (mean)
        - failures per park per maintenance type (mean)
    
    Returns: 1D array of concatenated features
    """
    features = []
    
    # Use case.D_t to get {period_name: [days_in_period]}
    periods = case.D_t
    
    # 1. Weather windows: operable hours per (vessel_type, park)
    for h in case.vessel_types:
        for w in case.wind_farms:
            period_windows = []
            for period_name, days_in_period in periods.items():
                windows_this_period = [
                    weather_windows.get((h.name, w.name, d, scenario_id), 0)
                    for d in days_in_period
                ]
                period_windows.append(np.mean(windows_this_period))
            
            # Features: mean, std, q10 across periods
            features.append(np.mean(period_windows))
            features.append(np.std(period_windows))
            features.append(np.quantile(period_windows, 0.1))
    
    # 2. Downtime costs per park
    for w in case.wind_farms:
        period_costs = []
        for period_name, days_in_period in periods.items():
            costs_this_period = [
                downtime_costs.get((w.name, d, scenario_id), 0)
                for d in days_in_period
            ]
            period_costs.append(np.mean(costs_this_period))
        
        features.append(np.mean(period_costs))
        features.append(np.std(period_costs))
    
    # 3. Failures per park per maintenance type
    for w in case.wind_farms:
        for m in case.maintenance_categories:
            period_failures = []
            for period_name, days_in_period in periods.items():
                failures_this_period = [
                    failures.get((w.name, m.name, d, scenario_id), 0)
                    for d in days_in_period
                ]
                period_failures.append(np.mean(failures_this_period))
            
            features.append(np.mean(period_failures))
    
    return np.array(features)


def perform_scenario_reduction(
    case, 
    scenario_ids, 
    weather_windows, 
    downtime_costs, 
    failures, 
    n_reduced_scenarios):
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
        feat = build_feature_vector(case, s_id, weather_windows, downtime_costs, failures)
        features_list.append(feat)
    
    X = np.array(features_list)  # shape: (n_scenarios, n_features)
    print(f"Feature matrix shape: {X.shape}")
    
    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-medoids clustering using pure NumPy/SciPy
    print(f"Running k-medoids with k={n_reduced_scenarios}...")
    medoid_indices = _kmedoids(X_scaled, n_clusters=n_reduced_scenarios, random_state=42)
    labels = _assign_clusters(X_scaled, medoid_indices)
    
    medoid_ids = [scenario_ids[idx] for idx in medoid_indices]
    
    # Compute weights: fraction of scenarios in each cluster
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
    
    return medoid_ids, weights, X_scaled


def _kmedoids(X, n_clusters, max_iter=100, random_state=42):
    """
    Simple k-medoids clustering using pure NumPy/SciPy.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Standardized feature matrix
    n_clusters : int
        Number of clusters
    max_iter : int
        Maximum iterations
    random_state : int
        Random seed
    
    Returns
    -------
    medoid_indices : ndarray of shape (n_clusters,)
        Indices of medoid points in X
    """
    n_samples = X.shape[0]
    rng = np.random.RandomState(random_state)
    
    # Initialize: k-medoids++ init (choose first medoid as furthest from center)
    center = X.mean(axis=0)
    distances_to_center = np.linalg.norm(X - center, axis=1)
    medoid_indices = np.array([np.argmax(distances_to_center)])
    
    # Greedily add remaining medoids (furthest from existing ones)
    while len(medoid_indices) < n_clusters:
        distances = np.zeros(n_samples)
        for i in range(n_samples):
            distances[i] = np.min([np.linalg.norm(X[i] - X[m]) for m in medoid_indices])
        # Probabilistically choose next medoid weighted by distance
        probs = distances / distances.sum()
        next_medoid = rng.choice(n_samples, p=probs)
        medoid_indices = np.append(medoid_indices, next_medoid)
    
    medoid_indices = np.unique(medoid_indices)  # Remove duplicates
    
    # Iterate: assign points and recompute medoids
    for iteration in range(max_iter):
        # Assign each point to nearest medoid
        labels = _assign_clusters(X, medoid_indices)
        
        # Recompute medoids: for each cluster, choose point with min sum-of-distances
        new_medoid_indices = []
        for k in range(n_clusters):
            mask = (labels == k)
            if not np.any(mask):
                # Empty cluster: keep old medoid
                new_medoid_indices.append(medoid_indices[k])
                continue
            
            cluster_points = X[mask]
            # Sum of distances from each point to all others in cluster
            distances = np.zeros(np.sum(mask))
            for i, idx in enumerate(np.where(mask)[0]):
                distances[i] = np.sum(np.linalg.norm(cluster_points - X[idx], axis=1))
            
            # Medoid is point with minimum sum-of-distances
            local_medoid_idx = np.argmin(distances)
            global_medoid_idx = np.where(mask)[0][local_medoid_idx]
            new_medoid_indices.append(global_medoid_idx)
        
        new_medoid_indices = np.array(new_medoid_indices)
        
        # Check convergence
        if np.array_equal(medoid_indices, new_medoid_indices):
            break
        
        medoid_indices = new_medoid_indices
    
    return medoid_indices


def _assign_clusters(X, medoid_indices):
    """
    Assign each point to nearest medoid.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix
    medoid_indices : ndarray
        Indices of medoids
    
    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster assignment for each point
    """
    n_samples = X.shape[0]
    labels = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        distances = np.array([np.linalg.norm(X[i] - X[m]) for m in medoid_indices])
        labels[i] = np.argmin(distances)
    return labels
