from typing import Dict, List, Tuple
import math
from haversine import haversine, Unit

def gen_patterns(weather_windows, case, scenarios):
    mcats = list(case.maintenance_categories)
    m_names = [m.name for m in mcats]
    m_durs = [float(m.duration) for m in mcats]
    n = len(mcats)
    
    patterns: List[Tuple[int, ...]] = []
    durations: List[float] = []
    
    def rec(i: int, rem: float, cur: List[int], dur: float):
        if i == n:
            if any(cur):  # dropp tom pattern
                patterns.append(tuple(cur))
                durations.append(dur)
            return
        
        di = m_durs[i]
        max_c = int(math.floor(rem / di))
        for c in range(max_c + 1):
            cur.append(c)
            rec(i+1, rem - c*di, cur, dur + c*di)
            cur.pop()
    rec(0, float(case.upper_bound_weather_window), [], 0.0)
    
    L: Dict[int, float] = {k: durations[k] for k in range(len(patterns))}
    P: Dict[Tuple[str, int], int] = {}
    for k, counts in enumerate(patterns):
        for name, c in  zip(m_names, counts):
            P[(name, k)] = c
    K: Dict[str, List[int]] = {v.name: [] for v in case.vessel_types}
    for k, counts in enumerate(patterns):
        active_idx = [i for i, c in enumerate(counts) if c]
        for h in case.vessel_types:
            if all(h.name in mcats[i].vessel_types for i in active_idx):
                K[h.name].append(k)
    
    KS_hbwds, KM_hwds = remove_inf_patterns(K, L, weather_windows, case, scenarios)
    KS_hbwds, KM_hwds = remove_dominated_patterns(KS_hbwds, KM_hwds, P, m_names)
    
    return KS_hbwds, KM_hwds, P


def remove_inf_patterns(K, L, weather_windows, case, scenarios):
    KS_hbwds = {(h.name, b.name, w.name, d, s): [] 
        for h in case.vessel_types if not h.multiday
        for b in case.bases
        for w in case.wind_farms
        for d in case.D
        for s in scenarios}
    KM_hwds = {(h.name, w.name, d, s): [] 
        for h in case.vessel_types if h.multiday
        for w in case.wind_farms 
        for d in case.D
        for s in scenarios}
    L_RT = {(h.name, b.name, w.name): 
        0 if h.multiday else 
        2 * haversine((b.lat, b.lon), (w.lat, w.lon), unit=Unit.KILOMETERS) / h.travel_speed
        for h in case.vessel_types
        for b in case.bases
        for w in case.wind_farms}
    # print("L_RT:", L_RT)
    
    for h in case.vessel_types:
        for w in case.wind_farms:
            for d in case.D:
                for s in scenarios:
                    if h.multiday:
                        for k in K[h.name]:
                            if (1 + case.work_friction) * L[k] <= weather_windows[(h.name, w.weather_location_id, d, s)]:
                                KM_hwds[h.name, w.name, d, s].append(k)
                    else:
                        for b in case.bases:
                            for k in K[h.name]:
                                if (1 + case.work_friction) * (L[k] + L_RT[h.name, b.name, w.name]) <= weather_windows[(h.name, w.weather_location_id, d, s)]:
                                    KS_hbwds[h.name, b.name, w.name, d, s].append(k)
    return KS_hbwds, KM_hwds


def remove_dominated_patterns(KS_hbwds, KM_hwds, P, m_names):
    def filter_list(pattern_ids):
        vectors = {k: [P[(m, k)] for m in m_names] for k in pattern_ids}
        kept = []
        for k1 in pattern_ids:
            v1 = vectors[k1]
            dominated = False
            for k2 in pattern_ids:
                if k1 == k2:
                    continue
                v2 = vectors[k2]
                if all(x <= y for x, y in zip(v1, v2)) and any(x < y for x, y in zip(v1, v2)):
                    dominated = True
                    break
            if not dominated:
                kept.append(k1)
        return kept
    
    KS_hbwds = {k: filter_list(ids) for k, ids in KS_hbwds.items()}
    KM_hwds = {k: filter_list(ids) for k, ids in KM_hwds.items()}
                
    return KS_hbwds, KM_hwds