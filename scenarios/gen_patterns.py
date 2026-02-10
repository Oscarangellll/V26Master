from typing import Dict, List, Tuple, Optional
import math
from dataclasses import dataclass
from gen_windows import find_weather_windows
from haversine import haversine, Unit

def gen_patterns(weather, case, data):
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
    
    rec(0, float(data.max_capacity), [], 0.0)
    
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
    
    KS_hbwds, KM_hwds = remove_inf_patterns(K, L, weather, case) 
    KS_hbwds, KM_hwds = remove_dominated_patterns(KS_hbwds, KM_hwds, P, m_names)
    
    return K, P, KS_hbwds, KM_hwds


def remove_inf_patterns(K, L, weather, case):
    KS_hbwds = {(h.name, b.name, w.name, d, s): [] 
        for h in case.vessel_types if not h.multiday
        for b in case.bases
        for w in case.wind_farms
        for d in case.D
        for s in case.scenarios}
    KM_hwds = {(h.name, w.name, d, s): [] 
        for h in case.vessel_types if h.multiday
        for w in case.wind_farms 
        for d in case.D
        for s in case.scenarios}
    
    L_RT = {(h.name, b.name, w.name): 
        0 if h.multiday else 
        2 * haversine((b.lat, b.lon), (w.lat, w.lon), unit=Unit.KILOMETERS) / h.travel_speed
        for h in case.vessel_types
        for b in case.bases
        for w in case.wind_farms}
    print("L_RT:", L_RT)

    for h in case.vessel_types:
        for w in case.wind_farms:
            for d in case.D:
                for s in case.scenarios:
                    if h.multiday:
                        windows = find_weather_windows(case, weather)[(h.name, w.name, d, s)]
                        for k in K[h.name]:
                            if L[k] <= windows:
                                KM_hwds[h.name, w.name, d, s].append(k)
                    else:
                        windows = find_weather_windows(case, weather)[(h.name, w.name, d, s)]
                        for b in case.bases:
                            for k in K[h.name]:
                                if L[k] + L_RT[h.name, b.name, w.name] <= windows:
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


####################
# Test/Sanity Check:
####################

@dataclass(frozen=True)
class MaintCat:
    name: str
    duration: float
    vessel_types: List[str]

@dataclass(frozen=True)
class VesselType:
    name: str
    multiday: bool
    travel_speed: float

@dataclass(frozen=True)
class Base:
    name: str
    lat: float
    lon: float

@dataclass(frozen=True)
class WindFarm:
    name: str
    lat: float
    lon: float

@dataclass
class Case:
    maintenance_categories: List[MaintCat]
    vessel_types: List[VesselType]
    bases: List[Base]
    wind_farms: List[WindFarm]
    D: List[int]
    scenarios: List[int]

@dataclass
class Data:
    max_capacity: float

# --- Minimal case ---
case = Case(
    maintenance_categories=[
        MaintCat(name="A", duration=2, vessel_types=["SOV"]),
        MaintCat(name="B", duration=3, vessel_types=["SOV"]),
        MaintCat(name="C", duration=1, vessel_types=["CTV", "SOV"]),
    ],
    vessel_types=[
        VesselType(name="CTV", multiday=False, travel_speed=30),
        VesselType(name="SOV", multiday=True, travel_speed=30),
    ],
    bases=[Base(name="B1", lat=54, lon=54)],
    wind_farms=[WindFarm(name="W1", lat=53.9, lon=53.9)],
    D=[0],
    scenarios=[0],
)
data = Data(max_capacity=4)
weather_windows = None  # ikke brukt i gen_patterns nå

# Override weather windows for test
def find_weather_windows(_case, _weather):
    return {
        (h.name, w.name, d, s): 3
        for h in _case.vessel_types
        for w in _case.wind_farms
        for d in _case.D
        for s in _case.scenarios
    }

K, P, _, _ = gen_patterns(weather_windows, case, data)

# finn alle pattern-id-er som faktisk finnes i P
pattern_ids = sorted({k for (_name, k) in P.keys()})
m_dur = {m.name: m.duration for m in case.maintenance_categories}

print("=== K ===")
print(K)

print("\n=== P and L ===")
for k in pattern_ids:
    a = P[("A", k)]
    b = P[("B", k)]
    c = P[("C", k)]
    Lk = a * m_dur["A"] + b * m_dur["B"] + c * m_dur["C"]
    print(f"pattern {k}: A={a}, B={b}, C={c}, L={Lk}")

# Recompute L from P for filtering
L = {k: sum(m_dur[m] * P[(m, k)] for m in m_dur) for k in pattern_ids}
m_names = [m.name for m in case.maintenance_categories]

KS_hbwds, KM_hwds = remove_inf_patterns(K, L, weather_windows, case)
print("\n=== AFTER remove_inf_patterns ===")
print("KS_hbwds:", KS_hbwds)
print("KM_hwds:", KM_hwds)

KS_hbwds, KM_hwds = remove_dominated_patterns(KS_hbwds, KM_hwds, P, m_names)
print("\n=== AFTER remove_dominated_patterns ===")
print("KS_hbwds:", KS_hbwds)
print("KM_hwds:", KM_hwds)
