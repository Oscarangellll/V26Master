from pathlib import Path
import yaml
from haversine import haversine, Unit

from data import FixedData

class CaseConfig:

    def __init__(self, case_path, wind_farm_names=None):
        data = FixedData()
        case_path = Path(case_path)
        
        with case_path.open() as f:
            case = yaml.safe_load(f)

        self.name = case_path.relative_to("cases").with_suffix("")
        
        if "vessel_types" in case:
            self.vessel_types = [
                h 
                for h in data.vessel_types 
                if h.name in case["vessel_types"]
            ]
        else:
            self.vessel_types = data.vessel_types

        self.max_multiday_vessels = case["max_multiday_vessels"]

        self.bases = [b for b in data.bases if b.name in case["bases"]]
        
        self.periods = case.get("periods",
            ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        )

        self.days_per_period = case.get("days_per_period", 30)

        # Use wind_farm_names override if provided (for coalition analysis),
        # otherwise use all wind farms from the YAML config
        wf_filter = wind_farm_names if wind_farm_names is not None else case["wind_farms"]
        self.wind_farms = [
            w 
            for w in data.wind_farms 
            if w.name in wf_filter
        ]
        self.coalition = "".join(w.name for w in self.wind_farms)
        
        self.all_wl_ids_for_iso = {
            iso: list({w.weather_location_id for w in data.wind_farms if w.iso == iso})
            for iso in list({w.iso for w in self.wind_farms})
        }
        
        if "maintenance_categories" in case:
            self.maintenance_categories = [
                m 
                for m in data.maintenance_categories 
                if m.name in case["maintenance_categories"]
            ]
        else:
            self.maintenance_categories = data.maintenance_categories

        self.power_curve = data.power_curve
        
        self.upper_bound_weather_window = data.upper_bound_weather_window
        
        self.n_vessels_ub = case["n_vessels_ub"]
        
        self.one_base = case["one_base"]

    # First stage sets
    @property
    def H(self):
        return [h.name for h in self.vessel_types]

    @property
    def H_S(self):
        return [h.name for h in self.vessel_types if not h.multiday]

    @property
    def H_M(self):
        return [h.name for h in self.vessel_types if h.multiday]
    
    @property
    def V(self):
        return {
            h: [f"{h}{i + 1}" for i in range(self.max_multiday_vessels)]
            for h in self.H_M
        }

    @property
    def B(self):
        return [b.name for b in self.bases]

    @property
    def T(self):
        return self.periods

    # First stage parameters
    @property
    def C_ST(self):
        return {
            (h.name, t): h.cost_ST(self.days_per_period)
            for h in self.vessel_types
            for t in self.T
        }

    @property
    def C_LT(self):
        return {
            h.name: h.cost_LT(self.days_per_period, len(self.T))
            for h in self.vessel_types
        }
        
    @property
    def C_B(self):
        return {b.name: b.cost for b in self.bases}

    @property
    def K_MAX(self):
        return {b.name: b.capacity for b in self.bases}

    @property
    def K_REQ(self):
        return {h.name: h.required_capacity for h in self.vessel_types}

    # Second stage sets
    @property 
    def W(self):
        return [w.name for w in self.wind_farms]

    @property
    def L(self):
        return self.B + self.W

    @property
    def M(self):
        return [m.name for m in self.maintenance_categories]

    @property
    def D(self):
        return [d + 1 for d in range(self.days_per_period * len(self.T))]

    @property
    def D_t(self):
        return {
            t: self.D[i * self.days_per_period:(i + 1) * self.days_per_period]
            for i, t in enumerate(self.T)
        }

    @property
    def D_T(self):
        return [d for d in self.D if d % self.days_per_period == 1 and d != 1]

    # Second stage parameters
    @property
    def N(self):
        return {h.name: h.n_teams for h in self.vessel_types}

    @property
    def P(self):
        P = {}

        for m in self.maintenance_categories:
            P[m.name, 1] = 2

        return P
    
    @property
    def C_RT(self):
        C_RT = {}

        for h in self.vessel_types:
            for b in self.bases:
                for w in self.wind_farms:
                    C_RT[h.name, b.name, w.name] = 2 * haversine((b.lat, b.lon), (w.lat, w.lon), unit=Unit.KILOMETERS) * h.cost_per_km

        return C_RT

    @property
    def C_T(self):
        C_T = {}
        
        for h in self.vessel_types:
            if h.multiday:
                for i in self.bases + self.wind_farms:
                    for j in self.bases + self.wind_farms:
                        if i != j:
                            C_T[h.name, i.name, j.name] = haversine((i.lat, i.lon), (j.lat, j.lon), unit=Unit.KILOMETERS) * h.cost_per_km

        return C_T

    @property
    def R(self):
        return {h.name: h.periodic_return for h in self.vessel_types if h.multiday}
































