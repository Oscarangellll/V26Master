import numpy as np
from scenarios.gen_patterns import gen_patterns

class ScenarioConfig:

    def __init__(self, case, weather_model, price_model, scenarios: list[int]):
        self.case = case
        self.scenarios = scenarios
        self.weather = {(s, loc) for s in scenarios for loc in case.locations}
        self.prices = {(s, iso3) for s in scenarios for iso3 in case.ISO_codes.keys()}
        
        for s in scenarios:
            for loc in case.locations:
                self.weather[(s, loc)] = weather_model.simulate(s, loc)
            for iso3 in case.ISO_codes.keys():
                weather_per_loc = {loc: self.weather[s, loc] for loc in case.ISO_codes[iso3]}
                for loc in case.ISO_codes[iso3]:
                    n_days = len(weather_per_loc[loc] / self.case.days_per_period)
                    daily_matrix = weather_per_loc[loc][:]]
                
                self.prices[(s, iso3)] = price_model.simulate(speed_averages, iso, periods, seed)

        K, P, KS_hbwds, KM_hwds = gen_patterns(self.weather, case, scenarios)
        
        self.weather = {"locationID"}
        for w in windfarms:
            self.weather[w.locationiD] = weather_model.simulate(w.loationID, seed)

    def make_singleday_pattern_set(self):
        K = {}
        
        for s in scenarios:
            for w in self.case.wind_farms:
                weather = weathermodel.simulate(location id, seed=s,
                
                weather_windows
        for h in self.case.vessel_types:
            if not h.multiday:
                for b in self.case.bases:
                    for w in self.case.wind_farms:
                        for d in self.case.D:
                            for s in self.scenarios:
                                K[h.name, b.name, w.name, d, s] = [1]

        return K

    def make_multiday_pattern_set(self):
        K = {}

        for h in self.case.vessel_types:
            if h.multiday:
                for w in self.case.wind_farms:
                    for d in self.case.D:
                        for s in self.scenarios:
                            K[h.name, w.name, d, s] = [1]

        return K


    def make_failures(self):
        F = {}

        p = [m.failure_rate / 365 for m in self.case.maintenance_categories]
        p.append(1 - sum(p))
        
        for s in self.scenarios:
            rng = np.random.default_rng(seed=s)

            for w in self.case.wind_farms:
                draws = rng.multinomial(w.n_turbines, p, size=len(self.case.D))
                draws = draws[:, :-1]

                for d_idx, d in enumerate(self.case.D):
                    for m_idx, m in enumerate(self.case.maintenance_categories):
                        F[w.name, m.name, d, s] = draws[d_idx, m_idx]

        return F

    def make_downtime_costs(self):
        C_D = {}

        for w in self.case.wind_farms:
            for d in self.case.D:
                for s in self.scenarios:
                    C_D[w.name, d, s] = 200

        return C_D
