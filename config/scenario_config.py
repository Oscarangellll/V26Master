import numpy as np
# from scenarios.gen_patterns import gen_patterns

class ScenarioConfig:

    def __init__(self, case, weather_model, price_model, scenarios: list[int]):
        self.case = case
        self.scenarios = scenarios
        self.weather = {(s, iso3, loc) for s in scenarios for iso3 in case.ISO_codes.keys() for loc in case.ISO_codes[iso3]}
        self.prices = {(s, iso3) for s in scenarios for iso3 in case.ISO_codes.keys()}
        
        for s in scenarios:
            for iso3 in case.ISO_codes.keys():
                for loc in case.ISO_codes[iso3]:
                    self.weather[(s, iso3, loc)] = weather_model.simulate(s, loc)
                #want to simulate prices based on weather at all locations in the iso3 code
                #make an ndarray of shape (n_hours, n_locations) to pass to price model
                iso3_wind_speeds = np.array([self.weather[s, iso3, loc][:0] for loc in case.ISO_codes[iso3]]).T #.T to get shape (n_hours, n_locations) instead of (n_locations, n_days)
                #make averages per day (24 values per day) to pass to price model
                iso3_wind_speeds = iso3_wind_speeds.reshape(-1, 24, iso3_wind_speeds.shape[1]).mean(axis=1) #shape (n_days, n_locations)
                self.prices[s, iso3] = price_model.simulate(s, iso3, iso3_wind_speeds)
                
                    
                    
                weather_per_loc = {loc: self.weather[s, loc] for loc in case.ISO_codes[iso3]}
                for loc in case.ISO_codes[iso3]:
                    n_days = len(weather_per_loc[loc] / self.case.days_per_period)
                    daily_matrix = weather_per_loc[loc][:]
                
                self.prices[(s, iso3)] = price_model.simulate(speed_averages, iso, periods, seed)

        K, P, KS_hbwds, KM_hwds = gen_patterns(self.weather, case, scenarios)
        
        self.weather = {"locationID"}
        for w in windfarms:
            self.weather[w.locationiD] = weather_model.simulate(w.loationID, seed)

    def make_singleday_pattern_set(self):
        K = {}
        
        # for s in scenarios:
        #     for w in self.case.wind_farms:
        #         weather = weathermodel.simulate(location id, seed=s,
                
        #         weather_windows
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


s = 1
iso3 = {"DEU": ["DEU_loc1", "DEU_loc2"]}
weather = {
    # wind speeds are random between 10 and 30, wave heights are random between 0 and 5. make for each hour of the year (24 * 365) and for each location in the iso3 code
    (s, "DEU", "DEU_loc1"): [10 + 20 * np.random.rand(24 * 365), 5 * np.random.rand(24 * 365)],
    (s, "DEU", "DEU_loc2"): [10 + 20 * np.random.rand(24 * 365), 5 * np.random.rand(24 * 365)],
}
iso3_wind_speeds = np.array([weather[s, "DEU", loc][0] for loc in iso3["DEU"]]).T #.T to get shape (n_hours, n_locations) instead of (n_locations, n_days)
print(iso3_wind_speeds.shape) #should be (24 * 365, 2)
print(iso3_wind_speeds[:5]) #print first 5 rows to check values
iso3_wind_speeds = iso3_wind_speeds.reshape(-1, 24, iso3_wind_speeds.shape[1]).mean(axis=1) #shape (n_days, n_locations)
print(iso3_wind_speeds.shape) #should be (365, 2)
print(iso3_wind_speeds[:5]) #print first 5 rows to check values