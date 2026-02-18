import numpy as np
from scenarios.gen_patterns import gen_patterns

# from scenarios.gen_patterns import gen_patterns

class ScenarioConfig:

    def __init__(self, case, weather_model, price_model, scenarios: list[int]):

        self.case = case        
        self.weather_model = weather_model
        self.price_model = price_model
        self.scenarios = scenarios

        weather = {}
        prices = {}
        
        for s in scenarios:
            rng = np.random.default_rng(seed=s)
            for iso in case.all_wl_ids_for_iso.keys():
                for loc in case.all_wl_ids_for_iso[iso]:
                    weather[(s, iso, loc)] = weather_model.simulate(loc, rng, case.periods, case.days_per_period)
                iso3_wind_speeds = np.array([weather[s, iso, loc][:,0] for loc in sorted(case.all_wl_ids_for_iso[iso])]).T #.T to get shape (n_hours, n_locations) instead of (n_locations, n_hours)
                iso3_wind_speeds = iso3_wind_speeds.reshape(-1, 24, iso3_wind_speeds.shape[1]).mean(axis=1) #shape (n_days, n_locations)
                prices[s, iso] = price_model.simulate(iso3_wind_speeds, iso, rng, case.periods, case.days_per_period)
        # print("rett før patterns skal lages")
        self.K_S, self.K_M, self.P = gen_patterns(weather, case, scenarios)
        # print("rett etter patterns er laget")
        self.C_D = self.make_downtime_costs(weather, prices)
        self.F = self.make_failures()
        # print("scencon prints:")
        # print(self.K_S)
        # print("------------------------------------")
        # print(self.K_M)
        # print("------------------------------------")
        # print(self.P)
        # print("------------------------------------")
        # print(prices)
        # print("------------------------------------")
        # print(self.C_D)
        # print("------------------------------------")
        # print(self.F)
        # print("------------------------------------")
        

    def make_singleday_pattern_set(self):
        K = {}
        
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

    def make_downtime_costs(self, weather, prices):
        C_D = {}
        for w in self.case.wind_farms:
            for s in self.scenarios:
                sim_speed = weather[(s, w.iso, w.weather_location_id)][:, 0]
            
                sim_power_output = self.case.power_curve(sim_speed) 
                
                n_days = len(sim_power_output) // 24
                sim_daily_power = sim_power_output.reshape(n_days, 24).mean(axis=1)
                sim_daily_power *= 24
                sim_downtime_cost = sim_daily_power * prices[(s, w.iso)]

                for d in self.case.D:
                    C_D[w.name, d, s] = sim_downtime_cost[d - 1] 
        
        return C_D

