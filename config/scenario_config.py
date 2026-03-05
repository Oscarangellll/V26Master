import numpy as np
from scenarios.gen_patterns import gen_patterns
from scenario_models import WeatherModel, PriceModel
from scenarios.gen_windows import find_weather_windows
from scenario_reduction import scenario_reduction as perform_scenario_reduction

class ScenarioConfig:

    def __init__(self, case, scenarios: list[int], scenario_reduction: bool = False):

        self.case = case        
        self.weather_model = WeatherModel()
        self.price_model = PriceModel()
        self.scenarios = scenarios
        self.scenario_reduction = scenario_reduction

        weather = {}
        prices = {}
        
        for s in scenarios:
            rng = np.random.default_rng(seed=s)
            for iso in case.all_wl_ids_for_iso.keys():
                for loc in case.all_wl_ids_for_iso[iso]:
                    weather[(s, iso, loc)] = self.weather_model.simulate(loc, rng, case.periods, case.days_per_period)
                iso3_wind_speeds = np.array([weather[s, iso, loc][:,0] for loc in sorted(case.all_wl_ids_for_iso[iso])]).T #.T to get shape (n_hours, n_locations) instead of (n_locations, n_hours)
                iso3_wind_speeds = iso3_wind_speeds.reshape(-1, 24, iso3_wind_speeds.shape[1]).mean(axis=1) #shape (n_days, n_locations)
                #print iso3 weather first 20 rows with corresponding scenario
                prices[s, iso] = self.price_model.simulate(iso3_wind_speeds, iso, rng, case.periods, case.days_per_period)
        
        weather_windows = find_weather_windows(case, weather, scenarios)
        downtime_costs = self.make_downtime_costs(weather, prices)
        failures = self.make_failures()
        
        if scenario_reduction:
            medoid_ids, weights, X_scaled = perform_scenario_reduction(
                case=case,
                scenario_ids=scenarios,
                weather_windows=weather_windows,
                downtime_costs=downtime_costs,
                failures=failures,
                n_reduced_scenarios=12
            )
            weather_windows_reduced = {k: v for k, v in weather_windows.items() if k[3] in medoid_ids}
            self.C_D = {k: v for k, v in downtime_costs.items() if k[2] in medoid_ids}
            self.F = {k: v for k, v in failures.items() if k[3] in medoid_ids}
        
            self.K_S, self.K_M, self.P = gen_patterns(weather_windows_reduced, case, scenarios)
            self.S = medoid_ids
            self.scenario_weights = {s: weights[s] for s in medoid_ids}
        else:
            self.K_S, self.K_M, self.P = gen_patterns(weather_windows, case, scenarios)
            self.C_D = downtime_costs
            self.F = failures
            self.S = scenarios
            self.scenario_weights = {s: 1 / len(scenarios) for s in scenarios}
            
    def get_KS_for_scenarios(self, scenario_list):
        for (h, b, w, d, s), value in self.K_S.items():
            if s in scenario_list:
                yield (h, b, w, d, s), value        

    def get_KM_for_scenarios(self, scenario_list):
        for (h, w, d, s), value in self.K_M.items():
            if s in scenario_list:
                yield (h, w, d, s), value
                
    def get_CD_for_scenarios(self, scenario_list):
        for (w, d, s), value in self.C_D.items():
            if s in scenario_list:
                yield (w, d, s), value

    def get_F_for_scenarios(self, scenario_list):
        for (w, m, d, s), value in self.F.items():
            if s in scenario_list:
                yield (w, m, d, s), value

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

