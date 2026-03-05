import numpy as np

from scenario_models import PriceModel, WeatherModel
from .patterns import gen_patterns
from .weather_windows import find_weather_windows
from .scenario_config import perform_scenario_reduction

class ScenarioConfig:
    def __init__(self, case, scenarios: list[int], scenario_reduction: bool = False):

        self.case = case        
        self.scenarios = scenarios
        self.scenario_reduction = scenario_reduction
        
        price_model = PriceModel()
        weather_model = WeatherModel()
        
        # {(s, iso, wl_id): (n_hours, 2) np.array
        weather = self._make_weather(weather_model) 
            
        # {(s, iso): (n_days, n_locations) np.array
        prices = self._make_prices(weather, price_model) 
        
        # {(w, m, d, s): int}
        failures = self._make_failures()

        # {(w, d, s): float}} 
        downtime_costs = self._make_downtime_costs(weather, prices)
        
        # {(h, w, d, s): int}
        weather_windows = find_weather_windows(self.case, weather, scenarios)
        
        if scenario_reduction:
            medoid_ids, weights, X_scaled = perform_scenario_reduction(
                case=self.case,
                scenario_ids=self.scenarios,
                weather_windows=weather_windows,
                downtime_costs=downtime_costs,
                failures=failures,
                n_reduced_scenarios=12
            )
            weather_windows_reduced = {k: v for k, v in weather_windows.items() if k[3] in medoid_ids}
            self.C_D = {k: v for k, v in downtime_costs.items() if k[2] in medoid_ids}
            self.F = {k: v for k, v in failures.items() if k[3] in medoid_ids}
        
            self.K_S, self.K_M, self.P = gen_patterns(weather_windows_reduced, self.case, self.scenarios)
            self.S = medoid_ids
            self.scenario_weights = {s: weights[s] for s in medoid_ids}
        else:
            self.K_S, self.K_M, self.P = gen_patterns(weather_windows, self.case, self.scenarios)
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


    def _make_weather(self, weather_model):
        weather = {}

        for s in self.scenarios:
            rng = np.random.default_rng(seed=s)

            for iso in self.case.all_wl_ids_for_iso.keys():
                for loc in self.case.all_wl_ids_for_iso[iso]:
                    weather[(s, iso, loc)] = weather_model.simulate(
                        loc, 
                        rng, 
                        self.case.periods, 
                        self.case.days_per_period
                    )

        return weather
               
    def _make_prices(self, weather, price_model):
        prices = {}

        for s in self.scenarios:
            rng = np.random.default_rng(seed=s)

            for iso in self.case.all_wl_ids_for_iso.keys():
            
                # Hourly wind speed per location
                #.T to get shape (n_hours, n_locations) instead of (n_locations, n_hours)
                # Shape (n_hours, n_locations)
                iso3_wind_speeds = np.array(
                    [weather[s, iso, wl_id][:,0] for wl_id in sorted(self.case.all_wl_ids_for_iso[iso])]
                ).T 
        
                # Average wind speed per day per location
                # Shape (n_days, n_locations)
                iso3_wind_speeds = iso3_wind_speeds.reshape(
                    -1, 24, iso3_wind_speeds.shape[1]
                ).mean(axis=1) 

                prices[s, iso] = price_model.simulate(
                    iso3_wind_speeds, 
                    iso, 
                    rng,    
                    self.case.periods,   
                    self.case.days_per_period
                )
        
        return prices

    def _make_failures(self):
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

    def _make_downtime_costs(self, weather, prices):
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

