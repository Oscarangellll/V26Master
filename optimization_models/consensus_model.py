from config import ScenarioConfig
from optimization_models import OptimizationModel


class ConsensusModel:
    def __init__(self, case, weather_model, price_model, judges):
        self.case = case
        self.weather_model = weather_model
        self.price_model = price_model
        self.judges = judges
        
        
        self.fixed_values_eta = {}

        self.models = {}

        for judge in judges:
            scenario = ScenarioConfig(case, weather_model, price_model, [judge])

            model = OptimizationModel(case, scenario)

            model.build_model()

            self.models[judge] = model


    def optimize(self):
       
        
        for iteration in range(2):
            print(f"Consensus iteration {iteration}")
            
            votes = {}

            for judge in self.judges:
                model = self.models[judge]

                for key, val in self.fixed_values_eta.items():
                    model.eta[key].LB = val
                    model.eta[key].UB = val
            
                model.optimize()
                
                for key, var in model.eta.items():
                    val = var.X
                    votes.setdefault(key, {})
                    votes[key][val] = votes[key].get(val, 0) + 1
            print(votes)
            
            threshold = 0.8

            for key, value_counts in votes.items():
                for val, count in value_counts.items():
                    if count / len(self.judges) >= threshold:
                        self.fixed_values_eta[key] = val

