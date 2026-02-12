from dataclasses import dataclass

import pandas as pd
import numpy as np

@dataclass
class VesselType:
    name: str
    required_capacity: int
    multiday: bool
    day_rate: float
    mob_rate: float 
    n_teams: int
    travel_speed: float
    max_wind: float
    max_wave: float
    shift_length: int = 12
    periodic_return: int | None = None

    def cost_ST(self, days):
        return self.day_rate * days + self.mob_rate 

    def cost_LT(self, days, n_periods):
        return self.day_rate * days * n_periods + self.mob_rate

@dataclass
class WindFarm:
    name: str
    lat: float
    lon: float
    n_turbines: int
    iso: str
    weather_location_id: int

@dataclass
class WeatherLocation:
    id: int
    lat: float
    lon: float

@dataclass
class Base:
    name: str
    lat: float
    lon: float
    capacity: int
    cost: float

@dataclass
class MaintenanceCategory:  
    name: str
    failure_rate: float # per year
    duration: float # in hours
    vessel_types: list[str] 

class FixedData:
    def __init__(self):
        self.vessel_types = [
            VesselType("CTV", 
                required_capacity=2,
                multiday = False,
                day_rate=10,
                mob_rate=200,
                n_teams=3,
                travel_speed=35,
                max_wind=25,
                max_wave=1.5,
                shift_length=12
            ),
            VesselType("SOV", 
                required_capacity=5,
                multiday = True,
                day_rate=30,
                mob_rate=300,
                n_teams=6,
                travel_speed=20,
                max_wind=30,
                max_wave=2,
                shift_length=12,
                periodic_return=13,
            )
        ]

        self.wind_farms = [
            WindFarm("A",
                lat=53.95,
                lon=6.65,
                n_turbines=100,
                iso="DEU",
                weather_location_id=1,
            ),
            WindFarm("B",
                lat=53.93, 
                lon=8.14,
                n_turbines=150,
                iso="DEU",
                weather_location_id=2,
            )
        ]
	
        self.weather_locations = [
            WeatherLocation(1, lat=54, lon=6.65),
            WeatherLocation(2, lat=55, lon=5.65)
        ]

        self.bases = [
            Base("1", 
                lat=40,
                lon=20,
                capacity=20,
                cost=300
            ),
            Base("2", 
                lat=20,
                lon=20,
                capacity=20,
                cost=400
            )
        ]

        self.maintenance_categories = [
            MaintenanceCategory("Annual Service", 
                failure_rate=0.2,
                duration=3,
                vessel_types=["CTV", "SOV"]
            )
        ]
        
        power_curve_data = pd.read_csv("data/power_curve.csv")
        self._speed = power_curve_data["speed"].to_numpy()
        self._power = power_curve_data["power"].to_numpy()
        
        self.upper_bound_weather_window = 15
    
    def power_curve(self, speed):
        return np.interp(
            speed, 
            self._speed,
            self._power
        )
        


d = FixedData()

