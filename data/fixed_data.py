from dataclasses import dataclass

import numpy as np
import pandas as pd

@dataclass
class VesselType:
    name: str
    required_capacity: int
    multiday: bool
    day_rate_ST: float
    day_rate_LT: float
    mob_rate: float
    n_teams: int
    travel_speed: float
    max_wave: float
    cost_per_km: float
    periodic_return: int | None = None

    def cost_ST(self, days):
        return self.day_rate_ST * days

    def cost_LT(self, days, n_periods):
        return self.day_rate_LT * days * n_periods
    
    def cost_mob(self):
        return self.mob_rate

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

class PowerCurve:
    def __init__(self):
        df = pd.read_csv("data/power_curve.csv")
        self._speed = df["speed"].to_numpy()
        self._power = df["power"].to_numpy() / 1000 # MW

    def __call__(self, speed):
        return np.interp(speed, self._speed, self._power, left=0.0, right=0.0)

class FixedData:
    vessel_types = [
        VesselType("CTV",
            required_capacity=1,
            multiday = False,
            day_rate_ST=2300,
            day_rate_LT=2012.5,
            mob_rate=57_500,
            n_teams=4,
            travel_speed=46,
            max_wave=1.8005,
            cost_per_km=7.4,
        ),
        VesselType("SOV",
            required_capacity=1,
            multiday = True,
            day_rate_ST=18400,
            day_rate_LT=16100,
            mob_rate=230_000,
            n_teams=12,
            travel_speed=28,
            max_wave=2.5005,
            cost_per_km=37.2,
            periodic_return=14,
        )
    ]

    wind_farms = [
        WindFarm("A",
            lat=55.3,
            lon=7.8,
            n_turbines=50,
            iso="DNK",
            weather_location_id=1,
        ),
        WindFarm("B",
            lat=54.64,
            lon=7.94,
            n_turbines=70,
            iso="DEU",
            weather_location_id=2,
        ),
        WindFarm("C",
            lat=54.08,
            lon=8.13,
            n_turbines=80,
            iso="DEU",
            weather_location_id=3,
        ),
        WindFarm("D",
            lat=53.93,
            lon=7.22,
            n_turbines=100,
            iso="DEU",
            weather_location_id=3,
        ),
        WindFarm("E",
            lat=53.64,
            lon=5.03,
            n_turbines=70,
            iso="NLD",
            weather_location_id=5,
        ),
        WindFarm("F",
            lat=52.55,
            lon=4.22,
            n_turbines=50,
            iso="DEU",
            weather_location_id=5
        ),
        WindFarm("G",
            lat=53.3,
            lon=1.37,
            n_turbines=60,
            iso="GBR",
            weather_location_id=6,
        ),
        WindFarm("H",
            lat=53.99,
            lon=0.48,
            n_turbines=80,
            iso="GBR",
            weather_location_id=6,
        ),
    ]

    weather_locations = [
        WeatherLocation(1, lat=55.53, lon=7.51),
        WeatherLocation(2, lat=54.68, lon=7.4),
        WeatherLocation(3, lat=54.12, lon=6.93),
        WeatherLocation(4, lat=52.83, lon=4.06),
        WeatherLocation(5, lat=53.62, lon=1.02),
    ]

    weather_from_year = 2010
    weather_to_year = 2025

    bases = [
        Base("1",
            lat=56.66,
            lon=8.21,
            capacity=100,
            cost=0
        ),
        Base("2",
            lat=55.48,
            lon=8.34,
            capacity=100,
            cost=0
        ),
        Base("3",
            lat=54.68,
            lon=8.74,
            capacity=100,
            cost=0
        ),
        Base("4",
            lat=53.87,
            lon=8.63,
            capacity=100,
            cost=0
        ),
        Base("5",
            lat=53.63,
            lon=7.14,
            capacity=100,
            cost=0
        ),
        Base("6",
            lat=52.88,
            lon=4.74,
            capacity=100,
            cost=0
        ),
        Base("7",
            lat=52.72,
            lon=1.59,
            capacity=100,
            cost=0
        ),
        Base("8",
            lat=54.35,
            lon=-0.47,
            capacity=100,
            cost=0
        ),
    ]

    maintenance_categories = [
        MaintenanceCategory("Manual Reset",
            failure_rate=7.5,
            duration=3,
            vessel_types=["CTV", "SOV"]
        ),
        MaintenanceCategory("Minor Repair",
            failure_rate=6,
            duration=3.75,
            vessel_types=["CTV", "SOV"]
        ),
        MaintenanceCategory("Medium Repair",
            failure_rate= 1.375,
            duration=4.4,
            vessel_types=["CTV", "SOV"]
        ),
        MaintenanceCategory("Major Repair",
            failure_rate=0.24,
            duration=4.33,
            vessel_types=["CTV", "SOV"]
        ),
    ]
    
    power_curve = PowerCurve()
    
    price_from_year = 2023
    price_to_year = 2025
    
    days_per_period = 30
    
    periods = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", 
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

    travel_threshold_hours = 12

    # 07:00 = 7 etc
    work_day_start = 7
    work_day_end = 19

    n_scenarios_to_generate = 300 
    generate_scenarios_seed = 676769 
    
    wind_speed_resolution = 1
    wave_height_resolution = 0.1

    work_friction = 0.0

data = FixedData()
