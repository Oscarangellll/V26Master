from dataclasses import dataclass
import hashlib

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
    cost_per_km: float
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

class PowerCurve:
    def __init__(self):
        df = pd.read_csv("data/power_curve.csv")
        self._speed = df["speed"].to_numpy() 
        self._power = df["power"].to_numpy() / 1000 # MW
        
    def __call__(self, speed): 
        return np.interp(speed, self._speed, self._power)

class FixedData:
    vessel_types = [
        VesselType("CTV", 
            required_capacity=1,
            multiday = False,
            day_rate=2_940,
            mob_rate=58_825,
            n_teams=3,
            travel_speed=35,
            max_wind=25,
            max_wave=1.5,
            cost_per_km=8,
            shift_length=10,
        ),
        VesselType("SOV", 
            required_capacity=1,
            multiday = True,
            day_rate=11_765,
            mob_rate=235_295,
            n_teams=7,
            travel_speed=20,
            max_wind=30,
            max_wave=2,
            cost_per_km=10,
            shift_length=12,
            periodic_return=14,
        )
    ]

    wind_farms = [
        WindFarm("C",
            lat=54,
            lon=6.61,
            n_turbines=100,
            iso="DEU",
            weather_location_id=3,
        ),
        WindFarm("B",
            lat=54.23, 
            lon=7.82,
            n_turbines=100,
            iso="DEU",
            weather_location_id=2,
        ),
        WindFarm("A",
            lat=55, 
            lon=7.8,
            n_turbines=100,
            iso="DEU",
            weather_location_id=1,
        ),
        WindFarm("D",
            lat=55.23, 
            lon=7.61,
            n_turbines=100,
            iso="DEU",
            weather_location_id=1,
        ),
        WindFarm("E",
            lat=54.68, 
            lon=7.4,
            n_turbines=100,
            iso="DEU",
            weather_location_id=2,
        ),
    ]
    
    iso_codes = sorted({w.iso for w in wind_farms})
    
    weather_locations = [
        WeatherLocation(1, lat=55.23, lon=7.61),
        WeatherLocation(2, lat=54.68, lon=7.4),
        WeatherLocation(3, lat=54.12, lon=6.48)
    ]

    weather_from_year = 2010
    weather_to_year = 2025

    bases = [
        Base("3", 
            lat=53.63,
            lon=7.14,
            capacity=100,
            cost=0
        ),
        Base("2", 
            lat=53.87,
            lon=8.63,
            capacity=100,
            cost=0
        ),
        Base("1", 
            lat=54.68,
            lon=8.74,
            capacity=100,
            cost=0
        )
    ]

    maintenance_categories = [
        MaintenanceCategory("Annual Service", 
            failure_rate=5,
            duration=2,
            vessel_types=["CTV", "SOV"]
        ),
        MaintenanceCategory("Manual Reset", 
            failure_rate=7.5,
            duration=3,
            vessel_types=["CTV", "SOV"]
        ),
        MaintenanceCategory("Minor Repair", 
            failure_rate=3,
            duration=7.5,
            vessel_types=["CTV", "SOV"]
        ),
        MaintenanceCategory("Medium Repair", 
            failure_rate=0.825,
            duration=7.33,
            vessel_types=["CTV", "SOV"]
        ),
        MaintenanceCategory("Severe Repair", 
                failure_rate=0.12,
                duration=8.66,
                vessel_types=["CTV", "SOV"]
            ),
        ]
    
    power_curve = PowerCurve()
    
    price_from_year = 2023
    price_to_year = 2025

    upper_bound_weather_window = 15

    travel_threshold_hours = 12
   
    def weather_location_hash(self, wl):
        s = f"{wl.id}_{wl.lat}_{wl.lon}_{self.weather_from_year}_{self.weather_to_year}"
        return hashlib.sha256(s.encode()).hexdigest()[:10]
    
    def weather_data_hash(self):
        s = ";".join(
            f"{wl.id}_{wl.lat}_{wl.lon}"
            for wl in sorted(self.weather_locations, key=lambda x: x.id)
        )
        s = f"{s}_{self.weather_from_year}_{self.weather_to_year}"
        return hashlib.sha256(s.encode()).hexdigest()[:10]
    
    def weather_model_hash(self, rs, rh):
        s = ";".join(
            f"{wl.id}_{wl.lat}_{wl.lon}"
            for wl in sorted(self.weather_locations, key=lambda x: x.id)
        )
        s = f"{s}_{self.weather_from_year}_{self.weather_to_year}_{rs}_{rh}"
        return hashlib.sha256(s.encode()).hexdigest()[:10]
    
    def price_data_hash(self):
        s = ";".join(self.iso_codes)
        s = f"{s}_{self.price_from_year}_{self.price_to_year}"
        return hashlib.sha256(s.encode()).hexdigest()[:10]

    def price_model_hash(self):
        s = ";".join(
            f"{wl.id}_{wl.lat}_{wl.lon}"
            for wl in sorted(self.weather_locations, key=lambda x: x.id)
        )
        s = f"{s}_{self.weather_from_year}_{self.weather_to_year}"
        s = f"{s}_{';'.join(self.iso_codes)}"
        s = f"{s}_{self.price_from_year}_{self.price_to_year}"
        return hashlib.sha256(s.encode()).hexdigest()[:10]
        
        
data = FixedData()
