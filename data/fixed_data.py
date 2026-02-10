from dataclasses import dataclass

@dataclass
class VesselType:
    name: str
    required_capacity: int
    multiday: bool
    day_rate: float
    mob_rate: float 
    n_teams: int
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
    ISO3: str
    n_turbines: int

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
    vessel_types: [str] 

class FixedData:
    def __init__(self):
        self.vessel_types = [
            VesselType("CTV", 
                required_capacity=2,
                multiday = False,
                day_rate=10,
                mob_rate=200,
                n_teams=3
            ),
            VesselType("SOV", 
                required_capacity=5,
                multiday = True,
                day_rate=30,
                mob_rate=300,
                n_teams=6,
                periodic_return=13,
            )
        ]

        self.wind_farms = [
            WindFarm("A",
                lat=53.95,
                lon=6.65,
                ISO3="GBR",
                n_turbines=100,
            ),
            WindFarm("B",
                lat=53.93, 
                lon=8.14,
                ISO3="DEU",
                n_turbines=150,
            )
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
                duration=5,
                vessel_types=["CTV", "SOV"]
            )
        ]


        #self.pattern

        #self.vessels

