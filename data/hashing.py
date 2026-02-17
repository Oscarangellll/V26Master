import hashlib

def hash_weather_location(wl, from_year, to_year):
    s = f"{wl.id}_{wl.lat}_{wl.lon}_{from_year}_{to_year}"
    return hashlib.sha256(s.encode()).hexdigest()[:10]

def hash_all_weather_locations(wls, from_year, to_year):
    s = ";".join(
        f"{wl.id}_{wl.lat}_{wl.lon}"
        for wl in sorted(wls, key=lambda x: x.id)
    )
    s = f"{s}_{from_year}_{to_year}"
    return hashlib.sha256(s.encode()).hexdigest()[:10]

def hash_weather_model(wls, from_year, to_year, rs, rh):
    s = ";".join(
        f"{wl.id}_{wl.lat}_{wl.lon}"
        for wl in sorted(wls, key=lambda x: x.id)
    )
    s = f"{s}_{from_year}_{to_year}_{rs}_{rh}"
    return hashlib.sha256(s.encode()).hexdigest()[:10]

def hash_electricity_prices(iso_codes, from_year, to_year):
    s = ";".join(sorted(iso_codes))
    s = f"{s}_{from_year}_{to_year}"
    return hashlib.sha256(s.encode()).hexdigest()[:10]
