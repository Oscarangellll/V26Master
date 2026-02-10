def _find_window(speed, height, vessel_type):
    max_speed = vessel_type.max_wind
    max_height = vessel_type.max_wave
    shift_limit = vessel_type.shift_length
    
    current_window = 0
    max_window = 0
    
    operable = (speed <= max_speed) & (height <= max_height)
    
    for hour in operable:
        if hour:
            current_window += 1
            max_window = max(current_window, max_window)
        else:
            current_window = 0

    max_window = min(max_window, shift_limit)
    return max_window

def find_weather_windows(case, weather):
    weather_windows = {}

    for i in case.wind_farms:
        for s, scenario_data in weather.groupby("Scenario"):
            for d, daily_data in scenario_data.groupby("Day"):
                #only assess hours 7 to 19 (operational hours)
                daily_data = daily_data[(daily_data["Hour"] >= 7) & (daily_data["Hour"] <= 19)]
                for h in case.vessel_types:
                    weather_windows[(h.name, i.name, d, s)] = _find_window(
                                    daily_data["Speed"],
                                    daily_data["Height"],
                                    h
                                )
    return weather_windows