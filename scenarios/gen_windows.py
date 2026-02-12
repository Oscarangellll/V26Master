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

def find_weather_windows(case, weather: dict):
    """
    weather: {(scenario, location): np.ndarray}
    ndarray shape: (T, 2) with columns [speed, height]
    """
    weather_windows = {}

    for w in case.wind_farms:
        loc = getattr(w, "location", w.name)
        for s in case.scenarios:
            arr = weather[(s, loc)]
            T = arr.shape[0]
            n_days = T // 24

            for d_idx in range(n_days):
                day = case.D[d_idx] if d_idx < len(case.D) else d_idx
                day_slice = arr[d_idx * 24:(d_idx + 1) * 24]

                # hours 7..19 inclusive
                speed = day_slice[7:20, 0]
                height = day_slice[7:20, 1]

                for h in case.vessel_types:
                    weather_windows[(h.name, w.name, day, s)] = _find_window(
                        speed, height, h
                    )
    return weather_windows