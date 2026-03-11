from data.fixed_data import data

def _find_daily_window(wind_speed, wave_height, vessel_type):
    # Finds length of weather window for a single day
    #
    # wind_speed, wave_height: list of consecutive wind speeds and wave height for a single day
    # wind_speed and wave_height can be truncated, eg. values only between 07:00 and 19:00 
    #
    # return: int 

    # Vessel limits
    max_speed = vessel_type.max_wind
    max_height = vessel_type.max_wave

    current_window = 0
    max_window = 0

    operable = (wind_speed <= max_speed) & (wave_height <= max_height)

    for hour in operable:
        if hour:
            current_window += 1
            max_window = max(current_window, max_window)
        else:
            current_window = 0

    return max_window

def find_weather_windows(case, weather, scenarios):
    """
    weather: {(s, iso, wl_id): np.array with shape (T, 2) with columns [speed, height]
    scenarios: list[int]
    """
    weather_windows = {}

    for w in case.wind_farms:
        for s in scenarios:
            arr = weather[(s, w.iso, w.weather_location_id)]
            T = arr.shape[0]
            n_days = T // 24

            for d_idx in range(n_days):
                day = case.D[d_idx] if d_idx < len(case.D) else d_idx
                day_slice = arr[d_idx * 24:(d_idx + 1) * 24]

                # + 1 because of slicing
                wind_speed = day_slice[data.work_day_start:data.work_day_end + 1, 0]
                wave_height = day_slice[data.work_day_start:data.work_day_end + 1, 1]

                for h in case.vessel_types:
                    weather_windows[(h.name, w.name, day, s)] = _find_daily_window(
                        wind_speed, wave_height, h
                    )
    return weather_windows
