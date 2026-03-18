from haversine import haversine, Unit

A = (55.0, 7.8) # C
b1 = (54.68, 8.74) # German Bight
B = (54.23, 7.82) # C
b2 = (53.87, 8.63) # German Bight
C = (54.0, 6.61) # C
b3 = (53.63, 7.14) # German Bight


#Distance matrix
distances = {
    ("A", "b1"): haversine(A, b1, unit=Unit.KILOMETERS),
    ("A", "b2"): haversine(A, b2, unit=Unit.KILOMETERS),
    ("A", "b3"): haversine(A, b3, unit=Unit.KILOMETERS),
    ("B", "b1"): haversine(B, b1, unit=Unit.KILOMETERS),
    ("B", "b2"): haversine(B, b2, unit=Unit.KILOMETERS),
    ("B", "b3"): haversine(B, b3, unit=Unit.KILOMETERS),
    ("C", "b1"): haversine(C, b1, unit=Unit.KILOMETERS),
    ("C", "b2"): haversine(C, b2, unit=Unit.KILOMETERS),
    ("C", "b3"): haversine(C, b3, unit=Unit.KILOMETERS)
}
# travel time matrix assuming 35 km/h for CTVs and 20 km/h for SOVs
travel_times = {
    ("A", "b1"): distances[("A", "b1")] / 35,
    ("A", "b2"): distances[("A", "b2")] / 35,
    ("A", "b3"): distances[("A", "b3")] / 35,
    ("B", "b1"): distances[("B", "b1")] / 35,
    ("B", "b2"): distances[("B", "b2")] / 35,
    ("B", "b3"): distances[("B", "b3")] / 35,
    ("C", "b1"): distances[("C", "b1")] / 35,
    ("C", "b2"): distances[("C", "b2")] / 35,
    ("C", "b3"): distances[("C", "b3")] / 35
}

#print in travel time matrix in nice readable format
print("Travel Time Matrix (hours):")
print("     b1       b2       b3")
for farm in ["A", "B", "C"]:
    times = [f"{travel_times[(farm, f'b{i}')]:.2f}" for i in range(1, 4)]
    print(f"{farm}  " + "  ".join(times))
