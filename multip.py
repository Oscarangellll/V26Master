import multiprocessing as mp
import numpy as np
from time import time
from haversine import haversine, Unit

# # Prepare data
# np.random.RandomState(100)
# arr = np.random.randint(0, 10, size=[12, 20_000_000]) # 12 rows, 200k columns
# data = arr.tolist()

# # Solution Without Paralleization
# def howmany_within_range(row, minimum, maximum):
#     """Returns how many numbers lie within `maximum` and `minimum` in a given `row`"""
#     count = 0
#     for n in row:
#         if minimum <= n <= maximum:
#             count = count + 1
#     return count

# def main():
#     start = time()
#     with mp.Pool(12) as pool:
#         results = [pool.apply(howmany_within_range, args=(row, 4, 8)) for row in data]
#     # results = []
#     # for row in data:
#     #     results.append(howmany_within_range(row, minimum=4, maximum=8))
#     print(results[:10])
#     print(f"Elapsed: {time() - start:.2f}s")


# if __name__ == "__main__":
#     mp.freeze_support()
#     main()

WindFarmA = (55.0, 7.8)
Base1 = (54.68, 8.74)
print(haversine(WindFarmA, Base1, unit=Unit.KILOMETERS))
speed = 35 # km/h
print(f"Travel time from Base 1 to Wind Farm A: {haversine(WindFarmA, Base1, unit=Unit.KILOMETERS) / speed:.2f} hours")