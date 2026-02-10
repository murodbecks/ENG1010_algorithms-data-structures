import os
import numpy as np

from utils import generate_drone_locations, measure_method, save_benchmark_results
from algorithm import find_closest_pair_brute_force, find_k_closest_pair_brute_force, find_closest_pair_improved, find_k_closest_pair_improved

benchmark_folder = "files/"
os.makedirs(benchmark_folder, exist_ok=True)

benchmark_file = os.path.join(benchmark_folder, "benchmark.jsonl")

width = 10_000

# print(f"Starting brute-force top-1")
# for num_locations in [1_000, 10_000, 25_000]:
#     for dimension in [2, 3]:
#         df = generate_drone_locations(num_locations, dimension, width)
#         columns = ['x', 'y'] if dimension == 2 else ['x', 'y', 'z']
#         k = 1
        
#         # converting to numpy and list to get linear access time
#         arr = df[columns].to_numpy()
#         idx = df['id'].tolist()

#         time_taken, memory_usage, success = measure_method(find_closest_pair_brute_force, arr, idx)
#         stats = {
#             "algorithm": "brute-force",
#             "num_locations": num_locations,
#             "k": k,
#             "width": width,
#             "dimension": dimension,
#             "time_taken_ms": time_taken * 1000,
#             "memory_usage_kb": memory_usage / 1024
#         }
#         save_benchmark_results(stats, benchmark_file)

# print("\nStarting brute-force top-k")
# for num_locations in [1_000, 10_000, 25_000]:
#     for dimension in [2, 3]:
#         for k in [2, 10, 100]:
#             df = generate_drone_locations(num_locations, dimension, width)
#             columns = ['x', 'y'] if dimension == 2 else ['x', 'y', 'z']
            
#             # converting to numpy and list to get linear access time
#             arr = df[columns].to_numpy()
#             idx = df['id'].tolist()

#             time_taken, memory_usage, success = measure_method(find_k_closest_pair_brute_force, k, arr, idx)
#             stats = {
#                 "algorithm": "brute-force",
#                 "num_locations": num_locations,
#                 "k": k,
#                 "width": width,
#                 "dimension": dimension,
#                 "time_taken_ms": time_taken * 1000,
#                 "memory_usage_kb": memory_usage / 1024
#             }
#             save_benchmark_results(stats, benchmark_file)


print("\nStarting optimized search top-1")
for num_locations in [1_000, 10_000, 25_000, 100_000]:
    for dimension in [2, 3]:
        df = generate_drone_locations(num_locations, dimension, width)
        columns = ['x', 'y'] if dimension == 2 else ['x', 'y', 'z']
        k = 1
        
        # converting to numpy and list to get linear access time
        arr = df[columns].to_numpy()
        idx = df['id'].tolist()

        # sorting values by X-axis for optimal performance
        sorted_indexes = np.argsort(arr[:, 0])
        arr_sorted_by_x = arr[sorted_indexes]
        idx_sorted_by_x = [idx[i] for i in sorted_indexes]

        time_taken, memory_usage, success = measure_method(find_closest_pair_improved, arr_sorted_by_x, idx_sorted_by_x)
        stats = {
            "algorithm": "optimized",
            "num_locations": num_locations,
            "k": k,
            "width": width,
            "dimension": dimension,
            "time_taken_ms": time_taken * 1000,
            "memory_usage_kb": memory_usage / 1024
        }
        save_benchmark_results(stats, benchmark_file)

print("\nStarting optimized search top-k")
for num_locations in [1_000, 10_000, 25_000, 100_000]:
    for dimension in [2, 3]:
        for k in [2, 10, 100]:
            df = generate_drone_locations(num_locations, dimension, width)
            columns = ['x', 'y'] if dimension == 2 else ['x', 'y', 'z']
            
            # converting to numpy and list to get linear access time
            arr = df[columns].to_numpy()
            idx = df['id'].tolist()

            # sorting values by X-axis for optimal performance
            sorted_indexes = np.argsort(arr[:, 0])
            arr_sorted_by_x = arr[sorted_indexes]
            idx_sorted_by_x = [idx[i] for i in sorted_indexes]

            time_taken, memory_usage, success = measure_method(find_closest_pair_improved, arr_sorted_by_x, idx_sorted_by_x)

            # time_taken, memory_usage, success = measure_method(find_k_closest_pair_improved, k, arr, idx)
            stats = {
                "algorithm": "optimized",
                "num_locations": num_locations,
                "k": k,
                "width": width,
                "dimension": dimension,
                "time_taken_ms": time_taken * 1000,
                "memory_usage_kb": memory_usage / 1024
            }
            save_benchmark_results(stats, benchmark_file)


print("Done!")