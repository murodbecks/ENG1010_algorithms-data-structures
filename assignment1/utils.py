import os
import time
import json
import tracemalloc
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Callable

np.random.seed(25)

def generate_drone_locations(num_locations: int, dimension: int, width: int = 10_000, 
                             save_path: str = "data/") -> pd.DataFrame:
    """Generate drone locations using numpy and saving it to the path."""
    # checking whether we are getting either 2D or 3D
    if dimension not in [2, 3]:
        print(f"Dimensions should be one of 2 or 3. Got: {dimension}")
        return None
    
    # creating a path for saving the data
    os.makedirs(save_path, exist_ok=True)
    csv_path = os.path.join(save_path, f"{dimension}D_{width}x{width}_{num_locations}.csv")

    # checking whether we already generated the data
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Found data from `{csv_path}`")
        return df

    # generating ids in d followed by number in 7 decimals format
    idx = [f"d{i:07d}" for i in range(num_locations)]

    # generating X and Y coordinates
    x_values = np.random.uniform(0, width, size=num_locations)
    y_values = np.random.uniform(0, width, size=num_locations)

    # creating pandas dataframe with `id`, `x` and `y` columns
    df = pd.DataFrame({"id": idx, "x": x_values, "y": y_values})

    # adding Z coordinates if we are generating 3D dimension
    if dimension == 3:
        z_values = np.random.uniform(0, width, size=num_locations)
        df['z'] = z_values
    
    # saving dataframe for future use
    print(f"Data is saved to {csv_path}")
    df.to_csv(csv_path, index=False)

    return df

def measure_method(method: Callable, *args) -> Tuple[float, int, bool]:
    """Measure time and memory usage of a method call"""
    # starting memory tracking
    tracemalloc.start()
    
    # measuring time
    start_time = time.perf_counter()
    try:
        result = method(*args)
        success = result is not None
    except Exception as e:
        result = None
        success = False
        print(f"Exception happened during function call: {e}")
    end_time = time.perf_counter()
    
    # getting memory usage
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    time_taken = end_time - start_time
    return time_taken, peak, success

def save_benchmark_results(stats: Dict, save_path: str) -> bool:
    """Save benchmark results to jsonl file."""
    try:
        # appending results to jsonl file
        with open(save_path, "a", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=True)
            f.write("\n")
        return True
    
    except Exception as e:
        print(f"Exception occured in saving to file: {e}")
        return False
