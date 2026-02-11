"""
Utility functions for dronz collision prevention system.
Handles data generation, file I/O, distance calculations, and benchmarking utilities.
"""

import os
import csv
import time
import random
import tracemalloc
from typing import List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import math


@dataclass
class Dronz:
    """Represents a single dronz with ID and location."""
    id: int
    coords: Tuple[float, ...]

    @property
    def dimension(self) -> int:
        return len(self.coords)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id


def euclidean_distance(p1: Dronz, p2: Dronz) -> float:
    """Calculate Euclidean distance between two dronz."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1.coords, p2.coords)))


def generate_dronz_data(
    n: int,
    dimension: int = 2,
    width: float = 10000.0,
    seed: Optional[int] = None
) -> List[Dronz]:
    """
    Generate random dronz locations.

    Args:
        n: Number of dronz to generate
        dimension: 2 for 2D, 3 for 3D
        width: Range for coordinates [0, width)
        seed: Random seed for reproducibility

    Returns:
        List of Dronz objects
    """
    if seed is not None:
        random.seed(seed)

    dronz_list = []
    for i in range(n):
        coords = tuple(random.uniform(0, width) for _ in range(dimension))
        dronz_list.append(Dronz(id=i, coords=coords))

    return dronz_list


def get_data_filename(n: int, dimension: int, seed: int = 42) -> str:
    """Generate consistent filename for dataset."""
    return f"dronz_{n}_{dimension}d_seed{seed}.csv"


def save_dronz_data(dronz_list: List[Dronz], filepath: str) -> None:
    """Save dronz data to CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        dimension = dronz_list[0].dimension if dronz_list else 2
        if dimension == 2:
            writer.writerow(['id', 'x', 'y'])
        else:
            writer.writerow(['id', 'x', 'y', 'z'])
        # Data
        for dronz in dronz_list:
            writer.writerow([dronz.id] + list(dronz.coords))


def load_dronz_data(filepath: str) -> List[Dronz]:
    """Load dronz data from CSV file."""
    dronz_list = []

    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        dimension = len(header) - 1

        for row in reader:
            dronz_id = int(row[0])
            coords = tuple(float(x) for x in row[1:])
            dronz_list.append(Dronz(id=dronz_id, coords=coords))

    return dronz_list


def get_or_generate_data(
    n: int,
    dimension: int,
    data_dir: str = "data",
    seed: int = 42,
    width: float = 10000.0
) -> List[Dronz]:
    """
    Load data if exists, otherwise generate and save.

    Args:
        n: Number of dronz
        dimension: 2D or 3D
        data_dir: Directory to store/load data
        seed: Random seed
        width: Coordinate range

    Returns:
        List of Dronz objects
    """
    filename = get_data_filename(n, dimension, seed)
    filepath = os.path.join(data_dir, filename)

    if os.path.exists(filepath):
        print(f"Loading existing data from {filepath}")
        return load_dronz_data(filepath)
    else:
        print(f"Generating new data: n={n}, dim={dimension}, seed={seed}")
        dronz_list = generate_dronz_data(n, dimension, width, seed)
        save_dronz_data(dronz_list, filepath)
        print(f"Saved to {filepath}")
        return dronz_list


@dataclass
class BenchmarkResult:
    """Stores benchmark results for an algorithm run."""
    algorithm: str
    n: int
    dimension: int
    k: int
    time_seconds: float
    memory_peak_mb: float
    result_distance: float
    result_pairs: List[Tuple[int, int]]


def benchmark(
    func: Callable,
    *args,
    **kwargs
) -> Tuple[Any, float, float]:
    """
    Benchmark a function for time and memory usage.

    Returns:
        Tuple of (result, time_seconds, peak_memory_mb)
    """
    # Start memory tracking
    tracemalloc.start()

    # Time the function
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()

    # Get memory stats
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    time_taken = end_time - start_time
    peak_memory_mb = peak / (1024 * 1024)

    return result, time_taken, peak_memory_mb


def save_benchmark_result(result: BenchmarkResult, filepath: str) -> None:
    """Append benchmark result to CSV file."""
    file_exists = os.path.exists(filepath)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                'algorithm', 'n', 'dimension', 'k',
                'time_seconds', 'memory_peak_mb',
                'result_distance', 'result_pairs'
            ])
        writer.writerow([
            result.algorithm,
            result.n,
            result.dimension,
            result.k,
            f"{result.time_seconds:.6f}",
            f"{result.memory_peak_mb:.4f}",
            f"{result.result_distance:.10f}",
            str(result.result_pairs)
        ])


def load_benchmark_results(filepath: str) -> List[BenchmarkResult]:
    """Load benchmark results from CSV file."""
    results = []
    if not os.path.exists(filepath):
        return results

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(BenchmarkResult(
                algorithm=row['algorithm'],
                n=int(row['n']),
                dimension=int(row['dimension']),
                k=int(row['k']),
                time_seconds=float(row['time_seconds']),
                memory_peak_mb=float(row['memory_peak_mb']),
                result_distance=float(row['result_distance']),
                result_pairs=eval(row['result_pairs'])
            ))
    return results