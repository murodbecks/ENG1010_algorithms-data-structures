"""
Main script for running dronz collision prevention experiments.
Benchmarks brute-force vs optimized algorithms across different dataset sizes.
"""

import os
import sys
from datetime import datetime
from typing import List

from utils import (
    get_or_generate_data,
    benchmark,
    BenchmarkResult,
    save_benchmark_result,
    Dronz
)
from algorithm import (
    brute_force_closest_pair,
    brute_force_top_k_pairs,
    optimized_closest_pair,
    optimized_top_k_pairs,
    kd_tree_closest_pair,
)


# Configuration
DATA_DIR = "data"
FILES_DIR = "files"
BENCHMARK_FILE = os.path.join(FILES_DIR, "benchmark_results.csv")

# Dataset sizes - removed 10M, it's impractical
DATASET_SIZES = [1_000, 10_000, 100_000, 1_000_000]

# Brute force limits
BRUTE_FORCE_LIMIT = 10_000  # Reduced from 25K - BF at 10K already takes ~60s

# K values - reduced to just 3 representative values
K_VALUES = [1, 10, 100]

# Skip slow algorithms for large datasets
KDTREE_TOPK_LIMIT = 100_000  # Skip kdtree_topk for n > 100K (too slow)

SEED = 42


def run_experiment_closest_pair(
    dronz_list: List[Dronz],
    n: int,
    dimension: int,
    run_brute_force: bool = True
) -> List[BenchmarkResult]:
    """Run closest pair experiments for a given dataset."""
    results = []

    # Brute Force
    if run_brute_force:
        print(f"  Running brute force closest pair (n={n}, dim={dimension})...")
        result, time_taken, memory_mb = benchmark(brute_force_closest_pair, dronz_list)
        if result:
            br = BenchmarkResult(
                algorithm="brute_force",
                n=n,
                dimension=dimension,
                k=1,
                time_seconds=time_taken,
                memory_peak_mb=memory_mb,
                result_distance=result[2],
                result_pairs=[(result[0][0], result[1][0])]
            )
            results.append(br)
            print(f"    Time: {time_taken:.4f}s, Memory: {memory_mb:.2f}MB, "
                  f"Distance: {result[2]:.6f}")

    # Optimized (Divide and Conquer) - always run, it's fast
    print(f"  Running D&C closest pair (n={n}, dim={dimension})...")
    result, time_taken, memory_mb = benchmark(optimized_closest_pair, dronz_list)
    if result:
        br = BenchmarkResult(
            algorithm="divide_conquer",
            n=n,
            dimension=dimension,
            k=1,
            time_seconds=time_taken,
            memory_peak_mb=memory_mb,
            result_distance=result[2],
            result_pairs=[(result[0][0], result[1][0])]
        )
        results.append(br)
        print(f"    Time: {time_taken:.4f}s, Memory: {memory_mb:.2f}MB, "
              f"Distance: {result[2]:.6f}")

    # KD-Tree - run for all sizes
    print(f"  Running KD-Tree closest pair (n={n}, dim={dimension})...")
    result, time_taken, memory_mb = benchmark(kd_tree_closest_pair, dronz_list)
    if result:
        br = BenchmarkResult(
            algorithm="kdtree",
            n=n,
            dimension=dimension,
            k=1,
            time_seconds=time_taken,
            memory_peak_mb=memory_mb,
            result_distance=result[2],
            result_pairs=[(result[0][0], result[1][0])]
        )
        results.append(br)
        print(f"    Time: {time_taken:.4f}s, Memory: {memory_mb:.2f}MB, "
              f"Distance: {result[2]:.6f}")

    return results


def run_experiment_top_k(
    dronz_list: List[Dronz],
    n: int,
    dimension: int,
    k: int,
    run_brute_force: bool = True,
    run_kdtree_topk: bool = True
) -> List[BenchmarkResult]:
    """Run top-k closest pairs experiments."""
    results = []

    # Brute Force
    if run_brute_force:
        print(f"  Running brute force top-{k} (n={n}, dim={dimension})...")
        result, time_taken, memory_mb = benchmark(brute_force_top_k_pairs, dronz_list, k)
        if result:
            br = BenchmarkResult(
                algorithm="brute_force_topk",
                n=n,
                dimension=dimension,
                k=k,
                time_seconds=time_taken,
                memory_peak_mb=memory_mb,
                result_distance=result[0][2] if result else 0,
                result_pairs=[(r[0][0], r[1][0]) for r in result]
            )
            results.append(br)
            print(f"    Time: {time_taken:.4f}s, Memory: {memory_mb:.2f}MB")

    # KD-Tree Top-K
    if run_kdtree_topk:
        print(f"  Running KD-Tree top-{k} (n={n}, dim={dimension})...")
        result, time_taken, memory_mb = benchmark(optimized_top_k_pairs, dronz_list, k)
        if result:
            br = BenchmarkResult(
                algorithm="kdtree_topk",
                n=n,
                dimension=dimension,
                k=k,
                time_seconds=time_taken,
                memory_peak_mb=memory_mb,
                result_distance=result[0][2] if result else 0,
                result_pairs=[(r[0][0], r[1][0]) for r in result]
            )
            results.append(br)
            print(f"    Time: {time_taken:.4f}s, Memory: {memory_mb:.2f}MB")

    return results


def verify_correctness(dronz_list: List[Dronz], k: int = 5) -> bool:
    """Verify that optimized algorithms produce same results as brute force."""
    print("\n=== Verifying Algorithm Correctness ===")

    # Test closest pair
    bf_result = brute_force_closest_pair(dronz_list)
    dc_result = optimized_closest_pair(dronz_list)
    kd_result = kd_tree_closest_pair(dronz_list)

    if bf_result:
        bf_dist = bf_result[2]

        if dc_result and abs(bf_dist - dc_result[2]) < 1e-9:
            print(f"✓ Closest pair D&C: PASS (distance={bf_dist:.10f})")
        else:
            print(f"✗ Closest pair D&C: FAIL")
            return False

        if kd_result and abs(bf_dist - kd_result[2]) < 1e-9:
            print(f"✓ Closest pair KD-Tree: PASS")
        else:
            print(f"✗ Closest pair KD-Tree: FAIL")
            return False

    # Test top-k
    bf_topk = brute_force_top_k_pairs(dronz_list, k)
    kd_topk = optimized_top_k_pairs(dronz_list, k)

    bf_dists = sorted([r[2] for r in bf_topk])
    kd_dists = sorted([r[2] for r in kd_topk])

    if len(bf_dists) == len(kd_dists) and all(abs(a - b) < 1e-9 for a, b in zip(bf_dists, kd_dists)):
        print(f"✓ Top-{k} KD-Tree: PASS")
    else:
        print(f"✗ Top-{k} KD-Tree: FAIL")
        return False

    return True


def run_all_experiments():
    """Run complete benchmark suite."""
    os.makedirs(FILES_DIR, exist_ok=True)

    print("=" * 60)
    print("DRONZ COLLISION PREVENTION - BENCHMARK SUITE")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print(f"Dataset sizes: {DATASET_SIZES}")
    print(f"K values: {K_VALUES}")
    print(f"Brute force limit: {BRUTE_FORCE_LIMIT}")
    print("=" * 60)

    # Correctness verification
    print("\n--- Correctness Verification (2D) ---")
    small_data_2d = get_or_generate_data(1000, dimension=2, seed=SEED)
    if not verify_correctness(small_data_2d, k=10):
        print("ERROR: Correctness verification failed for 2D!")
        sys.exit(1)

    print("\n--- Correctness Verification (3D) ---")
    small_data_3d = get_or_generate_data(1000, dimension=3, seed=SEED)
    if not verify_correctness(small_data_3d, k=10):
        print("ERROR: Correctness verification failed for 3D!")
        sys.exit(1)

    # Run experiments
    for dimension in [2, 3]:
        print(f"\n{'=' * 60}")
        print(f"EXPERIMENTS FOR {dimension}D")
        print("=" * 60)

        for n in DATASET_SIZES:
            print(f"\n--- Dataset: n={n:,}, dimension={dimension}D ---")

            dronz_list = get_or_generate_data(n, dimension=dimension, seed=SEED)

            run_bf = n <= BRUTE_FORCE_LIMIT
            run_kdtree_topk = n <= KDTREE_TOPK_LIMIT

            # Closest pair experiments
            print("\n[Closest Pair Experiments]")
            results = run_experiment_closest_pair(
                dronz_list, n, dimension, run_brute_force=run_bf
            )
            for r in results:
                save_benchmark_result(r, BENCHMARK_FILE)

            # Top-k experiments
            print("\n[Top-K Experiments]")
            for k_val in K_VALUES:
                results = run_experiment_top_k(
                    dronz_list, n, dimension, k_val,
                    run_brute_force=run_bf,
                    run_kdtree_topk=run_kdtree_topk
                )
                for r in results:
                    save_benchmark_result(r, BENCHMARK_FILE)

    print("\n" + "=" * 60)
    print(f"Experiments completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {BENCHMARK_FILE}")
    print("=" * 60)


def run_quick_test():
    """Run a quick test with smaller datasets."""
    print("Running quick test...")

    for dim in [2, 3]:
        print(f"\n--- {dim}D Quick Test ---")
        dronz_list = get_or_generate_data(5000, dimension=dim, seed=SEED)

        print("Brute Force Closest Pair:")
        result, time_taken, mem = benchmark(brute_force_closest_pair, dronz_list)
        print(f"  Distance: {result[2]:.6f}, Time: {time_taken:.4f}s")

        print("D&C Closest Pair:")
        result, time_taken, mem = benchmark(optimized_closest_pair, dronz_list)
        print(f"  Distance: {result[2]:.6f}, Time: {time_taken:.4f}s")

        print("KD-Tree Closest Pair:")
        result, time_taken, mem = benchmark(kd_tree_closest_pair, dronz_list)
        print(f"  Distance: {result[2]:.6f}, Time: {time_taken:.4f}s")

        print("Top-10 Brute Force:")
        result, time_taken, mem = benchmark(brute_force_top_k_pairs, dronz_list, 10)
        print(f"  Time: {time_taken:.4f}s")

        print("Top-10 KD-Tree:")
        result, time_taken, mem = benchmark(optimized_top_k_pairs, dronz_list, 10)
        print(f"  Time: {time_taken:.4f}s")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        run_quick_test()
    else:
        run_all_experiments()