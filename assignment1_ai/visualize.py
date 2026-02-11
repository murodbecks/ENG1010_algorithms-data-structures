"""
Visualization script for dronz collision prevention benchmark results.
Generates publication-ready graphs for the report.
"""

import os
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

# Configuration
FILES_DIR = "files"
BENCHMARK_FILE = os.path.join(FILES_DIR, "benchmark_results.csv")

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'brute_force': '#e74c3c',
    'divide_conquer': '#2ecc71',
    'kdtree': '#3498db',
    'brute_force_topk': '#e74c3c',
    'kdtree_topk': '#3498db',
    'grid_topk': '#9b59b6'
}
MARKERS = {
    'brute_force': 'o',
    'divide_conquer': 's',
    'kdtree': '^',
    'brute_force_topk': 'o',
    'kdtree_topk': '^',
    'grid_topk': 'd'
}
LABELS = {
    'brute_force': 'Brute Force O(n²)',
    'divide_conquer': 'Divide & Conquer O(n log n)',
    'kdtree': 'KD-Tree O(n log n)',
    'brute_force_topk': 'Brute Force O(n²)',
    'kdtree_topk': 'KD-Tree',
    'grid_topk': 'Grid-Based'
}


def load_results() -> List[Dict]:
    """Load benchmark results from CSV."""
    results = []
    with open(BENCHMARK_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'algorithm': row['algorithm'],
                'n': int(row['n']),
                'dimension': int(row['dimension']),
                'k': int(row['k']),
                'time_seconds': float(row['time_seconds']),
                'memory_peak_mb': float(row['memory_peak_mb']),
                'result_distance': float(row['result_distance'])
            })
    return results


def filter_results(results: List[Dict], **kwargs) -> List[Dict]:
    """Filter results by given criteria."""
    filtered = results
    for key, value in kwargs.items():
        if isinstance(value, list):
            filtered = [r for r in filtered if r.get(key) in value]
        else:
            filtered = [r for r in filtered if r.get(key) == value]
    return filtered


def plot_closest_pair_time_comparison(results: List[Dict]):
    """
    Plot 1: Time comparison for closest pair algorithms (2D and 3D).
    Shows scalability of brute force vs optimized algorithms.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, dim in enumerate([2, 3]):
        ax = axes[idx]

        for algo in ['brute_force', 'divide_conquer', 'kdtree']:
            data = filter_results(results, algorithm=algo, dimension=dim, k=1)
            if not data:
                continue

            ns = [d['n'] for d in data]
            times = [d['time_seconds'] for d in data]

            ax.plot(ns, times, marker=MARKERS[algo], color=COLORS[algo],
                    label=LABELS[algo], linewidth=2, markersize=8)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Number of Dronz (n)', fontsize=12)
        ax.set_ylabel('Execution Time (seconds)', fontsize=12)
        ax.set_title(f'Closest Pair - {dim}D', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, p: f'{int(x):,}' if x >= 1 else str(x)))

    plt.tight_layout()
    plt.savefig(os.path.join(FILES_DIR, 'closest_pair_time.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: closest_pair_time.png")


def plot_algorithm_speedup(results: List[Dict]):
    """
    Plot 2: Speedup of optimized algorithms over brute force.
    Demonstrates the practical benefit of O(n log n) vs O(n²).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.35
    dims = [2, 3]

    # Get common n values where brute force was run
    bf_data_2d = filter_results(results, algorithm='brute_force', dimension=2, k=1)
    common_ns = sorted(set(d['n'] for d in bf_data_2d))

    x = np.arange(len(common_ns))

    for i, dim in enumerate(dims):
        speedups = []
        for n in common_ns:
            bf = filter_results(results, algorithm='brute_force', dimension=dim, k=1, n=n)
            dc = filter_results(results, algorithm='divide_conquer', dimension=dim, k=1, n=n)

            if bf and dc:
                speedup = bf[0]['time_seconds'] / dc[0]['time_seconds']
                speedups.append(speedup)
            else:
                speedups.append(0)

        offset = (i - 0.5) * bar_width
        bars = ax.bar(x + offset, speedups, bar_width,
                      label=f'{dim}D', color=COLORS['divide_conquer'] if dim == 2 else COLORS['kdtree'])

        # Add value labels on bars
        for bar, speedup in zip(bars, speedups):
            if speedup > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                        f'{speedup:.0f}x', ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Number of Dronz (n)', fontsize=12)
    ax.set_ylabel('Speedup (Brute Force Time / D&C Time)', fontsize=12)
    ax.set_title('Divide & Conquer Speedup over Brute Force', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n:,}' for n in common_ns])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(FILES_DIR, 'speedup_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: speedup_comparison.png")


def plot_topk_scalability(results: List[Dict]):
    """
    Plot 3: Top-K algorithm scalability with different k values.
    Shows how execution time grows with k for optimized algorithm.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, dim in enumerate([2, 3]):
        ax = axes[idx]

        # For a fixed n, show how time varies with k
        # Use n=10000 where we have brute force comparison
        n_test = 10000

        for algo in ['brute_force_topk', 'kdtree_topk']:
            data = filter_results(results, algorithm=algo, dimension=dim, n=n_test)
            if not data:
                continue

            ks = sorted([d['k'] for d in data])
            times = [next(d['time_seconds'] for d in data if d['k'] == k) for k in ks]

            ax.plot(ks, times, marker=MARKERS[algo], color=COLORS[algo],
                    label=LABELS[algo], linewidth=2, markersize=8)

        ax.set_xlabel('k (number of pairs)', fontsize=12)
        ax.set_ylabel('Execution Time (seconds)', fontsize=12)
        ax.set_title(f'Top-K Performance (n={n_test:,}, {dim}D)', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FILES_DIR, 'topk_scalability.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: topk_scalability.png")


def plot_memory_usage(results: List[Dict]):
    """
    Plot 4: Memory usage comparison across algorithms and dataset sizes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Closest pair memory usage
    ax = axes[0]
    for algo in ['divide_conquer', 'kdtree']:
        for dim in [2, 3]:
            data = filter_results(results, algorithm=algo, dimension=dim, k=1)
            if not data:
                continue

            ns = [d['n'] for d in data]
            memory = [d['memory_peak_mb'] for d in data]

            linestyle = '-' if dim == 2 else '--'
            ax.plot(ns, memory, marker=MARKERS[algo], color=COLORS[algo],
                    label=f'{LABELS[algo]} ({dim}D)', linewidth=2,
                    markersize=8, linestyle=linestyle)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Dronz (n)', fontsize=12)
    ax.set_ylabel('Peak Memory (MB)', fontsize=12)
    ax.set_title('Memory Usage - Closest Pair', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: Top-K memory usage (k=100)
    ax = axes[1]
    k_test = 100

    for dim in [2, 3]:
        data = filter_results(results, algorithm='kdtree_topk', dimension=dim, k=k_test)
        if not data:
            continue

        ns = [d['n'] for d in data]
        memory = [d['memory_peak_mb'] for d in data]

        linestyle = '-' if dim == 2 else '--'
        color = COLORS['kdtree'] if dim == 2 else COLORS['divide_conquer']
        ax.plot(ns, memory, marker='^', color=color,
                label=f'KD-Tree Top-{k_test} ({dim}D)', linewidth=2,
                markersize=8, linestyle=linestyle)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Dronz (n)', fontsize=12)
    ax.set_ylabel('Peak Memory (MB)', fontsize=12)
    ax.set_title(f'Memory Usage - Top-{k_test} Pairs', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FILES_DIR, 'memory_usage.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: memory_usage.png")


def plot_2d_vs_3d_comparison(results: List[Dict]):
    """
    Plot 5: Direct comparison of 2D vs 3D performance.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Compare divide_conquer performance in 2D vs 3D
    data_2d = filter_results(results, algorithm='divide_conquer', dimension=2, k=1)
    data_3d = filter_results(results, algorithm='divide_conquer', dimension=3, k=1)

    ns_2d = [d['n'] for d in data_2d]
    times_2d = [d['time_seconds'] for d in data_2d]

    ns_3d = [d['n'] for d in data_3d]
    times_3d = [d['time_seconds'] for d in data_3d]

    ax.plot(ns_2d, times_2d, marker='s', color=COLORS['divide_conquer'],
            label='2D', linewidth=2, markersize=8)
    ax.plot(ns_3d, times_3d, marker='^', color=COLORS['kdtree'],
            label='3D', linewidth=2, markersize=8)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Dronz (n)', fontsize=12)
    ax.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax.set_title('Divide & Conquer: 2D vs 3D Performance', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add ratio annotations
    common_ns = sorted(set(ns_2d) & set(ns_3d))
    for n in common_ns:
        t2d = next(d['time_seconds'] for d in data_2d if d['n'] == n)
        t3d = next(d['time_seconds'] for d in data_3d if d['n'] == n)
        ratio = t3d / t2d
        ax.annotate(f'{ratio:.1f}x', xy=(n, t3d), xytext=(5, 5),
                    textcoords='offset points', fontsize=9, color='gray')

    plt.tight_layout()
    plt.savefig(os.path.join(FILES_DIR, '2d_vs_3d.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 2d_vs_3d.png")


def plot_complexity_verification(results: List[Dict]):
    """
    Plot 6: Empirical complexity verification.
    Plots time/n² for brute force and time/(n log n) for optimized.
    Should show roughly constant lines if complexity is correct.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Brute force - should be O(n²)
    ax = axes[0]
    for dim in [2, 3]:
        data = filter_results(results, algorithm='brute_force', dimension=dim, k=1)
        if not data:
            continue

        ns = np.array([d['n'] for d in data])
        times = np.array([d['time_seconds'] for d in data])

        # Normalize by n²
        normalized = times / (ns ** 2) * 1e9  # Convert to nanoseconds per n²

        ax.plot(ns, normalized, marker='o', label=f'{dim}D', linewidth=2, markersize=8)

    ax.set_xscale('log')
    ax.set_xlabel('Number of Dronz (n)', fontsize=12)
    ax.set_ylabel('Time / n² (normalized)', fontsize=12)
    ax.set_title('Brute Force: O(n²) Verification', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Divide & Conquer - should be O(n log n)
    ax = axes[1]
    for dim in [2, 3]:
        data = filter_results(results, algorithm='divide_conquer', dimension=dim, k=1)
        if not data:
            continue

        ns = np.array([d['n'] for d in data])
        times = np.array([d['time_seconds'] for d in data])

        # Normalize by n log n
        normalized = times / (ns * np.log2(ns)) * 1e6

        ax.plot(ns, normalized, marker='s', label=f'{dim}D', linewidth=2, markersize=8)

    ax.set_xscale('log')
    ax.set_xlabel('Number of Dronz (n)', fontsize=12)
    ax.set_ylabel('Time / (n log n) (normalized)', fontsize=12)
    ax.set_title('Divide & Conquer: O(n log n) Verification', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FILES_DIR, 'complexity_verification.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: complexity_verification.png")


def create_summary_table(results: List[Dict]):
    """Generate a summary table of key results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    # Closest pair summary
    print("\n--- Closest Pair (k=1) ---")
    print(f"{'Algorithm':<20} {'Dim':<5} {'n':<12} {'Time (s)':<12} {'Memory (MB)':<12}")
    print("-" * 65)

    for algo in ['brute_force', 'divide_conquer', 'kdtree']:
        for dim in [2, 3]:
            data = filter_results(results, algorithm=algo, dimension=dim, k=1)
            for d in sorted(data, key=lambda x: x['n']):
                print(f"{algo:<20} {dim:<5} {d['n']:<12,} {d['time_seconds']:<12.4f} {d['memory_peak_mb']:<12.4f}")

    # Key findings
    print("\n--- Key Findings ---")

    # Max speedup
    bf_10k_2d = filter_results(results, algorithm='brute_force', dimension=2, k=1, n=10000)
    dc_10k_2d = filter_results(results, algorithm='divide_conquer', dimension=2, k=1, n=10000)
    if bf_10k_2d and dc_10k_2d:
        speedup = bf_10k_2d[0]['time_seconds'] / dc_10k_2d[0]['time_seconds']
        print(f"• D&C speedup over brute force at n=10,000 (2D): {speedup:.0f}x")

    # Largest dataset processed
    dc_data = filter_results(results, algorithm='divide_conquer', k=1)
    if dc_data:
        max_n = max(d['n'] for d in dc_data)
        max_time = next(d['time_seconds'] for d in dc_data if d['n'] == max_n)
        print(f"• Largest dataset processed by D&C: n={max_n:,} in {max_time:.2f}s")

    # 3D overhead
    dc_2d = filter_results(results, algorithm='divide_conquer', dimension=2, k=1, n=100000)
    dc_3d = filter_results(results, algorithm='divide_conquer', dimension=3, k=1, n=100000)
    if dc_2d and dc_3d:
        overhead = dc_3d[0]['time_seconds'] / dc_2d[0]['time_seconds']
        print(f"• 3D overhead vs 2D at n=100,000: {overhead:.1f}x slower")


def main():
    """Generate all visualizations."""
    os.makedirs(FILES_DIR, exist_ok=True)

    print("Loading benchmark results...")
    results = load_results()
    print(f"Loaded {len(results)} benchmark records.\n")

    print("Generating visualizations...")
    plot_closest_pair_time_comparison(results)
    plot_algorithm_speedup(results)
    plot_topk_scalability(results)
    plot_memory_usage(results)
    plot_2d_vs_3d_comparison(results)
    plot_complexity_verification(results)

    create_summary_table(results)

    print("\n" + "=" * 80)
    print("All visualizations saved to files/ directory:")
    print("  • closest_pair_time.png    - Algorithm time comparison")
    print("  • speedup_comparison.png   - D&C speedup over brute force")
    print("  • topk_scalability.png     - Top-K performance analysis")
    print("  • memory_usage.png         - Memory consumption comparison")
    print("  • 2d_vs_3d.png             - Dimension impact on performance")
    print("  • complexity_verification.png - Empirical complexity analysis")
    print("=" * 80)


if __name__ == "__main__":
    main()