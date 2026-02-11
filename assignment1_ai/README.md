# Dronz Collision Prevention System

A Python application for detecting closest pairs of dronz (drone objects) using efficient algorithms. This project compares brute-force O(n²) approaches with optimized O(n log n) divide-and-conquer and KD-tree algorithms.

## Project Structure

```
├── data/                  # Generated datasets (CSV files)
├── files/                 # Output files (benchmarks, graphs)
├── algorithm.py           # Core algorithms (brute-force, D&C, KD-tree)
├── utils.py               # Utilities (data generation, I/O, benchmarking)
├── main.py                # Experiment runner
├── live.py                # Real-time tracking simulation (Task 4)
├── visualize.py           # Graph generation for reports
├── requirements.txt       # Python dependencies
└── README.md
```

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run Experiments

```bash
# Quick test (small datasets)
python main.py --quick

# Full benchmark suite (takes several hours for large datasets)
python main.py
```

### Generate Visualizations

```bash
# After running experiments
python visualize.py
```

### Live Tracking Demo (Task 4)

```bash
# Quick demo
python live.py --demo

# Full simulation
python live.py
```

## Algorithms

| Algorithm | Task | Time Complexity | Space Complexity |
|-----------|------|-----------------|------------------|
| Brute Force | Closest Pair | O(n²) | O(1) |
| Divide & Conquer | Closest Pair | O(n log n) | O(n) |
| KD-Tree | Closest Pair | O(n log n) avg | O(n) |
| Brute Force | Top-K Pairs | O(n² log k) | O(k) |
| KD-Tree | Top-K Pairs | O(n log n) avg | O(n + k) |

## Output Files

After running experiments and visualization:

- `files/benchmark_results.csv` - Raw benchmark data
- `files/closest_pair_time.png` - Algorithm time comparison
- `files/speedup_comparison.png` - Speedup analysis
- `files/topk_scalability.png` - Top-K performance
- `files/memory_usage.png` - Memory consumption
- `files/2d_vs_3d.png` - Dimension comparison
- `files/complexity_verification.png` - Empirical complexity verification

## Key Results

- **Divide & Conquer** achieves 800-900x speedup over brute force at n=10,000
- Successfully processes **1 million+ dronz** in under 30 seconds
- **3D** operations are approximately 2-3x slower than 2D
- Memory usage scales linearly O(n) for optimized algorithms
