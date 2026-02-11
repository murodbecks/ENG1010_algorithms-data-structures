"""
Live dronz tracking system with incremental updates.
Task 4: Handles dynamic dronz locations without recalculating from scratch.

Approach:
- Uses a spatial grid for O(1) cell lookups
- When a dronz moves, only update affected grid cells
- Maintains a sorted list of top-k pairs, updating incrementally
"""

import random
import time
import heapq
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from utils import Dronz, euclidean_distance, generate_dronz_data


@dataclass
class DronzPair:
    """Represents a pair of dronz with their distance."""
    id1: int
    id2: int
    distance: float

    def __lt__(self, other):
        # For max-heap behavior (negate for min comparison)
        return self.distance > other.distance

    def __eq__(self, other):
        return {self.id1, self.id2} == {other.id1, other.id2}

    def __hash__(self):
        return hash(frozenset([self.id1, self.id2]))

    @property
    def key(self) -> frozenset:
        return frozenset([self.id1, self.id2])


class SpatialGrid:
    """
    Spatial hash grid for efficient neighbor queries.
    Divides space into cells of size `cell_size`.
    """

    def __init__(self, cell_size: float, dimension: int = 3):
        self.cell_size = cell_size
        self.dimension = dimension
        self.grid: Dict[Tuple[int, ...], Set[int]] = defaultdict(set)
        self.dronz_positions: Dict[int, Dronz] = {}
        self.dronz_cells: Dict[int, Tuple[int, ...]] = {}

    def _get_cell(self, coords: Tuple[float, ...]) -> Tuple[int, ...]:
        """Get cell coordinates for a position."""
        return tuple(int(c // self.cell_size) for c in coords)

    def _get_neighbor_cells(self, cell: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """Get all neighboring cells (including self) - 27 cells in 3D, 9 in 2D."""
        neighbors = []
        ranges = [range(c - 1, c + 2) for c in cell]

        def generate_combos(idx, current):
            if idx == len(ranges):
                neighbors.append(tuple(current))
                return
            for val in ranges[idx]:
                generate_combos(idx + 1, current + [val])

        generate_combos(0, [])
        return neighbors

    def insert(self, dronz: Dronz) -> None:
        """Insert a dronz into the grid."""
        cell = self._get_cell(dronz.coords)
        self.grid[cell].add(dronz.id)
        self.dronz_positions[dronz.id] = dronz
        self.dronz_cells[dronz.id] = cell

    def remove(self, dronz_id: int) -> Optional[Dronz]:
        """Remove a dronz from the grid."""
        if dronz_id not in self.dronz_positions:
            return None

        dronz = self.dronz_positions[dronz_id]
        cell = self.dronz_cells[dronz_id]

        self.grid[cell].discard(dronz_id)
        if not self.grid[cell]:
            del self.grid[cell]

        del self.dronz_positions[dronz_id]
        del self.dronz_cells[dronz_id]

        return dronz

    def update(self, dronz_id: int, new_coords: Tuple[float, ...]) -> Dronz:
        """Update dronz position. Returns the updated Dronz object."""
        old_dronz = self.remove(dronz_id)
        new_dronz = Dronz(id=dronz_id, coords=new_coords)
        self.insert(new_dronz)
        return new_dronz

    def get_neighbors(self, dronz_id: int) -> List[Dronz]:
        """Get all dronz in neighboring cells."""
        if dronz_id not in self.dronz_cells:
            return []

        cell = self.dronz_cells[dronz_id]
        neighbor_cells = self._get_neighbor_cells(cell)

        neighbors = []
        for nc in neighbor_cells:
            for other_id in self.grid.get(nc, set()):
                if other_id != dronz_id:
                    neighbors.append(self.dronz_positions[other_id])

        return neighbors

    def get_all_dronz(self) -> List[Dronz]:
        """Get all dronz in the grid."""
        return list(self.dronz_positions.values())


class LiveDronzTracker:
    """
    Live tracking system for dronz that maintains top-k closest pairs
    with incremental updates.
    """

    def __init__(
        self,
        k: int = 10,
        dimension: int = 3,
        cell_size: float = 100.0,
        width: float = 10000.0
    ):
        """
        Initialize the tracker.

        Args:
            k: Number of top pairs to track
            dimension: 2 or 3
            cell_size: Size of spatial grid cells (should be ~expected closest distance)
            width: Coordinate range
        """
        self.k = k
        self.dimension = dimension
        self.width = width
        self.grid = SpatialGrid(cell_size, dimension)

        # Top-k pairs storage
        self.top_k_pairs: List[DronzPair] = []  # Max-heap
        self.pair_set: Set[frozenset] = set()  # Quick lookup

        # Track all pairs involving each dronz for quick invalidation
        self.dronz_pairs: Dict[int, Set[frozenset]] = defaultdict(set)

    def _add_pair_to_topk(self, pair: DronzPair) -> bool:
        """
        Try to add a pair to top-k.
        Returns True if pair was added.
        """
        if pair.key in self.pair_set:
            return False

        if len(self.top_k_pairs) < self.k:
            heapq.heappush(self.top_k_pairs, pair)
            self.pair_set.add(pair.key)
            self.dronz_pairs[pair.id1].add(pair.key)
            self.dronz_pairs[pair.id2].add(pair.key)
            return True
        elif pair.distance < self.top_k_pairs[0].distance:
            # Remove worst pair
            old_pair = heapq.heapreplace(self.top_k_pairs, pair)
            self.pair_set.remove(old_pair.key)
            self.dronz_pairs[old_pair.id1].discard(old_pair.key)
            self.dronz_pairs[old_pair.id2].discard(old_pair.key)

            # Add new pair
            self.pair_set.add(pair.key)
            self.dronz_pairs[pair.id1].add(pair.key)
            self.dronz_pairs[pair.id2].add(pair.key)
            return True

        return False

    def _remove_pairs_involving(self, dronz_id: int) -> List[frozenset]:
        """Remove all pairs involving a dronz from top-k."""
        removed = []
        pairs_to_remove = list(self.dronz_pairs[dronz_id])

        for pair_key in pairs_to_remove:
            if pair_key in self.pair_set:
                self.pair_set.remove(pair_key)
                removed.append(pair_key)

                # Remove from other dronz's pair set
                for other_id in pair_key:
                    if other_id != dronz_id:
                        self.dronz_pairs[other_id].discard(pair_key)

        self.dronz_pairs[dronz_id].clear()

        # Rebuild heap without removed pairs
        self.top_k_pairs = [p for p in self.top_k_pairs if p.key in self.pair_set]
        heapq.heapify(self.top_k_pairs)

        return removed

    def _find_local_pairs(self, dronz: Dronz) -> List[DronzPair]:
        """Find pairs between a dronz and its neighbors."""
        neighbors = self.grid.get_neighbors(dronz.id)
        pairs = []

        for neighbor in neighbors:
            dist = euclidean_distance(dronz, neighbor)
            pairs.append(DronzPair(
                id1=min(dronz.id, neighbor.id),
                id2=max(dronz.id, neighbor.id),
                distance=dist
            ))

        return pairs

    def initialize(self, dronz_list: List[Dronz]) -> None:
        """Initialize tracker with initial dronz positions."""
        print(f"Initializing tracker with {len(dronz_list)} dronz...")

        # Insert all dronz into grid
        for dronz in dronz_list:
            self.grid.insert(dronz)

        # Find initial top-k pairs
        # For each dronz, check pairs with neighbors
        all_pairs = []
        seen = set()

        for dronz in dronz_list:
            local_pairs = self._find_local_pairs(dronz)
            for pair in local_pairs:
                if pair.key not in seen:
                    seen.add(pair.key)
                    all_pairs.append(pair)

        # Sort and take top-k
        all_pairs.sort(key=lambda p: p.distance)
        for pair in all_pairs[:self.k]:
            self._add_pair_to_topk(pair)

        print(f"Initialization complete. Tracking {len(self.top_k_pairs)} pairs.")

    def update_dronz_position(
        self,
        dronz_id: int,
        new_coords: Tuple[float, ...]
    ) -> None:
        """
        Update a dronz's position incrementally.
        Only recalculates pairs involving this dronz and its neighbors.
        """
        # Remove old pairs involving this dronz from top-k
        self._remove_pairs_involving(dronz_id)

        # Update position in grid
        updated_dronz = self.grid.update(dronz_id, new_coords)

        # Find new candidate pairs with neighbors
        new_pairs = self._find_local_pairs(updated_dronz)

        # Try to add each to top-k
        for pair in sorted(new_pairs, key=lambda p: p.distance):
            self._add_pair_to_topk(pair)

        # If we don't have k pairs, need to do broader search
        if len(self.top_k_pairs) < self.k:
            self._refill_topk()

    def _refill_topk(self) -> None:
        """Refill top-k if we don't have enough pairs."""
        all_dronz = self.grid.get_all_dronz()
        seen = set(self.pair_set)

        candidates = []
        for dronz in all_dronz:
            for pair in self._find_local_pairs(dronz):
                if pair.key not in seen:
                    seen.add(pair.key)
                    candidates.append(pair)

        candidates.sort(key=lambda p: p.distance)
        for pair in candidates:
            if len(self.top_k_pairs) >= self.k:
                break
            self._add_pair_to_topk(pair)

    def get_top_k_pairs(self) -> List[Tuple[DronzPair, Dronz, Dronz]]:
        """Get current top-k closest pairs with full dronz info."""
        result = []
        for pair in sorted(self.top_k_pairs, key=lambda p: p.distance):
            d1 = self.grid.dronz_positions.get(pair.id1)
            d2 = self.grid.dronz_positions.get(pair.id2)
            if d1 and d2:
                result.append((pair, d1, d2))
        return result

    def get_worst_distance(self) -> float:
        """Get the distance of the k-th closest pair."""
        if self.top_k_pairs:
            return self.top_k_pairs[0].distance
        return float('inf')


def simulate_live_tracking(
    n_dronz: int = 1000,
    k: int = 10,
    dimension: int = 3,
    n_updates: int = 100,
    update_interval: float = 1.0,
    dronz_per_update: int = 10,
    movement_range: float = 50.0,
    width: float = 10000.0
):
    """
    Simulate live dronz tracking with periodic position updates.

    Args:
        n_dronz: Number of dronz
        k: Number of top pairs to track
        dimension: 2D or 3D
        n_updates: Number of update cycles
        update_interval: Seconds between updates
        dronz_per_update: How many dronz move each update
        movement_range: Max distance a dronz can move
        width: Coordinate space width
    """
    print("=" * 60)
    print("LIVE DRONZ TRACKING SIMULATION")
    print("=" * 60)
    print(f"Dronz: {n_dronz}, K: {k}, Dimension: {dimension}D")
    print(f"Updates: {n_updates}, Interval: {update_interval}s")
    print(f"Dronz per update: {dronz_per_update}")
    print("=" * 60)

    # Initialize
    dronz_list = generate_dronz_data(n_dronz, dimension, width, seed=42)

    # Cell size heuristic: based on average distance between random points
    cell_size = width / (n_dronz ** (1 / dimension)) * 2

    tracker = LiveDronzTracker(
        k=k,
        dimension=dimension,
        cell_size=cell_size,
        width=width
    )
    tracker.initialize(dronz_list)

    print(f"\nInitial Top-{k} Pairs:")
    for pair, d1, d2 in tracker.get_top_k_pairs():
        print(f"  Dronz {pair.id1} <-> {pair.id2}: {pair.distance:.4f}")

    # Simulation loop
    for update_num in range(n_updates):
        print(f"\n--- Update {update_num + 1}/{n_updates} ---")

        # Select random dronz to move
        moving_ids = random.sample(range(n_dronz), min(dronz_per_update, n_dronz))

        update_start = time.perf_counter()

        for dronz_id in moving_ids:
            current = tracker.grid.dronz_positions[dronz_id]

            # Generate new position (random walk within range)
            new_coords = tuple(
                max(0, min(width, c + random.uniform(-movement_range, movement_range)))
                for c in current.coords
            )

            tracker.update_dronz_position(dronz_id, new_coords)

        update_time = time.perf_counter() - update_start

        print(f"Updated {len(moving_ids)} dronz in {update_time * 1000:.2f}ms")
        print(f"Top-{k} pairs (worst distance: {tracker.get_worst_distance():.4f}):")

        for i, (pair, d1, d2) in enumerate(tracker.get_top_k_pairs()[:5]):
            print(f"  {i + 1}. Dronz {pair.id1} <-> {pair.id2}: {pair.distance:.4f}")

        if update_interval > 0 and update_num < n_updates - 1:
            time.sleep(update_interval)

    print("\n" + "=" * 60)
    print("Simulation Complete")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        # Quick demo mode
        simulate_live_tracking(
            n_dronz=500,
            k=5,
            dimension=3,
            n_updates=5,
            update_interval=0.5,
            dronz_per_update=20
        )
    else:
        # Full simulation
        simulate_live_tracking(
            n_dronz=10000,
            k=10,
            dimension=3,
            n_updates=50,
            update_interval=1.0,
            dronz_per_update=100
        )