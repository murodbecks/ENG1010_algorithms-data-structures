"""
Algorithms for finding closest dronz pairs.
Includes brute-force (O(n²)) and optimized divide-and-conquer (O(n log n)) solutions.
"""

import heapq
from typing import List, Tuple, Optional, Set
from utils import Dronz, euclidean_distance


# Type alias for pair result: ((id1, coords1), (id2, coords2), distance)
PairResult = Tuple[Tuple[int, Tuple[float, ...]], Tuple[int, Tuple[float, ...]], float]


def format_result(d1: Dronz, d2: Dronz, dist: float) -> PairResult:
    """Format a pair of dronz into standard result format."""
    # Ensure consistent ordering (lower ID first) for tie resolution
    if d1.id > d2.id:
        d1, d2 = d2, d1
    return ((d1.id, d1.coords), (d2.id, d2.coords), dist)


# =============================================================================
# BRUTE FORCE ALGORITHMS - O(n²)
# =============================================================================

def brute_force_closest_pair(dronz_list: List[Dronz]) -> Optional[PairResult]:
    """
    Find the closest pair of dronz using brute force.
    Time Complexity: O(n²)
    Space Complexity: O(1)

    Args:
        dronz_list: List of Dronz objects

    Returns:
        Tuple of ((id1, coords1), (id2, coords2), distance) or None if < 2 dronz
    """
    n = len(dronz_list)
    if n < 2:
        return None

    min_dist = float('inf')
    closest_pair = None

    for i in range(n):
        for j in range(i + 1, n):
            dist = euclidean_distance(dronz_list[i], dronz_list[j])
            if dist < min_dist or (dist == min_dist and closest_pair is None):
                min_dist = dist
                closest_pair = (dronz_list[i], dronz_list[j])
            # Tie resolution: prefer pair with smaller IDs
            elif dist == min_dist and closest_pair is not None:
                curr_ids = sorted([closest_pair[0].id, closest_pair[1].id])
                new_ids = sorted([dronz_list[i].id, dronz_list[j].id])
                if new_ids < curr_ids:
                    closest_pair = (dronz_list[i], dronz_list[j])

    return format_result(closest_pair[0], closest_pair[1], min_dist)


def brute_force_top_k_pairs(
    dronz_list: List[Dronz],
    k: int
) -> List[PairResult]:
    """
    Find the k closest pairs of dronz using brute force.
    Time Complexity: O(n² log k) using a max-heap of size k
    Space Complexity: O(k)

    Args:
        dronz_list: List of Dronz objects
        k: Number of closest pairs to find

    Returns:
        List of k closest pairs, sorted by distance (ascending)
    """
    n = len(dronz_list)
    if n < 2:
        return []

    # Max-heap (negate distance for max-heap behavior)
    # Heap element: (-distance, id1, id2, dronz1, dronz2)
    heap = []

    for i in range(n):
        for j in range(i + 1, n):
            dist = euclidean_distance(dronz_list[i], dronz_list[j])
            d1, d2 = dronz_list[i], dronz_list[j]

            # Ensure consistent ordering for heap comparison
            if d1.id > d2.id:
                d1, d2 = d2, d1

            entry = (-dist, d1.id, d2.id, d1, d2)

            if len(heap) < k:
                heapq.heappush(heap, entry)
            elif -dist > heap[0][0]:  # dist < -heap[0][0] (current max)
                heapq.heapreplace(heap, entry)

    # Extract and sort results
    results = []
    for neg_dist, id1, id2, d1, d2 in heap:
        results.append(format_result(d1, d2, -neg_dist))

    # Sort by distance, then by IDs for tie resolution
    results.sort(key=lambda x: (x[2], x[0][0], x[1][0]))

    return results


# =============================================================================
# OPTIMIZED CLOSEST PAIR - O(n log n) using Divide and Conquer
# =============================================================================

def _closest_pair_strip(
    strip: List[Dronz],
    delta: float,
    dimension: int
) -> Tuple[Optional[Dronz], Optional[Dronz], float]:
    """
    Find closest pair in the strip region.
    The strip is sorted by y-coordinate.
    We only need to check at most 7 points ahead (for 2D) or 15 (for 3D).
    """
    min_dist = delta
    best_pair = (None, None)

    # Points to check: 7 for 2D, 15 for 3D (based on packing argument)
    max_check = 7 if dimension == 2 else 15

    for i in range(len(strip)):
        j = i + 1
        while j < len(strip) and (strip[j].coords[1] - strip[i].coords[1]) < min_dist:
            if j - i > max_check:
                break
            dist = euclidean_distance(strip[i], strip[j])
            if dist < min_dist:
                min_dist = dist
                best_pair = (strip[i], strip[j])
            j += 1

    return best_pair[0], best_pair[1], min_dist


def _divide_and_conquer_closest(
    points_x: List[Dronz],
    points_y: List[Dronz],
    dimension: int
) -> Tuple[Dronz, Dronz, float]:
    """
    Recursive divide and conquer for closest pair.

    Args:
        points_x: Points sorted by x-coordinate
        points_y: Points sorted by y-coordinate
        dimension: 2 or 3

    Returns:
        Tuple of (dronz1, dronz2, distance)
    """
    n = len(points_x)

    # Base case: use brute force for small inputs
    if n <= 3:
        min_dist = float('inf')
        best = (None, None)
        for i in range(n):
            for j in range(i + 1, n):
                dist = euclidean_distance(points_x[i], points_x[j])
                if dist < min_dist:
                    min_dist = dist
                    best = (points_x[i], points_x[j])
        return best[0], best[1], min_dist

    # Divide
    mid = n // 2
    mid_point = points_x[mid]

    # Split points_y into left and right based on x-coordinate
    left_x = points_x[:mid]
    right_x = points_x[mid:]

    left_set = set(d.id for d in left_x)
    left_y = [d for d in points_y if d.id in left_set]
    right_y = [d for d in points_y if d.id not in left_set]

    # Conquer
    d1_l, d2_l, dist_l = _divide_and_conquer_closest(left_x, left_y, dimension)
    d1_r, d2_r, dist_r = _divide_and_conquer_closest(right_x, right_y, dimension)

    # Find minimum of left and right
    if dist_l < dist_r:
        delta = dist_l
        best_pair = (d1_l, d2_l)
    else:
        delta = dist_r
        best_pair = (d1_r, d2_r)

    # Combine: check strip
    strip = [d for d in points_y if abs(d.coords[0] - mid_point.coords[0]) < delta]

    d1_s, d2_s, dist_s = _closest_pair_strip(strip, delta, dimension)

    if dist_s < delta:
        return d1_s, d2_s, dist_s

    return best_pair[0], best_pair[1], delta


def optimized_closest_pair(dronz_list: List[Dronz]) -> Optional[PairResult]:
    """
    Find closest pair using divide and conquer.
    Time Complexity: O(n log n)
    Space Complexity: O(n)

    Args:
        dronz_list: List of Dronz objects

    Returns:
        Tuple of ((id1, coords1), (id2, coords2), distance)
    """
    if len(dronz_list) < 2:
        return None

    dimension = dronz_list[0].dimension

    # Pre-sort by x and y
    points_x = sorted(dronz_list, key=lambda d: d.coords[0])
    if dimension == 2:
        points_y = sorted(dronz_list, key=lambda d: d.coords[1])
    else:
        points_y = sorted(dronz_list, key=lambda d: (d.coords[1], d.coords[2]))

    d1, d2, dist = _divide_and_conquer_closest(points_x, points_y, dimension)

    return format_result(d1, d2, dist)


# =============================================================================
# KD-TREE IMPLEMENTATION
# =============================================================================

class KDNode:
    """Node for KD-Tree."""
    __slots__ = ['dronz', 'left', 'right', 'axis']

    def __init__(self, dronz: Dronz, left=None, right=None, axis=0):
        self.dronz = dronz
        self.left = left
        self.right = right
        self.axis = axis


def build_kd_tree(dronz_list: List[Dronz], depth: int = 0) -> Optional[KDNode]:
    """
    Build a KD-Tree from list of dronz.
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    if not dronz_list:
        return None

    dimension = dronz_list[0].dimension
    axis = depth % dimension

    # Sort and find median
    dronz_list = sorted(dronz_list, key=lambda d: d.coords[axis])
    mid = len(dronz_list) // 2

    return KDNode(
        dronz=dronz_list[mid],
        left=build_kd_tree(dronz_list[:mid], depth + 1),
        right=build_kd_tree(dronz_list[mid + 1:], depth + 1),
        axis=axis
    )


def _kd_tree_nearest_k(
    node: Optional[KDNode],
    target: Dronz,
    heap: List[Tuple[float, int, Dronz]],
    k: int,
    exclude_ids: Set[int]
) -> None:
    """
    Find k nearest neighbors in KD-Tree using iterative deepening.
    Uses a max-heap to track k closest points.

    Args:
        node: Current KD-Tree node
        target: Target dronz to find neighbors for
        heap: Max-heap of (-distance, id, dronz)
        k: Number of neighbors to find
        exclude_ids: Set of IDs to exclude (e.g., the target itself)
    """
    if node is None:
        return

    current = node.dronz

    # Process current node if not excluded
    if current.id not in exclude_ids:
        dist = euclidean_distance(target, current)

        if len(heap) < k:
            heapq.heappush(heap, (-dist, current.id, current))
        elif dist < -heap[0][0]:
            heapq.heapreplace(heap, (-dist, current.id, current))

    # Determine which subtree to search first
    axis = node.axis
    diff = target.coords[axis] - current.coords[axis]

    if diff <= 0:
        first, second = node.left, node.right
    else:
        first, second = node.right, node.left

    # Search closer subtree first
    _kd_tree_nearest_k(first, target, heap, k, exclude_ids)

    # Check if we need to search the other subtree
    # We need to search if: distance to splitting plane < current worst distance
    # OR we haven't found k neighbors yet
    worst_dist = -heap[0][0] if len(heap) == k else float('inf')
    if abs(diff) < worst_dist:
        _kd_tree_nearest_k(second, target, heap, k, exclude_ids)


def kd_tree_closest_pair(dronz_list: List[Dronz]) -> Optional[PairResult]:
    """
    Find closest pair using KD-Tree.
    Time Complexity: O(n log n) average
    Space Complexity: O(n)
    """
    if len(dronz_list) < 2:
        return None

    tree = build_kd_tree(dronz_list.copy())

    min_dist = float('inf')
    best_pair = None

    for dronz in dronz_list:
        heap = []
        _kd_tree_nearest_k(tree, dronz, heap, 1, {dronz.id})

        if heap:
            dist = -heap[0][0]
            neighbor = heap[0][2]
            if dist < min_dist:
                min_dist = dist
                best_pair = (dronz, neighbor)

    if best_pair:
        return format_result(best_pair[0], best_pair[1], min_dist)
    return None


# =============================================================================
# OPTIMIZED TOP-K USING KD-TREE
# =============================================================================

def optimized_top_k_pairs(
    dronz_list: List[Dronz],
    k: int
) -> List[PairResult]:
    """
    Find top-k closest pairs using KD-Tree for efficient neighbor searches.

    Strategy:
    1. Build KD-Tree
    2. For each point, find its nearest neighbors (search for sqrt(k) + buffer neighbors)
    3. Collect all candidate pairs
    4. Use a heap to extract top-k

    Time Complexity: O(n log n + n * sqrt(k) * log n) average
    Space Complexity: O(n + k)

    Args:
        dronz_list: List of Dronz objects
        k: Number of closest pairs to find

    Returns:
        List of k closest pairs, sorted by distance (ascending)
    """
    n = len(dronz_list)
    if n < 2:
        return []

    max_pairs = n * (n - 1) // 2
    k = min(k, max_pairs)

    # Build KD-Tree
    tree = build_kd_tree(dronz_list.copy())

    # Determine how many neighbors to search per point
    # We need enough to cover the k closest pairs
    # Heuristic: search for at least sqrt(2k) + some buffer per point
    import math
    neighbors_per_point = max(int(math.sqrt(2 * k)) + 10, k)
    neighbors_per_point = min(neighbors_per_point, n - 1)

    # Collect all candidate pairs
    seen_pairs: Set[frozenset] = set()
    # Max-heap for top-k: (-distance, id1, id2, dronz1, dronz2)
    result_heap = []

    for dronz in dronz_list:
        # Find nearest neighbors
        neighbor_heap = []
        _kd_tree_nearest_k(tree, dronz, neighbor_heap, neighbors_per_point, {dronz.id})

        # Process each neighbor
        for neg_dist, neighbor_id, neighbor in neighbor_heap:
            pair_key = frozenset([dronz.id, neighbor_id])

            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            dist = -neg_dist
            d1, d2 = (dronz, neighbor) if dronz.id < neighbor.id else (neighbor, dronz)
            entry = (-dist, d1.id, d2.id, d1, d2)

            if len(result_heap) < k:
                heapq.heappush(result_heap, entry)
            elif -dist > result_heap[0][0]:  # dist < current worst
                heapq.heapreplace(result_heap, entry)

    # Extract and sort results
    results = []
    for neg_dist, id1, id2, d1, d2 in result_heap:
        results.append(format_result(d1, d2, -neg_dist))

    results.sort(key=lambda x: (x[2], x[0][0], x[1][0]))

    return results


# =============================================================================
# GRID-BASED TOP-K (Alternative approach)
# =============================================================================

def grid_based_top_k_pairs(
    dronz_list: List[Dronz],
    k: int,
    initial_cell_size: Optional[float] = None
) -> List[PairResult]:
    """
    Find top-k closest pairs using spatial grid hashing.

    Strategy:
    1. Place all points in a grid
    2. For each point, only compute distances to points in same/adjacent cells
    3. Adaptively refine grid if needed

    Time Complexity: O(n) average for uniformly distributed points
    Space Complexity: O(n + k)

    Args:
        dronz_list: List of Dronz objects
        k: Number of closest pairs to find
        initial_cell_size: Initial grid cell size (auto-computed if None)

    Returns:
        List of k closest pairs, sorted by distance (ascending)
    """
    from collections import defaultdict

    n = len(dronz_list)
    if n < 2:
        return []

    max_pairs = n * (n - 1) // 2
    k = min(k, max_pairs)

    dimension = dronz_list[0].dimension

    # Compute bounding box
    min_coords = [min(d.coords[i] for d in dronz_list) for i in range(dimension)]
    max_coords = [max(d.coords[i] for d in dronz_list) for i in range(dimension)]
    ranges = [max_coords[i] - min_coords[i] for i in range(dimension)]

    # Initial cell size heuristic: aim for ~20-50 points per cell on average
    if initial_cell_size is None:
        volume = 1.0
        for r in ranges:
            volume *= max(r, 1.0)
        points_per_cell = 30
        cell_volume = volume * points_per_cell / n
        initial_cell_size = cell_volume ** (1.0 / dimension)

    def get_cell(coords: Tuple[float, ...], cell_size: float) -> Tuple[int, ...]:
        return tuple(int((coords[i] - min_coords[i]) / cell_size) for i in range(dimension))

    def get_neighbor_cells(cell: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """Get all adjacent cells including self."""
        neighbors = []

        def recurse(idx, current):
            if idx == dimension:
                neighbors.append(tuple(current))
                return
            for delta in [-1, 0, 1]:
                recurse(idx + 1, current + [cell[idx] + delta])

        recurse(0, [])
        return neighbors

    # Build grid
    cell_size = initial_cell_size
    grid = defaultdict(list)
    for dronz in dronz_list:
        cell = get_cell(dronz.coords, cell_size)
        grid[cell].append(dronz)

    # Collect candidate pairs from adjacent cells
    seen_pairs: Set[frozenset] = set()
    result_heap = []

    for dronz in dronz_list:
        cell = get_cell(dronz.coords, cell_size)
        neighbor_cells = get_neighbor_cells(cell)

        for nc in neighbor_cells:
            for other in grid.get(nc, []):
                if other.id >= dronz.id:  # Avoid duplicates
                    continue

                pair_key = frozenset([dronz.id, other.id])
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                dist = euclidean_distance(dronz, other)
                d1, d2 = (other, dronz) if other.id < dronz.id else (dronz, other)
                entry = (-dist, d1.id, d2.id, d1, d2)

                if len(result_heap) < k:
                    heapq.heappush(result_heap, entry)
                elif -dist > result_heap[0][0]:
                    heapq.heapreplace(result_heap, entry)

    # If we didn't find enough pairs, the grid was too coarse
    # Fall back to checking more cells or use brute force for remaining
    if len(result_heap) < k:
        # Just compute all pairs we haven't seen
        for i, d1 in enumerate(dronz_list):
            for j in range(i + 1, n):
                d2 = dronz_list[j]
                pair_key = frozenset([d1.id, d2.id])
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                dist = euclidean_distance(d1, d2)
                if d1.id > d2.id:
                    d1, d2 = d2, d1
                entry = (-dist, d1.id, d2.id, d1, d2)

                if len(result_heap) < k:
                    heapq.heappush(result_heap, entry)
                elif -dist > result_heap[0][0]:
                    heapq.heapreplace(result_heap, entry)

    # Extract and sort results
    results = []
    for neg_dist, id1, id2, d1, d2 in result_heap:
        results.append(format_result(d1, d2, -neg_dist))

    results.sort(key=lambda x: (x[2], x[0][0], x[1][0]))

    return results