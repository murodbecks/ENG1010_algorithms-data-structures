import numpy as np
import pandas as pd
from tqdm import tqdm

from typing import Tuple

def get_squared_distance(location_1: np.array, location_2: np.array) -> float:
    """Get a squared Euclidian distance."""
    # since location 1 and location 2 are numpy arrays, we can take the difference and calculate dot product with itself
    diff = location_1 - location_2
    return np.dot(diff, diff)

def find_closest_pair_brute_force(arr: np.array, idx: list):
    """Find a closest pair among given locations in brute-force way."""
    # initializing closest pair and the smallest distance to keep track.
    closest_pair_info = None
    smallest_distance = float('inf') # it is set to infinity to be replaced by first distance. 

    n = len(arr)

    # tracking the progress knowing total number of steps
    with tqdm(total=n*(n-1)//2, desc=f"brute-force top-1 with {n} locations") as pbar:
        # looping through every location
        for i in range(n):
            loc1 = arr[i]

            # inner looping all locations up to that location
            for j in range(i):
                loc2 = arr[j]
                
                # measuring the distance
                distance = get_squared_distance(loc1, loc2)
                
                # if distance is smaller, we will update info and smallest distance
                if distance < smallest_distance:
                    smallest_distance = distance
                    closest_pair_info = (idx[j], idx[i], distance**0.5)
            
            pbar.update(i)

    # getting Euclidian distance from squared one
    closest_pair_info = [(closest_pair_info[0], closest_pair_info[1], closest_pair_info[2]**0.5)]
    return closest_pair_info

def combine_new_pair(k: int, k_closest_pairs: list, new_pair: Tuple[str, str, float]) -> Tuple[list, float]:
    """Combine new pair to sorted k pairs."""
    # initializinf necessary variables
    new_loc1, new_loc2, new_distance = new_pair
    updated_k_closest_pairs = []

    # variable for whether we added new pair. The idea is to just extend the list if we already added a new variable
    is_new_pair_added = False

    i = 0
    # looping through list
    while (not is_new_pair_added) and i < len(k_closest_pairs):
        loc1, loc2, distance = k_closest_pairs[i]
        # if new distance is smaller, add it first
        if distance > new_distance:
            updated_k_closest_pairs.append((new_loc1, new_loc2, new_distance))
            is_new_pair_added = True
        
        updated_k_closest_pairs.append((loc1, loc2, distance))
        
        i += 1

    # adding a new pair, if it is still not added
    if not is_new_pair_added:
        updated_k_closest_pairs.append((new_loc1, new_loc2, new_distance))

    updated_k_closest_pairs.extend(k_closest_pairs[i:])

    # returning k pairs (excluding last pair in the sorted list)
    updated_k_closest_pairs = updated_k_closest_pairs[:k]

    # keeping kth distance infinity unless `k_closest_pairs` list is full
    kth_closest_pair_distance = updated_k_closest_pairs[-1][-1] if len(updated_k_closest_pairs) == k else float('inf')

    return updated_k_closest_pairs, kth_closest_pair_distance

def find_k_closest_pair_brute_force(k: int, arr: np.array, idx: list):
    """Find k closest pair in brute-force way."""
    # initializing closest pair and the smallest distance to keep track.
    k_closest_pairs = []
    kth_closest_pair_distance = float('inf') # it is set to infinity to be replaced by first k distances.

    n = len(arr)

    with tqdm(total=n*(n-1)//2, desc=f"brute-force top-{k} with {n} locations") as pbar:
        # looping through every location
        for i in range(n):
            loc1 = arr[i]

            # inner looping all locations up to that location
            for j in range(i):
                loc2 = arr[j]

                # measuring the distance
                distance = get_squared_distance(loc1, loc2)
                
                # if distance is smaller than kth smallest distance, we will update info and smallest distance
                if distance < kth_closest_pair_distance:
                    k_closest_pairs, kth_closest_pair_distance = combine_new_pair(k, k_closest_pairs, (idx[j], idx[i], distance))

            pbar.update(i)

     # getting Euclidian distance from squared ones
    k_closest_pairs = [(loc1, loc2, distance**0.5) for loc1, loc2, distance in k_closest_pairs]
    return k_closest_pairs

def find_distances_brute_force(arr: np.array, idx: list) -> list:
    """Measure a distance in a base case (array with 2 or 3 length)."""
    n = len(idx)

    # returning id1, id2 and distance if we got 2 points
    if n == 2:
        return [(idx[0], idx[1], get_squared_distance(arr[0], arr[1]))]
    
    # returning sorted list of ids and squared distances
    elif n == 3:
        # starting with initial pair
        distance_info = [(idx[0], idx[1], get_squared_distance(arr[0], arr[1]))]

        # looping through remaining 2 pairs
        for i, j in [(0, 2), (1, 2)]:
            new_distance = get_squared_distance(arr[i], arr[j])

            # creating new list and combining existing info
            new_distance_info = []
            is_new_pair_added = False
            for idx1, idx2, distance in distance_info:
                if new_distance < distance and (not is_new_pair_added):
                    new_distance_info.append((idx[i], idx[j], new_distance))
                    is_new_pair_added = True
                    
                new_distance_info.append((idx1, idx2, distance))

            # adding new pair if it is not added
            if not is_new_pair_added:
                new_distance_info.append((idx[i], idx[j], new_distance))

            # renewing `distance_info`
            distance_info = new_distance_info
            
        return distance_info
    
    # returning error message and `None` otherwise
    else:
        print(f"Array length should be either 2 or 3, got {n}")
        return None

def combine_pair_infos(k: int, distance_info_left: list, distance_info_right: list) -> Tuple[list, float]:
    """Combining 2 sorted lists and returning k values."""
    k_closest_pairs = []

    # looking through left and right lists unless either finished or we already have k values
    i, j = 0, 0
    while (i < len(distance_info_left)) and (j < len(distance_info_right)) and (len(k_closest_pairs) < k):
        # if left value is smaller, we will add it first and increment left index by 1
        if distance_info_left[i][-1] < distance_info_right[j][-1]:
            k_closest_pairs.append(distance_info_left[i])
            i += 1
        
        # else we will add right one first and increment right index by 1
        else:
            k_closest_pairs.append(distance_info_right[j])
            j+=1
    
    # if we already got k pairs, return it
    if len(k_closest_pairs) == k:
        return k_closest_pairs, k_closest_pairs[-1][-1]**0.5

    # adding remaining values from left list
    if i < len(distance_info_left):
        k_closest_pairs.extend(distance_info_left[i:])

    # adding remaining values from right list
    if j < len(distance_info_right):
        k_closest_pairs.extend(distance_info_right[j:])

    # getting Euclidian distance from square one or returning infinity if `k_closes_pairs` is still not full 
    kth_closest_distance = k_closest_pairs[k-1][-1]**0.5 if len(k_closest_pairs) >= k else float('inf')
    return k_closest_pairs[:k], kth_closest_distance

def find_k_closest_pair_improved(k: int, arr: np.array, idx: list):
    """Finding k closest pairs in devide and conquer way."""
    # ====BASE CASE====
    if len(arr) <= 3:
        return find_distances_brute_force(arr, idx)[:k]

    # ====DEVIDE====
    # # sorting values by X-axis
    # sorted_indexes = np.argsort(arr[:, 0])
    # arr_sorted_by_x = arr[sorted_indexes]
    # idx_sorted_by_x = [idx[i] for i in sorted_indexes]

    arr_sorted_by_x = arr
    idx_sorted_by_x = idx

    # deviding the list by median value
    median_idx = len(arr_sorted_by_x) // 2

    arr_left = arr_sorted_by_x[:median_idx]
    idx_left = idx_sorted_by_x[:median_idx]

    arr_right = arr_sorted_by_x[median_idx:]
    idx_right = idx_sorted_by_x[median_idx:]

    # X coordinate that we split the points
    x_border = arr_left[-1][0]

    # recursively calling itself untill reaching base case
    distance_info_left = find_k_closest_pair_improved(k, arr_left, idx_left)
    distance_info_right = find_k_closest_pair_improved(k, arr_right, idx_right)

    # ====CONQUER====
    # combining left and right distances first
    k_closest_pairs, kth_closest_distance = combine_pair_infos(k, distance_info_left, distance_info_right)

    # border relations

    # filtering points from `x_border - kth_closest_distance` to `x_border + kth_closest_distance`
    filtering_mask = (arr_sorted_by_x[:, 0] >= x_border - kth_closest_distance) & (arr_sorted_by_x[:, 0] <= x_border + kth_closest_distance)
    filtered_arr = arr_sorted_by_x[filtering_mask]
    filtered_idx = [idx_sorted_by_x[i] for i in range(len(idx_sorted_by_x)) if filtering_mask[i]]

    # sorting through Y-axis
    sorted_indexes_by_y = np.argsort(filtered_arr[:, 1])
    filtered_arr_sorted_by_y = filtered_arr[sorted_indexes_by_y]
    filtered_idx_sorted_by_y = [filtered_idx[i] for i in sorted_indexes_by_y]

    len_filtered = len(filtered_arr_sorted_by_y)

    # looping through each point
    for i in range(len_filtered):
        loc1 = filtered_arr_sorted_by_y[i]
        idx1 = filtered_idx_sorted_by_y[i]

        # filtering for getting points on another side (by x_border)
        loc_mask = filtered_arr_sorted_by_y[:, 0] <= x_border if loc1[0] > x_border else filtered_arr_sorted_by_y[:, 0] > x_border

        # filtering for getting points between `y` and `y + kth_closest_distance` (because we are looping in y sorted points and started from bottom)
        loc_mask = loc_mask & (filtered_arr_sorted_by_y[:, 1] <= loc1[1] + kth_closest_distance) & (filtered_arr_sorted_by_y[:, 1] > loc1[1])

        # filtering further for points between `z - kth_closest_distance` and `z + kth_closest_distance`
        if arr.shape[1] == 3:
            loc_mask = loc_mask & (filtered_arr_sorted_by_y[:, 2] <= loc1[2] + kth_closest_distance) & (filtered_arr_sorted_by_y[:, 2] >= loc1[2] - kth_closest_distance) 

        # finding all egligible points for the point
        egligible_arr = filtered_arr_sorted_by_y[loc_mask]
        egligible_idx = [filtered_idx_sorted_by_y[i] for i in range(len_filtered) if loc_mask[i]]

        for loc2, idx2 in zip(egligible_arr, egligible_idx):
            # getting a squared distance
            distance = get_squared_distance(loc1, loc2)
            
            # if the distance is smaller than `kth_closest_distance`, we will add it to `k_closest_pairs`
            if distance < kth_closest_distance**2:
                k_closest_pairs, kth_closest_distance = combine_pair_infos(k, k_closest_pairs, [(idx1, idx2, distance)])

    # I was about to do this step, but since we are calling it recursively, the square will end up exploding. better not to do.
    # k_closest_pairs = [(loc1, loc2, distance**0.5) for loc1, loc2, distance in k_closest_pairs]
    return k_closest_pairs

def find_closest_pair_improved(arr: np.array, idx: list):
    """Find a closest pair among given locations in brute-force way."""

    # this is special case of `find_k_closest_pair_improved` with `k=1` so some tricks exists but unfortunately we are short on time
    k_closest_pairs = find_k_closest_pair_improved(1, arr, idx)
    closest_pair_info = [(k_closest_pairs[0][0], k_closest_pairs[0][1], k_closest_pairs[0][2]**0.5)]
    return closest_pair_info