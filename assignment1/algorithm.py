import numpy as np
import pandas as pd
from tqdm import tqdm

from typing import Tuple

def get_squared_distance(location_1: np.array, location_2: np.array) -> float:
    """Get a squared Euclidian distance."""
    # since location 1 and location 2 are numpy arrays, we can take the difference and calculate dot product with itself
    diff = location_1 - location_2
    return np.dot(diff, diff)

def find_closest_pair_brute_force(df: pd.DataFrame, columns: list):
    """Finding closest pair among given locations in brute-force way."""
    # converting to numpy and list to get linear access time
    arr = df[columns].to_numpy()
    idx = df['id'].tolist()

    # initializing closest pair and the smallest distance to keep track.
    closest_pair_info = None
    smallest_distance = float('inf') # it is set to infinity to be replaced by first distance. 

    n = len(df)

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

def add_new_pair(k: int, k_closest_pairs: list, new_pair: Tuple[str, str, float]) -> Tuple[list, float]:
    """Add new pair to sorted k pairs."""
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
    return updated_k_closest_pairs, updated_k_closest_pairs[-1][-1] 

def find_k_closest_pair_brute_force(k: int, df: pd.DataFrame, columns: list):
    """Find k closest pair in brute-force way."""
    # converting to numpy and list to get linear access time
    arr = df[columns].to_numpy()
    idx = df['id'].tolist()

    # initializing closest pair and the smallest distance to keep track.
    k_closest_pairs = None
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
                    k_closest_pairs, kth_closest_pair_distance = add_new_pair(k, k_closest_pairs, (idx[j], idx[i], distance))

                    # keeping kth distance infinity unless `k_closest_pairs` list is full
                    if len(k_closest_pairs) < k:
                        kth_closest_pair_distance = float('inf')

            pbar.update(i)

     # getting Euclidian distance from squared ones
    k_closest_pairs = [(loc1, loc2, distance**0.5) for loc1, loc2, distance in k_closest_pairs]
    return k_closest_pairs

def find_closest_pair_improved(df: pd.DataFrame, columns: list):
    pass

def find_k_closest_pair_improved(k: int, df: pd.DataFrame, columns: list):
    pass