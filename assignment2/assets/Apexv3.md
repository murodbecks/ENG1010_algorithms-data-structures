# Apex v3: Hidden Scenario Robustness & Dynamic Analysis

## Context
The current `Apex.py` bot scores very well on known public maps (48/55) but fails catastrophically on hidden evaluation maps (scoring 6/25). 

The root cause of this failure is **brittle map-name checking**. The bot relies on hardcoded string checks (e.g., `map_key.startswith("s_")` or `map_name == "cube"`) to determine its behavior, weights, and whether it is in a Solo, Duel, or Battle Royale scenario. When placed in a hidden map with a different naming convention, it applies the wrong heuristics, misses solo-boosting opportunities, and loses.

## Goal
Refactor `Apex.py` to be completely robust against unknown map names by dynamically analyzing the map structure and the actual game state (number of active opponents) rather than relying on `map_name`.

## Tasks

### 1. State-Based Mode Switching (Crucial)
Instead of using `map_key.startswith("s_")` to determine if we are in a solo game, we must look at the actual number of opponents.
- **Solo Mode:** Triggered if `len(opponents) == 0`. All solo-specific logic (aggressive boosting, branch profiling, cluster scoring) must activate based on this condition, regardless of what `map_name` is.
- **Duel Mode:** Triggered if `len(opponents) == 1`. Continue to use Voronoi territory logic.
- **Battle Royale Mode:** Triggered if `len(opponents) > 1`. 

### 2. Dynamic Map Classification
Remove hardcoded map name checks inside `_get_weights` and the main `get_move` function. Instead, create a lightweight map analyzer that runs once (or caches its result) to classify the map structure based on:
- **Wall Density:** What percentage of the map is walls/trails? 
- **Choke Points / Corridors:** Run a quick scan to see if the map consists mostly of 1-cell wide paths (like `path` or `maze`) or open areas (like `arena` or `cube`).
- **Classification Categories:** Based on structural math, classify the map internally as:
  - `OPEN` (like arena/orbit/cube)
  - `CORRIDOR` (like path/maze)
  - `CHOICE/FLOODFILL` (dense items, distinct branching paths)

### 3. Update Weight Retrieval
Rewrite `_get_weights(self, map_name, opponent_count)` to be `_get_weights(self, map_class, opponent_count)`.
- Apply the weights previously reserved for `"s_path"` or `"maze"` to any map dynamically classified as `CORRIDOR`.
- Apply the weights previously reserved for `"arena"` or `"cube"` to any map dynamically classified as `OPEN`.
- Ensure that the `opponent_count` modifiers correctly override behaviors (e.g., zeroing out `danger` and `territory` weights if `opponent_count == 0`).

### 4. Safety & Time Limits
Dynamic map classification must be fast. The engine enforces a strict 0.5s turn limit.
- Compute the map classification only once per game using a cached dictionary or instance variable (e.g., `if not self.map_classified: self._classify_map()`).
- Do not let the map analysis exceed 50ms.

### 5. Retain Duel Perfection
Do not alter the core movement simulation, flood fill, or Voronoi math. The bot currently scores perfectly in Duels, and we must not regress this. Only change **how** and **when** these features are activated.

## Evaluation Commands for Codex
Use the following commands to test your implementations using the provided virtual environment:

Evaluate on standard public tests:
```bash
conda run -p ./.venv bash ./evaluate.sh Apex
```

Evaluate on secret/hidden tests:
```bash
conda run -p ./.venv bash ./secret_evaluate.sh Apex
```

Do not consider the task complete until the secret evaluation score improves substantially without regressing the public duel scores.