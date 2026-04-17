# Apex v4: Algorithmic Optimization & True Multi-Agent Tactics

## Context
The bot has successfully integrated dynamic map profiling (scoring 46.75 on public and 13.67 on secret maps). To maximize the score, we must resolve performance bottlenecks that limit our lookahead depth, and fix flaws in our Battle Royale territory calculations. 

## Goal
Optimize redundant BFS searches to allow deeper lookaheads (increasing solo item collection) and implement a true multi-source Voronoi for Battle Royale to ensure perfect spatial dominance.

## Tasks

### 1. Unified BFS Item Evaluation (Performance & Depth)
**Problem:** `_best_item_score`, `_local_item_density`, and `_solo_cluster_score` all perform nearly identical BFS traversals. On large maps, this causes massive overhead and limits our `max_depth`.
**Action:** 
- Merge these three functions into a single function: `_evaluate_items_unified(self, board, start, rows, cols, max_depth, start_time, map_profile, is_solo)`.
- This function should traverse the board **once** up to `max_depth` (which can now be safely increased by +10 across all weights) and return a dictionary or tuple containing `item_score`, `item_distance`, `density_score`, and `cluster_score`.
- Update `get_move` to unpack these values from the single call.

### 2. True Multi-Player Voronoi Territory
**Problem:** In `_territory_score`, the bot only initializes the BFS queue with `opponents[:2]`. In a 4-player Battle Royale, ignoring the 3rd opponent leads to catastrophic miscalculations of safe territory.
**Action:**
- Rewrite `_territory_score` to perform a true multi-source Voronoi.
- Initialize the `owner` map and BFS queue with `my_start` and **all** alive opponents in the `opponents` list.
- The score returned should be `my_cells - max(opp_cells_list)`. We want to maximize the territory we own compared to the opponent who owns the most territory.

### 3. Smart Phase Logic (Proactive vs Reactive)
**Problem:** Phase (`P`) is heavily penalized in Solo mode (e.g., `ability_bonus -= 70` in corridors). The bot only uses it as a panic button.
**Action:**
- Rewrite the `ability_bonus` calculation for `candidate["action"] == "P"`.
- Remove the flat massive penalties.
- Instead, give Phase a significant **bonus** if `candidate["phase_needed"]` is True AND it bridges the bot into a completely new area (measured by `space_future` being significantly higher than `space_now`) OR it allows the bot to grab a Diamond (`direct_reward >= 50`).

### 4. End-Game Preservation (Win-Condition Enforcement)
**Problem:** The bot plays just as aggressively when it is winning as when it is losing, sometimes dying to greedy item grabs late in the game.
**Action:**
- In `get_move`, detect if it is the End-Game: `battle_mode == True` AND `opponent_count == 1`.
- Check if our score (`info["my_score"]`) is significantly higher than the remaining opponent's score (assuming we can track or estimate this, or simply prioritize survival over items).
- If it is the 1v1 phase of a Battle Royale, dynamically alter the active `weights`: Multiply `danger` by 1.5, `escape` by 1.5, and multiply `item` and `density` by 0.2. Force the bot to starve out the opponent rather than risk head-to-head collisions.

## Evaluation
After implementing these changes, verify that the bot does not timeout (exceed 0.5s) on any map, and run the evaluation scripts:
```bash
conda run -p ./.venv bash ./evaluate.sh Apex
conda run -p ./.venv bash ./secret_evaluate.sh Apex
```
Ensure public scores do not regress but improve to the max, and secret scores improve.