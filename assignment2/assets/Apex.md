# Tron Snake Bot — Tuning Plan For Higher Score

## Current Status

`bots/Apex.py` already exists and is past the skeleton stage. Do not
restart from Phase 0 unless the file becomes corrupted.

**Latest verified result on 2026-04-15**
- Solo: `700 / 1000` → `7.00 / 10`
- Duel: `6000 / 6000` → `30.00 / 30`
- Battle Royale: `1100 / 1500` → `11.00 / 15`
- Total automated score: `48.00 / 55.00`

This is the current best known Apex result. The main remaining ceiling is
no longer duel play. The next work should focus on solo score-race maps
and a short list of BR outliers.

## Required Python Version

Always run the bot with Python 3.11.

**Primary evaluation command**
```bash
conda run -n tron_snake bash ./evaluate.sh Apex
```

**Quick headless test**
```bash
conda run -n tron_snake python runner.py -m arena.txt -b Apex Dummy Dummy Scout
```

**Quick GUI test**
```bash
conda run -n tron_snake python main.py
```

## Verified Engine Facts

These points were checked against the actual code in `game_logic.py`.

1. `get_move(self, board, player_id, pos, info)` receives:
   - `board`
   - `player_id`
   - `pos`
   - `info` with keys:
     - `map_name`
     - `my_score`
     - `my_direction`
     - `phase_charges`
     - `emp_charges`
     - `emp_radius`
     - `stun_duration`
     - `active_emps`
     - `opponents`

2. The board may contain live player heads as `p1` to `p4`.
   - Trails are `t1` to `t4`.
   - Walls are `#`.
   - Timed walls are integers.

3. `+` and `P` do **not** include a direction.
   - They always move two cells in the current direction from `info['my_direction']`.

4. Timed walls shown to the bot are compressed integers `1..9`.
   - Internally the engine stores turns in multiples of 10.
   - Treating board value `n` as roughly `n * 10` turns remaining is the best available estimate.

5. Movement is simultaneous.
   - Walking into an opponent’s current head cell is not the same thing as hitting a committed wall.
   - Head-to-head resolution happens after all moves are chosen.
   - Still treat nearby heads as dangerous.

6. Any exception inside `get_move` destroys the bot.
   - Keep the top-level `try/except`.

## Current Apex Architecture

`bots/Apex.py` already includes:

1. Candidate simulation for `N/S/E/W/+ /P`
2. Flood fill for immediate space and future space
3. Item BFS with extra weight for clustered items
4. Opponent danger map
5. Exact 1v1 Voronoi balance for duels
6. Map-specific weight tuning
7. Opening discipline for `cube` / open maps
8. Hidden-own-trail tracking when head overlays mask committed cells
9. Conservative EMP usage
10. Enemy EMP avoidance with stun-runway scoring on `maze`, `gate`, and `treasure`
11. Fallback logic for no-good-move states

This means the next improvements should be **tuning and targeted heuristics**,
not broad rewrites.

## Recent Lessons

1. A visible `pX` head can hide one of Apex's own committed trail cells in
   the board copy. Tracking known committed trail cells materially improved
   collision safety.
2. The biggest remaining duel leaks were not normal pathing mistakes. They
   were delayed deaths after enemy EMP stuns forced Apex to keep moving in a
   bad direction.
3. Broad EMP fear on open BR maps caused regressions.
   - Strong EMP escape / stun-runway logic helps on `maze`, `gate`, and
     `treasure`.
   - The same logic should stay lighter on `arena`, `cube`, and `orbit`.

## Measured Weak Spots

These came from the full evaluator run.

### Solo

Weak maps:
- `s_path_2`
- `s_floodfill_0`
- `s_floodfill_1`
- `s_floodfill_2`
- `s_choice_2`

Observed pattern:
- The bot survives well but sometimes loses the score race.
- It boosts aggressively in safe open space, but not always toward the
  richest item branch.
- These maps now look more like structural score-race problems than
  survival problems.

### Duel

Current status:
- No failing duel matchups in the latest full evaluation.

Observed pattern:
- Duel is now effectively solved for the automated suite.
- Preserve this strength. Do not trade it away for small solo or BR gains.

### Battle Royale

Weak lobbies:
- `arena.txt` with `Apex Rogue Viper Blaze`
- `treasure.txt` with `Apex Rogue Stingray Stingray`
- `treasure.txt` with `Apex Viper Viper Rogue`
- `maze.txt` with `Apex Viper Rogue Rogue`
- `gate.txt` with `Apex Stingray Stingray Blaze`
- `cube.txt` with `Apex Viper Blaze Blaze`

Observed pattern:
- The bot is solid when it can carve space.
- The remaining failures are concentrated in crowded contests where local
  space and item timing collide.
- EMP handling in BR must stay selective. Over-broad EMP panic hurts open
  maps like `arena`.

## Priority Order For Next Improvements

Do these in order. Re-run the evaluator after each major step.

### Priority 1: Solo Score Race Tuning

This is likely the highest ROI for the next score jump.

**Goals**
- Convert survival into points faster
- Commit to richer branches earlier
- Boost only when it meaningfully improves score without collapsing mobility

**Tasks**
1. Add an item-density metric, not just nearest-item score.
   - Score a region by total nearby coin/diamond value over the next
     `k` cells.
   - Prefer moves that enter richer regions when survival is already safe.

2. Improve branch commitment for `s_choice_*`.
   - Detect first meaningful fork.
   - For each branch, estimate:
     - reachable space
     - total item value before next choke point
     - number of future exits
   - Choose the branch with the best combined value.

3. Improve `s_path_*` corridor policy.
   - Prefer monotonic forward progress.
   - Penalize unnecessary turns and wasted phase usage.
   - Boost only when the corridor ahead stays safe after landing.

4. Improve `s_floodfill_*` sweeping.
   - Bias toward the densest remaining item zone inside safe space.
   - Avoid wandering in empty open areas while Drunk bots farm item-rich zones.

5. Reduce opponent fear on solo maps.
   - Danger weighting should be lower in `s_*` maps unless a collision is immediate.

**Validation**
- Re-run full evaluation
- Expect Solo to move from `675` toward `750+`

### Priority 2: Battle Royale Outlier Fixes

This is now the best source of points after solo tuning.

**Goals**
- Convert the remaining BR rank-3 and rank-2 results into top-2 or wins
- Preserve the current perfect duel score

**Tasks**
1. Improve `treasure` crowd handling.
   - Decide earlier whether the center diamond is still worth contesting.
   - If not, preserve lane ownership and force a later cleanup.

2. Improve `maze` BR local compression handling.
   - Detect when two opponents can pinch the same pocket.
   - Prefer multi-exit zones over raw territory when two enemies are close.

3. Improve `gate` BR stun survival.
   - Keep the strong corridor EMP handling from duel.
   - Avoid overcommitting into single-lane pressure when two enemies are nearby.

4. Improve `cube` contested opening against double-`Blaze`.
   - The current opening is safe, but not yet score-dominant enough.
   - Tune first 15-20 turns only; do not rewrite the whole map policy.

**Validation**
- Re-test the exact failing BR lineups before the full evaluator

### Priority 3: Treasure Map Mode Switching

`treasure` appears in both duel and battle royale and is worth special handling.

**Goals**
- Avoid losing to bots that rush the central item pile better
- Stay alive while still scoring efficiently

**Tasks**
1. Add a `treasure` mode switch:
   - `race` mode when you can arrive first or safely contest
   - `harvest` mode when opponents beat you to the center

2. In `harvest` mode:
   - take flank items
   - preserve escape routes
   - avoid crowded central contests

3. In `race` mode:
   - allow more aggressive boosting
   - but only if landing mobility remains acceptable

**Validation**
- Re-test `treasure` vs `Rogue` and `Blaze`
- Re-test battle royale `treasure` lineups

### Priority 4: Timed Wall Precision

Current timed-wall handling is serviceable, but still heuristic.

**Tasks**
1. Keep using `cell * 10` as the best estimate for true open time.
2. For `gate` and `maze`, add a “worth waiting” concept.
   - Only plan around a future-opening wall if the waiting area is safe.
3. Avoid overvaluing future openings when a strong safe path exists now.

**Validation**
- Re-test `gate`
- Confirm we do not regress current good `gate` results

### Priority 5: Instrumentation And Profiling

Do this after behavior improvements start to stack up.

**Tasks**
1. Add lightweight optional debugging helpers for top candidate scores.
2. Profile worst-case turns on larger maps.
3. Keep `get_move` well under the engine limits:
   - soft warning at `0.5s`
   - hard death at `2.0s`

**Validation**
- No warnings
- No crashes
- Stable evaluation across repeated runs

## Suggested Regression Suite

Run these targeted checks before the full evaluator.

**Solo**
```bash
conda run -n tron_snake python runner.py -m s_path_2.txt -b Apex Drunk Drunk Drunk
conda run -n tron_snake python runner.py -m s_floodfill_0.txt -b Apex Drunk Drunk Drunk
conda run -n tron_snake python runner.py -m s_floodfill_1.txt -b Apex Drunk Drunk Drunk
conda run -n tron_snake python runner.py -m s_floodfill_2.txt -b Apex Drunk Drunk Drunk
conda run -n tron_snake python runner.py -m s_choice_2.txt -b Apex Drunk Drunk Drunk
```

**Duel**
```bash
conda run -n tron_snake python runner.py -m maze.txt -b Apex Dummy Dummy Blaze
conda run -n tron_snake python runner.py -m gate.txt -b Viper Dummy Dummy Apex
```

**Battle Royale**
```bash
conda run -n tron_snake python runner.py -m arena.txt -b Apex Rogue Viper Blaze
conda run -n tron_snake python runner.py -m treasure.txt -b Apex Rogue Stingray Stingray
conda run -n tron_snake python runner.py -m treasure.txt -b Apex Viper Viper Rogue
conda run -n tron_snake python runner.py -m maze.txt -b Apex Viper Rogue Rogue
conda run -n tron_snake python runner.py -m gate.txt -b Apex Stingray Stingray Blaze
conda run -n tron_snake python runner.py -m cube.txt -b Apex Viper Blaze Blaze
```

## Target Score

Near-term target:
- Solo: `750+ / 1000`
- Duel: `6000 / 6000`
- Battle Royale: `1150+ / 1500`
- Total: `48+ / 55`

Stretch target:
- Total: `50+ / 55`

## Phase 10: Code Simplification & Hyperparameter Tuning

**Goal:** The bot currently uses over 15 hand-tuned heuristic weights per map type. We are hitting the ceiling of manual tuning. We need to automate weight optimization and clean up dead code.

### Tasks
1. **Automated Weight Tuning Script:** Create a standalone Python script (e.g., `tune.py`) that uses a library like Optuna, or a simple random-search/hill-climbing algorithm, to play headless matches and optimize the `_get_weights` dictionaries.
2. **Prune Redundant Heuristics:** Evaluate if `density_score` and `item_score` are duplicating efforts. Remove or merge heuristics that correlate too highly to speed up computation.

## Phase 11: Multi-Target Routing (Solo Score Maxing)

**Goal:** The bot currently uses a greedy BFS to find the *nearest* or *densest* items. On `s_floodfill` and `s_path` maps, this causes it to miss optimal scoring lines. We need a Traveling Salesperson Problem (TSP) style approach for items.

### Tasks
1. **Cluster Tracking:** Instead of just scoring the nearest item, identify clusters of diamonds/coins. 
2. **Multi-point Pathing:** In Solo maps, use a lightweight A* or Dijkstra search to find a path that intersects the *maximum value of items within 15-20 steps*, rather than just the immediate next item.
3. **Dead-end Sweep:** If an item is in a dead-end, calculate exactly how many steps it takes to enter and exit. If `steps * 2 < time_limit` and it doesn't trap the bot, grab it. Otherwise, ignore it completely.

## Phase 12: Adversarial Intention Prediction (Battle Royale Outliers)

**Goal:** The current `_build_danger_map` assumes enemies move randomly to any valid adjacent square. We can predict their moves better to win crowded BR maps.

### Tasks
1. **1-Ply Enemy Simulation:** For the closest 1-2 opponents, run a simplified version of our own `get_move` scoring on *their* valid moves to predict where they are most likely to go (usually toward space or high-value items).
2. **Trap Setting:** If an opponent is predicted to go for a specific diamond, and we are closer, boost to take it and cut off their escape route simultaneously.
3. **Drafting:** In `treasure`, if two enemies are fighting for the center, intentionally hang back just outside the danger zone, let them use their EMPs/Phases, and then sweep in to take the remaining territory.

## Phase 13: Time-Limit Wall Exploitation

**Goal:** Turn timed walls into weapons on `gate` and `maze`.

### Tasks
1. **Pre-Staging:** If a wall has a countdown of `3`, and we are `3` steps away, path directly *into* it so we walk through on the exact turn it vanishes.
2. **Opponent Baiting:** Use EMPs specifically to stun opponents *while* they are waiting for a timed wall to open, guaranteeing they miss the timing window and get trapped by trailing bodies.

## Guardrails

1. Do not throw away the existing duel strength just to chase solo score.
2. Change one behavior family at a time and re-evaluate.
3. Prefer small, measurable tuning passes over full rewrites.
4. Keep the top-level fail-safe in `get_move`.
5. If a tweak helps one map but hurts many others, revert it.

## Experiment Log

Track results after each meaningful tuning pass.

| Pass | Solo | Duel | BR | Total | Notes |
|------|------|------|----|-------|-------|
| Baseline 2026-04-15 | 675 | 4500 | 850 | 37.75 | Initial Apex implementation |
| Pass 1 2026-04-15 | 675 | 5000 | 975 | 41.50 | Added branch-aware boost control, richer item density, exact 1v1 Voronoi, crowd and escape scoring |
| Pass 2 2026-04-15 | 675 | 5400 | 1000 | 43.75 | Added cube/open-map opening discipline and same-axis approach penalties; reverted failed treasure override |
| Pass 3 2026-04-15 | 675 | 5800 | 1175 | 47.50 | Fixed hidden-own-trail collisions and improved orbit risk handling |
| Pass 4 2026-04-15 | 675 | 6000 | 1100 | 47.75 | Added selective enemy EMP avoidance and stun-runway scoring; narrowed hard EMP logic to corridor/item maps to avoid BR open-map regressions |
| Pass 5 2026-04-15 | 700 | 6000 | 1100 | 48.00 | Added solo-only boost viability scoring and explicit penalties for boosting over intermediate items; `s_floodfill_1` improved from rank 3 to rank 2 |
| Pass 6 2026-04-15 | 700 | 6000 | 1100 | 48.00 | Replaced failed timed-wall staging with lightweight solo cluster lookahead and branch-value bias for `s_choice_*` / `s_floodfill_*`; raw solo scores shifted but weighted evaluator stayed flat |
