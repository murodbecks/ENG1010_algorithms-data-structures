# Botir_v3.md — Targeted Fixes to Push from 47 to 50+

## CONTEXT
`bots/Botir.py` scores 47.25/55. We need 50+.
This document contains SPECIFIC bugs to fix and improvements to make.

Before making ANY changes:
1. Read ALL files in `bots/` — especially `Scout.py`, `Rogue.py`,
   `Stingray.py`, `Viper.py`, `Blaze.py` — understand their logic fully.
2. Read ALL `.txt` files in `maps/` — understand each map's layout.
3. Run `source .venv/bin/activate && ./evaluate.sh Botir` and save the
   full output to `baseline_results.txt` so you can compare before/after.
4. Identify which SPECIFIC matchups/maps score 0 (rank 4 or 2 in duels).

---

## BUG 1 (CRITICAL): Voronoi Weight is 0 for Most Duel Maps

This is the single biggest scoring bug. Duels are worth 30 points — the
largest category — and territory control is the key to winning duels.

### The Problem
In `_scenario_weights`, the flow is:

```python
elif len(opponents) <= 1:
    weights.update({
        ...
        "voronoi": 0.0,     # <-- DEFAULT IS ZERO
    })
    if map_name in {"gate", "cube", "orbit", ...}:
        weights["voronoi"] = 1.7   # <-- only these 3 maps get voronoi!
```

So for `arena`, `maze`, `treasure` duels — which are 3 out of 6 duel maps
(= 30 out of 60 duel matches) — Voronoi scoring is COMPLETELY DISABLED.
The bot ignores territory control on half the duel maps.

### The Fix
Set `"voronoi": 1.5` for ALL duel scenarios (len(opponents) <= 1), not
just gate/cube/orbit. Then tune per-map:

```python
elif len(opponents) <= 1:
    weights.update({
        "area": 3.2,
        "density": 2.5,
        "reachable_value": 0.10,
        "nearest_item": 1.0,
        "turn_points": 7.0,
        "threat": 10.5,
        "voronoi": 1.5,       # ENABLE FOR ALL DUELS
        "boost_bonus": 6.0,
    })
    # Tighter maps benefit more from territory control
    if map_name in ("gate", "cube", "orbit"):
        weights["voronoi"] = 2.0
        weights["threat"] = 11.5
    elif map_name in ("maze",):
        weights["voronoi"] = 2.2   # Maze has corridors — cutting off is huge
        weights["area"] = 3.8
    elif map_name in ("arena",):
        weights["voronoi"] = 1.5
    elif map_name in ("treasure",):
        weights["voronoi"] = 1.3
        weights["density"] = 3.5   # Treasure has lots of items
```

---

## BUG 2 (CRITICAL): Voronoi is Never Computed in Battle Royale

The current code only runs Voronoi when `len(opponents) == 1`:

```python
if len(opponents) == 1:
    my_cells, opp_cells = self._voronoi_score(...)
```

In Battle Royale, there are 3 opponents. The bot completely ignores
territory in BR. It should compute territory against the NEAREST opponent
at minimum.

### The Fix
In `_score_candidate`, change the Voronoi section:

```python
if opponents and scenario["voronoi"] > 0:
    # Find nearest alive opponent
    nearest_opp = min(
        opponents,
        key=lambda o: abs(dest[0] - o["pos"][0]) + abs(dest[1] - o["pos"][1])
    )
    my_cells, opp_cells = self._voronoi_score(
        board, dest, nearest_opp["pos"], candidate["blocked"]
    )
    score += scenario["voronoi"] * (my_cells - opp_cells)
```

And set a small voronoi weight for BR in the default weights:
```python
# Default weights (used for BR and multi-opponent)
"voronoi": 0.4,
```

---

## BUG 3: Solo `s_path` Maps Use Chain Scoring Instead of Greedy Route

For `s_path` maps, the `_guided_commands` method calls
`_guided_path_choice_commands` which uses item-chain scoring with depth 2
over at most 9 items. But `s_path` maps are LINEAR corridors — there's
typically ONE correct direction to go. The chain scoring is overkill and
can pick wrong targets.

### The Fix
For ALL solo maps (s_path, s_floodfill, s_choice), use the SAME unified
greedy nearest-item planner. Remove the special `_guided_path_choice_commands`
branch. The `_pick_solo_target` + `_path_to_target` approach already works
well — just use it for everything:

```python
def _guided_commands(self, board, pos, my_dir, map_name):
    if not map_name.startswith("s_"):
        return set()

    # Unified solo planner for ALL solo map types
    if self.opening_dir is None:
        self.opening_dir = self._scan_opening_direction(board, pos)

    if not self.item_positions:
        # No items left — just survive. Prefer forward or opening dir.
        commands = set()
        if self.opening_dir:
            commands.add(self.opening_dir)
        return commands

    if self.solo_target not in self.item_positions:
        self.solo_target = self._pick_solo_target(board, pos)

    if self.solo_target is None:
        return {self.opening_dir} if self.opening_dir else set()

    path = self._path_to_target(board, pos, self.solo_target,
                                 allow_soon_safe=True)
    if len(path) < 2:
        self.solo_target = None
        return {self.opening_dir} if self.opening_dir else set()

    first = self._step_direction(path[0], path[1])
    commands = {first}

    # Boost if next 2 steps are same direction as current
    if first == my_dir and len(path) >= 3:
        second = self._step_direction(path[1], path[2])
        if second == first:
            commands.add("+")

    return commands
```

Delete the `_guided_path_choice_commands` and `_item_chain_score` methods
entirely. They add complexity without benefit.

---

## BUG 4: `_pick_solo_target` Doesn't Consider Route Efficiency

The current `_pick_solo_target` scores items by `value / (distance + 1)`.
This is greedy-nearest which can be suboptimal. For example, it might go
for a coin at distance 2 instead of a diamond at distance 4.

### The Fix
Weight diamonds MUCH more heavily and factor in density of nearby items:

```python
def _pick_solo_target(self, board, pos):
    if not self.item_positions:
        return None

    reachable = self._distance_map(board, pos, frozenset(),
                                    allow_soon_safe=True)
    best = None
    best_score = -1e18

    for target in self.item_positions:
        dist = reachable.get(target)
        if dist is None:
            continue
        value = self._cell_value(board[target[1]][target[0]])

        # Base score: value efficiency
        score = value / (dist + 1.0)

        # Bonus for diamonds (always worth pursuing)
        if value >= 50:
            score += 8.0

        # Bonus for very close items (grab them on the way)
        if dist <= 2:
            score += 5.0

        # Cluster bonus: how many other items are near this target?
        cluster_bonus = 0
        target_dm = self._distance_map(board, target, frozenset())
        for other in self.item_positions:
            if other == target:
                continue
            od = target_dm.get(other)
            if od is not None and od <= 5:
                cluster_bonus += self._cell_value(
                    board[other[1]][other[0]]) / (od + 1.0)
        score += cluster_bonus * 0.15

        if score > best_score:
            best_score = score
            best = target

    return best
```

WARNING: The cluster bonus requires computing `_distance_map` for each
candidate target. If there are many items (>15), this could be slow.
Limit it: only compute cluster bonus for the top 8 candidates by base
score. Or skip cluster bonus if `len(self.item_positions) > 20`.

---

## IMPROVEMENT 5: Smarter Duel Opening Based on Opponent Direction

In duels, Players 1 and 4 start facing opposite directions (1→South,
4→North). On the first few turns, the bot should:
1. Identify where the real opponent is (not the Dummy bots)
2. Move TOWARD items and AWAY from opponent's predicted path
3. Try to claim the larger side of the map

### Implementation
Add a duel-specific opening heuristic:

```python
def _duel_opening_boost(self, board, pos, my_dir, opponents, candidates):
    """On early duel turns, bias toward claiming territory fast."""
    if self.turn_count > 8:
        return {}
    if len(opponents) != 1:
        return {}

    opp = opponents[0]
    opp_pos = opp["pos"]

    # Prefer directions that move AWAY from opponent
    bias = {}
    for cand in candidates:
        dx = cand["dest"][0] - opp_pos[0]
        dy = cand["dest"][1] - opp_pos[1]
        separation = abs(dx) + abs(dy)
        bias[cand["command"]] = separation * 0.5

    return bias
```

Add these biases to the candidate scores in `_score_candidate`.

---

## IMPROVEMENT 6: Don't Boost Into Small Areas

The current code penalizes boost when area < 8 or < 14, but it should
also check: after boosting, how many EXITS does the destination have?
Boosting skips a cell, and that skipped cell becomes your trail. This
can block an exit.

```python
if candidate["boost"]:
    # The intermediate cell becomes trail — check if that blocks exits
    if len(candidate["traversed"]) >= 2:
        intermediate = candidate["traversed"][0]
        # Count how many exits the destination has WITHOUT the intermediate
        extended_blocked = candidate["blocked"] | {intermediate}
        exits_after = self._exit_count(board, dest, extended_blocked)
        if exits_after == 0 and area < 20:
            score -= 100.0  # Boost seals us into a dead end
```

---

## IMPROVEMENT 7: Opponent-Specific Counter Strategies

After reading the enemy bot source code, implement counters. You must
READ the actual source files to determine what each bot does. Here's what
to look for and how to counter common patterns:

### If a bot is aggressive (chases you):
- Increase threat weight, play defensively
- Use EMP when it approaches
- Lead it into dead ends

### If a bot focuses on items:
- Race for high-value items, use boost to beat it
- Cut off its path to item clusters

### If a bot is territorial:
- Contest territory early with boost
- Don't let it wall you off

Add a method that infers opponent behavior from their moves:

```python
def __init__(self):
    ...
    self.opponent_history = {}  # id -> list of positions

def _track_opponents(self, opponents):
    for opp in opponents:
        oid = opp["id"]
        if oid not in self.opponent_history:
            self.opponent_history[oid] = []
        self.opponent_history[oid].append(opp["pos"])
        # Keep only last 10 positions
        if len(self.opponent_history[oid]) > 10:
            self.opponent_history[oid].pop(0)
```

Call `self._track_opponents(opponents)` at the start of `get_move`.

---

## IMPROVEMENT 8: Timed Wall Countdown is Wrong

In `_is_soon_safe`:
```python
def _is_soon_safe(self, cell, distance):
    if not isinstance(cell, int):
        return False
    if cell <= 1:
        return True
    return distance >= cell * 5
```

But from game.py, timed walls display `ceil(remaining_ticks / 10)`. So:
- cell=1 means 1-10 ticks remaining
- cell=2 means 11-20 ticks remaining

The `distance >= cell * 5` heuristic is wrong. If cell=2 and distance=10,
we assume it'll be open, but it might need 20 ticks. Since distance is
in BFS steps (1 step = 1 tick for normal movement), we should use:

```python
def _is_soon_safe(self, cell, distance):
    if not isinstance(cell, int):
        return False
    # cell=N means up to N*10 ticks remaining, at least (N-1)*10+1
    # We need distance >= max_remaining_ticks to be safe
    max_remaining = cell * 10
    # Add safety buffer of 3 ticks
    return distance >= max_remaining + 3
```

This is more conservative but avoids walking into walls that haven't
opened yet.

---

## IMPROVEMENT 9: Smarter EMP Timing

Current EMP logic fires whenever an opponent is within radius. But EMP
takes 5 ticks to detonate. The opponent will likely move away.

Better strategy: fire EMP when the opponent is in a CORRIDOR (few exits)
so they can't escape the blast radius in 5 ticks.

```python
def _should_use_emp(self, candidate, opponents, map_name, board):
    if not opponents or map_name.startswith("s_"):
        return False

    dest = candidate["dest"]
    for opp in opponents:
        ox, oy = opp["pos"]
        cheb = max(abs(dest[0] - ox), abs(dest[1] - oy))

        if cheb > self.EMP_RADIUS:
            continue

        # Check if opponent is in a tight space (hard to escape)
        opp_area = len(self._distance_map(board, opp["pos"], frozenset()))
        opp_exits = self._exit_count(board, opp["pos"], frozenset())

        # Fire if opponent is cornered or very close
        if cheb <= 1:
            return True
        if opp_exits <= 2 and opp_area < 25:
            return True
        if cheb <= 2 and opp_exits <= 2:
            return True

    # Also fire if multiple opponents in range
    in_range = sum(
        1 for opp in opponents
        if max(abs(dest[0] - opp["pos"][0]),
               abs(dest[1] - opp["pos"][1])) <= self.EMP_RADIUS
    )
    return in_range >= 2
```

---

## IMPROVEMENT 10: Safe Candidate Always Available

Sometimes ALL candidates score very negative and the bot picks a terrible
move. Add an absolute safety check: if the best candidate has area <= 2,
and Phase is available and gives area > 10, ALWAYS phase.

```python
# After scoring all candidates, before returning:
if best["area"] <= 3 and phase_charges > 0:
    phase_cand = next((c for c in candidates if c["phase"]), None)
    if phase_cand and phase_cand["area"] > best["area"] * 3:
        best = phase_cand
        move = "P"
```

---

## WEIGHT ADJUSTMENTS SUMMARY

```python
# Default (Battle Royale)
"area": 2.8, "density": 3.5, "reachable_value": 0.15,
"nearest_item": 1.4, "turn_points": 6.5, "threat": 10.0,
"boost_bonus": 5.0, "phase_bonus": 2.0, "voronoi": 0.4

# Solo path
"area": 1.5, "density": 6.0, "reachable_value": 0.25,
"nearest_item": 2.5, "turn_points": 5.0, "threat": 1.5,
"boost_bonus": 14.0, "voronoi": 0.0

# Solo floodfill
"area": 2.5, "density": 5.5, "reachable_value": 0.25,
"nearest_item": 2.0, "turn_points": 5.0, "threat": 1.5,
"boost_bonus": 14.0, "voronoi": 0.0

# Solo choice
"area": 2.0, "density": 6.5, "reachable_value": 0.30,
"nearest_item": 2.2, "turn_points": 5.0, "threat": 1.5,
"boost_bonus": 14.0, "voronoi": 0.0

# Duel (all maps)
"area": 3.2, "density": 2.5, "reachable_value": 0.10,
"nearest_item": 1.0, "turn_points": 7.0, "threat": 10.5,
"boost_bonus": 6.0, "voronoi": 1.5
# Then per-map adjustments as shown in Bug 1 fix
```

---

## TESTING PROCEDURE

Test incrementally after each fix:

```bash
source .venv/bin/activate

# 1. After fixing Voronoi bug, test duels specifically:
for map in arena maze treasure gate cube orbit; do
  for bot in Scout Rogue Stingray Viper Blaze; do
    echo "=== $map vs $bot (pos1) ==="
    python runner.py -m ${map}.txt -b Botir Dummy Dummy $bot --max-ticks 10000
    echo "=== $map vs $bot (pos4) ==="
    python runner.py -m ${map}.txt -b $bot Dummy Dummy Botir --max-ticks 10000
  done
done

# 2. After fixing solo planner, test all solo maps:
for map in s_path_0 s_path_1 s_path_2 s_path_3 \
           s_floodfill_0 s_floodfill_1 s_floodfill_2 \
           s_choice_0 s_choice_1 s_choice_2; do
  echo "=== $map ==="
  python runner.py -m ${map}.txt -b Botir Drunk Drunk Drunk --max-ticks 10000
done

# 3. Full evaluation
./evaluate.sh Botir
```

IMPORTANT: After each major change, run `./evaluate.sh Botir` and compare
the output to `baseline_results.txt`. If a change makes things WORSE on
some maps, revert it for those maps by adding map-specific conditions.

---

## PRIORITY ORDER OF FIXES

Implement in this order (highest impact first):

1. **Bug 1: Enable Voronoi for all duel maps** (~2-4 pts potential)
2. **Bug 2: Enable Voronoi in Battle Royale** (~0.5-1 pt)
3. **Bug 3: Unify solo planner** (~0.5-1 pt)
4. **Improvement 8: Fix timed wall safety check** (~0.5 pt)
5. **Improvement 6: Boost dead-end check** (~0.5 pt)
6. **Bug 4: Better solo target selection** (~0.5 pt)
7. **Improvement 10: Phase safety override** (~0.5 pt)
8. **Improvement 9: Smarter EMP** (~0.3 pt)
9. **Improvement 7: Opponent tracking** (enables future tuning)
10. **Improvement 5: Duel opening** (~0.3 pt)

Fixes 1-3 alone should push past 50. The rest are insurance.

---

## CONSTRAINTS REMINDER

- File: `bots/Botir.py`, class `Bot`, method `get_move`
- Python 3.11 stdlib only (collections, heapq, math)
- Must return within 0.3s (leave margin under 0.5s)
- Grid max 50×50 = 2500 cells — BFS is fast, don't worry
- Never crash. Wrap anything risky in try/except.
- `board[y][x]`, `pos = (x, y)`. N=y-1, S=y+1, W=x-1, E=x+1.
- Cells: `.` `#` `c` `D` `t1-t4` `p1-p4` or int (timed wall)

## NOW: Apply fixes in priority order. Test after each. Target: 50+/55.