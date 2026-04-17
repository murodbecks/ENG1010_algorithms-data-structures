# Botir_v2.md — Upgrade Instructions for Codex CLI Agent

## CONTEXT
The file `bots/Botir.py` exists and scores 45/55. We need to improve it to
55/55. DO NOT rewrite from scratch — refactor and enhance the existing bot.

Before making changes:
1. Read EVERY file in `bots/` — especially `Scout.py`, `Rogue.py`,
   `Stingray.py`, `Viper.py`, `Blaze.py`, `Drunk.py`, `Dummy.py`
2. Read EVERY `.txt` file in `maps/` to understand map layouts
3. Read `game.py`, `config.py`, `loader.py` to understand engine internals
4. Run the evaluation: `source .venv/bin/activate && ./evaluate.sh Botir`
   and note WHICH specific maps/matchups score poorly

---

## DIAGNOSIS OF CURRENT WEAKNESSES

### Problem 1: Solo Item Collection is Suboptimal
The `_guided_commands` method uses a chain-scoring heuristic of depth 2 over
at most 9 items. This is far too shallow. On solo maps, the only goal is to
collect ALL items as fast as possible while surviving. The bot needs a proper
TSP-like greedy route planner that visits items in an efficient order.

**Fix**: Replace `_guided_commands` with a full greedy-nearest-unvisited
planner for solo maps. At each turn, BFS to the nearest uncollected item,
go there, repeat. Use a persistent `self.solo_target` to avoid recomputing
every tick. Only recompute when the target is collected or unreachable.

### Problem 2: Boost is Underused on Solo Maps
Boosting gives +2 survival points per turn instead of +1, AND moves 2 cells.
On solo maps with no threats, the bot should boost whenever possible (when
both cells ahead are safe and the direction is correct). This nearly doubles
survival score over the game.

**Fix**: On solo maps, when the chosen direction matches `my_dir` and the
next 2 cells in that direction are both safe, prefer boost (`'+'`) over
normal move. Give boost a much larger bonus on solo maps.

### Problem 3: No Territory Cutting in Duels
In 1v1 duels, the winning strategy is often to cut the map in half so the
opponent has less space and dies first. The current bot doesn't do this at all.

**Fix**: Implement Voronoi territory scoring. For each candidate move:
- BFS from your destination AND from the opponent's position simultaneously
- Count cells you reach first vs cells opponent reaches first
- Strongly prefer moves that maximize YOUR territory
- When territories are close, prefer moves that create a "wall" between you
  and the opponent

### Problem 4: No Opponent-Specific Counters
The 5 enemy bots (Scout, Rogue, Stingray, Viper, Blaze) each have different
strategies. The current bot treats them all the same.

**Fix**: After reading each bot's source code, implement specific counters:
- Detect which bot you're facing from `info['opponents']` positions/behavior
  (you won't have the bot name, but you can use map_name + player positions
  to infer the scenario from the evaluation table)
- For duels: `info['opponents']` has exactly 1 alive opponent (the other 2
  are Dummy and die immediately). Adjust aggression based on observed
  behavior.

### Problem 5: Timed Walls Not Exploited
The bot treats timed walls as permanent obstacles. But timed walls with low
remaining time will open soon. The bot should:
- For pathfinding: treat timed walls with displayed value `1` (meaning ≤10
  ticks remaining) as "soon walkable" — include them in BFS with a distance
  penalty of +5 or so
- Never walk INTO a timed wall, but plan paths that route through cells that
  WILL become open by the time you arrive

**Fix**: In `_distance_map` and `_is_safe_cell` variants, add a
`_is_soon_safe(cell, distance)` that returns True if `isinstance(cell, int)`
and `cell * 10 <= distance + buffer`.

### Problem 6: Trap Detection is Too Simple
The current area threshold penalties (-250 for area ≤ blocked+2, -90 for ≤6,
-30 for ≤12) are static. The bot needs:
- Compare area to the NUMBER OF REMAINING TURNS (if area < remaining_ticks,
  you WILL die — avoid at all costs)
- A "tunnel detector" — if a move leads into a corridor with only 1 exit,
  measure the corridor length. If it's short, avoid it unless there's a
  valuable item inside.

### Problem 7: Head-to-Head Collision Avoidance
When two players move to the same cell, BOTH die. The current bot checks
opponent proximity but doesn't explicitly avoid cells that an opponent is
likely to move into this exact tick.

**Fix**: For each alive opponent, compute their most likely next position
(current_pos + direction_delta). Mark that cell as HIGH danger (penalty ≈ 8).
Also mark cells reachable by opponent boost (2 cells in their direction).

---

## SPECIFIC IMPROVEMENTS TO IMPLEMENT

### Improvement A: Persistent Solo Route Planner

```python
def __init__(self):
    self.turn_count = 0
    self.prev_pos = None
    self.prev_move = None
    self.solo_target = None       # NEW: current target for solo maps
    self.solo_visited = set()     # NEW: items we've already collected
    self.item_positions = None    # NEW: cache of all item positions on map
```

On solo maps (map_name starts with `s_`):
1. On first turn, scan entire board for all `'c'` and `'D'` positions.
   Store as `self.item_positions`.
2. Each turn, remove collected items from the set (check if board cell at
   known item position is no longer `'c'`/`'D'`).
3. If `self.solo_target` is None or already collected or unreachable, pick
   a new target:
   - BFS from current pos to ALL remaining items
   - Pick the one with best `value / (distance + 1)` ratio
   - Tiebreak: prefer diamonds, then closer items
4. BFS path to target, return the first step direction.
5. If the first step matches `my_dir` and boost is safe, return `'+'`.
6. Override the candidate scoring: the guided direction gets +200 bonus
   (not just +55).

### Improvement B: Voronoi Territory Control for Duels

Add a method:

```python
def _voronoi_score(self, board, my_pos, opp_pos, blocked):
    """Simultaneous BFS from both positions.
    Returns (my_cells, opp_cells) count."""
    rows = len(board)
    cols = len(board[0])
    visited = {}  # cell -> 'me' or 'opp'
    queue = deque()

    visited[my_pos] = 'me'
    visited[opp_pos] = 'opp'
    queue.append((my_pos, 'me', 0))
    queue.append((opp_pos, 'opp', 0))

    my_count = 0
    opp_count = 0

    while queue:
        (x, y), owner, dist = queue.popleft()
        if owner == 'me':
            my_count += 1
        else:
            opp_count += 1

        for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
            nx, ny = x + dx, y + dy
            nxt = (nx, ny)
            if not (0 <= nx < cols and 0 <= ny < rows):
                continue
            if nxt in visited or nxt in blocked:
                continue
            cell = board[ny][nx]
            if cell in ('.', 'c', 'D'):
                visited[nxt] = owner
                queue.append((nxt, owner, dist + 1))

    return my_count, opp_count
```

In `_score_candidate`, for duel maps (exactly 1 alive opponent):
- Compute Voronoi for each candidate destination vs opponent position
- Add `voronoi_weight * (my_cells - opp_cells)` to score
- `voronoi_weight` should be ~1.5 for duel maps

### Improvement C: Smarter Boost Logic

```python
# In _score_candidate, replace the simple boost_bonus with:
if candidate["boost"]:
    # Boost is great when:
    # 1. It collects an item (immediate_gain > 0)
    # 2. Area after boost is still large (> 15)
    # 3. On solo maps (always prefer boost for +2 survival)
    if is_solo:
        score += 15.0  # Big bonus on solo maps
    if candidate["immediate_gain"] > 0:
        score += 12.0
    if area >= 15:
        score += scenario["boost_bonus"]
    elif area < 8:
        score -= 60.0  # Boosting into tight space is suicidal
```

### Improvement D: Better Phase Usage

Phase should ONLY be used when:
1. All normal moves have area < 5 (you're about to be trapped)
2. Phase leads to area > 20 (it actually saves you)
3. Phase reaches a diamond that's otherwise unreachable

```python
if candidate["phase"]:
    # Only valuable as escape or for high-value unreachable items
    normal_areas = [c["area"] for c in candidates if not c.get("phase")]
    max_normal_area = max(normal_areas) if normal_areas else 0

    if max_normal_area < 5 and area > max_normal_area * 2:
        score += 80.0  # Life-saving phase
    elif candidate["immediate_gain"] >= 50:
        score += 40.0  # Diamond grab
    elif area <= max_normal_area:
        score -= 50.0  # Phase doesn't help, save the charge
```

### Improvement E: Opening Move Hardcoding

For solo maps, analyze the map layout on turn 1 and determine the optimal
opening direction. This can be done by:
1. Scan which direction from start has items
2. For `s_path` maps: detect which way the path goes and follow it
3. Store the planned opening direction in `self.opening_dir`

```python
if self.turn_count == 1 and map_name.startswith("s_"):
    # Scan all 4 directions for item density within distance 10
    best_dir = None
    best_density = -1
    for d, (dx, dy) in self.DIRS.items():
        density = 0
        cx, cy = pos
        for step in range(1, 11):
            nx, ny = cx + dx * step, cy + dy * step
            if not (0 <= nx < cols and 0 <= ny < rows):
                break
            cell = board[ny][nx]
            if cell == '#' or (isinstance(cell, str) and cell.startswith('t')):
                break
            if cell == 'c':
                density += 20
            elif cell == 'D':
                density += 50
        if density > best_density:
            best_density = density
            best_dir = d
    if best_dir:
        self.opening_dir = best_dir
```

### Improvement F: Predict Opponent Next Position

```python
def _predict_opponent_next(self, opp, board):
    """Predict where opponent will be next tick."""
    ox, oy = opp['pos']
    d = opp.get('direction', 'N')
    dx, dy = self.DIRS.get(d, (0, 0))
    nx, ny = ox + dx, oy + dy
    rows = len(board)
    cols = len(board[0])
    if 0 <= nx < cols and 0 <= ny < rows and self._is_safe_cell(board[ny][nx]):
        return (nx, ny)
    return (ox, oy)  # If forward is blocked, they'll turn — uncertain
```

Add predicted positions to the threat map with high weight (6+).

### Improvement G: Adaptive Threat Weights by Game Phase

```python
def _game_phase(self, board, opponents):
    """Determine game phase: early, mid, late."""
    total_cells = len(board) * len(board[0])
    # Count free cells
    free = sum(
        1 for y in range(len(board)) for x in range(len(board[0]))
        if board[y][x] in ('.', 'c', 'D')
    )
    ratio = free / max(total_cells, 1)
    alive = len(opponents)

    if self.turn_count < 15:
        return 'early'
    if ratio < 0.3 or alive <= 1:
        return 'late'
    return 'mid'
```

- **Early game**: focus on items, low threat weight
- **Mid game**: balanced
- **Late game**: survival is paramount, high threat weight, maximize area

---

## WEIGHT TUNING

After implementing the above, tune these weights by running evaluation
repeatedly:

### Solo Maps (target: near-perfect collection)
```python
"area": 1.5,          # Less important — just don't trap yourself
"density": 7.0,       # Very important — collect items efficiently
"reachable_value": 0.3,
"nearest_item": 3.0,  # Chase items aggressively
"turn_points": 5.0,   # Boost whenever possible for +2
"threat": 1.0,        # Drunk bots are harmless
"boost_bonus": 15.0,  # Always boost on solo
```

### Duel Maps (target: outlast + outcollect)
```python
"area": 3.5,           # Territory is crucial
"density": 2.0,
"reachable_value": 0.08,
"nearest_item": 1.0,
"turn_points": 7.0,
"threat": 12.0,        # Don't die
"boost_bonus": 6.0,
"voronoi": 1.5,        # NEW: territory control
```

### Battle Royale (target: survive + collect)
```python
"area": 3.0,
"density": 2.5,
"reachable_value": 0.1,
"nearest_item": 1.2,
"turn_points": 7.0,
"threat": 14.0,         # Very defensive
"boost_bonus": 4.0,
"voronoi": 0.5,
```

---

## TESTING PROCEDURE

After each change, test incrementally:

```bash
source .venv/bin/activate

# Test solo maps first (easiest to debug)
python runner.py -m s_path_0.txt -b Botir Drunk Drunk Drunk --max-ticks 10000
python runner.py -m s_path_1.txt -b Botir Drunk Drunk Drunk --max-ticks 10000
python runner.py -m s_path_2.txt -b Botir Drunk Drunk Drunk --max-ticks 10000
python runner.py -m s_path_3.txt -b Botir Drunk Drunk Drunk --max-ticks 10000
python runner.py -m s_floodfill_0.txt -b Botir Drunk Drunk Drunk --max-ticks 10000
python runner.py -m s_floodfill_1.txt -b Botir Drunk Drunk Drunk --max-ticks 10000
python runner.py -m s_floodfill_2.txt -b Botir Drunk Drunk Drunk --max-ticks 10000
python runner.py -m s_choice_0.txt -b Botir Drunk Drunk Drunk --max-ticks 10000
python runner.py -m s_choice_1.txt -b Botir Drunk Drunk Drunk --max-ticks 10000
python runner.py -m s_choice_2.txt -b Botir Drunk Drunk Drunk --max-ticks 10000

# Test duels against each bot
for bot in Scout Rogue Stingray Viper Blaze; do
  echo "=== vs $bot on arena ==="
  python runner.py -m arena.txt -b Botir Dummy Dummy $bot --max-ticks 10000
  python runner.py -m arena.txt -b $bot Dummy Dummy Botir --max-ticks 10000
done

# Full evaluation
./evaluate.sh Botir
```

Check which specific matchups lose and focus fixes there.

---

## CRITICAL CONSTRAINTS (DO NOT VIOLATE)

1. File must be `bots/Botir.py` with `class Bot` and `def get_move`
2. Only Python 3.11 stdlib imports (collections, heapq, math, etc.)
3. `get_move` must return within 0.3 seconds (leave margin)
4. Never crash — wrap risky logic in try/except returning a safe fallback
5. Never return an invalid move string
6. The 50x50 grid means BFS is at most 2500 nodes — keep it fast
7. Don't store huge data structures between turns (256MB memory limit)
8. `board[y][x]` — y is row, x is column. `pos = (x, y)`.

---

## SUMMARY OF CHANGES TO MAKE

1. ✅ Add persistent solo route planner (self.solo_target, self.item_positions)
2. ✅ Add Voronoi territory scoring for duels
3. ✅ Improve boost logic (much more aggressive on solo maps)
4. ✅ Improve phase logic (only for emergencies or diamonds)
5. ✅ Add opening move analysis for solo maps
6. ✅ Add opponent next-position prediction to threat map
7. ✅ Add game phase detection for adaptive weights
8. ✅ Tune weights per scenario
9. ✅ Add timed wall awareness to pathfinding
10. ✅ Read all enemy bot files and implement any specific counters possible

## NOW: Apply all improvements to `bots/Botir.py`, test with evaluation,
## and iterate until score improves. Target: 55/55.