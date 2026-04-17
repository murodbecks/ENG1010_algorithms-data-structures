# Botir.md — Full Instructions for Codex CLI Agent

## GOAL
Create a single file `bots/Botir.py` that contains a `class Bot` with a method
`def get_move(self, board, player_id, pos, info)` and achieves the highest
possible score across Solo, Duel, and Battle Royale scenarios in the Tron Snake
Arena game (ENG1010 assignment).

The bot must run under **Python 3.11** only (use `.venv/` virtualenv).
The file must be self-contained — no external imports beyond the Python 3.11
standard library (collections, heapq, math, etc.). No numpy, no torch, no
third-party packages.

---

## STEP 0 — READ THE EXISTING CODEBASE FIRST

Before writing any code, read and understand these files in the repo:

1. `game.py` — the full game engine (I pasted it below for reference)
2. `bots/Scout.py`, `bots/Rogue.py`, `bots/Stingray.py`, `bots/Viper.py`,
   `bots/Blaze.py` — the 5 enemy bots you must beat
3. `bots/Drunk.py` — the drunk bot used in solo maps
4. `bots/Dummy.py` — the dummy bot used in duels (does nothing)
5. `maps/` folder — all `.txt` map files, especially the solo ones:
   `s_path_0.txt` through `s_path_3.txt`, `s_floodfill_0.txt` through
   `s_floodfill_2.txt`, `s_choice_0.txt` through `s_choice_2.txt`
6. `config.py` — game constants
7. `loader.py` — how maps are loaded
8. `runner.py` — CLI runner
9. `evaluate.sh` — the evaluation script

Understanding the enemy bots is CRITICAL. Analyze their strategies so you can
counter them.

---

## STEP 1 — UNDERSTAND THE GAME RULES

### Board
- 2D grid, max 50×50.
- `board[y][x]` gives the cell content.
- Cell values:
  - `'.'` = empty (safe to walk)
  - `'#'` = permanent wall (deadly, can phase through)
  - integer `1`–`9` = timed wall, turns remaining ÷ 10 rounded up (deadly,
    will become `'.'` eventually; the actual remaining ticks are stored
    internally)
  - `'t1'`..`'t4'` = player trail (deadly)
  - `'c'` = coin (+20 pts)
  - `'D'` = diamond (+50 pts)
  - `'p1'`..`'p4'` = player head positions (added to the view grid)

### Movement
- Return one of: `'N'`, `'S'`, `'E'`, `'W'`, `'+'`, `'P'`
- Can prefix with `'X'` to fire EMP: `'XN'`, `'XS'`, `'XE'`, `'XW'`, `'X+'`,
  `'XP'`
- **N** = y-1, **S** = y+1, **W** = x-1, **E** = x+1
- **Boost (`'+'`)**: move 2 cells in current direction. Both cells must be safe.
- **Phase (`'P'`)**: move 2 cells in current direction. First cell can be
  anything (ignored), second cell must be safe. Limited to 3 uses per game.
- **EMP (`'X'` prefix)**: activates 5×5 blast around your head. Detonates after
  5 ticks. Stuns enemies for 7 turns. +50 pts per enemy hit. 1 charge,
  recharges after detonation + cooldown.
- Moving backwards into your own trail = instant death.
- Time limit: 0.5s soft (warning), 2s hard (instant death). 5 warnings = death.

### Scoring
- +1 per turn survived (+2 if boosting or phasing)
- Coin: +20, Diamond: +50
- EMP hit: +50 per opponent stunned
- Survival bonuses: last alive +50, 2nd last +25, 3rd +10

### Key mechanic details from game.py
- Moves are simultaneous. All players move at once.
- Head-to-head collision on same cell kills BOTH players.
- When 2+ bots contest same coin/diamond, lower player_id gets priority.
- Trail is committed AFTER all moves resolve (so you can't collide with a trail
  that was just placed this tick by another player moving to a new position).
- EMP follows the owner's head position each tick.
- Stunned players keep moving in their last direction (loss_of_control).
- Players 1,2 start facing South; Players 3,4 start facing North.

---

## STEP 2 — THE STRATEGY (what the friend did to get 55/55)

The friend's approach:
1. Simulated 1000 games to study game logs
2. Analyzed the 5 enemy bot scripts to understand their behavior
3. Fed bot scripts + map pictures to an AI to generate counter-strategies
4. For some maps, hardcoded opening moves to guarantee optimal collection

### Your bot MUST implement these core algorithms:

### A. BFS/A* Pathfinding
- Use BFS (breadth-first search) to find shortest path to targets (coins,
  diamonds, safe cells).
- A cell is **safe** if it's `'.'`, `'c'`, or `'D'`.
- A cell is **dangerous** if it's `'#'`, a timed wall (integer), any trail
  (`'t1'`–`'t4'`), out of bounds, or an opponent head.
- Timed walls with low remaining value (≤ 2) can be treated as "soon safe"
  for planning.

### B. Flood Fill for Space Control
- After finding candidate moves, use flood fill from each candidate next
  position to count reachable empty cells.
- NEVER move into a direction with very few reachable cells (you'll trap
  yourself).
- This is the #1 survival mechanism.

### C. Target Prioritization
- Diamonds (+50) > Coins (+20) > Survival
- But only pursue a target if:
  - You are the closest player to it (BFS distance), OR
  - You have lower player_id (tiebreaker advantage)
  - The path doesn't lead to a dead-end (flood fill check)

### D. Opponent Awareness
- Track opponent positions from `info['opponents']`
- Avoid cells adjacent to opponent heads (they might move there)
- In duels, actively try to cut off opponent's space (Voronoi territory)
- Consider opponent direction to predict their next move

### E. EMP Usage
- Use EMP when an opponent is within ~3 Manhattan distance and you can keep
  them in range for 5 ticks
- Combine EMP with movement: `'XN'`, `'XE'`, etc.
- Don't waste EMP when no opponents are nearby

### F. Phase Usage
- Save phase charges for emergencies (about to be trapped)
- Use phase to skip over walls or trails when it leads to a much better
  position (e.g., reaching a diamond behind a wall)
- Only 3 per game — use wisely

### G. Boost Usage
- Use boost to grab distant coins/diamonds faster (+2 survival points too)
- Use boost to escape dangerous situations
- Make sure BOTH cells in boost path are safe

### H. Map-Specific Strategies
- Check `info['map_name']` to detect map type
- **s_path_N**: These are corridor/path maps. Follow the path, collect all
  items. Basically a traversal problem — use BFS to find the optimal route
  through the path. May need to go in a specific direction at start.
- **s_floodfill_N**: Open area maps. Use flood fill to maximize territory
  while collecting items.
- **s_choice_N**: Maps with branching choices. Evaluate each branch's total
  reward (sum of coins + diamonds) vs risk, pick the best one.
- **arena/maze/treasure/gate/cube/orbit**: Competitive maps. Focus on survival
  + items + opponent avoidance.

### I. Anti-Opponent Bot Strategies
After reading the 5 enemy bot scripts, implement counters:
- **Scout**: likely a simple pathfinding bot → out-collect it
- **Rogue**: likely aggressive → avoid it, let it crash into walls
- **Stingray**: likely uses EMP → stay out of EMP range (>2 cells away)
- **Viper**: likely cuts off space → maintain large territory
- **Blaze**: likely uses boost aggressively → watch for fast approaches

---

## STEP 3 — IMPLEMENTATION STRUCTURE

```python
from collections import deque
import heapq

class Bot:
    def __init__(self):
        self.turn_count = 0
        self.prev_pos = None
        self.prev_direction = None

    def get_move(self, board, player_id, pos, info):
        self.turn_count += 1
        rows = len(board)
        cols = len(board[0])
        x, y = pos
        my_dir = info['my_direction']
        phase_charges = info['phase_charges']
        emp_charges = info['emp_charges']
        opponents = info['opponents']
        active_emps = info['active_emps']
        map_name = info['map_name']

        # 1. Build safety grid
        # 2. Find all safe moves (N/S/E/W that don't immediately kill us)
        # 3. For each safe move, flood fill to count reachable area
        # 4. Find all collectibles, BFS to nearest high-value target
        # 5. If good target found and path is safe, move toward it
        # 6. Otherwise, pick move with largest flood fill area
        # 7. Consider boost if beneficial
        # 8. Consider phase if trapped
        # 9. Consider EMP if opponent nearby
        # 10. Return the chosen move string

        # ... (implement all the above)

        return move
```

---

## STEP 4 — DETAILED IMPLEMENTATION REQUIREMENTS

### 4.1 Safe Move Detection
```python
DIRS = {'N': (0, -1), 'S': (0, 1), 'W': (-1, 0), 'E': (1, 0)}

def is_safe(board, rows, cols, nx, ny):
    """Check if a cell is safe to move into."""
    if nx < 0 or nx >= cols or ny < 0 or ny >= rows:
        return False
    cell = board[ny][nx]
    if cell == '.' or cell == 'c' or cell == 'D':
        return True
    return False

def get_opposite(direction):
    opposites = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
    return opposites.get(direction)
```

- Filter out the **reverse** of current direction (instant death).
- Filter out any move that leads to a wall, trail, or out of bounds.

### 4.2 Flood Fill
```python
def flood_fill_count(board, rows, cols, start_x, start_y, danger_set):
    """Count reachable safe cells from a starting position."""
    visited = set()
    queue = deque([(start_x, start_y)])
    visited.add((start_x, start_y))
    count = 0
    while queue:
        cx, cy = queue.popleft()
        count += 1
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) not in visited and 0 <= nx < cols and 0 <= ny < rows:
                if (nx, ny) not in danger_set:
                    cell = board[ny][nx]
                    if cell == '.' or cell == 'c' or cell == 'D':
                        visited.add((nx, ny))
                        queue.append((nx, ny))
    return count
```

- The danger_set should include opponent head positions and cells adjacent to
  opponent heads.

### 4.3 BFS to Target
```python
def bfs_path(board, rows, cols, start, targets, danger_set):
    """BFS from start to nearest target. Returns (target_pos, distance, first_step_direction)."""
    sx, sy = start
    visited = {(sx, sy)}
    queue = deque()
    # Initialize with all 4 directions
    for d, (dx, dy) in DIRS.items():
        nx, ny = sx + dx, sy + dy
        if 0 <= nx < cols and 0 <= ny < rows and (nx, ny) not in danger_set:
            cell = board[ny][nx]
            if cell == '.' or cell == 'c' or cell == 'D':
                if (nx, ny) in targets:
                    return ((nx, ny), 1, d)
                visited.add((nx, ny))
                queue.append((nx, ny, d, 1))
    while queue:
        cx, cy, first_dir, dist = queue.popleft()
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) not in visited and 0 <= nx < cols and 0 <= ny < rows:
                if (nx, ny) not in danger_set:
                    cell = board[ny][nx]
                    if cell == '.' or cell == 'c' or cell == 'D':
                        if (nx, ny) in targets:
                            return ((nx, ny), dist + 1, first_dir)
                        visited.add((nx, ny))
                        queue.append((nx, ny, first_dir, dist + 1))
    return None
```

### 4.4 Opponent Danger Zones
```python
def get_danger_set(opponents, board, rows, cols):
    """Get set of cells to avoid (near opponent heads)."""
    danger = set()
    for opp in opponents:
        if opp['alive']:
            ox, oy = opp['pos']
            danger.add((ox, oy))
            # Add adjacent cells (opponent might move there)
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = ox + dx, oy + dy
                if 0 <= nx < cols and 0 <= ny < rows:
                    danger.add((nx, ny))
    return danger
```

### 4.5 Move Selection Logic (Priority Order)
1. **Eliminate suicide moves** (reverse direction, walls, trails, out of bounds)
2. **If only one safe move**, take it immediately
3. **Flood fill** each candidate move → reject any with reachable area < 5
   (unless all are small)
4. **Find collectibles** (diamonds first, then coins) → BFS to closest one
   that we can reach before opponents
5. **If good target exists** and the first step direction is a safe move with
   decent flood fill, go for it
6. **Otherwise** pick the safe move with the **largest flood fill** area
7. **Boost**: if current direction's next 2 cells are both safe AND there's a
   collectible at distance 2 in that direction, use `'+'`
8. **Phase**: if ALL normal moves have tiny flood fill (< 3) and phasing in
   current direction lands on a safe cell with better flood fill, use `'P'`
9. **EMP**: if an opponent is within Manhattan distance ≤ 3 and alive, and we
   have charges, prefix the move with `'X'`

### 4.6 Timed Wall Awareness
```python
# Timed walls appear as integers on the board
# A cell with value 1 means it opens in ~10 ticks
# A cell with value 2 means ~20 ticks, etc.
# For pathfinding, treat timed walls with value 1 as "soon walkable"
# but don't walk into them NOW
```

### 4.7 Solo Map Handling
For solo maps (`s_path_*`, `s_floodfill_*`, `s_choice_*`):
- No real opponents (Drunk bots are random and die quickly)
- Focus 100% on collecting ALL coins and diamonds
- Use BFS to plan optimal collection route
- **s_path** maps: these are narrow corridor maps. Plan a route that visits
  all items. Think of it as a traversal — go one direction, collect everything,
  come back.
- **s_floodfill** maps: open areas. Use a greedy nearest-item BFS loop.
- **s_choice** maps: branching paths with different rewards. Evaluate total
  value of each branch, pick the richest one first.

### 4.8 Duel Map Handling
For duel maps (arena, maze, treasure, gate, cube, orbit):
- Only 1 real opponent (Players 2,3 are Dummy — they don't move and die
  immediately)
- Balance between collecting items and controlling territory
- Try to cut off opponent's space using Voronoi-style territory control
- Survive longer than the opponent for +50 bonus

### 4.9 Battle Royale Handling
- 3 real opponents
- More defensive play
- Prioritize survival and opportunistic item collection
- Avoid head-to-head areas
- Use EMP when multiple opponents cluster near you

---

## STEP 5 — TESTING

After generating `bots/Botir.py`, test it:

```bash
# Activate venv
source .venv/bin/activate

# Quick test on one map
python runner.py -m s_path_0.txt -b Botir Drunk Drunk Drunk --max-ticks 10000

# Quick duel test
python runner.py -m arena.txt -b Botir Dummy Dummy Scout --max-ticks 10000

# Full evaluation
chmod +x evaluate.sh
./evaluate.sh Botir
```

Watch for:
- Bot crashing (any exception = instant death)
- Bot timing out (> 0.5s per move gets warnings)
- Bot going backwards (check reverse direction logic)
- Bot getting trapped (flood fill not working)

---

## STEP 6 — PERFORMANCE REQUIREMENTS

- **get_move must return within 0.3s** (leave margin under 0.5s limit)
- Grid is max 50×50 = 2500 cells. BFS/flood fill on this is fast.
- Don't use recursion for flood fill (stack overflow risk on large maps).
  Use iterative BFS with deque.
- Don't deepcopy the board unnecessarily.
- Keep `__init__` state minimal to avoid memory limit (256MB).

---

## STEP 7 — FINAL CHECKLIST

- [ ] File is `bots/Botir.py`
- [ ] Contains `class Bot` with `def get_move(self, board, player_id, pos, info)`
- [ ] Returns one of: `'N'`, `'S'`, `'E'`, `'W'`, `'+'`, `'P'`, or prefixed
      with `'X'` (e.g., `'XN'`, `'X+'`)
- [ ] Never returns invalid string
- [ ] Never moves backwards into own trail
- [ ] Uses BFS pathfinding to targets
- [ ] Uses flood fill to avoid traps
- [ ] Handles all cell types correctly
- [ ] Handles solo/duel/battle royale modes
- [ ] Handles map-specific strategies via `info['map_name']`
- [ ] Uses boost strategically
- [ ] Uses phase as emergency escape
- [ ] Uses EMP when opponents are close
- [ ] Runs within 0.3s per call
- [ ] No external dependencies (only stdlib: collections, heapq, math)
- [ ] No crashes, no exceptions

---

## IMPORTANT EDGE CASES TO HANDLE

1. **First turn**: `prev_pos` is None, be careful with direction logic
2. **All moves are deadly**: if no safe move exists, try Phase. If no phase
   charges, pick the "least bad" option (e.g., timed wall about to open).
   As absolute last resort, just go in current direction.
3. **Opponent on same cell contest**: lower player_id wins the item. If your
   ID is higher, don't race for contested items.
4. **Stunned state**: when stunned, the engine forces your current direction.
   You can't change it. Your bot still gets called but the return is ignored
   during stun. Still return a valid move to avoid errors.
5. **Board cell types**: cells can be strings (`'.'`, `'#'`, `'c'`, `'D'`,
   `'t1'`–`'t4'`, `'p1'`–`'p4'`) OR integers (timed walls). Always check
   `isinstance(cell, int)` for timed walls.
6. **Boost path validation**: for boost (`'+'`), BOTH the intermediate cell
   and the destination cell must be safe. If either is a wall/trail, you die.
7. **Phase landing validation**: for phase (`'P'`), the intermediate cell is
   ignored but the DESTINATION cell must be safe. If destination is a wall,
   you die.

---

## REFERENCE: info DICTIONARY STRUCTURE

```python
info = {
    'map_name': str,           # e.g., 's_path_0', 'arena'
    'my_score': int,           # current score
    'my_direction': str,       # 'N', 'S', 'E', or 'W'
    'phase_charges': int,      # remaining phase uses (0-3)
    'emp_charges': int,        # remaining EMP charges (0 or 1)
    'emp_radius': int,         # EMP blast radius (from config)
    'stun_duration': int,      # EMP stun duration (from config)
    'active_emps': [           # list of active EMPs on the field
        {'pos': (x, y), 'timer': int},
        ...
    ],
    'opponents': [             # list of opponent info
        {
            'id': int,         # player id (1-4)
            'pos': (x, y),     # current position
            'alive': bool,     # whether alive
            'direction': str   # current facing direction
        },
        ...
    ]
}
```

---

## REFERENCE: BOARD COORDINATE SYSTEM

- `board[y][x]` — y is row (vertical), x is column (horizontal)
- `pos = (x, y)` — position tuple is (column, row)
- North = y decreases, South = y increases
- West = x decreases, East = x increases
- `board[0][0]` is top-left corner

---

## NOW: Generate the complete `bots/Botir.py` file

Read all bot files in `bots/` folder first to understand enemy strategies.
Read all map files in `maps/` folder to understand map layouts.
Then implement the bot following all the above specifications.
Make it robust, fast, and competitive.
```

Save this as `Botir.md` in your project root, then run:

```bash
source .venv/bin/activate
codex -a Botir.md
```

Or if you're using the Codex CLI with a prompt:

```bash
codex --approval-mode full-auto "Read the file Botir.md in the project root. Follow ALL instructions in it. First read every file in bots/ and maps/ folders to understand enemy bots and map layouts. Then create bots/Botir.py following the complete specification in Botir.md. Test it with: python runner.py -m s_path_0.txt -b Botir Drunk Drunk Drunk --max-ticks 10000"
```

**Key tips to maximize your score:**

1. **The biggest wins come from reading the enemy bot source code** — Codex has access to your repo, so telling it to read `bots/Scout.py`, `bots/Rogue.py`, etc. lets it build counter-strategies
2. **Solo maps are free points** — if the bot just survives and collects items efficiently, that's 10 easy points
3. **Flood fill is the #1 survival technique** — never pick a move that traps you in a small area
4. **Iterate** — after the first generation, run `./evaluate.sh Botir`, check which scenarios score low, and ask Codex to improve those specific cases