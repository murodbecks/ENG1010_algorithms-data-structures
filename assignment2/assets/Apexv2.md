# Tron Snake Bot — Tuning Plan (v2)

## Current Status

**Latest verified result on 2026-04-15**
- Solo: `675 / 1000` → `6.75 / 10`
- Duel: `6000 / 6000` → `30.00 / 30`
- Battle Royale: `1100 / 1500` → `11.00 / 15`
- Total automated score: `47.75 / 55.00`

Duel is solved. The remaining ceiling is **Solo (+325 available)** and
**Battle Royale (+400 available)**. Solo has the best ROI because the
opponents are Drunk bots and the outcomes are deterministic and
reproducible.

## Required Python Version

Always run the bot with Python 3.11.

```bash
conda run -n tron_snake bash ./evaluate.sh Apex
```

## Verified Engine Facts (unchanged)

1. `get_move(self, board, player_id, pos, info)` receives `board`,
   `player_id`, `pos`, `info`.
2. `info` keys: `map_name`, `my_score`, `my_direction`, `phase_charges`,
   `emp_charges`, `emp_radius`, `stun_duration`, `active_emps`,
   `opponents`.
3. Board cells: `.` empty, `#` wall, integer = timed wall, `c` coin,
   `D` diamond, `t1`–`t4` trail, `p1`–`p4` head.
4. `+` and `P` move two cells in `info['my_direction']`.
5. Timed wall board value `n` ≈ `n * 10` turns remaining.
6. Any exception destroys the bot. Keep the top-level `try/except`.

## What Changed Since v1

- Branch profiling, item density, Voronoi, EMP avoidance, crowd/escape
  scoring, opening discipline, hidden-trail tracking all shipped.
- Duel went from 4500 → 6000 (perfect). **Do not regress this.**
- Solo stayed flat at 675. This is the primary growth target.
- BR improved from 850 → 1100 but plateaued.

## Score Anatomy — Where Points Come From

Understanding the point sources is critical for prioritizing work.

### Solo point budget (per map)

| Source              | Typical | Max potential | Notes |
|---------------------|---------|---------------|-------|
| Survival turns (+1) | ~80     | ~150+         | live longer |
| Boost turns (+2)    | ~20     | ~150+         | boost every safe turn |
| Coins collected     | ~40     | ~200+         | depends on map |
| Diamonds collected  | ~0–50   | ~100+         | depends on map |
| Last alive bonus    | +50     | +50           | almost guaranteed vs Drunk |

**Key insight**: boosting on every safe turn effectively doubles survival
income. On a 150-turn game, that is +150 extra points — almost as much
as collecting all coins on the map. This is the single largest untapped
solo point source.

### BR point budget (per match)

Placement dominates. Getting 1st (+100) vs 2nd (+50) is worth far more
than any in-game item collection. Survival and territory control matter
more than item racing in BR.

---

## Priority 1: Solo Boost Maximization (HIGH ROI)

**Current problem**: Apex boosts occasionally but not systematically. On
solo maps with Drunk bots, danger is near zero, so the bot should boost
on almost every turn where it is safe.

**Goal**: Boost on every turn where:
- The landing cell (2 ahead) is safe
- The intermediate cell (1 ahead) is safe
- Flood fill from landing is above threshold
- No fork is skipped (existing branch logic)
- No upcoming turn within 1 cell of landing

**Tasks**

1. Add a `_solo_boost_viable(board, pos, direction, rows, cols)` check:
   - Verify both cells ahead are walkable
   - Verify landing has ≥2 safe neighbors (not a dead-end)
   - Verify no item is skipped on intermediate cell that we can't
     collect (if intermediate has a coin, boosting skips it — weigh
     this tradeoff)

2. On `s_*` maps, when `_solo_boost_viable` is true, give boost a large
   bonus (e.g., +25) so it almost always wins over normal moves.

3. Special case: if the intermediate cell has a coin/diamond, prefer
   normal move to collect it, THEN boost next turn. Track this with a
   `self.boost_deferred` flag.

4. On `s_path_*` maps: boost aggressively along straight corridors.
   The branch profiler already computes `straight_run`. If
   `straight_run >= 2`, boost is always safe.

5. On `s_floodfill_*` maps: boost when heading into open space. Avoid
   boosting toward walls or into corners.

**Expected gain**: +60 to +120 solo points across 10 maps.

**Validation**
```bash
conda run -n tron_snake python runner.py -m s_path_0.txt -b Apex Drunk Drunk Drunk
conda run -n tron_snake python runner.py -m s_path_1.txt -b Apex Drunk Drunk Drunk
conda run -n tron_snake python runner.py -m s_path_2.txt -b Apex Drunk Drunk Drunk
conda run -n tron_snake python runner.py -m s_path_3.txt -b Apex Drunk Drunk Drunk
conda run -n tron_snake python runner.py -m s_floodfill_0.txt -b Apex Drunk Drunk Drunk
conda run -n tron_snake python runner.py -m s_floodfill_1.txt -b Apex Drunk Drunk Drunk
conda run -n tron_snake python runner.py -m s_floodfill_2.txt -b Apex Drunk Drunk Drunk
conda run -n tron_snake python runner.py -m s_choice_0.txt -b Apex Drunk Drunk Drunk
conda run -n tron_snake python runner.py -m s_choice_1.txt -b Apex Drunk Drunk Drunk
conda run -n tron_snake python runner.py -m s_choice_2.txt -b Apex Drunk Drunk Drunk
```

Compare per-map scores before and after. Every map should improve or hold.

---

## Priority 2: Solo Space-Filling Traversal (HIGH ROI)

**Current problem**: On `s_floodfill_*` maps, the bot wanders reactively
toward nearby items but does not plan a traversal that covers the most
ground. This leaves items uncollected and shortens survival.

**Goal**: Approximate a Hamiltonian-path-like sweep through open areas.

**Tasks**

1. Implement a `_sweep_direction(board, pos, rows, cols)` heuristic:
   - Divide the reachable area into quadrants or strips
   - Determine which strip/quadrant has the most unvisited cells and
     items
   - Return a directional bias toward that region

2. On `s_floodfill_*` maps, add a `sweep_bias` term to scoring:
   - Moves that advance toward the densest unvisited region get a bonus
   - Moves that revisit already-trailed areas get a penalty

3. Track visited cells across turns with `self.visited_cells = set()`.
   Update it every turn. Use it to bias toward unvisited territory.

4. Wall-hugging heuristic: when in open space with no clear item target,
   prefer moves that keep a wall on one side. This naturally creates
   efficient space-filling patterns without complex planning.
   - Check: does the move keep exactly 1 wall/trail neighbor?
   - If so, small bonus.

**Expected gain**: +30 to +80 solo points, concentrated on floodfill maps.

**Validation**: Same solo test suite as Priority 1.

---

## Priority 3: Solo Item Skip Prevention

**Current problem**: The bot sometimes boosts over coins or takes a
suboptimal path that misses a coin by 1 cell. On solo maps every missed
coin is -20 points with no way to recover.

**Tasks**

1. Add an `intermediate_item_loss` penalty to boost candidates:
   - If boosting, check the intermediate cell
   - If it contains `c` or `D`, compute: can we collect it with a
     normal move this turn and still boost next turn?
   - If yes, penalize the boost by the item value
   - If the item is a diamond (+50), almost never boost over it

2. Add a `nearby_uncollected` scan after choosing a direction:
   - BFS depth 1–2 from landing
   - If a coin/diamond is 1 step off the chosen path and the detour
     does not reduce flood fill below threshold, prefer the detour

3. On `s_choice_*` maps, at fork points, estimate total item value down
   each branch to depth 10–15. Commit to the richer branch. Once
   committed, don't reverse unless the branch is exhausted.
   - Store `self.committed_branch_direction` and
     `self.committed_branch_turns_remaining`
   - Reset when the branch is fully explored (flood fill from current
     matches only the committed direction's reachable set)

**Expected gain**: +20 to +50 solo points.

---

## Priority 4: Opponent Behavior Modeling for BR (MEDIUM ROI)

**Current problem**: Apex treats all opponents identically. But the
assignment names specific bots (Scout, Rogue, Stingray, Viper, Blaze)
that likely have predictable behaviors.

**Tasks**

1. Observe each bot's behavior by running visual games:
   ```bash
   conda run -n tron_snake python main.py
   ```
   Note for each bot:
   - Does it chase items aggressively? (Scout, Rogue?)
   - Does it use EMP? (Blaze?)
   - Does it play defensively / wall-hug? (Viper?)
   - Does it boost frequently? (Stingray?)

2. Store observations in a `BOT_PROFILES` dict:
   ```python
   BOT_PROFILES = {
       "Scout": {"aggressive": False, "emp_user": False, ...},
       "Blaze": {"aggressive": True, "emp_user": True, ...},
       ...
   }
   ```

3. Adjust danger weighting based on opponent type:
   - Against EMP-using bots: increase `emp_threat` weight
   - Against aggressive chasers: increase `danger` and `crowd` weights
   - Against passive bots: decrease danger, increase item pursuit

4. The opponent bot names may be available in `info['opponents']` entries
   or may need to be inferred from behavior patterns. Check the engine
   code to see if bot names are exposed.

5. If bot names are NOT exposed, implement simple behavior fingerprinting:
   - Track each opponent's movement pattern over 5–10 turns
   - Classify: "chaser" (moves toward items/players), "wanderer"
     (random), "hugger" (follows walls), "aggressive" (uses EMP/boost)
   - Store classification in `self.opponent_profiles`

**Expected gain**: +25 to +75 BR points from better opponent-specific play.

**Validation**: Re-test the 6 failing BR lineups individually.

---

## Priority 5: Anticipatory Territory Denial in BR (MEDIUM ROI)

**Current problem**: In BR, Apex reacts to opponents but does not
proactively cut off their space.

**Tasks**

1. On open BR maps (`arena`, `cube`, `orbit`), add a
   `_territory_denial_score(board, pos, candidate, opponents, rows, cols)`:
   - Simulate: if I move here, does it reduce the nearest opponent's
     flood fill by more than it reduces mine?
   - Prefer moves that partition the board in our favor

2. Implementation:
   - After choosing top-2 candidates by existing scoring, compare their
     Voronoi impact against the closest opponent
   - If one candidate cuts the opponent's territory by ≥20% more, bias
     toward it (+15 to +25 bonus)

3. Early-game land grab (turns 1–15):
   - On `arena` and `cube`, prefer moves toward the board center
   - Avoid committing to a corner or edge in the first 10 turns
   - The current `center_bonus` helps but may not be strong enough
     in BR specifically

4. Mid-game seal (turns 15–40):
   - If Apex controls ≥55% of Voronoi space, switch to conservative
     play: maximize own flood fill, stop chasing items in opponent
     territory
   - If Apex controls <45%, play more aggressively toward contested
     zones

**Expected gain**: +25 to +50 BR points.

---

## Priority 6: Score-Aware Endgame Play (LOW-MEDIUM ROI)

**Current problem**: Apex plays the same way regardless of score position.
In BR, when already ahead, it should play conservatively. When behind, it
should take calculated risks.

**Tasks**

1. Read `info['my_score']` and `info['opponents'][*]['score']` each turn.

2. Compute score rank (1st, 2nd, 3rd, 4th).

3. Strategy adjustments:
   - **If 1st place and alive opponents ≤ 2**: maximize survival.
     Increase `space` weight by 50%, decrease `item` weight by 30%.
     Do NOT boost into risky territory.
   - **If last place**: increase `item` weight by 40%, accept slightly
     lower flood fill for item access. Consider using EMP more
     aggressively for the +50 stun bonus.
   - **If close to an opponent** (within 20 points): prioritize
     placement. Try to outlast them rather than outscore them.

4. Survival countdown: if only 2 players remain, the last alive gets
   +50. At this point, pure survival (maximize flood fill, avoid all
   risks) is worth more than any single item.

**Expected gain**: +15 to +40 BR points.

---

## Priority 7: Phase Usage Optimization (LOW ROI, HIGH SAFETY)

**Current problem**: Phase is used reactively when trapped. It could be
used proactively for shortcuts.

**Tasks**

1. On solo maps, scan for "phase shortcuts":
   - Is there a wall/trail exactly 1 cell thick with valuable items on
     the other side?
   - Would phasing through it give access to ≥30 new cells or a
     diamond?
   - If yes, and `phase_charges > 1`, consider proactive phase

2. Save at least 1 phase charge as an emergency escape on all map types.

3. On BR maps, phase is most valuable as an escape. Do not use it
   proactively in BR unless the gain is ≥50 points worth.

4. Add `self.phase_budget`:
   - Solo: can use all 3 proactively if the gains are clear
   - Duel: save 1 for emergency, use up to 2 proactively
   - BR: save 2 for emergencies, use at most 1 proactively

**Expected gain**: +10 to +30 across all modes.

---

## Priority 8: Code Cleanup Without Score Regression

**Current problem**: `Apex.py` has accumulated complexity. Some scoring
terms may be redundant or counterproductive.

**Tasks**

1. **Identify dead weight**: temporarily zero out each scoring term one
   at a time and re-run evaluation. If score holds or improves, the
   term is not helping.
   - Candidates to test: `orbit_risk`, `edge_bias`,
     `opening_axis_pressure`, `center_bonus`, `straight_bonus`

2. **Merge redundant BFS passes**: `_best_item_score`,
   `_local_item_density`, and `_branch_profile` all run separate BFS
   from similar starting points. Merge into a single BFS that collects
   all needed data in one pass. This also helps with time budget.

3. **Simplify weight dict**: if ablation shows that `future_space` and
   `density` have very similar effects, merge them into one term with
   one weight.

4. **Profile time budget**:
   - Add `time.time()` measurement around the main loop
   - Log worst-case turn times
   - Target: 95th percentile under 200ms on 50×50 boards
   - If any BFS is taking too long, cap its depth

**Guardrail**: Run full evaluation before AND after cleanup. If total
score drops by more than 0.5 points, revert.

---

## Priority 9: Hidden Scenario Robustness (DEFENSIVE)

25 points come from hidden scenarios. The assignment guarantees:
- Hidden solo maps are variations of path/floodfill/choice with names
  like `s_path_N`
- Hidden duel/BR may have new maps or new bots

**Tasks**

1. Ensure map-name detection is robust:
   - `s_path` prefix → path weights
   - `s_floodfill` prefix → floodfill weights
   - `s_choice` prefix → choice weights
   - Unknown `s_*` prefix → use floodfill weights as default (most
     general)
   - Unknown non-`s_` map → use arena/open weights as default

2. Ensure the bot handles edge cases:
   - Very small maps (5×5)
   - Very large maps (50×50)
   - Maps with no items at all
   - Maps where all opponents are already dead on turn 1
   - Maps where spawn positions are in corners vs center

3. Test with synthetic scenarios:
   ```bash
   # Test on every available map with Drunk bots to verify no crashes
   for map in maps/*.txt; do
     conda run -n tron_snake python runner.py -m "$map" -b Apex Drunk Drunk Drunk
   done
   ```

**Expected gain**: defensive — prevents losing points on hidden tests.

---

## Anti-Regression Checklist

Before committing any change, verify:

- [ ] `./evaluate.sh Apex` total ≥ 47.75
- [ ] Duel score = 6000
- [ ] No crashes on any map
- [ ] No timeout warnings

If any check fails, revert the change before trying the next one.

---

## Suggested Implementation Order

````
Priority 1 (Solo Boost)     → expect +60–120 solo
    ↓ evaluate
Priority 2 (Space Filling)  → expect +30–80 solo
    ↓ evaluate
Priority 3 (Item Skip)      → expect +20–50 solo
    ↓ evaluate
Priority 4 (Bot Modeling)   → expect +25–75 BR
    ↓ evaluate
Priority 5 (Territory)      → expect +25–50 BR
    ↓ evaluate
Priority 6 (Score-Aware)    → expect +15–40 BR
    ↓ evaluate
Priority 7 (Phase)          → expect +10–30 all
    ↓ evaluate
Priority 8 (Cleanup)        → expect +0–10 (time savings)
    ↓ evaluate
Priority 9 (Robustness)     → defensive
    ↓ final evaluate
```

## Target Scores

| Milestone | Solo | Duel | BR | Total |
|-----------|------|------|----|-------|
| Current   | 675  | 6000 | 1100 | 47.75 |
| After P1–P3 | 850+ | 6000 | 1100 | 49.50+ |
| After P4–P6 | 850+ | 6000 | 1250+ | 51.00+ |
| Stretch   | 900+ | 6000 | 1350+ | 52.50+ |

## Experiment Log

| Pass | Solo | Duel | BR | Total | Notes |
|------|------|------|----|-------|-------|
| v1 Baseline | 675 | 6000 | 1100 | 47.75 | Starting point for v2 plan |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |

## Guardrails

1. **Never regress duel.** It is perfect. Do not change duel-specific
   logic unless a change is proven neutral on all 60 duel matchups.
2. **One priority at a time.** Evaluate after each. Do not stack
   untested changes.
3. **Prefer additive scoring terms over rewrites.** Add a new bonus/
   penalty rather than restructuring the main scoring loop.
4. **Keep the try/except.** Any uncaught exception kills the bot.
5. **If a change helps solo but hurts BR by more points, revert it.**
   Use map-specific weights to isolate changes.