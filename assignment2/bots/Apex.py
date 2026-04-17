
from collections import deque


MOVE_DELTAS = {
    "N": (0, -1),
    "S": (0, 1),
    "E": (1, 0),
    "W": (-1, 0),
}

ITEM_VALUES = {
    "c": 20,
    "D": 50,
}

OPEN_CELLS = {".", "c", "D"}
MAP_OPEN = "OPEN"
MAP_CORRIDOR = "CORRIDOR"
MAP_BRANCHING = "BRANCHING"


class Bot:
    def __init__(self):
        self.turn_count = 0
        self.prev_pos = None
        self.prev_direction = "S"
        self.known_trail_cells = set()
        self.map_profile = None
        self.saw_real_battle = False

    def get_move(self, board, player_id, pos, info):
        self.turn_count += 1

        fallback = info.get("my_direction") or self.prev_direction or "N"
        if fallback not in MOVE_DELTAS:
            fallback = "N"

        try:
            rows, cols = self._parse_board(board)
            if rows == 0 or cols == 0:
                return fallback

            map_name = str(info.get("map_name", "")).lower()
            map_key = map_name[:-4] if map_name.endswith(".txt") else map_name
            map_profile = self._apply_map_name_hints(self._get_map_profile(board, rows, cols), map_key)
            map_class = map_profile["class"]
            my_direction = info.get("my_direction", fallback)
            if my_direction not in MOVE_DELTAS:
                my_direction = fallback

            opponents = [
                opp
                for opp in info.get("opponents", [])
                if opp.get("alive") and opp.get("pos") is not None
            ]
            active_emps = [
                emp
                for emp in info.get("active_emps", [])
                if emp.get("pos") is not None
            ]
            opponent_count = len(opponents)
            is_solo = self._is_solo_like_state(pos, opponents, rows, cols, map_profile)
            strategic_opponent_count = 0 if is_solo else opponent_count
            duel_mode = (not is_solo) and opponent_count == 1
            battle_mode = (not is_solo) and opponent_count >= 2
            if opponent_count >= 3 and self.turn_count >= 10:
                self.saw_real_battle = True

            weights = self._apply_profile_weight_overrides(
                self._get_weights(map_class, strategic_opponent_count),
                map_profile,
                strategic_opponent_count,
            )
            if duel_mode and self.saw_real_battle and self.turn_count >= 28:
                weights = dict(weights)
                if info.get("my_score", 0) >= 160:
                    weights["danger"] *= 1.5
                    weights["escape"] *= 1.5
                    weights["item"] *= 0.2
                    weights["density"] *= 0.2
                else:
                    weights["danger"] *= 1.25
                    weights["escape"] *= 1.25
                    weights["item"] *= 0.55
                    weights["density"] *= 0.55
            danger_map = self._build_danger_map(board, pos, opponents, rows, cols)
            open_now = self._count_open_cells(board, rows, cols)
            safe_threshold = max(10, int(open_now * weights["safe_ratio"]))
            emp_radius = info.get("emp_radius", 3)
            stun_duration = info.get("stun_duration", 5)

            candidates = self._get_candidates(
                board,
                pos,
                rows,
                cols,
                my_direction,
                info.get("phase_charges", 0),
            )

            if not candidates:
                return self._least_bad_move(board, pos, rows, cols, my_direction)

            solo_boost_setup = None
            if is_solo:
                for boost_candidate in candidates:
                    if boost_candidate["action"] != "+":
                        continue
                    boost_space_now = self._flood_fill_count(
                        board,
                        boost_candidate["landing"],
                        rows,
                        cols,
                        start_time=boost_candidate["steps"],
                        timed_policy="blocked",
                    )
                    boost_mobility = self._count_safe_neighbors(
                        board,
                        boost_candidate["landing"],
                        rows,
                        cols,
                        start_time=boost_candidate["steps"],
                    )
                    boost_branch = self._branch_profile(
                        board,
                        pos,
                        boost_candidate,
                        rows,
                        cols,
                        map_profile,
                    )
                    mid_pos = boost_candidate["path"][0] if boost_candidate["path"] else None
                    mid_value = 0
                    if mid_pos is not None:
                        mid_cell = self._get_cell(board, mid_pos[0], mid_pos[1])
                        if mid_cell in ITEM_VALUES:
                            mid_value = ITEM_VALUES[mid_cell]
                    solo_boost_setup = {
                        "move_dir": boost_candidate["move_dir"],
                        "mid_pos": mid_pos,
                        "mid_value": mid_value,
                        "viable": self._solo_boost_viable(
                            boost_candidate,
                            map_profile,
                            boost_branch,
                            boost_space_now,
                            safe_threshold,
                            boost_mobility,
                        ),
                    }
                    break

            best = None
            best_score = float("-inf")

            use_potential_timing = map_profile["timed_map"] or map_class == MAP_CORRIDOR
            max_item_depth = weights["item_depth"]
            depth_bonus = 10 if is_solo else 0
            hard_emp_mode = duel_mode or ((not is_solo) and (map_class == MAP_CORRIDOR or map_profile["timed_map"] or map_profile["center_race"]))
            imminent_emp_centers = [
                emp["pos"]
                for emp in active_emps
                if emp.get("timer") == 1 and emp.get("pos") is not None and emp.get("pos") != pos
            ] if hard_emp_mode else []
            imminent_emp_escape = False
            if imminent_emp_centers:
                for candidate in candidates:
                    landing = candidate["landing"]
                    if all(
                        max(abs(landing[0] - center[0]), abs(landing[1] - center[1])) > emp_radius
                        for center in imminent_emp_centers
                    ):
                        imminent_emp_escape = True
                        break

            branch_values = {}
            if is_solo and weights["solo_branch_depth"] > 0:
                branch_values = self._solo_branch_values(
                    board,
                    pos,
                    rows,
                    cols,
                    my_direction,
                    map_profile,
                    weights["solo_branch_depth"],
                )

            for candidate in candidates:
                landing = candidate["landing"]
                arrival_steps = candidate["steps"]

                space_now = self._flood_fill_count(
                    board,
                    landing,
                    rows,
                    cols,
                    start_time=arrival_steps,
                    timed_policy="blocked",
                )
                space_future = self._flood_fill_count(
                    board,
                    landing,
                    rows,
                    cols,
                    start_time=arrival_steps,
                    timed_policy="future" if use_potential_timing else "soon",
                )
                mobility = self._count_safe_neighbors(
                    board,
                    landing,
                    rows,
                    cols,
                    start_time=arrival_steps,
                )
                if is_solo:
                    item_eval = self._evaluate_items_unified(
                        board,
                        landing,
                        rows,
                        cols,
                        max_depth=max(
                            max_item_depth + 6 + depth_bonus,
                            weights["solo_cluster_depth"] + depth_bonus,
                        ),
                        start_time=arrival_steps,
                        map_profile=map_profile,
                        is_solo=True,
                        item_depth=max_item_depth + depth_bonus,
                        density_depth=max_item_depth + 6 + depth_bonus,
                        cluster_depth=weights["solo_cluster_depth"] + depth_bonus,
                    )
                    item_score = item_eval["item_score"]
                    item_distance = item_eval["item_distance"]
                    density_score = item_eval["density_score"]
                else:
                    item_score, item_distance = self._best_item_score(
                        board,
                        landing,
                        rows,
                        cols,
                        max_depth=max_item_depth,
                        start_time=arrival_steps,
                        map_profile=map_profile,
                    )
                    density_score = self._local_item_density(
                        board,
                        landing,
                        rows,
                        cols,
                        max_depth=max_item_depth + 6,
                        start_time=arrival_steps,
                        map_profile=map_profile,
                    )
                route_score = 0.0
                if is_solo:
                    route_score = self._solo_route_score(
                        board,
                        landing,
                        rows,
                        cols,
                        max_depth=weights["solo_route_depth"],
                        start_time=arrival_steps,
                        map_profile=map_profile,
                    )
                cluster_score = 0.0
                if is_solo:
                    cluster_score = item_eval["cluster_score"]
                branch = self._branch_profile(
                    board,
                    pos,
                    candidate,
                    rows,
                    cols,
                    map_profile,
                )
                escape_score = self._two_ply_escape_score(
                    board,
                    landing,
                    rows,
                    cols,
                    start_time=arrival_steps,
                    map_profile=map_profile,
                )
                danger = self._danger_penalty(candidate, danger_map, opponents)
                crowd_pressure = self._crowd_pressure(landing, opponents)
                axis_pressure = self._opening_axis_pressure(
                    pos,
                    candidate,
                    opponents,
                    map_profile,
                )
                edge_bias = self._edge_bias(landing, rows, cols)
                orbit_risk = self._orbit_risk(
                    landing,
                    rows,
                    cols,
                    self.turn_count,
                    len(opponents),
                )
                stun_runway = self._stun_runway(
                    board,
                    pos,
                    candidate,
                    rows,
                    cols,
                    stun_duration,
                    map_profile,
                )
                emp_threat = self._emp_threat(
                    board,
                    pos,
                    candidate,
                    rows,
                    cols,
                    active_emps,
                    emp_radius,
                    stun_duration,
                    map_profile,
                    stun_runway,
                )
                inside_imminent_emp = False
                if imminent_emp_centers:
                    inside_imminent_emp = any(
                        max(abs(landing[0] - center[0]), abs(landing[1] - center[1])) <= emp_radius
                        for center in imminent_emp_centers
                    )
                if imminent_emp_escape:
                    for center in imminent_emp_centers:
                        if max(abs(landing[0] - center[0]), abs(landing[1] - center[1])) <= emp_radius:
                            emp_threat += 40.0
                            if stun_runway < stun_duration:
                                emp_threat += (stun_duration - stun_runway + 1) * 6.0
                            break

                if duel_mode:
                    territory = self._voronoi_balance(
                        board,
                        landing,
                        opponents[0]["pos"],
                        rows,
                        cols,
                        map_profile,
                        my_start_time=arrival_steps,
                    )
                else:
                    territory = self._territory_score(
                        board,
                        landing,
                        rows,
                        cols,
                        opponents,
                        map_profile,
                        my_start_time=arrival_steps,
                    )
                center_bonus = self._center_bonus(landing, rows, cols)
                direct_reward = candidate["direct_reward"]
                straight_bonus = 1 if candidate["move_dir"] == my_direction else 0
                crisis_penalty = 0
                if space_now < safe_threshold:
                    crisis_penalty = (safe_threshold - space_now) * weights["trap"]

                ability_bonus = 0
                if candidate["action"] == "+":
                    ability_bonus += 6
                    if candidate["move_dir"] == my_direction:
                        ability_bonus += 2
                    if is_solo and danger < 6 and space_now >= safe_threshold:
                        ability_bonus += 8
                    if branch["skipped_forks"] > 0:
                        ability_bonus -= 12 * branch["skipped_forks"]
                    if branch["upcoming_turn"] > 0:
                        ability_bonus -= 10
                    if escape_score < 4:
                        ability_bonus -= 8
                elif candidate["action"] == "P":
                    if is_solo:
                        breakthrough_gain = max(0, space_future - space_now)
                        breakthrough_threshold = max(12, safe_threshold // 2)
                        if candidate["phase_needed"]:
                            ability_bonus += 8
                            if direct_reward >= 50:
                                ability_bonus += 28
                            elif direct_reward >= 20:
                                ability_bonus += 6
                            if breakthrough_gain >= breakthrough_threshold:
                                ability_bonus += min(32.0, breakthrough_gain * 0.8)
                                if map_class == MAP_CORRIDOR:
                                    ability_bonus += 8
                            elif breakthrough_gain >= max(6, breakthrough_threshold // 2):
                                ability_bonus += 8
                            else:
                                ability_bonus -= 6
                            if mobility <= 1 and direct_reward < 50 and breakthrough_gain < breakthrough_threshold:
                                ability_bonus -= 10
                        else:
                            ability_bonus -= 12
                    else:
                        if candidate["phase_needed"]:
                            ability_bonus += 12
                        else:
                            ability_bonus -= 10

                score = (
                    space_now * weights["space"]
                    + space_future * weights["future_space"]
                    + mobility * weights["mobility"]
                    + item_score * weights["item"]
                    + density_score * weights["density"]
                    + route_score * weights["solo_route"]
                    + cluster_score * weights["solo_cluster"]
                    + territory * weights["territory"]
                    + center_bonus * weights["center"]
                    + direct_reward * weights["direct"]
                    + straight_bonus * weights["straight"]
                    + escape_score * weights["escape"]
                    + ability_bonus
                    - danger * weights["danger"]
                    - crowd_pressure * weights["crowd"]
                    - emp_threat * weights["emp_threat"]
                    - crisis_penalty
                )

                if is_solo and map_class == MAP_BRANCHING:
                    score += branch["landing_fork"] * 8
                    score += branch["ahead_value"] * 0.65
                    score += branch["upcoming_fork"] * 3
                    score -= branch["skipped_forks"] * 8
                elif is_solo and map_class == MAP_CORRIDOR:
                    score += branch["ahead_value"] * 0.35
                    score += min(branch["straight_run"], 6) * 1.5
                    if candidate["action"] in ["+", "P"]:
                        score -= branch["upcoming_turn"] * 12
                    if candidate["move_dir"] == self.prev_direction:
                        score += 2
                elif is_solo and map_profile["dense_items"]:
                    score += density_score * 0.35

                if map_profile["open_battle_opening"] and battle_mode and self.turn_count <= 18:
                    score += edge_bias * 6
                    score -= axis_pressure * 3.0
                    if candidate["action"] in ["+", "P"]:
                        score -= 14

                if map_class == MAP_OPEN and battle_mode and self.turn_count <= 10:
                    score -= axis_pressure * 1.5

                if map_profile["orbit_like"]:
                    score -= orbit_risk
                    if self.turn_count <= 20 and candidate["action"] in ["+", "P"]:
                        score -= 10

                if battle_mode and crowd_pressure > 0:
                    score -= max(0.0, 5 - escape_score) * 3.0

                if inside_imminent_emp and not imminent_emp_escape:
                    score += stun_runway * 28.0
                    score -= max(0, stun_duration - stun_runway) * 20.0
                    if candidate["action"] in ["+", "P"]:
                        score -= 8.0

                if is_solo:
                    if candidate["action"] == "+":
                        solo_boost_viable = self._solo_boost_viable(
                            candidate,
                            map_profile,
                            branch,
                            space_now,
                            safe_threshold,
                            mobility,
                        )
                        if solo_boost_viable:
                            score += weights["solo_boost"]
                            score += min(branch["straight_run"], 3) * 3.0
                            if map_class == MAP_CORRIDOR and branch["straight_run"] >= 2:
                                score += 8.0
                            if map_profile["dense_items"] and space_now >= safe_threshold * 2:
                                score += 6.0
                        else:
                            score -= 18.0

                        if solo_boost_setup and solo_boost_setup["mid_value"] > 0:
                            score -= solo_boost_setup["mid_value"] * weights["solo_skip_item"]
                            if solo_boost_setup["mid_value"] >= 50:
                                score -= 18.0
                    elif (
                        solo_boost_setup
                        and solo_boost_setup["viable"]
                        and solo_boost_setup["mid_value"] > 0
                        and candidate["action"] in MOVE_DELTAS
                        and candidate["move_dir"] == solo_boost_setup["move_dir"]
                        and landing == solo_boost_setup["mid_pos"]
                    ):
                        score += solo_boost_setup["mid_value"] * 1.05 + 12.0

                    if branch_values:
                        branch_average = sum(branch_values.values()) / len(branch_values)
                        branch_peak = max(branch_values.values())
                        branch_delta = branch_values.get(candidate["move_dir"], branch_average) - branch_average
                        score += branch_delta * weights["solo_branch"]
                        if map_class == MAP_BRANCHING and candidate["move_dir"] in branch_values:
                            if branch_values[candidate["move_dir"]] == branch_peak and branch_peak - branch_average >= 3.0:
                                score += 5.0
                            elif branch_peak - branch_values[candidate["move_dir"]] >= 5.0:
                                score -= 4.0

                # Prefer simpler safe moves when scores are close.
                if candidate["action"] in MOVE_DELTAS:
                    score += 1.5

                if item_distance is not None and space_now < safe_threshold:
                    score -= max(0, safe_threshold - space_now) * 0.5

                candidate["score"] = score
                candidate["space_now"] = space_now
                candidate["space_future"] = space_future
                candidate["mobility"] = mobility
                candidate["danger"] = danger
                candidate["item_score"] = item_score
                candidate["density_score"] = density_score
                candidate["route_score"] = route_score
                candidate["cluster_score"] = cluster_score
                candidate["territory"] = territory
                candidate["escape_score"] = escape_score
                candidate["crowd_pressure"] = crowd_pressure
                candidate["axis_pressure"] = axis_pressure
                candidate["stun_runway"] = stun_runway
                candidate["emp_threat"] = emp_threat

                if score > best_score:
                    best_score = score
                    best = candidate

            if best is None:
                return fallback

            action = best["action"]
            if self._should_emp(board, pos, best, opponents, info, danger_map, map_profile):
                action = "X" + action

            self.prev_pos = pos
            if best["move_dir"] in MOVE_DELTAS:
                self.prev_direction = best["move_dir"]
            else:
                self.prev_direction = my_direction
            self.known_trail_cells.update(self._committed_trail_cells(pos, best))

            return action
        except Exception:
            return fallback

    def _parse_board(self, board):
        rows = len(board)
        cols = len(board[0]) if rows else 0
        return rows, cols

    def _next_pos(self, pos, move):
        dx, dy = MOVE_DELTAS[move]
        return pos[0] + dx, pos[1] + dy

    def _in_bounds(self, x, y, rows, cols):
        return 0 <= x < cols and 0 <= y < rows

    def _get_cell(self, board, x, y):
        return board[y][x]

    def _is_head(self, cell):
        return isinstance(cell, str) and len(cell) >= 2 and cell[0] == "p" and cell[1:].isdigit()

    def _is_trail(self, cell):
        return isinstance(cell, str) and cell.startswith("t")

    def _is_walkable_now(self, board, x, y, rows, cols, allow_heads=True):
        if not self._in_bounds(x, y, rows, cols):
            return False
        cell = self._get_cell(board, x, y)
        if cell == "#":
            return False
        if isinstance(cell, int) and cell > 0:
            return False
        if self._is_trail(cell):
            return False
        if self._is_head(cell):
            if (x, y) in self.known_trail_cells:
                return False
            return allow_heads
        return True

    def _is_walkable_future(self, board, x, y, rows, cols, arrival_time, timed_policy):
        if not self._in_bounds(x, y, rows, cols):
            return False
        cell = self._get_cell(board, x, y)
        if cell == "#":
            return False
        if self._is_trail(cell):
            return False
        if self._is_head(cell):
            return False
        if isinstance(cell, int) and cell > 0:
            open_time = cell * 10
            if timed_policy == "blocked":
                return False
            if timed_policy == "soon":
                return arrival_time >= open_time
            if timed_policy == "future":
                return arrival_time >= max(6, open_time - 5)
            return False
        return True

    def _count_open_cells(self, board, rows, cols):
        count = 0
        for y in range(rows):
            for x in range(cols):
                cell = board[y][x]
                if cell in OPEN_CELLS or self._is_head(cell):
                    count += 1
        return count

    def _get_map_profile(self, board, rows, cols):
        if self.map_profile is None:
            self.map_profile = self._classify_map(board, rows, cols)
        return self.map_profile

    def _is_solo_like_state(self, pos, opponents, rows, cols, map_profile):
        if not opponents:
            return True
        if len(opponents) < 3:
            return False

        nearest_opponent = min(self._manhattan(pos, opp["pos"]) for opp in opponents)
        distance_threshold = max(18, min(rows, cols) - 3)

        if nearest_opponent < distance_threshold:
            return False
        if map_profile["timed_map"] or map_profile["center_race"]:
            return False

        return map_profile["wall_density"] >= 0.45 or map_profile["item_density"] >= 0.02

    def _apply_map_name_hints(self, map_profile, map_key):
        if not map_key:
            return map_profile

        hinted = dict(map_profile)

        if map_key.startswith("s_path"):
            hinted["class"] = MAP_CORRIDOR
            hinted["dense_items"] = True
        elif map_key.startswith("s_choice"):
            hinted["class"] = MAP_BRANCHING
            hinted["dense_items"] = True
        elif map_key.startswith("s_floodfill"):
            hinted["class"] = MAP_BRANCHING
            hinted["dense_items"] = True
        elif map_key == "maze":
            hinted["class"] = MAP_CORRIDOR
            hinted["open_battle_opening"] = False
        elif map_key == "gate":
            hinted["class"] = MAP_CORRIDOR
            hinted["timed_map"] = True
            hinted["open_battle_opening"] = False
        elif map_key == "treasure":
            hinted["class"] = MAP_OPEN
            hinted["center_race"] = True
            hinted["open_battle_opening"] = False
        elif map_key == "orbit":
            hinted["class"] = MAP_OPEN
            hinted["orbit_like"] = True
            hinted["open_battle_opening"] = False
        elif map_key in {"arena", "cube"}:
            hinted["class"] = MAP_OPEN
            hinted["open_battle_opening"] = True

        return hinted

    def _classify_map(self, board, rows, cols):
        total_cells = max(1, rows * cols)
        open_cells = 0
        wall_cells = 0
        timed_cells = 0
        item_cells = 0
        center_item_value = 0
        max_local_item_value = 0
        corridorish = 0
        branching = 0
        center_x = (cols - 1) / 2.0
        center_y = (rows - 1) / 2.0

        for y in range(rows):
            for x in range(cols):
                cell = board[y][x]
                if cell == "#":
                    wall_cells += 1
                    continue
                if isinstance(cell, int) and cell > 0:
                    timed_cells += 1
                    continue

                open_cells += 1
                local_item_value = ITEM_VALUES.get(cell, 0)
                if cell in ITEM_VALUES:
                    item_cells += 1
                    if abs(x - center_x) <= 2 and abs(y - center_y) <= 2:
                        center_item_value += ITEM_VALUES[cell]

                open_neighbors = 0
                for dx, dy in MOVE_DELTAS.values():
                    nx, ny = x + dx, y + dy
                    if not self._in_bounds(nx, ny, rows, cols):
                        continue
                    neighbor = board[ny][nx]
                    if neighbor == "#" or (isinstance(neighbor, int) and neighbor > 0):
                        continue
                    open_neighbors += 1
                    local_item_value += ITEM_VALUES.get(neighbor, 0)

                max_local_item_value = max(max_local_item_value, local_item_value)
                if open_neighbors <= 2:
                    corridorish += 1
                elif open_neighbors >= 3:
                    branching += 1

        wall_density = wall_cells / total_cells
        timed_density = timed_cells / total_cells
        item_density = item_cells / max(1, open_cells)
        corridor_ratio = corridorish / max(1, open_cells)
        branch_ratio = branching / max(1, open_cells)

        if item_density >= 0.05 and branch_ratio >= 0.15:
            map_class = MAP_BRANCHING
        elif corridor_ratio >= 0.52 or (wall_density >= 0.50 and branch_ratio < 0.45):
            map_class = MAP_CORRIDOR
        elif item_density >= 0.025 and branch_ratio >= 0.75:
            map_class = MAP_BRANCHING
        else:
            map_class = MAP_OPEN

        timed_map = timed_density >= 0.02
        open_battle_opening = map_class == MAP_OPEN and wall_density <= 0.16 and item_density <= 0.01 and timed_density < 0.08
        orbit_like = map_class == MAP_OPEN and timed_density >= 0.08 and wall_density <= 0.20
        center_race = map_class == MAP_OPEN and center_item_value >= 50 and item_density <= 0.02
        dense_items = item_density >= 0.03 or max_local_item_value >= 90

        return {
            "class": map_class,
            "wall_density": wall_density,
            "timed_density": timed_density,
            "item_density": item_density,
            "corridor_ratio": corridor_ratio,
            "branch_ratio": branch_ratio,
            "timed_map": timed_map,
            "open_battle_opening": open_battle_opening,
            "orbit_like": orbit_like,
            "center_race": center_race,
            "dense_items": dense_items,
        }

    def _get_safe_moves(self, board, pos, rows, cols):
        safe = []
        x, y = pos
        for move, (dx, dy) in MOVE_DELTAS.items():
            nx, ny = x + dx, y + dy
            if self._is_walkable_now(board, nx, ny, rows, cols, allow_heads=True):
                safe.append(move)
        return safe

    def _get_direction(self, pos, next_pos):
        dx = next_pos[0] - pos[0]
        dy = next_pos[1] - pos[1]
        for move, delta in MOVE_DELTAS.items():
            if delta == (dx, dy):
                return move
        return None

    def _committed_trail_cells(self, pos, candidate):
        path = candidate.get("path", [])
        if not path:
            return {pos}
        return {pos, *path[:-1]}

    def _simulate_action(self, board, pos, action, current_direction, rows, cols):
        direction = action if action in MOVE_DELTAS else current_direction
        if direction not in MOVE_DELTAS:
            return None

        steps = []
        if action in MOVE_DELTAS:
            steps = [("normal", direction)]
        elif action == "+":
            steps = [("normal", direction), ("normal", direction)]
        elif action == "P":
            steps = [("phase", direction), ("normal", direction)]
        else:
            return None

        current = pos
        direct_reward = 0
        phase_needed = False
        traversed = []

        for step_type, step_dir in steps:
            dx, dy = MOVE_DELTAS[step_dir]
            nx, ny = current[0] + dx, current[1] + dy
            if not self._in_bounds(nx, ny, rows, cols):
                return None

            cell = self._get_cell(board, nx, ny)
            if step_type != "phase":
                if cell == "#":
                    return None
                if isinstance(cell, int) and cell > 0:
                    return None
                if self._is_trail(cell):
                    return None
                if self._is_head(cell) and (nx, ny) in self.known_trail_cells:
                    return None
            else:
                if cell == "#" or (isinstance(cell, int) and cell > 0) or self._is_trail(cell):
                    phase_needed = True

            if cell in ITEM_VALUES:
                direct_reward += ITEM_VALUES[cell]

            traversed.append((nx, ny))
            current = (nx, ny)

        if action == "P" and not phase_needed:
            phase_needed = False

        return {
            "action": action,
            "landing": current,
            "steps": len(steps),
            "move_dir": direction,
            "direct_reward": direct_reward,
            "path": traversed,
            "phase_needed": phase_needed,
        }

    def _get_candidates(self, board, pos, rows, cols, my_direction, phase_charges):
        candidates = []

        for move in ("N", "S", "E", "W"):
            simulated = self._simulate_action(board, pos, move, my_direction, rows, cols)
            if simulated is not None:
                candidates.append(simulated)

        boosted = self._simulate_action(board, pos, "+", my_direction, rows, cols)
        if boosted is not None:
            candidates.append(boosted)

        if phase_charges > 0:
            phased = self._simulate_action(board, pos, "P", my_direction, rows, cols)
            if phased is not None:
                candidates.append(phased)

        return candidates

    def _count_safe_neighbors(self, board, start, rows, cols, start_time):
        total = 0
        x, y = start
        for dx, dy in MOVE_DELTAS.values():
            nx, ny = x + dx, y + dy
            if self._is_walkable_future(
                board,
                nx,
                ny,
                rows,
                cols,
                start_time + 1,
                "soon",
            ):
                total += 1
        return total

    def _future_neighbors(self, board, pos, rows, cols, arrival_time, timed_policy, exclude=None):
        neighbors = []
        x, y = pos
        for move, (dx, dy) in MOVE_DELTAS.items():
            nx, ny = x + dx, y + dy
            next_pos = (nx, ny)
            if exclude is not None and next_pos == exclude:
                continue
            if self._is_walkable_future(
                board,
                nx,
                ny,
                rows,
                cols,
                arrival_time + 1,
                timed_policy,
            ):
                neighbors.append((move, next_pos))
        return neighbors

    def _flood_fill_count(self, board, start, rows, cols, start_time, timed_policy):
        queue = deque([(start[0], start[1], start_time)])
        seen = {start}
        count = 0

        while queue:
            x, y, steps = queue.popleft()
            count += 1
            for dx, dy in MOVE_DELTAS.values():
                nx, ny = x + dx, y + dy
                next_pos = (nx, ny)
                if next_pos in seen:
                    continue
                if not self._is_walkable_future(
                    board,
                    nx,
                    ny,
                    rows,
                    cols,
                    steps + 1,
                    timed_policy,
                ):
                    continue
                seen.add(next_pos)
                queue.append((nx, ny, steps + 1))

        return count

    def _evaluate_items_unified(
        self,
        board,
        start,
        rows,
        cols,
        max_depth,
        start_time,
        map_profile,
        is_solo,
        item_depth=None,
        density_depth=None,
        cluster_depth=None,
    ):
        if max_depth <= 0:
            return {
                "item_score": 0.0,
                "item_distance": None,
                "density_score": 0.0,
                "cluster_score": 0.0,
            }

        item_depth = max_depth if item_depth is None else max(0, min(item_depth, max_depth))
        density_depth = max_depth if density_depth is None else max(0, min(density_depth, max_depth))
        if is_solo:
            cluster_depth = max_depth if cluster_depth is None else max(0, min(cluster_depth, max_depth))
        else:
            cluster_depth = 0

        timed_policy = "future" if (map_profile["timed_map"] or map_profile["class"] == MAP_CORRIDOR) else "soon"
        queue = deque([(start, 0)])
        seen = {start}
        best_score = 0.0
        aggregate_score = 0.0
        best_distance = None
        items_seen = 0
        density = 0.0
        cluster = 0.0

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue

            for _, next_pos in self._future_neighbors(
                board,
                current,
                rows,
                cols,
                start_time + depth,
                timed_policy,
            ):
                if next_pos in seen:
                    continue

                seen.add(next_pos)
                step = depth + 1
                cell = self._get_cell(board, next_pos[0], next_pos[1])

                if item_depth > 0 and step <= item_depth and cell in ITEM_VALUES:
                    value = ITEM_VALUES[cell]
                    item_value = value / step
                    if cell == "D":
                        item_value += 2
                    if item_value > best_score:
                        best_score = item_value
                        best_distance = step
                    if items_seen < 8:
                        aggregate_score += value / (step ** 1.2)
                    items_seen += 1

                if density_depth > 0 and step <= density_depth and cell in ITEM_VALUES:
                    density += ITEM_VALUES[cell] / (step ** 1.15)

                if cluster_depth > 0 and step <= cluster_depth:
                    if cell in ITEM_VALUES:
                        local_cluster = 0.0
                        for _, around in self._future_neighbors(
                            board,
                            next_pos,
                            rows,
                            cols,
                            start_time + step,
                            timed_policy,
                        ):
                            around_cell = self._get_cell(board, around[0], around[1])
                            if around_cell in ITEM_VALUES:
                                local_cluster += ITEM_VALUES[around_cell] * 0.2
                        cluster += (ITEM_VALUES[cell] + local_cluster) / (step ** 0.85)

                    onward = self._future_neighbors(
                        board,
                        next_pos,
                        rows,
                        cols,
                        start_time + step,
                        timed_policy,
                        exclude=current,
                    )
                    cluster += max(0, len(onward) - 1) * (0.7 / step)
                    if len(onward) == 0 and cell not in ITEM_VALUES:
                        cluster -= 1.0 / step

                    if map_profile["dense_items"] and step <= max(1, cluster_depth // 2):
                        cluster += min(3, len(onward)) * (0.45 / step)
                    if map_profile["class"] == MAP_BRANCHING and len(onward) >= 2:
                        cluster += 1.4 / step

                if step < max_depth:
                    queue.append((next_pos, step))

        return {
            "item_score": best_score + aggregate_score * 0.4,
            "item_distance": best_distance,
            "density_score": density,
            "cluster_score": cluster,
        }

    def _best_item_score(self, board, start, rows, cols, max_depth, start_time, map_profile):
        queue = deque([(start[0], start[1], 0)])
        seen = {start}
        best_score = 0.0
        aggregate_score = 0.0
        best_distance = None
        items_seen = 0

        timed_policy = "future" if map_profile["timed_map"] else "soon"

        while queue:
            x, y, depth = queue.popleft()
            if depth >= max_depth:
                continue

            for _, (dx, dy) in MOVE_DELTAS.items():
                nx, ny = x + dx, y + dy
                next_pos = (nx, ny)
                if next_pos in seen:
                    continue
                arrival_time = start_time + depth + 1
                if not self._is_walkable_future(
                    board,
                    nx,
                    ny,
                    rows,
                    cols,
                    arrival_time,
                    timed_policy,
                ):
                    continue

                seen.add(next_pos)
                cell = self._get_cell(board, nx, ny)
                if cell in ITEM_VALUES:
                    value = ITEM_VALUES[cell]
                    score = value / (depth + 1)
                    if cell == "D":
                        score += 2
                    if score > best_score:
                        best_score = score
                        best_distance = depth + 1
                    if items_seen < 8:
                        aggregate_score += value / ((depth + 1) ** 1.2)
                    items_seen += 1
                queue.append((nx, ny, depth + 1))

        return best_score + aggregate_score * 0.4, best_distance

    def _local_item_density(self, board, start, rows, cols, max_depth, start_time, map_profile):
        queue = deque([(start[0], start[1], 0)])
        seen = {start}
        density = 0.0
        timed_policy = "future" if (map_profile["timed_map"] or map_profile["class"] == MAP_CORRIDOR) else "soon"

        while queue:
            x, y, depth = queue.popleft()
            if depth >= max_depth:
                continue

            for dx, dy in MOVE_DELTAS.values():
                nx, ny = x + dx, y + dy
                next_pos = (nx, ny)
                if next_pos in seen:
                    continue
                arrival_time = start_time + depth + 1
                if not self._is_walkable_future(
                    board,
                    nx,
                    ny,
                    rows,
                    cols,
                    arrival_time,
                    timed_policy,
                ):
                    continue

                seen.add(next_pos)
                cell = self._get_cell(board, nx, ny)
                if cell in ITEM_VALUES:
                    density += ITEM_VALUES[cell] / ((depth + 1) ** 1.15)
                queue.append((nx, ny, depth + 1))

        return density

    def _solo_route_score(self, board, start, rows, cols, max_depth, start_time, map_profile):
        if max_depth <= 0:
            return 0.0

        timed_policy = "future" if (map_profile["timed_map"] or map_profile["class"] == MAP_CORRIDOR) else "soon"
        beam_width = 12 if map_profile["class"] == MAP_BRANCHING else 10
        states = [
            {
                "pos": start,
                "score": 0.0,
                "seen": {start},
                "prev": None,
            }
        ]
        best_score = 0.0

        for depth in range(max_depth):
            next_states = []
            step_index = depth + 1

            for state in states:
                x, y = state["pos"]
                for dx, dy in MOVE_DELTAS.values():
                    nx, ny = x + dx, y + dy
                    next_pos = (nx, ny)
                    if next_pos in state["seen"]:
                        continue
                    if not self._is_walkable_future(
                        board,
                        nx,
                        ny,
                        rows,
                        cols,
                        start_time + step_index,
                        timed_policy,
                    ):
                        continue

                    cell = self._get_cell(board, nx, ny)
                    step_score = 0.0
                    if cell in ITEM_VALUES:
                        step_score += ITEM_VALUES[cell] / (step_index ** 0.7)
                        if cell == "D":
                            step_score += 4.0

                    onward = self._future_neighbors(
                        board,
                        next_pos,
                        rows,
                        cols,
                        start_time + step_index,
                        timed_policy,
                        exclude=state["pos"],
                    )
                    step_score += max(0, len(onward) - 1) * 0.8
                    if len(onward) == 0 and step_index < max_depth:
                        step_score -= 3.0

                    if map_profile["class"] == MAP_BRANCHING and len(onward) >= 2:
                        step_score += 2.5
                    if map_profile["dense_items"] and len(onward) >= 2:
                        step_score += 1.2

                    new_score = state["score"] + step_score
                    next_states.append(
                        {
                            "pos": next_pos,
                            "score": new_score,
                            "seen": state["seen"] | {next_pos},
                            "prev": state["pos"],
                        }
                    )
                    if new_score > best_score:
                        best_score = new_score

            if not next_states:
                break

            next_states.sort(key=lambda s: s["score"], reverse=True)
            states = next_states[:beam_width]

        return best_score

    def _solo_cluster_score(self, board, start, rows, cols, max_depth, start_time, map_profile):
        if max_depth <= 0:
            return 0.0
        result = self._evaluate_items_unified(
            board,
            start,
            rows,
            cols,
            max_depth=max_depth,
            start_time=start_time,
            map_profile=map_profile,
            is_solo=True,
            item_depth=0,
            density_depth=0,
            cluster_depth=max_depth,
        )
        return result["cluster_score"]

    def _solo_branch_values(self, board, pos, rows, cols, my_direction, map_profile, max_depth):
        if max_depth <= 0:
            return {}

        values = {}
        for move in MOVE_DELTAS:
            candidate = self._simulate_action(board, pos, move, my_direction, rows, cols)
            if candidate is None:
                continue
            landing = candidate["landing"]
            branch = self._branch_profile(board, pos, candidate, rows, cols, map_profile)
            branch_space = self._flood_fill_count(
                board,
                landing,
                rows,
                cols,
                start_time=candidate["steps"],
                timed_policy="blocked",
            )
            cluster = self._solo_cluster_score(
                board,
                landing,
                rows,
                cols,
                max_depth=max_depth,
                start_time=candidate["steps"],
                map_profile=map_profile,
            )
            values[move] = (
                cluster
                + branch["ahead_value"] * 0.8
                + branch["landing_fork"] * 7.0
                + branch["upcoming_fork"] * 2.5
                - branch["skipped_forks"] * 3.5
                + min(branch["straight_run"], 4) * 0.9
                + min(40, branch_space) * 0.12
            )

        return values

    def _branch_profile(self, board, origin, candidate, rows, cols, map_profile):
        timed_policy = "future" if (map_profile["timed_map"] or map_profile["class"] == MAP_CORRIDOR) else "soon"
        path = candidate["path"]
        prev = origin
        skipped_forks = 0
        landing_fork = 0
        ahead_value = 0.0

        for idx, cell in enumerate(path):
            if self._get_cell(board, cell[0], cell[1]) in ITEM_VALUES:
                ahead_value += ITEM_VALUES[self._get_cell(board, cell[0], cell[1])]
            options = self._future_neighbors(
                board,
                cell,
                rows,
                cols,
                idx + 1,
                timed_policy,
                exclude=prev,
            )
            if len(options) >= 2:
                if idx == len(path) - 1:
                    landing_fork = 1
                else:
                    skipped_forks += 1
            prev = cell

        straight_run = 0
        upcoming_turn = 0
        upcoming_fork = 0
        current = candidate["landing"]
        prev = path[-2] if len(path) >= 2 else origin
        move_dir = candidate["move_dir"]
        dx, dy = MOVE_DELTAS[move_dir]

        for step in range(1, 5):
            nx, ny = current[0] + dx, current[1] + dy
            next_pos = (nx, ny)
            if not self._is_walkable_future(
                board,
                nx,
                ny,
                rows,
                cols,
                candidate["steps"] + step,
                timed_policy,
            ):
                break

            next_pos = (nx, ny)
            straight_run += 1
            cell_value = self._get_cell(board, nx, ny)
            if cell_value in ITEM_VALUES:
                ahead_value += ITEM_VALUES[cell_value] / (step + 1)

            options = self._future_neighbors(
                board,
                next_pos,
                rows,
                cols,
                candidate["steps"] + step,
                timed_policy,
                exclude=current,
            )
            forward = (nx + dx, ny + dy)
            forward_is_open = self._is_walkable_future(
                board,
                forward[0],
                forward[1],
                rows,
                cols,
                candidate["steps"] + step + 1,
                timed_policy,
            )

            if len(options) >= 2:
                upcoming_fork = 1
                break
            if options and not forward_is_open:
                upcoming_turn = 1
                break

            prev = current
            current = next_pos

        return {
            "skipped_forks": skipped_forks,
            "landing_fork": landing_fork,
            "ahead_value": ahead_value,
            "straight_run": straight_run,
            "upcoming_turn": upcoming_turn,
            "upcoming_fork": upcoming_fork,
        }

    def _two_ply_escape_score(self, board, start, rows, cols, start_time, map_profile):
        timed_policy = "future" if (map_profile["timed_map"] or map_profile["class"] == MAP_CORRIDOR) else "soon"
        queue = deque([(start, 0)])
        seen = {(start, 0)}
        endpoints = set()
        first_step_cells = set()

        while queue:
            pos, depth = queue.popleft()
            if depth == 2:
                endpoints.add(pos)
                continue

            for _, next_pos in self._future_neighbors(
                board,
                pos,
                rows,
                cols,
                start_time + depth,
                timed_policy,
            ):
                state = (next_pos, depth + 1)
                if state in seen:
                    continue
                seen.add(state)
                if depth == 0:
                    first_step_cells.add(next_pos)
                if depth + 1 >= 1:
                    endpoints.add(next_pos)
                queue.append((next_pos, depth + 1))

        return len(endpoints) + 0.5 * len(first_step_cells)

    def _path_to_direction(self, pos, path):
        if len(path) < 2:
            return None
        return self._get_direction(pos, path[1])

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _find_cells(self, board, rows, cols, target_value):
        found = []
        for y in range(rows):
            for x in range(cols):
                if board[y][x] == target_value:
                    found.append((x, y))
        return found

    def _edge_bias(self, pos, rows, cols):
        edge_distance = min(pos[0], pos[1], cols - 1 - pos[0], rows - 1 - pos[1])
        return max(0.0, 6.0 - edge_distance)

    def _orbit_risk(self, pos, rows, cols, turn_count, opponent_count):
        if rows == 0 or cols == 0:
            return 0.0
        cx = (cols - 1) / 2.0
        cy = (rows - 1) / 2.0
        cheb = max(abs(pos[0] - cx), abs(pos[1] - cy))
        risk = 0.0

        if turn_count <= 20 and cheb < 12:
            risk += (12 - cheb) * (2.0 if opponent_count <= 1 else 3.5)
        elif turn_count <= 35 and cheb < 10:
            risk += (10 - cheb) * (1.5 if opponent_count <= 1 else 2.5)

        return risk

    def _stun_runway(self, board, origin, candidate, rows, cols, stun_duration, map_profile):
        direction = candidate["move_dir"]
        if direction not in MOVE_DELTAS:
            return 0

        timed_policy = "soon" if (map_profile["timed_map"] or map_profile["class"] == MAP_CORRIDOR or map_profile["orbit_like"]) else "blocked"
        blocked = self._committed_trail_cells(origin, candidate)
        current = candidate["landing"]
        runway = 0

        for step in range(1, max(1, stun_duration) + 1):
            nx = current[0] + MOVE_DELTAS[direction][0]
            ny = current[1] + MOVE_DELTAS[direction][1]
            if (nx, ny) in blocked:
                break
            if not self._is_walkable_future(
                board,
                nx,
                ny,
                rows,
                cols,
                candidate["steps"] + step,
                timed_policy,
            ):
                break
            runway += 1
            current = (nx, ny)

        return runway

    def _solo_boost_viable(self, candidate, map_profile, branch, space_now, safe_threshold, mobility):
        if candidate["action"] != "+":
            return False
        if mobility < 2:
            return False
        if space_now < max(8, safe_threshold):
            return False
        if branch["skipped_forks"] > 0:
            return False
        if map_profile["class"] == MAP_BRANCHING and (branch["landing_fork"] > 0 or branch["upcoming_fork"] > 0):
            return False
        if branch["upcoming_turn"] > 0:
            if map_profile["class"] == MAP_CORRIDOR:
                return branch["straight_run"] >= 2
            return mobility >= 3 and space_now >= max(12, safe_threshold + 2)
        if map_profile["class"] == MAP_CORRIDOR:
            return branch["straight_run"] >= 1
        return True

    def _emp_threat(
        self,
        board,
        origin,
        candidate,
        rows,
        cols,
        active_emps,
        radius,
        stun_duration,
        map_profile,
        stun_runway,
    ):
        if not active_emps:
            return 0.0

        landing = candidate["landing"]
        threat = 0.0

        for emp in active_emps:
            center = emp.get("pos")
            timer = emp.get("timer", 99)
            if center is None or center == origin:
                continue
            if timer > 3:
                continue

            cheb = max(abs(landing[0] - center[0]), abs(landing[1] - center[1]))
            center_shift = max(0, min(2, timer - 1))
            effective_radius = radius + center_shift

            if cheb > effective_radius + 1:
                continue

            if timer == 1:
                if cheb <= radius:
                    threat += 32.0 + max(0, radius - cheb) * 10.0
                elif cheb == radius + 1:
                    threat += 6.0
                else:
                    threat += 2.0
                if cheb <= radius:
                    if stun_runway < stun_duration:
                        threat += (stun_duration - stun_runway + 1) * 14.0
                    if candidate["action"] in ["+", "P"]:
                        threat += 8.0
            elif timer == 2:
                if cheb <= radius:
                    threat += 12.0 + max(0, radius - cheb) * 4.0
                elif cheb == radius + 1:
                    threat += 5.0
                else:
                    threat += 1.5
                if cheb <= radius + 1 and stun_runway < max(2, stun_duration - 1):
                    threat += (max(2, stun_duration - 1) - stun_runway + 1) * 5.0
            else:
                if cheb <= radius:
                    threat += 4.0 + max(0, radius - cheb) * 2.0
                else:
                    threat += 1.0

            if (map_profile["class"] == MAP_CORRIDOR or map_profile["center_race"]) and cheb <= effective_radius:
                threat *= 1.15

        return threat

    def _opening_axis_pressure(self, pos, candidate, opponents, map_profile):
        if self.turn_count > 16 or map_profile["class"] != MAP_OPEN:
            return 0.0

        landing = candidate["landing"]
        penalty = 0.0

        for opp in opponents:
            ox, oy = opp["pos"]
            dx0 = pos[0] - ox
            dy0 = pos[1] - oy
            dx1 = landing[0] - ox
            dy1 = landing[1] - oy
            opp_dir = opp.get("direction")

            if abs(dx0) <= 2 and abs(dy0) >= 8:
                approaching = (dy0 < 0 and opp_dir == "N") or (dy0 > 0 and opp_dir == "S")
                if approaching:
                    if abs(dx1) <= abs(dx0):
                        penalty += 1.5
                    if abs(dy1) < abs(dy0):
                        penalty += 2.0
                    if candidate["move_dir"] in ["N", "S"]:
                        penalty += 1.5

            if abs(dy0) <= 2 and abs(dx0) >= 8:
                approaching = (dx0 < 0 and opp_dir == "W") or (dx0 > 0 and opp_dir == "E")
                if approaching:
                    if abs(dy1) <= abs(dy0):
                        penalty += 1.5
                    if abs(dx1) < abs(dx0):
                        penalty += 2.0
                    if candidate["move_dir"] in ["E", "W"]:
                        penalty += 1.5

        return penalty

    def _build_danger_map(self, board, my_pos, opponents, rows, cols):
        danger = {}
        for opp in opponents:
            ox, oy = opp["pos"]
            base = (ox, oy)
            danger[base] = danger.get(base, 0) + 6

            opp_dir = opp.get("direction")
            if opp_dir in MOVE_DELTAS:
                fx, fy = self._next_pos(base, opp_dir)
                if self._in_bounds(fx, fy, rows, cols):
                    danger[(fx, fy)] = danger.get((fx, fy), 0) + 8

                bx, by = fx, fy
                if self._is_walkable_now(board, bx, by, rows, cols, allow_heads=True):
                    fx2, fy2 = bx + MOVE_DELTAS[opp_dir][0], by + MOVE_DELTAS[opp_dir][1]
                    if self._in_bounds(fx2, fy2, rows, cols):
                        if self._is_walkable_now(board, fx2, fy2, rows, cols, allow_heads=True):
                            danger[(fx2, fy2)] = danger.get((fx2, fy2), 0) + 3

            for move in ("N", "S", "E", "W"):
                nx, ny = ox + MOVE_DELTAS[move][0], oy + MOVE_DELTAS[move][1]
                if self._is_walkable_now(board, nx, ny, rows, cols, allow_heads=True):
                    danger[(nx, ny)] = danger.get((nx, ny), 0) + 9
                    for dx, dy in MOVE_DELTAS.values():
                        ax, ay = nx + dx, ny + dy
                        if self._in_bounds(ax, ay, rows, cols):
                            danger[(ax, ay)] = danger.get((ax, ay), 0) + 1

        mx, my = my_pos
        danger.pop((mx, my), None)
        return danger

    def _danger_penalty(self, candidate, danger_map, opponents):
        penalty = 0.0
        landing = candidate["landing"]
        path = candidate["path"]

        penalty += danger_map.get(landing, 0)
        for step in path[:-1]:
            penalty += danger_map.get(step, 0) * 0.35

        for opp in opponents:
            ox, oy = opp["pos"]
            dist = abs(landing[0] - ox) + abs(landing[1] - oy)
            if dist == 0:
                penalty += 20
            elif dist == 1:
                penalty += 8
            elif dist == 2:
                penalty += 3

        return penalty

    def _territory_score(self, board, my_start, rows, cols, opponents, map_profile, my_start_time=0):
        if not opponents:
            return 0.0

        live_positions = [opp["pos"] for opp in opponents if opp.get("pos") is not None]
        if not live_positions:
            return 0.0

        my_dist = self._distance_map(
            board,
            my_start,
            rows,
            cols,
            map_profile,
            start_time=my_start_time,
        )
        opp_dists = [
            self._distance_map(board, pos, rows, cols, map_profile, start_time=0)
            for pos in live_positions
        ]

        counts = [0] * (len(opp_dists) + 1)
        all_cells = set(my_dist)
        for dist_map in opp_dists:
            all_cells.update(dist_map)

        for cell in all_cells:
            best_distance = my_dist.get(cell)
            owner = 0 if best_distance is not None else None
            tied = False

            for idx, dist_map in enumerate(opp_dists, start=1):
                dist = dist_map.get(cell)
                if dist is None:
                    continue
                if best_distance is None or dist < best_distance:
                    best_distance = dist
                    owner = idx
                    tied = False
                elif dist == best_distance:
                    tied = True

            if owner is not None and not tied:
                counts[owner] += 1

        my_cells = counts[0]
        opp_cells = max(counts[1:], default=0)
        return my_cells - opp_cells

    def _distance_map(self, board, start, rows, cols, map_profile, start_time=0):
        timed_policy = "soon" if map_profile["timed_map"] else "blocked"
        queue = deque([(start[0], start[1], 0)])
        dist = {start: 0}

        while queue:
            x, y, steps = queue.popleft()
            for dx, dy in MOVE_DELTAS.values():
                nx, ny = x + dx, y + dy
                next_pos = (nx, ny)
                if next_pos in dist:
                    continue
                if not self._is_walkable_future(
                    board,
                    nx,
                    ny,
                    rows,
                    cols,
                    start_time + steps + 1,
                    timed_policy,
                ):
                    continue
                dist[next_pos] = steps + 1
                queue.append((nx, ny, steps + 1))

        return dist

    def _voronoi_balance(self, board, my_start, opp_start, rows, cols, map_profile, my_start_time=0):
        my_dist = self._distance_map(board, my_start, rows, cols, map_profile, start_time=my_start_time)
        opp_dist = self._distance_map(board, opp_start, rows, cols, map_profile, start_time=0)
        my_cells = 0
        opp_cells = 0

        for pos in set(my_dist) | set(opp_dist):
            if pos == my_start:
                my_cells += 1
                continue
            md = my_dist.get(pos)
            od = opp_dist.get(pos)
            if md is None:
                opp_cells += 1
            elif od is None:
                my_cells += 1
            elif md < od:
                my_cells += 1
            elif od < md:
                opp_cells += 1

        return my_cells - opp_cells

    def _crowd_pressure(self, landing, opponents):
        pressure = 0.0
        close_count = 0
        for opp in opponents:
            ox, oy = opp["pos"]
            dist = abs(landing[0] - ox) + abs(landing[1] - oy)
            if dist <= 6:
                pressure += max(0, 7 - dist)
            if dist <= 4:
                close_count += 1
        if close_count >= 2:
            pressure += (close_count - 1) * 5
        return pressure

    def _center_bonus(self, pos, rows, cols):
        cx = (cols - 1) / 2.0
        cy = (rows - 1) / 2.0
        dist = abs(pos[0] - cx) + abs(pos[1] - cy)
        return max(0.0, (rows + cols) / 4.0 - dist)

    def _should_emp(self, board, pos, candidate, opponents, info, danger_map, map_profile):
        if info.get("emp_charges", 0) <= 0 or not opponents:
            return False

        if len(opponents) == 0:
            return False

        if map_profile["open_battle_opening"] and len(opponents) >= 2 and self.turn_count <= 20:
            return False

        landing = candidate["landing"]
        radius = info.get("emp_radius", 3)

        nearby = 0
        trapped_targets = 0
        for opp in opponents:
            ox, oy = opp["pos"]
            if max(abs(landing[0] - ox), abs(landing[1] - oy)) <= radius:
                nearby += 1
                safe_count = 0
                for dx, dy in MOVE_DELTAS.values():
                    nx, ny = ox + dx, oy + dy
                    if self._is_walkable_now(board, nx, ny, len(board), len(board[0]), allow_heads=True):
                        safe_count += 1
                if safe_count <= 2:
                    trapped_targets += 1

        if nearby >= 2:
            return True
        if trapped_targets >= 1 and nearby >= 1:
            return True

        if candidate["danger"] >= 14 and nearby >= 1 and candidate["space_now"] >= 10:
            return True

        return False

    def _apply_profile_weight_overrides(self, weights, map_profile, opponent_count):
        tuned = dict(weights)

        if opponent_count <= 0:
            return tuned

        if map_profile["center_race"]:
            tuned.update(
                {
                    "future_space": 0.2,
                    "mobility": 9.0,
                    "item": 13.0,
                    "density": 1.1,
                    "danger": 2.0,
                    "direct": 0.8,
                    "space": 1.7,
                    "center": 0.4,
                    "item_depth": 20,
                    "safe_ratio": 0.14,
                    "escape": 4.0,
                    "crowd": 1.6,
                    "territory": 0.32,
                    "emp_threat": 1.2,
                }
            )
        elif map_profile["timed_map"] and not map_profile["orbit_like"]:
            tuned.update(
                {
                    "space": 2.5,
                    "future_space": 0.45,
                    "danger": 2.7,
                    "mobility": 11.0,
                    "item": 7.0,
                    "density": 0.5,
                    "center": 0.4,
                    "direct": 0.55,
                    "safe_ratio": 0.20,
                    "territory": 0.34,
                    "escape": 4.6,
                    "crowd": 1.2,
                    "emp_threat": 1.35,
                }
            )

        if opponent_count == 1:
            tuned["territory"] *= 1.4
        else:
            tuned["danger"] *= 1.2
            tuned["territory"] *= 0.6

        return tuned

    def _least_bad_move(self, board, pos, rows, cols, current_direction):
        simulated = []
        for action in ("N", "S", "E", "W", "+", "P"):
            candidate = self._simulate_action(board, pos, action, current_direction, rows, cols)
            if candidate is not None:
                simulated.append(candidate)

        if simulated:
            simulated.sort(key=lambda c: (c["direct_reward"], c["action"] in MOVE_DELTAS), reverse=True)
            return simulated[0]["action"]

        safe_moves = self._get_safe_moves(board, pos, rows, cols)
        if safe_moves:
            return safe_moves[0]
        return current_direction if current_direction in MOVE_DELTAS else "N"

    def _get_weights(self, map_class, opponent_count):
        weights = {
            "space": 1.8,
            "future_space": 0.2,
            "mobility": 9.0,
            "item": 10.0,
            "density": 0.65,
            "danger": 2.0,
            "territory": 0.16,
            "center": 0.4,
            "direct": 0.55,
            "straight": 1.0,
            "escape": 3.5,
            "crowd": 0.8,
            "trap": 0.8,
            "emp_threat": 0.7,
            "solo_boost": 0.0,
            "solo_skip_item": 1.0,
            "solo_route": 0.0,
            "solo_route_depth": 0,
            "solo_cluster": 0.0,
            "solo_cluster_depth": 0,
            "solo_branch": 0.0,
            "solo_branch_depth": 0,
            "safe_ratio": 0.14,
            "item_depth": 16,
        }

        if opponent_count == 0 and map_class == MAP_CORRIDOR:
            weights.update(
                {
                    "space": 1.1,
                    "item": 12.0,
                    "density": 1.0,
                    "direct": 0.7,
                    "straight": 3.0,
                    "item_depth": 40,
                    "safe_ratio": 0.10,
                    "danger": 0.8,
                    "territory": 0.0,
                    "center": 0.2,
                    "escape": 2.0,
                    "crowd": 0.25,
                    "emp_threat": 0.35,
                    "solo_boost": 28.0,
                    "solo_skip_item": 1.15,
                    "solo_route": 0.7,
                    "solo_route_depth": 18,
                    "solo_cluster": 0.25,
                    "solo_cluster_depth": 18,
                    "solo_branch": 0.35,
                    "solo_branch_depth": 14,
                }
            )
        elif opponent_count == 0 and map_class == MAP_BRANCHING:
            weights.update(
                {
                    "space": 2.2,
                    "future_space": 0.30,
                    "item": 11.5,
                    "density": 1.15,
                    "mobility": 11.0,
                    "item_depth": 24,
                    "danger": 0.7,
                    "territory": 0.0,
                    "center": 0.2,
                    "escape": 2.9,
                    "crowd": 0.2,
                    "emp_threat": 0.35,
                    "solo_boost": 15.0,
                    "solo_skip_item": 1.15,
                    "solo_route": 1.0,
                    "solo_route_depth": 14,
                    "solo_cluster": 0.75,
                    "solo_cluster_depth": 15,
                    "solo_branch": 1.0,
                    "solo_branch_depth": 12,
                }
            )
        elif opponent_count == 0 and map_class == MAP_OPEN:
            weights.update(
                {
                    "space": 2.6,
                    "future_space": 0.35,
                    "item": 10.0,
                    "density": 1.2,
                    "mobility": 12.0,
                    "safe_ratio": 0.15,
                    "danger": 0.7,
                    "territory": 0.0,
                    "center": 0.15,
                    "item_depth": 24,
                    "escape": 3.0,
                    "crowd": 0.2,
                    "emp_threat": 0.35,
                    "solo_boost": 18.0,
                    "solo_skip_item": 1.1,
                    "solo_route": 1.0,
                    "solo_route_depth": 12,
                    "solo_cluster": 0.8,
                    "solo_cluster_depth": 14,
                    "solo_branch": 0.55,
                    "solo_branch_depth": 10,
                }
            )
        elif map_class == MAP_OPEN:
            weights.update(
                {
                    "space": 2.2,
                    "danger": 3.0,
                    "territory": 0.36,
                    "center": 0.8,
                    "item": 7.0,
                    "density": 0.4,
                    "safe_ratio": 0.18,
                    "escape": 4.0,
                    "crowd": 1.4,
                    "emp_threat": 0.3,
                }
            )
        elif map_class == MAP_BRANCHING:
            weights.update(
                {
                    "item": 11.0,
                    "density": 1.0,
                    "direct": 0.8,
                    "space": 1.9,
                    "item_depth": 22,
                    "escape": 4.1,
                    "crowd": 1.5,
                    "territory": 0.32,
                    "emp_threat": 0.95,
                    "danger": 2.4,
                }
            )
        elif map_class == MAP_CORRIDOR:
            weights.update(
                {
                    "space": 2.5,
                    "future_space": 0.45,
                    "danger": 2.7,
                    "mobility": 11.0,
                    "item": 7.0,
                    "density": 0.5,
                    "safe_ratio": 0.20,
                    "territory": 0.34,
                    "escape": 4.6,
                    "crowd": 1.2,
                    "emp_threat": 1.35,
                }
            )

        if opponent_count == 0:
            weights["danger"] = 0.0
            weights["territory"] = 0.0
            weights["center"] *= 0.5
            weights["crowd"] = 0.0

        return weights
