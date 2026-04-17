import heapq
from collections import deque


class Bot:
    DIRS = {
        "N": (0, -1),
        "S": (0, 1),
        "E": (1, 0),
        "W": (-1, 0),
    }

    EMP_RADIUS = 3
    HUGE_PENALTY = 10**9

    def __init__(self):
        self.turn_count = 0
        self.prev_pos = None
        self.prev_move = None
        self.current_map = None
        self.solo_target = None
        self.item_positions = None
        self.opening_dir = None
        self.topology = None

    def get_move(self, board, player_id, pos, info):
        map_name = info.get("map_name", "")
        self._sync_state_for_map(board, map_name)
        self.turn_count += 1

        opponents = [opp for opp in info.get("opponents", []) if opp.get("alive")]
        my_dir = info.get("my_direction", "S")
        phase_charges = info.get("phase_charges", 0)
        emp_charges = info.get("emp_charges", 0)

        candidates = self._generate_candidates(board, pos, my_dir, phase_charges)
        if not candidates:
            move = self._fallback_move(board, pos, my_dir, phase_charges)
            self.prev_pos = pos
            self.prev_move = move
            return move

        threat_map = self._build_threat_map(board, opponents)
        opponent_maps = self._build_opponent_maps(board, opponents)
        guided_commands = self._guided_commands(board, pos, my_dir, map_name)
        scenario = self._scenario_weights(map_name, opponents, self._get_topology(board))

        for cand in candidates:
            dist_map = self._distance_map(board, cand["dest"], cand["blocked"])
            cand["analysis"] = self._scan_metrics(board, dist_map, player_id, opponent_maps)
            cand["area"] = cand["analysis"][0]

        max_normal_area = max((cand["area"] for cand in candidates if not cand["phase"]), default=0)
        best = None
        best_score = -10**18

        for cand in candidates:
            score = self._score_candidate(
                board=board,
                player_id=player_id,
                candidate=cand,
                opponents=opponents,
                opponent_maps=opponent_maps,
                active_emps=info.get("active_emps", []),
                threat_map=threat_map,
                scenario=scenario,
                guided_commands=guided_commands,
                map_name=map_name,
                max_normal_area=max_normal_area,
            )
            cand["score"] = score
            if score > best_score:
                best_score = score
                best = cand

        move = best["command"]
        if emp_charges > 0 and self._should_use_emp(best, opponents, map_name, board):
            move = "X" + move

        self.prev_pos = pos
        self.prev_move = move
        return move

    def _sync_state_for_map(self, board, map_name):
        if map_name != self.current_map:
            self.current_map = map_name
            self.turn_count = 0
            self.prev_pos = None
            self.prev_move = None
            self.solo_target = None
            self.item_positions = None
            self.opening_dir = None
            self.topology = None

        if map_name.startswith("s_") and self.item_positions is None:
            self.item_positions = set()
            for y, row in enumerate(board):
                for x, cell in enumerate(row):
                    if cell in ("c", "D"):
                        self.item_positions.add((x, y))

        if map_name.startswith("s_") and self.item_positions is not None:
            self.item_positions = {
                cell for cell in self.item_positions
                if self._cell_value(board[cell[1]][cell[0]]) > 0
            }
            if self.solo_target not in self.item_positions:
                self.solo_target = None

    def _scenario_weights(self, map_name, opponents, topology):
        weights = {
            "area": 2.6,
            "density": 4.0,
            "reachable_value": 0.18,
            "nearest_item": 1.6,
            "turn_points": 6.0,
            "threat": 9.0,
            "boost_bonus": 5.0,
            "phase_bonus": 2.0,
            "voronoi": 0.0,
        }

        if map_name.startswith("s_path"):
            weights.update({
                "area": 2.4,
                "density": 4.6,
                "reachable_value": 0.24,
                "nearest_item": 2.1,
                "turn_points": 1.5,
                "threat": 2.5,
                "boost_bonus": 1.0,
                "phase_bonus": 1.0,
            })
        elif map_name.startswith("s_choice"):
            weights.update({
                "area": 2.6,
                "density": 5.0,
                "reachable_value": 0.28,
                "nearest_item": 2.0,
                "turn_points": 1.5,
                "threat": 2.5,
                "boost_bonus": 1.2,
                "phase_bonus": 1.0,
            })
        elif map_name.startswith("s_floodfill"):
            weights.update({
                "area": 3.0,
                "density": 5.2,
                "reachable_value": 0.24,
                "nearest_item": 1.5,
                "turn_points": 4.0,
                "threat": 2.0,
            })
        elif len(opponents) <= 1:
            weights.update({
                "area": 3.1,
                "density": 2.8,
                "reachable_value": 0.10,
                "nearest_item": 0.8,
                "turn_points": 7.0,
                "threat": 10.5,
                "voronoi": 0.0,
            })

        if map_name.startswith("s_") or topology is None:
            return weights

        openness = topology["openness_ratio"]
        corridor = topology["corridor_density"]
        diamonds = topology["diamond_count"]
        centrality = topology["diamond_centrality"]

        if corridor >= 0.45:
            weights["area"] += 0.5
            weights["density"] -= 1.2
            weights["reachable_value"] -= 0.08
            weights["nearest_item"] -= 0.5
            weights["threat"] += 0.9
            weights["boost_bonus"] -= 2.0
            if len(opponents) <= 1:
                weights["voronoi"] = 0.2
        elif openness >= 0.86 and corridor <= 0.08:
            weights["area"] += 0.5
            weights["density"] -= 1.8
            weights["reachable_value"] -= 0.12
            weights["nearest_item"] -= 0.8
            weights["turn_points"] += 0.8
            weights["threat"] += 0.8
            weights["boost_bonus"] += 2.0
            if len(opponents) <= 1:
                weights["voronoi"] = 0.35

            if diamonds > 0 and centrality >= 0.55:
                weights["density"] += 1.1
                weights["reachable_value"] += 0.16
                weights["nearest_item"] += 1.2
                weights["threat"] -= 0.7
                weights["boost_bonus"] -= 1.0
        elif diamonds > 0:
            weights["density"] += 0.7
            weights["reachable_value"] += 0.10
            weights["nearest_item"] += 0.8
            weights["area"] += 0.2
            weights["threat"] -= 0.4

        return weights

    def _generate_candidates(self, board, pos, my_dir, phase_charges):
        candidates = []

        for direction in self._ordered_dirs(my_dir):
            cand = self._simulate_action(board, pos, my_dir, direction)
            if cand is not None:
                candidates.append(cand)

        boost = self._simulate_action(board, pos, my_dir, "+")
        if boost is not None:
            candidates.append(boost)

        if phase_charges > 0:
            phase = self._simulate_action(board, pos, my_dir, "P")
            if phase is not None:
                candidates.append(phase)

        return candidates

    def _ordered_dirs(self, my_dir):
        if my_dir == "N":
            return ("N", "E", "W", "S")
        if my_dir == "S":
            return ("S", "W", "E", "N")
        if my_dir == "E":
            return ("E", "N", "S", "W")
        return ("W", "S", "N", "E")

    def _simulate_action(self, board, pos, my_dir, command):
        blocked_positions = []
        traversed = []
        immediate_gain = 0

        if command in self.DIRS:
            steps = [("normal", command)]
            new_dir = command
            boost = False
            phase = False
        elif command == "+":
            steps = [("normal", my_dir), ("normal", my_dir)]
            new_dir = my_dir
            boost = True
            phase = False
        else:
            steps = [("phase", my_dir), ("normal", my_dir)]
            new_dir = my_dir
            boost = False
            phase = True

        cx, cy = pos
        rows = len(board)
        cols = len(board[0])

        for step_type, direction in steps:
            dx, dy = self.DIRS[direction]
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < cols and 0 <= ny < rows):
                return None

            cell = board[ny][nx]
            if step_type != "phase" and not self._is_safe_cell(cell):
                return None

            blocked_positions.append((cx, cy))
            traversed.append((nx, ny))
            immediate_gain += self._cell_value(cell)
            cx, cy = nx, ny

        return {
            "command": command,
            "dest": (cx, cy),
            "dir": new_dir,
            "blocked": frozenset(blocked_positions),
            "traversed": tuple(traversed),
            "immediate_gain": immediate_gain,
            "turn_points": 2 if command in {"+", "P"} else 1,
            "boost": boost,
            "phase": phase,
        }

    def _score_candidate(
        self,
        board,
        player_id,
        candidate,
        opponents,
        opponent_maps,
        active_emps,
        threat_map,
        scenario,
        guided_commands,
        map_name,
        max_normal_area,
    ):
        area, density, reachable_value, nearest_dist, nearest_value, opportunity = candidate["analysis"]
        if area == 0:
            return -self.HUGE_PENALTY

        is_solo = map_name.startswith("s_")
        is_solo_path_choice = map_name.startswith("s_path") or map_name.startswith("s_choice")
        dest = candidate["dest"]
        score = 0.0

        score += candidate["immediate_gain"] * 2.4
        score += scenario["turn_points"] * candidate["turn_points"]
        score += scenario["area"] * area
        score += scenario["density"] * density
        score += scenario["reachable_value"] * reachable_value
        score += 8.0 * opportunity

        if candidate["command"] in guided_commands:
            score += 200.0 if is_solo else 55.0
        elif candidate["dir"] in guided_commands:
            score += 80.0 if is_solo else 18.0

        if nearest_dist is not None:
            score += scenario["nearest_item"] * (nearest_value - 2.0 * nearest_dist)

        if len(opponents) == 1 and scenario["voronoi"] > 0:
            opp = opponents[0]
            if abs(dest[0] - opp["pos"][0]) + abs(dest[1] - opp["pos"][1]) <= 8:
                my_cells, opp_cells = self._voronoi_score(
                    board,
                    dest,
                    opp["pos"],
                    candidate["blocked"],
                )
                score += scenario["voronoi"] * (my_cells - opp_cells)

        threat_penalty = self._candidate_threat_penalty(
            candidate,
            opponents,
            active_emps,
            threat_map,
            area,
        )
        score -= scenario["threat"] * threat_penalty

        if area <= len(candidate["blocked"]) + 2:
            score -= 250.0
        elif area <= 6:
            score -= 90.0
        elif area <= 12:
            score -= 30.0

        exits = self._exit_count(board, dest, candidate["blocked"])
        if exits <= 1 and area < 18:
            score -= 45.0 if candidate["immediate_gain"] == 0 else 18.0

        if candidate["boost"]:
            exits_after = None
            if len(candidate["traversed"]) >= 2:
                intermediate = candidate["traversed"][0]
                exits_after = self._exit_count(board, dest, candidate["blocked"] | {intermediate})
            else:
                exits_after = exits

            if is_solo:
                if exits_after >= 2 and area >= 24:
                    score += 12.0
                elif exits_after >= 2:
                    score += 4.0
                else:
                    score -= 18.0
                if is_solo_path_choice and candidate["immediate_gain"] < 50:
                    score -= 90.0 if candidate["immediate_gain"] == 0 else 45.0
            if candidate["immediate_gain"] > 0:
                score += 12.0
            if area >= 15:
                score += scenario["boost_bonus"]
            elif area < 8:
                score -= 60.0
            elif area < 14:
                score -= 25.0
            if exits_after == 0 and area < 20:
                score -= 70.0
            elif exits_after == 1 and area < 32:
                score -= 24.0

        if candidate["phase"]:
            if is_solo_path_choice:
                if max_normal_area < 4 and area > max_normal_area * 2:
                    score += 45.0
                elif candidate["immediate_gain"] >= 50 and area > max_normal_area + 8:
                    score += 20.0
                else:
                    score -= 180.0
                if area <= max_normal_area + 8:
                    score -= 180.0
                if candidate["immediate_gain"] < 50:
                    score -= 60.0
            elif max_normal_area < 5 and area > max_normal_area * 2:
                score += 80.0
            elif candidate["immediate_gain"] >= 50:
                score += 40.0
            elif area <= max_normal_area:
                score -= 50.0
            else:
                score += scenario["phase_bonus"]

        if self.prev_pos is not None and dest == self.prev_pos and area < 14:
            score -= 18.0

        return score

    def _guided_commands(self, board, pos, my_dir, map_name):
        if not map_name.startswith("s_"):
            return set()

        if map_name.startswith("s_path") or map_name.startswith("s_choice"):
            return self._guided_path_choice_commands(board, pos, my_dir, map_name)

        if self.opening_dir is None:
            self.opening_dir = self._scan_opening_direction(board, pos)

        if self.item_positions is None or not self.item_positions:
            return {self.opening_dir} if self.opening_dir else set()

        if self.solo_target not in self.item_positions:
            self.solo_target = self._pick_solo_target(board, pos)

        if self.solo_target is None:
            return {self.opening_dir} if self.opening_dir else set()

        path = self._path_to_target(board, pos, self.solo_target, allow_soon_safe=True)
        if len(path) < 2:
            self.solo_target = None
            return {self.opening_dir} if self.opening_dir else set()

        first = self._step_direction(path[0], path[1])
        commands = {first}
        if first == my_dir and len(path) >= 3:
            second = self._step_direction(path[1], path[2])
            if second == first:
                commands.add("+")
        return commands

    def _guided_path_choice_commands(self, board, pos, my_dir, map_name):
        path = self._best_solo_route_path(
            board,
            pos,
            map_name,
            allow_soon_safe=False,
            depth=3,
            candidate_limit=6,
        )
        if len(path) < 2:
            return set()

        first = self._step_direction(path[0], path[1])
        commands = {first}
        if len(path) >= 3:
            second = self._step_direction(path[1], path[2])
            if first == second == my_dir:
                commands.add("+")
        return commands

    def _item_chain_score(self, current_cell, current_value, current_dist, items, pair_maps, depth):
        score = current_value * 4.4 - current_dist * 2.2
        if depth <= 0:
            return score

        best_follow = 0.0
        dist_map = pair_maps[current_cell]
        for next_cell, next_value, _ in items:
            if next_cell == current_cell:
                continue
            step_dist = dist_map.get(next_cell)
            if step_dist is None:
                continue
            follow = 0.72 * self._item_chain_score(
                next_cell,
                next_value,
                step_dist,
                items,
                pair_maps,
                depth - 1,
            )
            if follow > best_follow:
                best_follow = follow
        return score + best_follow

    def _pick_solo_target(self, board, pos):
        if not self.item_positions:
            return None

        reachable = self._distance_map(board, pos, frozenset(), allow_soon_safe=True)
        if self.current_map and self.current_map.startswith("s_floodfill"):
            return self._pick_floodfill_target(board, pos, reachable)

        best = None
        best_key = None

        for target in self.item_positions:
            dist = reachable.get(target)
            if dist is None:
                continue
            value = self._cell_value(board[target[1]][target[0]])
            ratio = value / (dist + 1.0)
            first_dir = self._first_direction_toward(board, pos, target, allow_soon_safe=True)
            open_bonus = 1 if first_dir is not None and first_dir == self.opening_dir else 0
            key = (ratio, value, open_bonus, -dist)
            if best_key is None or key > best_key:
                best_key = key
                best = target

        return best

    def _pick_floodfill_target(self, board, pos, reachable):
        scored = []
        for target in self.item_positions:
            dist = reachable.get(target)
            if dist is None:
                continue
            value = self._cell_value(board[target[1]][target[0]])
            score = value / (dist + 1.0)
            if value >= 50:
                score += 4.0
            if dist <= 2:
                score += 3.0
            first_dir = self._first_direction_toward(board, pos, target, allow_soon_safe=True)
            if first_dir is not None and first_dir == self.opening_dir:
                score += 0.5
            scored.append((score, target, dist, value))

        if not scored:
            return None

        scored.sort(key=lambda item: (item[0], item[3], -item[2]), reverse=True)
        top_candidates = scored[:8]
        best_target = None
        best_score = -10**18

        for base_score, target, _, _ in top_candidates:
            target_dm = self._distance_map(board, target, frozenset(), allow_soon_safe=True)
            cluster_bonus = 0.0
            for other in self.item_positions:
                if other == target:
                    continue
                od = target_dm.get(other)
                if od is not None and od <= 5:
                    cluster_bonus += self._cell_value(board[other[1]][other[0]]) / (od + 1.0)

            score = base_score + 0.12 * cluster_bonus
            if score > best_score:
                best_score = score
                best_target = target

        return best_target

    def _get_topology(self, board):
        if self.topology is None:
            self.topology = self._analyze_topology(board)
        return self.topology

    def _analyze_topology(self, board):
        rows = len(board)
        cols = len(board[0])
        open_cells = 0
        corridor_cells = 0
        diamond_cells = []

        for y, row in enumerate(board):
            for x, cell in enumerate(row):
                if cell == "#":
                    continue
                open_cells += 1
                if cell == "D":
                    diamond_cells.append((x, y))

                neighbors = 0
                for dx, dy in self.DIRS.values():
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < cols and 0 <= ny < rows and board[ny][nx] != "#":
                        neighbors += 1
                if neighbors <= 2:
                    corridor_cells += 1

        openness_ratio = open_cells / (rows * cols) if rows and cols else 0.0
        corridor_density = corridor_cells / open_cells if open_cells else 0.0

        if diamond_cells:
            cx = (cols - 1) / 2.0
            cy = (rows - 1) / 2.0
            avg_dist = sum(abs(x - cx) + abs(y - cy) for x, y in diamond_cells) / len(diamond_cells)
            diamond_centrality = max(0.0, 1.0 - (2.0 * avg_dist / max(1.0, rows + cols)))
        else:
            diamond_centrality = 0.0

        return {
            "openness_ratio": openness_ratio,
            "corridor_density": corridor_density,
            "diamond_count": len(diamond_cells),
            "diamond_centrality": diamond_centrality,
        }

    def _best_solo_route_path(self, board, start, map_name, allow_soon_safe, depth, candidate_limit):
        memo = {}
        path_cache = {}
        space_cache = {}

        def route_value(path, taken):
            value = 0
            collected = []
            for cell in path[1:]:
                if cell in taken:
                    continue
                cell_value = self._cell_value(board[cell[1]][cell[0]])
                if cell_value:
                    value += cell_value
                    collected.append(cell)
            return value, frozenset(collected)

        def candidate_bonus(target, reachable):
            value = self._cell_value(board[target[1]][target[0]])
            dist = reachable[target]
            bonus = 0.0
            if value >= 50:
                bonus += 8.0
            if map_name.startswith("s_choice") and dist <= 4:
                bonus += 2.5
            if self.opening_dir is not None:
                first_dir = self._first_direction_toward(board, start, target, allow_soon_safe=allow_soon_safe)
                if first_dir == self.opening_dir:
                    bonus += 1.0
            return bonus

        def search(pos, blocked, taken, remaining_depth):
            key = (pos, blocked, taken, remaining_depth)
            cached = memo.get(key)
            if cached is not None:
                return cached

            reachable = self._distance_map(board, pos, blocked, allow_soon_safe=allow_soon_safe)
            items = []
            for cell, dist in reachable.items():
                if cell == pos or cell in taken:
                    continue
                value = self._cell_value(board[cell[1]][cell[0]])
                if not value:
                    continue
                pre_score = value * 4.6 - dist * 2.2 + candidate_bonus(cell, reachable)
                items.append((pre_score, cell, dist, value))

            if not items or remaining_depth <= 0:
                area = len(reachable)
                exits = self._exit_count(board, pos, blocked)
                result = (0.16 * area + 5.0 * exits, [pos])
                memo[key] = result
                return result

            items.sort(key=lambda item: (item[0], item[3], -item[2]), reverse=True)
            best_score = -10**18
            best_path = [pos]

            for _, target, dist, _ in items[:candidate_limit]:
                path_key = (pos, target, blocked, allow_soon_safe)
                path = path_cache.get(path_key)
                if path is None:
                    path = self._path_to_target_with_blocked(
                        board,
                        pos,
                        target,
                        blocked,
                        allow_soon_safe=allow_soon_safe,
                    )
                    path_cache[path_key] = path
                if len(path) < 2:
                    continue

                gained_value, collected = route_value(path, taken)
                new_taken = taken | collected
                new_blocked = blocked | frozenset(path[:-1])
                end = path[-1]

                area_key = (end, new_blocked, allow_soon_safe)
                next_space = space_cache.get(area_key)
                if next_space is None:
                    next_space = self._distance_map(board, end, new_blocked, allow_soon_safe=allow_soon_safe)
                    space_cache[area_key] = next_space
                area = len(next_space)
                exits = self._exit_count(board, end, new_blocked)

                local_score = gained_value * 4.8 - dist * 2.0 + 0.18 * area + 7.0 * exits
                if exits == 0:
                    local_score -= 140.0
                elif exits == 1 and area < 16:
                    local_score -= 45.0

                future_score, future_path = search(end, new_blocked, new_taken, remaining_depth - 1)
                total = local_score + 0.72 * future_score
                if total > best_score:
                    best_score = total
                    best_path = path + future_path[1:]

            result = (best_score, best_path)
            memo[key] = result
            return result

        return search(start, frozenset(), frozenset(), depth)[1]

    def _path_to_target_with_blocked(self, board, start, target, blocked, allow_soon_safe=False):
        if target is None:
            return [start]
        if start == target:
            return [start]

        dist_map, parents = self._distance_map(
            board,
            start,
            blocked,
            allow_soon_safe=allow_soon_safe,
            return_parents=True,
        )
        if target not in dist_map:
            return [start]

        path = [target]
        cur = target
        while parents[cur] is not None:
            cur = parents[cur]
            path.append(cur)
        path.reverse()
        return path

    def _path_to_target(self, board, start, target, allow_soon_safe=False):
        if target is None:
            return [start]
        if start == target:
            return [start]

        dist_map, parents = self._distance_map(
            board,
            start,
            frozenset(),
            allow_soon_safe=allow_soon_safe,
            return_parents=True,
        )
        if target not in dist_map:
            return [start]

        path = [target]
        cur = target
        while parents[cur] is not None:
            cur = parents[cur]
            path.append(cur)
        path.reverse()
        return path

    def _step_direction(self, src, dst):
        dx = dst[0] - src[0]
        dy = dst[1] - src[1]
        for direction, (mx, my) in self.DIRS.items():
            if (dx, dy) == (mx, my):
                return direction
        return "N"

    def _candidate_threat_penalty(self, candidate, opponents, active_emps, threat_map, area):
        dest = candidate["dest"]
        penalty = float(threat_map.get(dest, 0))

        for cell in candidate["traversed"]:
            penalty += 0.35 * threat_map.get(cell, 0)

        for opp in opponents:
            ox, oy = opp["pos"]
            cheb = max(abs(dest[0] - ox), abs(dest[1] - oy))
            manhattan = abs(dest[0] - ox) + abs(dest[1] - oy)

            if cheb == 0:
                penalty += 12.0
            elif cheb == 1:
                penalty += 4.0
            elif manhattan <= 3:
                penalty += 1.0

        for emp in active_emps:
            ex, ey = emp.get("pos", (-999, -999))
            timer = emp.get("timer", 99)
            cheb = max(abs(dest[0] - ex), abs(dest[1] - ey))
            if cheb <= self.EMP_RADIUS:
                if timer <= 1:
                    penalty += 20.0
                elif timer <= 2:
                    penalty += 10.0
                else:
                    penalty += 2.5

        if area < 10:
            penalty += 1.5
        return penalty

    def _distance_map(self, board, start, blocked, allow_soon_safe=False, return_parents=False):
        if start in blocked:
            return ({}, {}) if return_parents else {}

        rows = len(board)
        cols = len(board[0])
        pq = [(0, start)]
        dist_map = {start: 0}
        parents = {start: None}

        while pq:
            dist, (x, y) = heapq.heappop(pq)
            if dist != dist_map[(x, y)]:
                continue

            for dx, dy in self.DIRS.values():
                nx, ny = x + dx, y + dy
                nxt = (nx, ny)
                if not (0 <= nx < cols and 0 <= ny < rows):
                    continue
                if nxt in blocked:
                    continue

                cell = board[ny][nx]
                if nxt == start:
                    step_cost = 1
                else:
                    step_cost = self._plan_step_cost(cell, dist + 1, allow_soon_safe)
                if step_cost is None:
                    continue

                new_dist = dist + step_cost
                if new_dist >= dist_map.get(nxt, 10**9):
                    continue

                dist_map[nxt] = new_dist
                parents[nxt] = (x, y)
                heapq.heappush(pq, (new_dist, nxt))

        if return_parents:
            return dist_map, parents
        return dist_map

    def _scan_metrics(self, board, dist_map, player_id, opponent_maps):
        area = len(dist_map)
        density = 0.0
        reachable_value = 0
        nearest_dist = None
        nearest_value = 0
        opportunity = 0.0

        for (x, y), dist in dist_map.items():
            value = self._cell_value(board[y][x])
            if not value:
                continue

            reachable_value += value
            density += value / (dist + 1)

            if nearest_dist is None or dist < nearest_dist or (dist == nearest_dist and value > nearest_value):
                nearest_dist = dist
                nearest_value = value

            best_opp_dist = None
            best_opp_id = None
            for opp_id, opp_map in opponent_maps.items():
                opp_dist = opp_map.get((x, y))
                if opp_dist is None:
                    continue
                if best_opp_dist is None or opp_dist < best_opp_dist or (opp_dist == best_opp_dist and opp_id < best_opp_id):
                    best_opp_dist = opp_dist
                    best_opp_id = opp_id

            if best_opp_dist is None:
                factor = 1.4
            elif dist < best_opp_dist:
                factor = 1.8
            elif dist == best_opp_dist and player_id < best_opp_id:
                factor = 1.35
            elif dist <= best_opp_dist + 1:
                factor = 0.45
            else:
                factor = 0.12

            opportunity += factor * value / (dist + 1)

        return area, density, reachable_value, nearest_dist, nearest_value, opportunity

    def _build_opponent_maps(self, board, opponents):
        maps = {}
        for opp in opponents:
            maps[opp["id"]] = self._distance_map(board, opp["pos"], frozenset())
        return maps

    def _build_threat_map(self, board, opponents):
        threat = {}
        rows = len(board)
        cols = len(board[0])

        for opp in opponents:
            ox, oy = opp["pos"]
            self._add_threat(threat, (ox, oy), 4)

            for cell, value in self._predict_opponent_cells(opp, board):
                self._add_threat(threat, cell, value)

            valid_moves = []
            for direction, (dx, dy) in self.DIRS.items():
                nx, ny = ox + dx, oy + dy
                if 0 <= nx < cols and 0 <= ny < rows and self._is_safe_cell(board[ny][nx]):
                    valid_moves.append((direction, nx, ny))

            forward = opp.get("direction")
            for direction, nx, ny in valid_moves:
                base = 5 if direction == forward else 3
                self._add_threat(threat, (nx, ny), base)

                bdx, bdy = self.DIRS[direction]
                bx, by = nx + bdx, ny + bdy
                if 0 <= bx < cols and 0 <= by < rows and self._is_safe_cell(board[by][bx]):
                    self._add_threat(threat, (bx, by), 1)

        return threat

    def _predict_opponent_cells(self, opp, board):
        rows = len(board)
        cols = len(board[0])
        cells = []
        direction = opp.get("direction")
        if direction not in self.DIRS:
            return cells

        ox, oy = opp["pos"]
        dx, dy = self.DIRS[direction]
        first = (ox + dx, oy + dy)
        second = (ox + 2 * dx, oy + 2 * dy)

        if 0 <= first[0] < cols and 0 <= first[1] < rows and self._is_safe_cell(board[first[1]][first[0]]):
            cells.append((first, 8))
            if 0 <= second[0] < cols and 0 <= second[1] < rows and self._is_safe_cell(board[second[1]][second[0]]):
                cells.append((second, 4))

        return cells

    def _should_use_emp(self, candidate, opponents, map_name, board):
        if not opponents or map_name.startswith("s_"):
            return False

        dest = candidate["dest"]
        nearby = 0
        close = 0
        for opp in opponents:
            ox, oy = opp["pos"]
            cheb = max(abs(dest[0] - ox), abs(dest[1] - oy))
            if cheb <= self.EMP_RADIUS:
                nearby += 1
            if cheb <= 1:
                close += 1

        if nearby == 0:
            return False

        area = len(self._distance_map(board, dest, candidate["blocked"]))
        if area < 10:
            return False

        if close > 0:
            return True
        if len(opponents) == 1 and nearby >= 1:
            return True
        return nearby >= 2

    def _voronoi_score(self, board, my_pos, opp_pos, blocked):
        rows = len(board)
        cols = len(board[0])
        my_dist = {my_pos: 0}
        opp_dist = {opp_pos: 0}
        queue = deque([(my_pos, "me"), (opp_pos, "opp")])

        while queue:
            (x, y), owner = queue.popleft()
            dist = my_dist[(x, y)] if owner == "me" else opp_dist[(x, y)]

            for dx, dy in self.DIRS.values():
                nx, ny = x + dx, y + dy
                nxt = (nx, ny)
                if not (0 <= nx < cols and 0 <= ny < rows):
                    continue
                if nxt in blocked:
                    continue
                if nxt != my_pos and nxt != opp_pos and not self._is_safe_cell(board[ny][nx]):
                    continue

                if owner == "me":
                    if nxt in my_dist:
                        continue
                    my_dist[nxt] = dist + 1
                    queue.append((nxt, "me"))
                else:
                    if nxt in opp_dist:
                        continue
                    opp_dist[nxt] = dist + 1
                    queue.append((nxt, "opp"))

        my_count = 0
        opp_count = 0
        all_cells = set(my_dist) | set(opp_dist)
        for cell in all_cells:
            mine = my_dist.get(cell)
            theirs = opp_dist.get(cell)
            if mine is None:
                opp_count += 1
            elif theirs is None:
                my_count += 1
            elif mine < theirs:
                my_count += 1
            elif theirs < mine:
                opp_count += 1

        return my_count, opp_count

    def _fallback_move(self, board, pos, my_dir, phase_charges):
        for direction in self._ordered_dirs(my_dir):
            dx, dy = self.DIRS[direction]
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= ny < len(board) and 0 <= nx < len(board[0]) and self._is_safe_cell(board[ny][nx]):
                return direction

        if phase_charges > 0:
            phase = self._simulate_action(board, pos, my_dir, "P")
            if phase is not None:
                return "P"

        return my_dir if my_dir in self.DIRS else "N"

    def _cell_value(self, cell):
        if cell == "c":
            return 20
        if cell == "D":
            return 50
        return 0

    def _is_safe_cell(self, cell):
        return cell in (".", "c", "D")

    def _is_soon_safe(self, cell, distance):
        if not isinstance(cell, int):
            return False
        if cell <= 1:
            return distance >= 5
        if cell == 2:
            return distance >= 12
        return False

    def _plan_step_cost(self, cell, distance, allow_soon_safe):
        if self._is_safe_cell(cell):
            return 1
        if allow_soon_safe and self._is_soon_safe(cell, distance):
            return 6 + 2 * int(cell)
        return None

    def _exit_count(self, board, pos, blocked):
        rows = len(board)
        cols = len(board[0])
        count = 0
        for dx, dy in self.DIRS.values():
            nx, ny = pos[0] + dx, pos[1] + dy
            nxt = (nx, ny)
            if 0 <= nx < cols and 0 <= ny < rows and nxt not in blocked and self._is_safe_cell(board[ny][nx]):
                count += 1
        return count

    def _first_direction_toward(self, board, start, target, allow_soon_safe=False):
        path = self._path_to_target(board, start, target, allow_soon_safe=allow_soon_safe)
        if len(path) < 2:
            return None
        return self._step_direction(path[0], path[1])

    def _scan_opening_direction(self, board, pos):
        rows = len(board)
        cols = len(board[0])
        best_dir = None
        best_density = -1

        for direction, (dx, dy) in self.DIRS.items():
            density = 0.0
            cx, cy = pos
            for step in range(1, 11):
                nx = cx + dx * step
                ny = cy + dy * step
                if not (0 <= nx < cols and 0 <= ny < rows):
                    break
                cell = board[ny][nx]
                if cell == "#" or (isinstance(cell, str) and cell.startswith("t")):
                    break
                density += self._cell_value(cell)
                if isinstance(cell, int):
                    density -= 2.0 * int(cell)
            if density > best_density:
                best_density = density
                best_dir = direction

        return best_dir

    def _add_threat(self, threat, cell, value):
        threat[cell] = threat.get(cell, 0) + value
