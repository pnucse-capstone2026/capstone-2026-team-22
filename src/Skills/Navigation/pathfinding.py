"""LocalMap 격자 A* 와 통행 비용."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from core.map_types import BlockType, EntityType
from core.portal import LocalMap

Coord = Tuple[int, int]

# LocalMap.to_matrix("entity") 코드
ENTITY_NPC = 1
ENTITY_OBJECT = 2
ENTITY_CUT_TREE = 3


@dataclass
class TraversalCosts:
    """A*에서 사용하는 지형별 이동 비용."""

    free: int = 1
    grass: int = 1
    water: int = 1
    warp: int = 1
    cut_tree: int = 1
    ledge: int = 1

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name, value in self.as_dict().items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} 비용은 int여야 합니다: {value!r}")
            if value < 0:
                raise ValueError(f"{name} 비용은 0 이상이어야 합니다: {value}")

    def as_dict(self) -> Dict[str, int]:
        return {
            "free": self.free,
            "grass": self.grass,
            "water": self.water,
            "warp": self.warp,
            "cut_tree": self.cut_tree,
            "ledge": self.ledge,
        }


@dataclass
class PathResult:
    """A* 결과 좌표와 누적 비용."""

    path: List[Coord]
    cost: int


def face_action(from_pos: Coord, toward: Coord) -> Optional[str]:
    dx = int(toward[0]) - int(from_pos[0])
    dy = int(toward[1]) - int(from_pos[1])
    if abs(dx) > abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    if dy != 0:
        return "DOWN" if dy > 0 else "UP"
    return None


def is_standable_cell(local_map: LocalMap, x: int, y: int) -> bool:
    cell = local_map.get_cell(x, y)
    if cell is None:
        return False
    if cell.block_type is BlockType.WALL:
        return False
    if cell.entity in (EntityType.NPC, EntityType.OBJECT, EntityType.CUT_TREE):
        return False
    return True


def inside(index: Tuple[int, int], rows: int, cols: int) -> bool:
    return 0 <= index[0] < rows and 0 <= index[1] < cols


def ledge_landing(
    ledge_index: Tuple[int, int],
    dr: int,
    dc: int,
    cell_val: int,
) -> Optional[Tuple[int, int]]:
    if cell_val == LocalMap.MAT_LEDGE_D and dr == 1 and dc == 0:
        return ledge_index[0] + 1, ledge_index[1]
    if cell_val == LocalMap.MAT_LEDGE_U and dr == -1 and dc == 0:
        return ledge_index[0] - 1, ledge_index[1]
    if cell_val == LocalMap.MAT_LEDGE_L and dr == 0 and dc == -1:
        return ledge_index[0], ledge_index[1] - 1
    if cell_val == LocalMap.MAT_LEDGE_R and dr == 0 and dc == 1:
        return ledge_index[0], ledge_index[1] + 1
    return None


# 길을 막은 오브젝트(몬스터볼 등)를 "주워서 치울 수 있다" 고 볼 때의 추가 비용.
OBJECT_PICKUP_COST = 50


def normal_step_cost(
    costs: TraversalCosts,
    cell_val: int,
    entity_val: int,
    *,
    is_surfing: bool,
    can_cut: bool,
    objects_passable: bool = False,
) -> Optional[int]:
    """일반 한 칸의 통행 가능 여부와 비용."""
    if cell_val == LocalMap.MAT_WARP:
        return costs.warp
    if entity_val == ENTITY_OBJECT and objects_passable:
        return costs.free + OBJECT_PICKUP_COST
    if entity_val in (ENTITY_NPC, ENTITY_OBJECT):
        return None
    if entity_val == ENTITY_CUT_TREE:
        return costs.cut_tree if can_cut else None
    if cell_val == LocalMap.MAT_FREE:
        return costs.free
    if cell_val == LocalMap.MAT_GRASS:
        return costs.grass
    if cell_val == LocalMap.MAT_WATER:
        return costs.water if is_surfing else None
    return None


def minimum_enabled_cost(
    costs: TraversalCosts,
    *,
    is_surfing: bool,
    can_cut: bool,
) -> int:
    """현재 통행 가능한 이동 중 가장 작은 비용."""
    enabled = [costs.free, costs.grass, costs.warp, costs.ledge]
    if is_surfing:
        enabled.append(costs.water)
    if can_cut:
        enabled.append(costs.cut_tree)
    return min(enabled)


def coords_to_actions(start_pos: Coord, path: List[Coord]) -> List[str]:
    full_path = [start_pos] + path
    actions: List[str] = []
    for current, next_pos in zip(full_path, full_path[1:]):
        current_x, current_y = current
        next_x, next_y = next_pos
        dx = next_x - current_x
        dy = next_y - current_y
        if dx > 0 and dy == 0:
            actions.append("RIGHT")
        elif dx < 0 and dy == 0:
            actions.append("LEFT")
        elif dy > 0 and dx == 0:
            actions.append("DOWN")
        elif dy < 0 and dx == 0:
            actions.append("UP")
        else:
            raise ValueError(
                f"A* 경로에 대각선/비정상 이동이 있습니다: "
                f"{current} -> {next_pos}"
            )
    return actions


def a_star(
    local_map: LocalMap,
    start: Coord,
    goals: List[Coord],
    *,
    costs: TraversalCosts,
    is_surfing: bool = False,
    can_cut: bool = False,
    force_goal_passable: bool = False,
    on_debug=None,
    objects_passable: bool = False,
) -> Optional[PathResult]:
    """LocalMap A*."""
    matrix, (origin_x, origin_y) = local_map.to_matrix("block")
    entity_matrix, _ = local_map.to_matrix("entity")

    if matrix.size == 0:
        return None

    matrix = matrix.copy()
    entity_matrix = entity_matrix.copy()
    rows, cols = matrix.shape

    # (``warp_candidate``)은 block_type 이 바닥 그대로라, 지나가는 길로 쓰이면 밟는 순간 다른 층으로 넘어간다 — 동굴에서 1F 를 다 보기 전에 B1F 로…
    for (x, y), cell in (getattr(local_map, "grid", None) or {}).items():
        if getattr(cell, "warp_candidate", False):
            idx = (y - origin_y, x - origin_x)
            if inside(idx, rows, cols):
                matrix[idx] = LocalMap.MAT_WARP

    if on_debug is not None:
        on_debug(matrix, entity_matrix, origin_x, origin_y)

    def to_idx(pos: Coord) -> Tuple[int, int]:
        return pos[1] - origin_y, pos[0] - origin_x

    def to_pos(index: Tuple[int, int]) -> Coord:
        return index[1] + origin_x, index[0] + origin_y

    start_idx = to_idx(start)
    valid_goals = {
        to_idx(goal) for goal in goals if inside(to_idx(goal), rows, cols)
    }

    if not inside(start_idx, rows, cols) or not valid_goals:
        return None

    if force_goal_passable:
        for goal_idx in valid_goals:
            matrix[goal_idx] = LocalMap.MAT_WARP
            entity_matrix[goal_idx] = 0

    if start_idx in valid_goals:
        return PathResult(path=[], cost=0)

    min_step_cost = minimum_enabled_cost(
        costs, is_surfing=is_surfing, can_cut=can_cut
    )

    def heuristic(index: Tuple[int, int]) -> int:
        distance = min(
            abs(index[0] - goal[0]) + abs(index[1] - goal[1])
            for goal in valid_goals
        )
        return ((distance + 1) // 2) * min_step_cost

    queue: List[Tuple[int, int, int, Tuple[int, int]]] = []
    sequence = 0
    heapq.heappush(queue, (heuristic(start_idx), 0, sequence, start_idx))

    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    cost_so_far: Dict[Tuple[int, int], int] = {start_idx: 0}
    reached_goal: Optional[Tuple[int, int]] = None
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        _, queued_cost, _, current = heapq.heappop(queue)
        current_cost = cost_so_far.get(current)
        if current_cost is None or queued_cost != current_cost:
            continue
        if current in valid_goals:
            reached_goal = current
            break

        for dr, dc in directions:
            first_neighbor = (current[0] + dr, current[1] + dc)
            if not inside(first_neighbor, rows, cols):
                continue

            neighbor = first_neighbor
            current_val = int(matrix[current])
            cell_val = int(matrix[first_neighbor])
            entity_val = int(entity_matrix[first_neighbor])

            if cell_val == LocalMap.MAT_WARP:
                if first_neighbor not in valid_goals:
                    continue
                if current_val == LocalMap.MAT_WARP:
                    continue

            step_cost = normal_step_cost(
                costs,
                cell_val,
                entity_val,
                is_surfing=is_surfing,
                can_cut=can_cut,
                objects_passable=objects_passable,
            )
            if step_cost is None:
                jump_neighbor = ledge_landing(
                    first_neighbor, dr, dc, cell_val
                )
                if jump_neighbor is None or not inside(jump_neighbor, rows, cols):
                    continue
                landing_cost = normal_step_cost(
                    costs,
                    int(matrix[jump_neighbor]),
                    int(entity_matrix[jump_neighbor]),
                    is_surfing=is_surfing,
                    can_cut=can_cut,
                )
                if landing_cost is None:
                    continue
                neighbor = jump_neighbor
                step_cost = costs.ledge

            new_cost = current_cost + step_cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                sequence += 1
                heapq.heappush(
                    queue,
                    (new_cost + heuristic(neighbor), new_cost, sequence, neighbor),
                )

    if reached_goal is None:
        return None

    path_indices: List[Tuple[int, int]] = []
    current = reached_goal
    while current != start_idx:
        path_indices.append(current)
        previous = came_from.get(current)
        if previous is None:
            return None
        current = previous

    path_indices.reverse()
    return PathResult(
        path=[to_pos(index) for index in path_indices],
        cost=cost_so_far[reached_goal],
    )
