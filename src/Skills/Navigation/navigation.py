"""Skills.Navigation — Portal 기반 Global A* + LocalMap A*."""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import networkx as nx

from core.map_types import BlockType, EntityType
from core.portal import (
    ROLE_ENTRY,
    ROLE_EXIT,
    GlobalMap,
    LocalMap,
    PortalKey,
)

from .pathfinding import (
    ENTITY_CUT_TREE,
    ENTITY_NPC,
    ENTITY_OBJECT,
    PathResult,
    TraversalCosts,
    a_star as grid_a_star,
    coords_to_actions,
    face_action,
    is_standable_cell,
    normal_step_cost,
)

Coord = Tuple[int, int]

# 경로 계산 과정(행렬 덤프·경로 시각화)을 터미널에 찍을지.
debug_flags: dict[str, bool] = {
    "matrix": False,
}


@dataclass
class NavStep:
    """고수준 Navigation 미션의 한 단계."""

    kind: str
    map_id: Optional[int] = None
    npc_name: Optional[str] = None


@dataclass
class NavMission:
    """맵 이동 + Local 목표를 묶은 Navigation 미션."""

    steps: List[NavStep]
    summary: str


class Navigation:
    """Portal 그래프와 LocalMap A*를 결합한 경로 계획기."""

    # 길을 막은 오브젝트(몬스터볼)를 치울 때: 누를 A 횟수 · 같은 것을 시도할 최대 횟수
    OBJECT_PICKUP_A_PRESSES = 6
    OBJECT_PICKUP_MAX_TRIES = 2
    # 오브젝트 앞까지 **걸어가는** 계획의 상한.
    OBJECT_PICKUP_MAX_WALKS = 8

    START_NODE = "__NAV_START__"
    END_NODE = "__NAV_END__"
    # 맵 간 hop 1회 ≈ 로컬 맨해튼보다 항상 크게 (같은 맵 목표 우선).
    _MAP_HOP_PENALTY = 1000
    _GRASS_MAP_HOP_PENALTY = _MAP_HOP_PENALTY

    def __init__(
        self,
        global_map: GlobalMap,
        costs: Optional[TraversalCosts] = None,
    ):
        self.global_map = global_map
        self.costs = costs if costs is not None else TraversalCosts()
        self.costs.validate()

        # 실제 경로. PortalKey를 그대로 저장한다.
        self.portal_queue: Deque[PortalKey] = deque()

        # portal_queue에서 "다음에 실제로 통과할 EXIT"만 뽑아 둔 큐.
        self.gate_queue: Deque[PortalKey] = deque()

        self.action_queue: Deque[str] = deque()
        # 길을 막은 오브젝트를 치우려고 시도한 횟수 (맵, 좌표) -> 횟수
        self._object_tries: Dict[Tuple[int, Coord], int] = {}
        # 그 오브젝트 앞까지 걸어가려고 계획한 횟수 (맵, 좌표) -> 횟수
        self._object_walks: Dict[Tuple[int, Coord], int] = {}

        self.target_map_id: Optional[int] = None
        self._active_map_id: Optional[int] = None

        # 마지막 전역 경로 탐색에 사용한 이동 능력.
        self.is_surfing: bool = False
        self.can_cut: bool = False

        # 같은 맵 Portal 간 A* 생성 과정을 보고 싶을 때 True로 설정한다.
        self.debug_internal_edges: bool = False

    def set_internal_edge_debug(self, enabled: bool = True) -> None:
        """Portal 내부 간선 생성/실패 로그를 켜거나 끈다."""
        self.debug_internal_edges = bool(enabled)

    # ==================================================================
    # 전역 Portal 경로
    # ==================================================================
    def set_global_destination(
        self,
        current_map_id: int,
        target_map_id: int,
        start_pos: Optional[Coord] = None,
        is_surfing: bool = False,
        can_cut: bool = False,
    ) -> bool:
        """현재 위치에서 목표 map_id까지 Portal 단위 경로를 계산한다."""
        current_map_id = int(current_map_id)
        target_map_id = int(target_map_id)

        current_name = self.global_map.map_name(current_map_id)
        target_name = self.global_map.map_name(target_map_id)

        print(
            f"\n[Global] Portal 경로 탐색 요청: "
            f"{current_name}(ID:{current_map_id}) -> "
            f"{target_name}(ID:{target_map_id})"
        )

        self._clear_route()
        self.target_map_id = target_map_id
        self._active_map_id = current_map_id
        self.is_surfing = bool(is_surfing)
        self.can_cut = bool(can_cut)

        if current_map_id == target_map_id:
            print("[Global] 이미 목표 맵에 있습니다.")
            return True

        # 각 LocalMap 안에서 ENTRY → EXIT 이동 가능성을 방향별 A*로 갱신.
        self.update_all_internal_edges(
            is_surfing=is_surfing,
            can_cut=can_cut,
        )

        gate_graph = self.global_map.gate_graph

        # 현재 위치에서는 실제로 나갈 수 있는 EXIT만 START 후보가 된다.
        start_portals = self._exit_portal_nodes(current_map_id)
        target_portals = self._entry_portal_nodes(target_map_id)

        if not start_portals:
            print("[Global] 현재 맵에 등록된 EXIT Portal이 없습니다.")
            return False
        if not target_portals:
            print("[Global] 목표 맵에 등록된 ENTRY Portal이 없습니다.")
            return False

        # START/END는 탐색 중에만 잠시 사용하는 가상 노드다.
        self._remove_virtual_nodes(gate_graph)
        gate_graph.add_node(self.START_NODE)
        gate_graph.add_node(self.END_NODE)

        connected_start_count = 0

        try:
            local_map = self.global_map.get_local_map(current_map_id)

            for portal_node in start_portals:
                self._ensure_portal_node(portal_node)
                raw_portal_coords = self.global_map.portal_coords(portal_node)

                if not raw_portal_coords:
                    continue

                # 경계 밖 warp 좌표는 맵 안쪽 activation 좌표로 바꿔 A*를 계산한다.
                portal_coords = (
                    self._portal_activation_coords(portal_node, local_map)
                    if local_map is not None
                    else list(raw_portal_coords)
                )
                if not portal_coords:
                    continue

                # start_pos가 없으면 하위 호환 모드로 모든 Portal을 후보로 둔다.
                if start_pos is None:
                    result: Optional[PathResult] = PathResult([], 0)
                elif local_map is None:
                    result = None
                else:
                    result = self._a_star(
                        local_map,
                        start_pos,
                        portal_coords,
                        is_surfing=is_surfing,
                        can_cut=can_cut,
                        debug=False,
                        force_goal_passable=True,
                    )

                if result is None:
                    continue

                gate_graph.add_edge(
                    self.START_NODE,
                    portal_node,
                    weight=result.cost,
                    edge_type="start",
                    raw_portal_coords=tuple(raw_portal_coords),
                    activation_coords=tuple(portal_coords),
                    costs=self.costs.as_dict(),
                    is_surfing=bool(is_surfing),
                    can_cut=bool(can_cut),
                )
                connected_start_count += 1

            if connected_start_count == 0:
                print("[Global] 현재 위치에서 도달 가능한 Portal이 없습니다.")
                return False

            for portal_node in target_portals:
                self._ensure_portal_node(portal_node)
                gate_graph.add_edge(
                    portal_node,
                    self.END_NODE,
                    weight=0,
                    edge_type="end",
                )

            try:
                full_path = self._global_a_star_path(
                    gate_graph,
                    current_map_id=current_map_id,
                    target_map_id=target_map_id,
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                print(
                    "[Global] Portal 탐색 실패: 관측된 Portal 간에 "
                    "물리적으로 연결된 경로가 없습니다."
                )
                if self.debug_internal_edges:
                    print(
                        "[Global] 시작 연결 Portal 수:",
                        connected_start_count,
                    )
                    print(
                        "[Global] 현재 맵 Portal:",
                        [
                            self._portal_text(node)
                            for node in start_portals
                        ],
                    )
                    print(
                        "[Global] 목표 맵 Portal:",
                        [
                            self._portal_text(node)
                            for node in target_portals
                        ],
                    )
                return False

            portal_path = [
                node
                for node in full_path
                if node not in (self.START_NODE, self.END_NODE)
            ]

            if not portal_path:
                print("[Global] 사용할 Portal 경로가 없습니다.")
                return False

            self.portal_queue = deque(portal_path)
            self.action_queue.clear()
            self._refresh_gate_queue()

            route_map_ids = self._collapsed_map_route(portal_path)
            route_names = [
                self.global_map.map_name(map_id)
                for map_id in route_map_ids
            ]

            print(
                "[Global] Portal 탐색 성공! 전체 맵 경로: "
                + " -> ".join(route_names)
            )
            print("[Global] Portal 경로:")
            for index, node in enumerate(portal_path, start=1):
                print(f"  {index}. {self._portal_text(node)}")

            return True

        finally:
            self._remove_virtual_nodes(gate_graph)

    def _global_a_star_path(
        self,
        graph: nx.DiGraph,
        *,
        current_map_id: int,
        target_map_id: int,
    ) -> List[object]:
        """Portal 그래프에서 START부터 END까지 Global A*를 수행한다."""
        heuristic = self._build_global_heuristic(
            graph,
            current_map_id=int(current_map_id),
            target_map_id=int(target_map_id),
        )

        return list(
            nx.astar_path(
                graph,
                self.START_NODE,
                self.END_NODE,
                heuristic=heuristic,
                weight="weight",
            )
        )

    def _build_global_heuristic(
        self,
        graph: nx.DiGraph,
        *,
        current_map_id: int,
        target_map_id: int,
    ):
        """NetworkX A*가 사용할 admissible/consistent 휴리스틱 생성."""
        map_hops = self._map_hop_distances_to_target(target_map_id)
        min_transition_cost = self._minimum_global_transition_cost(graph)

        def heuristic(node: object, _goal: object) -> float:
            if node == self.END_NODE:
                return 0.0

            if node == self.START_NODE:
                node_map_id = int(current_map_id)
            elif self._is_portal_key(node):
                node_map_id = int(node[0])
            else:
                # 탐색 중 임시 노드나 알 수 없는 노드는 안전하게 0.
                return 0.0

            if node_map_id == int(target_map_id):
                return 0.0

            remaining_hops = map_hops.get(node_map_id)
            if remaining_hops is None:
                # map graph에 정보가 없다고 경로가 없다고 단정하면 안 된다.
                return 0.0

            return float(remaining_hops) * min_transition_cost

        return heuristic

    def _map_hop_distances_to_target(
        self,
        target_map_id: int,
    ) -> Dict[int, int]:
        """각 map_id에서 목표 map까지의 최소 방향성 hop 수."""
        target_map_id = int(target_map_id)
        map_graph = getattr(self.global_map, "graph", None)

        if map_graph is None or target_map_id not in map_graph:
            return {target_map_id: 0}

        try:
            reversed_graph = map_graph.reverse(copy=False)
            lengths = nx.single_source_shortest_path_length(
                reversed_graph,
                target_map_id,
            )
        except (nx.NetworkXError, AttributeError, TypeError):
            return {target_map_id: 0}

        result: Dict[int, int] = {target_map_id: 0}
        for map_id, distance in lengths.items():
            try:
                result[int(map_id)] = int(distance)
            except (TypeError, ValueError):
                continue
        return result

    @staticmethod
    def _minimum_global_transition_cost(graph: nx.DiGraph) -> float:
        """관측된 map 간 transition 간선 비용의 최솟값."""
        minimum: Optional[float] = None

        for _, _, data in graph.edges(data=True):
            if data.get("edge_type") != "transition":
                continue

            raw_weight = data.get("weight", 1)
            try:
                cost = float(raw_weight)
            except (TypeError, ValueError):
                continue

            if cost < 0:
                raise ValueError(
                    "Global A*는 음수 transition 비용을 지원하지 않습니다: "
                    f"{raw_weight!r}"
                )

            if minimum is None or cost < minimum:
                minimum = cost

        return 0.0 if minimum is None else minimum

    def update_all_internal_edges(
        self,
        is_surfing: bool = False,
        can_cut: bool = False,
    ) -> None:
        """현재 능력으로 모든 LocalMap의 내부 Portal 간선을 갱신."""
        try:
            map_ids = list(self.global_map.graph.nodes)
        except RuntimeError:
            map_ids = []

        for map_id in map_ids:
            self.update_internal_edges(
                int(map_id),
                is_surfing=is_surfing,
                can_cut=can_cut,
            )

    def _portal_activation_coords(
        self,
        portal_node: PortalKey,
        local_map: LocalMap,
    ) -> List[Coord]:
        """Portal을 작동시키기 위해 A*가 실제로 도달해야 할 맵 내부 좌표."""
        portal = self.global_map.get_portal(portal_node)
        if portal is None or not self._portal_has_role(portal_node, ROLE_EXIT):
            return []

        size = self.global_map.map_size_cells.get(int(portal.map_id))
        bounds = local_map.bounds()
        result: List[Coord] = []

        for raw_x, raw_y in portal.warp_coords:
            x, y = int(raw_x), int(raw_y)

            if size is not None:
                width, height = int(size[0]), int(size[1])
                if width <= 0 or height <= 0:
                    continue
                x = min(max(x, 0), width - 1)
                y = min(max(y, 0), height - 1)
            elif bounds is not None:
                min_x, min_y, max_x, max_y = bounds
                # 맵 밖임이 명확한 좌표만 관측 범위 안으로 접는다.
                if x < min_x:
                    x = min_x
                elif portal.direction == "EAST" and x > max_x:
                    x = max_x
                if y < min_y:
                    y = min_y
                elif portal.direction == "SOUTH" and y > max_y:
                    y = max_y

            result.append((x, y))

        return list(dict.fromkeys(result))

    def _portal_final_exit_action(
        self,
        portal_node: PortalKey,
        local_map: LocalMap,
    ) -> Optional[str]:
        """activation 좌표 도착 뒤 맵 밖으로 나가기 위한 마지막 입력."""
        portal = self.global_map.get_portal(portal_node)
        if portal is None or not self._portal_has_role(portal_node, ROLE_EXIT):
            return None

        size = self.global_map.map_size_cells.get(int(portal.map_id))
        if size is not None:
            width, height = int(size[0]), int(size[1])
            for x, y in portal.warp_coords:
                x, y = int(x), int(y)
                if y < 0:
                    return "UP"
                if y >= height:
                    return "DOWN"
                if x < 0:
                    return "LEFT"
                if x >= width:
                    return "RIGHT"
            return None

        # map_size 정보가 없는 경우에도 음수 경계는 확정 가능하다.
        for x, y in portal.warp_coords:
            if int(y) < 0:
                return "UP"
            if int(x) < 0:
                return "LEFT"

        bounds = local_map.bounds()
        if bounds is not None:
            min_x, min_y, max_x, max_y = bounds
            for x, y in portal.warp_coords:
                x, y = int(x), int(y)
                if portal.direction == "SOUTH" and y > max_y:
                    return "DOWN"
                if portal.direction == "EAST" and x > max_x:
                    return "RIGHT"

        return None

    def _portal_internal_start_coords(
        self,
        portal_node: PortalKey,
        local_map: LocalMap,
        *,
        is_surfing: bool,
        can_cut: bool,
    ) -> List[Coord]:
        """ENTRY에서 같은 맵 내부 이동을 시작할 실제 arrival 좌표."""
        portal = self.global_map.get_portal(portal_node)
        if portal is None or not self._portal_has_role(portal_node, ROLE_ENTRY):
            return []

        candidates = {
            (int(x), int(y))
            for x, y in (
                list(getattr(portal, "access_coords", ()) or ())
                + list(getattr(portal, "arrival_coords", ()) or ())
                + list(getattr(portal, "warp_coords", ()) or ())
            )
        }

        return sorted(
            coord
            for coord in candidates
            if self._coord_can_start_internal(
                local_map,
                coord,
                is_surfing=is_surfing,
                can_cut=can_cut,
                allow_warp=True,
            )
        )

    def _coord_can_start_internal(
        self,
        local_map: LocalMap,
        coord: Coord,
        *,
        is_surfing: bool,
        can_cut: bool,
        allow_warp: bool,
    ) -> bool:
        """좌표가 관측된 내부 출발점으로 사용 가능한지 검사한다."""
        cell = local_map.get_cell(int(coord[0]), int(coord[1]))
        if cell is None:
            return False

        cell_val = int(LocalMap.cell_to_mat(cell))
        if cell_val == LocalMap.MAT_WARP:
            return bool(allow_warp)

        entity_matrix, (origin_x, origin_y) = local_map.to_matrix("entity")
        row = int(coord[1]) - int(origin_y)
        col = int(coord[0]) - int(origin_x)
        if not (0 <= row < entity_matrix.shape[0] and 0 <= col < entity_matrix.shape[1]):
            return False

        entity_val = int(entity_matrix[row, col])
        return self._normal_step_cost(
            cell_val,
            entity_val,
            is_surfing=is_surfing,
            can_cut=can_cut,
        ) is not None

    def update_internal_edges(
        self,
        map_id: int,
        is_surfing: bool = False,
        can_cut: bool = False,
    ) -> None:
        """같은 LocalMap의 ENTRY → EXIT 내부 간선만 A*로 갱신한다."""
        map_id = int(map_id)
        local_map = self.global_map.get_local_map(map_id)
        if local_map is None:
            return

        entry_nodes = self._entry_portal_nodes(map_id)
        exit_nodes = self._exit_portal_nodes(map_id)
        gate_graph = self.global_map.gate_graph

        for portal_node in entry_nodes + exit_nodes:
            self._ensure_portal_node(portal_node)

        # 이전 A* 결과만 제거한다. 실제 EXIT → ENTRY transition은 보존한다.
        for source, destination, data in list(gate_graph.edges(data=True)):
            if data.get("edge_type") != "internal":
                continue
            if self._is_portal_key(source) and int(source[0]) == map_id:
                gate_graph.remove_edge(source, destination)

        if not entry_nodes or not exit_nodes:
            return

        # 방향성 언덕 때문에 각 ENTRY → EXIT를 독립적으로 검사한다.
        for source in entry_nodes:
            source_coords = self._portal_internal_start_coords(
                source,
                local_map,
                is_surfing=is_surfing,
                can_cut=can_cut,
            )
            if not source_coords:
                if self.debug_internal_edges:
                    print(
                        "[internal-debug] ENTRY 시작점 없음:",
                        self._portal_text(source),
                    )
                continue

            for destination in exit_nodes:
                destination_raw_coords = self.global_map.portal_coords(destination)
                if not destination_raw_coords:
                    continue

                destination_coords = self._portal_activation_coords(
                    destination,
                    local_map,
                )
                if not destination_coords:
                    continue

                shortest_result = self._shortest_portal_path(
                    local_map,
                    source_coords,
                    destination_coords,
                    is_surfing=is_surfing,
                    can_cut=can_cut,
                )

                if shortest_result is None:
                    if self.debug_internal_edges:
                        print(
                            "[internal-debug] 실패:",
                            self._portal_text(source),
                            "arrival=",
                            source_coords,
                            "->",
                            self._portal_text(destination),
                            "activation=",
                            destination_coords,
                            "raw_exit=",
                            destination_raw_coords,
                        )
                    continue

                gate_graph.add_edge(
                    source,
                    destination,
                    weight=shortest_result.cost,
                    edge_type="internal",
                    path_length=len(shortest_result.path),
                    path=tuple(shortest_result.path),
                    source_entry_coords=tuple(source_coords),
                    destination_exit_coords=tuple(destination_raw_coords),
                    destination_activation_coords=tuple(destination_coords),
                    destination_exit_action=self._portal_final_exit_action(
                        destination, local_map
                    ),
                    costs=self.costs.as_dict(),
                    is_surfing=bool(is_surfing),
                    can_cut=bool(can_cut),
                )

                if self.debug_internal_edges:
                    print(
                        "[internal-debug] 생성:",
                        self._portal_text(source),
                        "arrival=",
                        source_coords,
                        "->",
                        self._portal_text(destination),
                        "activation=",
                        destination_coords,
                        "raw_exit=",
                        destination_raw_coords,
                        f"cost={shortest_result.cost}",
                    )

    def notify_map_change(self, current_map_id: int) -> None:
        """실제 map_id가 바뀐 뒤 Portal 경로 앞부분을 현재 맵에 맞춘다."""
        self._sync_route_to_map(int(current_map_id))

    # ==================================================================
    # 현재 맵 Local A*
    # ==================================================================
    def plan_local_path_to_next_map(
        self,
        current_map_id: int,
        start_pos: Coord,
        is_surfing: Optional[bool] = None,
        can_cut: Optional[bool] = None,
    ) -> bool:
        """현재 위치에서 Portal 전역 경로가 지정한 다음 출구까지 계획."""
        current_map_id = int(current_map_id)

        if is_surfing is None:
            is_surfing = self.is_surfing
        else:
            self.is_surfing = bool(is_surfing)

        if can_cut is None:
            can_cut = self.can_cut
        else:
            self.can_cut = bool(can_cut)
        self._sync_route_to_map(current_map_id)

        if not self.portal_queue:
            return False

        target_portal = self._current_map_exit_portal(current_map_id)
        if target_portal is None:
            print("[Local] 현재 맵에서 사용할 Portal을 찾지 못했습니다.")
            return False

        raw_target_positions = self.global_map.portal_coords(target_portal)
        if not raw_target_positions:
            print(f"[Local] Portal 좌표가 없습니다: {target_portal}")
            return False

        local_map = self.global_map.get_local_map(current_map_id)
        if local_map is None:
            print("[Local] 현재 LocalMap이 없습니다.")
            return False

        target_positions = self._portal_activation_coords(
            target_portal,
            local_map,
        )
        if not target_positions:
            print(f"[Local] Portal activation 좌표가 없습니다: {target_portal}")
            return False
        final_exit_action = self._portal_final_exit_action(
            target_portal,
            local_map,
        )

        next_map_id = self._next_map_after_current(current_map_id)
        current_name = self.global_map.map_name(current_map_id)
        next_name = (
            self.global_map.map_name(next_map_id)
            if next_map_id is not None
            else self.global_map.map_name(current_map_id)
        )

        print(
            f"\n[Local] Portal 경로 탐색: {current_name} -> {next_name}"
        )
        print(f"[Local] 목표 Portal: {self._portal_text(target_portal)}")
        print(
            f"[Local] activation={target_positions}, "
            f"raw_warp={raw_target_positions}, "
            f"final_action={final_exit_action}"
        )

        result = self._a_star(
            local_map,
            start_pos,
            target_positions,
            is_surfing=bool(is_surfing),
            can_cut=bool(can_cut),
            debug=True,
            force_goal_passable=True,
        )

        if result is None:
            # 길이 없다 — 몬스터볼 같은 오브젝트가 막고 있는지 본다
            if self.plan_through_blocking_object(
                current_map_id,
                start_pos,
                list(target_positions),
                is_surfing=bool(is_surfing),
                can_cut=bool(can_cut),
            ):
                return True
            print(
                f"[Local] A* 길찾기 실패! "
                f"({start_pos} -> {target_positions})"
            )
            return False

        path_coords = result.path
        self.action_queue.clear()
        actions = self._coords_to_actions(start_pos, path_coords)
        if final_exit_action is not None:
            actions.append(final_exit_action)
        self.action_queue.extend(actions)

        print(
            f"[Local] A* 성공! 비용 {result.cost}, "
            f"좌표 {len(path_coords)}개, 액션 {len(actions)}개 생성"
        )
        if debug_flags.get("matrix", False):
            self.debug_print_grid(
                local_map,
                start_pos,
                target_positions,
                path_coords,
            )
        return True

    def plan_local_path_to_coords(
        self,
        current_map_id: int,
        start_pos: Coord,
        goal_coords: List[Coord],
        *,
        label: str = "local",
        is_surfing: Optional[bool] = None,
        can_cut: Optional[bool] = None,
        force_goal_passable: bool = True,
        face_toward: Optional[Coord] = None,
        mash_a: int = 0,
    ) -> bool:
        """현재 맵에서 목표 좌표(들)까지 Local A*."""
        current_map_id = int(current_map_id)
        if is_surfing is None:
            is_surfing = self.is_surfing
        if can_cut is None:
            can_cut = self.can_cut

        local_map = self.global_map.get_local_map(current_map_id)
        if local_map is None:
            print("[Local] 현재 LocalMap이 없습니다.")
            return False

        goals = [(int(x), int(y)) for x, y in goal_coords]
        if not goals:
            print(f"[Local] {label}: 목표 좌표 없음")
            return False

        print(f"\n[Local] {label}: {start_pos} -> {goals}")
        result = self._a_star(
            local_map,
            start_pos,
            goals,
            is_surfing=bool(is_surfing),
            can_cut=bool(can_cut),
            debug=True,
            force_goal_passable=force_goal_passable,
        )
        if result is None:
            # 길이 없다 — 몬스터볼 같은 오브젝트가 막고 있는지 본다
            if self.plan_through_blocking_object(
                current_map_id,
                start_pos,
                goals,
                is_surfing=bool(is_surfing),
                can_cut=bool(can_cut),
            ):
                return True
            print(f"[Local] {label} A* 실패 ({start_pos} -> {goals})")
            return False

        path_coords = result.path
        self.action_queue.clear()
        actions = self._coords_to_actions(start_pos, path_coords)

        stand = path_coords[-1] if path_coords else start_pos
        if face_toward is not None:
            face = self._face_action(stand, face_toward)
            if face:
                actions.append(face)
        for _ in range(max(0, int(mash_a))):
            actions.append("A")

        self.action_queue.extend(actions)
        print(
            f"[Local] {label} A* 성공! 비용 {result.cost}, "
            f"액션 {len(actions)}개 (A×{mash_a})"
        )
        return True

    def plan_through_blocking_object(
        self,
        current_map_id: int,
        start_pos: Coord,
        goal_coords: List[Coord],
        *,
        is_surfing: bool,
        can_cut: bool,
        final_action: Optional[str] = None,
    ) -> bool:
        """길이 없을 때만: **몬스터볼 같은 오브젝트가 길을 막고 있으면** 주워서 치운다."""
        from core.config import cheat
        if not cheat("blocking_object_interact", True):
            return False                    # cheats.blocking_object_interact
        current_map_id = int(current_map_id)
        local_map = self.global_map.get_local_map(current_map_id)
        if local_map is None:
            return False
        result = self._a_star(
            local_map,
            start_pos,
            goal_coords,
            is_surfing=bool(is_surfing),
            can_cut=bool(can_cut),
            force_goal_passable=True,
            objects_passable=True,
        )
        if result is None or not result.path:
            return False

        prev = tuple(start_pos)
        blocker: Optional[Coord] = None
        stand = prev
        for coord in result.path:
            cell = local_map.grid.get(tuple(coord))
            if cell is not None and getattr(cell, "entity", None) is EntityType.OBJECT:
                blocker = tuple(coord)
                stand = prev
                break
            prev = tuple(coord)
        if blocker is None:
            return False

        key = (current_map_id, blocker)
        tries = self._object_tries.get(key, 0)
        if tries >= self.OBJECT_PICKUP_MAX_TRIES:
            return False
        adjacent = tuple(start_pos) == tuple(stand)
        if not adjacent:
            walks = self._object_walks.get(key, 0)
            if walks >= self.OBJECT_PICKUP_MAX_WALKS:
                return False                # 몇 번을 가도 앞에 못 섰다
            self._object_walks[key] = walks + 1
        if adjacent:
            # **바로 앞에 서서 누를 때만 한 번으로 센다.** 걸어가는 도중에 전투로 끊기면
            # A 를 누르지도 못했는데 횟수만 올라가, 타워처럼 전투가 잦은 곳에서는 볼을
            # 한 번도 못 줍고 포기했다 (포켓몬 타워 6층).
            self._object_tries[key] = tries + 1
            if tries >= 1:
                # 앞에서 한 번 눌렀는데 그대로 있다 = 줍거나 치울 수 없는 오브젝트다.
                self._remember_blocking_object(local_map, current_map_id, blocker)

        walk = result.path[: result.path.index(blocker)]
        actions = self._coords_to_actions(tuple(start_pos), walk) if walk else []
        face = self._face_action(stand, blocker)
        if face:
            actions.append(face)
        actions.extend(["A"] * self.OBJECT_PICKUP_A_PRESSES)
        self.action_queue.clear()
        self.action_queue.extend(actions)
        print(
            f"[Local] 길을 막은 오브젝트 {blocker} — 앞({stand})에서 A 로 치웁니다 "
            f"({self._object_tries.get(key, 0) + (0 if adjacent else 1)}/{self.OBJECT_PICKUP_MAX_TRIES})"
        )
        return True

    def _remember_blocking_object(self, local_map: Any, map_id: int, coord: Coord) -> None:
        """상호작용해도 안 비키는 오브젝트를 **사람처럼 이름을 붙여** 남긴다."""
        from core.config import cheat
        if not cheat("blocking_object_record", True):
            return                          # cheats.blocking_object_record
        names = (getattr(self.global_map, "npc_object_names", None) or {}).get(int(map_id)) or {}
        name = names.get(tuple(coord))
        if not name or not hasattr(local_map, "blocking_objects"):
            return
        if local_map.blocking_objects.get(tuple(coord)) == name:
            return
        local_map.blocking_objects[tuple(coord)] = name
        local_map.set_npc_name(coord[0], coord[1], name)
        print(f"[Local] {name} {tuple(coord)} 는 상호작용해도 비키지 않습니다 — 길을 막는 오브젝트로 남깁니다")
        try:
            from core import runlog
            runlog.log("blocking_object", map_id=int(map_id), 이름=name, 좌표=list(coord))
        except Exception:
            pass

    def forget_object_tries(self) -> None:
        """오브젝트 치우기 시도 횟수를 지운다 (새 이벤트 — 이제는 비킬 수 있다)."""
        self._object_tries.clear()
        self._object_walks.clear()

    def plan_local_path_to_npc(
        self,
        current_map_id: int,
        start_pos: Coord,
        npc_name: str,
        *,
        mash_a: int = 12,
        is_surfing: Optional[bool] = None,
        can_cut: Optional[bool] = None,
    ) -> bool:
        """NPC 이름 → 인접 칸까지 이동 후 바라보고 A 연타."""
        local_map = self.global_map.get_local_map(int(current_map_id))
        if local_map is None:
            print("[Local] LocalMap 없음")
            return False
        npc_cells = local_map.npc_coords(npc_name)
        if not npc_cells:
            print(f"[Local] NPC '{npc_name}' 좌표 없음 (npc 명령으로 목록 확인)")
            return False

        # 시작에서 가장 가까운 NPC 칸
        tx, ty = min(
            npc_cells,
            key=lambda c: abs(c[0] - start_pos[0]) + abs(c[1] - start_pos[1]),
        )
        # 배경 오브젝트(자판기·표지판 — hidden_objects.json)는 바로 앞에서만 조사된다
        bg_objects = (getattr(self.global_map, "hidden_objects", None) or {}).get(int(current_map_id)) or {}
        approach = self._npc_interaction_stand_tiles(
            local_map, (tx, ty), across_counter=(tx, ty) not in bg_objects,
        )
        if start_pos == (tx, ty):
            # 같은 칸이면 사방 중 하나(이미 서 있으면 face만)
            approach = [start_pos]
        if not approach:
            print(f"[Local] NPC ({tx},{ty}) 대화 가능 칸 없음")
            return False

        # 시작에서 가까운 대화 칸 우선 (카운터 앞 등)
        approach = sorted(
            approach,
            key=lambda c: abs(c[0] - start_pos[0]) + abs(c[1] - start_pos[1]),
        )
        print(
            f"[Local] NPC ({tx},{ty}) 대화 후보: {approach}"
        )

        return self.plan_local_path_to_coords(
            current_map_id,
            start_pos,
            approach,
            label=f"npc:{npc_name}",
            is_surfing=is_surfing,
            can_cut=can_cut,
            force_goal_passable=True,
            face_toward=(tx, ty),
            mash_a=mash_a,
        )

    def find_nearest_observed_grass(
        self,
        current_map_id: int,
        start_pos: Coord,
    ) -> Optional[Tuple[int, Coord, float]]:
        """Known GlobalMap 전체에서 최근접 관측 GRASS."""
        cur = int(current_map_id)
        sx, sy = int(start_pos[0]), int(start_pos[1])
        best: Optional[Tuple[float, int, Coord]] = None

        for mid in self.global_map.graph.nodes:
            mid_i = int(mid)
            local_map = self.global_map.get_local_map(mid_i)
            if local_map is None:
                continue
            grasses = local_map.grass_coords()
            if not grasses:
                continue

            if mid_i == cur:
                goal = min(
                    grasses,
                    key=lambda c: abs(c[0] - sx) + abs(c[1] - sy),
                )
                score = float(abs(goal[0] - sx) + abs(goal[1] - sy))
            else:
                route = self.global_map.get_route(cur, mid_i)
                if route is False:
                    continue
                hops = max(0, len(route) - 1)
                bounds = local_map.bounds()
                if bounds is not None:
                    cx = (bounds[0] + bounds[2]) // 2
                    cy = (bounds[1] + bounds[3]) // 2
                    goal = min(
                        grasses,
                        key=lambda c: abs(c[0] - cx) + abs(c[1] - cy),
                    )
                    local_est = float(abs(goal[0] - cx) + abs(goal[1] - cy))
                else:
                    goal = grasses[0]
                    local_est = 0.0
                score = float(self._MAP_HOP_PENALTY * hops) + local_est

            if best is None or score < best[0]:
                best = (score, mid_i, goal)

        if best is None:
            return None
        score, map_id, goal = best
        return map_id, goal, score

    @staticmethod
    def is_pokecenter_map_name(name: str) -> bool:
        n = (name or "").lower().replace("_", " ")
        return "pokemon center" in n or "pokecenter" in n

    @staticmethod
    def _is_nurse_npc_name(name: str) -> bool:
        u = (name or "").upper()
        return u.endswith("_NURSE") or "POKECENTER_NURSE" in u

    def resolve_pokecenter_nurse_name(self, map_id: int) -> Optional[str]:
        """관측 NPC 우선, 없으면 npc_objects 정적 const."""
        mid = int(map_id)
        local_map = self.global_map.get_local_map(mid)
        if local_map is not None:
            for name in local_map.list_npc_names():
                if self._is_nurse_npc_name(name):
                    return name
        static = getattr(self.global_map, "npc_object_names", {}) or {}
        for name in static.get(mid, {}).values():
            if self._is_nurse_npc_name(str(name)):
                return str(name)
        return None

    def find_nearest_pokecenter(
        self,
        current_map_id: int,
        start_pos: Coord,
    ) -> Optional[Tuple[int, str, float]]:
        """Known GlobalMap에서 최근접 Pokemon Center + Nurse 이름."""
        cur = int(current_map_id)
        sx, sy = int(start_pos[0]), int(start_pos[1])
        best: Optional[Tuple[float, int, str]] = None

        for mid in self.global_map.graph.nodes:
            mid_i = int(mid)
            name = self.global_map.map_name(mid_i)
            if not self.is_pokecenter_map_name(name):
                continue
            nurse = self.resolve_pokecenter_nurse_name(mid_i)
            if not nurse:
                continue

            if mid_i == cur:
                local_map = self.global_map.get_local_map(mid_i)
                nurse_cells = (
                    local_map.npc_coords(nurse) if local_map is not None else []
                )
                if nurse_cells:
                    tx, ty = min(
                        nurse_cells,
                        key=lambda c: abs(c[0] - sx) + abs(c[1] - sy),
                    )
                    score = float(abs(tx - sx) + abs(ty - sy))
                else:
                    score = 0.0
            else:
                route = self.global_map.get_route(cur, mid_i)
                if route is False:
                    continue
                hops = max(0, len(route) - 1)
                score = float(self._MAP_HOP_PENALTY * hops)

            if best is None or score < best[0]:
                best = (score, mid_i, nurse)

        if best is None:
            return None
        score, map_id, nurse = best
        return map_id, nurse, score

    def plan_to_nearest_grass(
        self,
        current_map_id: int,
        start_pos: Coord,
    ) -> Optional[NavMission]:
        """Known 최근접 GRASS — 필요 시 맵 Nav 후 Local grass."""
        found = self.find_nearest_observed_grass(current_map_id, start_pos)
        if found is None:
            return None
        grass_map, grass_xy, score = found
        name = self.global_map.map_name(grass_map)
        steps: List[NavStep] = []
        if int(grass_map) != int(current_map_id):
            steps.append(NavStep(kind="map", map_id=int(grass_map)))
        steps.append(NavStep(kind="grass"))
        return NavMission(
            steps=steps,
            summary=(
                f"[nav grass] 목표 {name}(ID:{grass_map}) {grass_xy} "
                f"(score={score:.0f})"
            ),
        )

    def plan_to_nearest_heal(
        self,
        current_map_id: int,
        start_pos: Coord,
    ) -> Optional[NavMission]:
        """Known 최근접 Pokemon Center → Nurse 대화(힐)."""
        found = self.find_nearest_pokecenter(current_map_id, start_pos)
        if found is None:
            return None
        center_map, nurse, score = found
        name = self.global_map.map_name(center_map)
        steps: List[NavStep] = []
        if int(center_map) != int(current_map_id):
            steps.append(NavStep(kind="map", map_id=int(center_map)))
        steps.append(NavStep(kind="heal", npc_name=nurse))
        return NavMission(
            steps=steps,
            summary=(
                f"[nav heal] 목표 {name}(ID:{center_map}) nurse={nurse!r} "
                f"(score={score:.0f})"
            ),
        )

    def plan_to_npc(
        self,
        current_map_id: int,
        npc_name: str,
    ) -> Optional[NavMission]:
        """현재 맵 NPC로 Local Nav + 대화."""
        name = str(npc_name or "").strip()
        if not name:
            return None
        local_map = self.global_map.get_local_map(int(current_map_id))
        if local_map is None:
            return None
        if not local_map.npc_coords(name):
            return None
        return NavMission(
            steps=[NavStep(kind="npc", npc_name=name)],
            summary=f"[nav npc] 목표 {name!r} (현재 맵)",
        )

    def plan_local_path_to_grass(
        self,
        current_map_id: int,
        start_pos: Coord,
        *,
        is_surfing: Optional[bool] = None,
        can_cut: Optional[bool] = None,
        goal: Optional[Coord] = None,
    ) -> bool:
        """현재 맵에서 가장 가까운(또는 지정) GRASS 셀로 이동."""
        local_map = self.global_map.get_local_map(int(current_map_id))
        if local_map is None:
            print("[Local] LocalMap 없음")
            return False
        grasses = local_map.grass_coords()
        if not grasses:
            print("[Local] 관측된 GRASS 없음")
            return False
        if goal is not None and tuple(goal) in set(grasses):
            chosen = (int(goal[0]), int(goal[1]))
        else:
            chosen = min(
                grasses,
                key=lambda c: abs(c[0] - start_pos[0]) + abs(c[1] - start_pos[1]),
            )
        return self.plan_local_path_to_coords(
            current_map_id,
            start_pos,
            [chosen],
            label=f"grass@{chosen}",
            is_surfing=is_surfing,
            can_cut=can_cut,
            force_goal_passable=True,
            mash_a=0,
        )

    @staticmethod
    def _face_action(from_pos: Coord, toward: Coord) -> Optional[str]:
        return face_action(from_pos, toward)

    @staticmethod
    def _is_standable_cell(local_map: LocalMap, x: int, y: int) -> bool:
        return is_standable_cell(local_map, x, y)

    @staticmethod
    def _is_blocking_cell(local_map: LocalMap, x: int, y: int) -> bool:
        """카운터/벽 등 — 통과 불가로 대화 시야만 가로막는 칸."""
        cell = local_map.get_cell(x, y)
        if cell is None:
            return False
        if cell.block_type is BlockType.WALL:
            return True
        if cell.entity in (EntityType.NPC, EntityType.OBJECT, EntityType.CUT_TREE):
            return True
        return False

    @classmethod
    def _adjacent_stand_tiles(cls, local_map: LocalMap, npc_pos: Coord) -> List[Coord]:
        """NPC 칸의 4방 중 통행 가능 칸."""
        x, y = int(npc_pos[0]), int(npc_pos[1])
        out: List[Coord] = []
        # 변수명 nx 는 networkx import 를 가리므로 쓰지 않는다.
        for ax, ay in ((x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)):
            if cls._is_standable_cell(local_map, ax, ay):
                out.append((ax, ay))
        return out

    @staticmethod
    def _is_warp_cell(local_map: LocalMap, x: int, y: int) -> bool:
        """계단·문·워프 패널 칸 (밟아서 확정됐거나 화면으로 본 것)."""
        cell = local_map.get_cell(x, y)
        return cell is not None and (
            cell.block_type is BlockType.WARP or bool(getattr(cell, "warp_candidate", False))
        )

    @classmethod
    def _npc_interaction_stand_tiles(
        cls,
        local_map: LocalMap,
        npc_pos: Coord,
        *,
        max_counter_gap: int = 1,
        across_counter: bool = True,
    ) -> List[Coord]:
        """NPC와 대화할 서 있는 칸."""
        x, y = int(npc_pos[0]), int(npc_pos[1])
        seen: set[Coord] = set()
        out: List[Coord] = []

        def _add(coord: Coord) -> None:
            if coord in seen or cls._is_warp_cell(local_map, coord[0], coord[1]):
                return
            seen.add(coord)
            out.append(coord)

        for tile in cls._adjacent_stand_tiles(local_map, npc_pos):
            _add(tile)

        if not across_counter:
            return out

        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            walls = 0
            for dist in range(1, max_counter_gap + 2):
                nx, ny = x + dx * dist, y + dy * dist
                if cls._is_standable_cell(local_map, nx, ny):
                    if walls >= 1:
                        _add((nx, ny))
                    break
                if cls._is_blocking_cell(local_map, nx, ny):
                    walls += 1
                    if walls > max_counter_gap:
                        break
                    continue
                # 미관측 등 — 더 이상 레이 진행하지 않음
                break

        return out

    def pop_action(self) -> Optional[str]:
        if self.action_queue:
            return self.action_queue.popleft()
        return None

    # ==================================================================
    # A*
    # ==================================================================
    def _a_star(
        self,
        local_map: LocalMap,
        start: Coord,
        goals: List[Coord],
        is_surfing: bool = False,
        can_cut: bool = False,
        debug: bool = False,
        force_goal_passable: bool = False,
        objects_passable: bool = False,
    ) -> Optional[PathResult]:
        """LocalMap A*."""

        def _on_debug(matrix, entity_matrix, origin_x, origin_y) -> None:
            if debug and debug_flags.get("matrix", False):
                self._debug_print_a_star_matrix(
                    local_map,
                    matrix,
                    entity_matrix,
                    origin_x,
                    origin_y,
                    start,
                    goals,
                    is_surfing=is_surfing,
                    can_cut=can_cut,
                )

        return grid_a_star(
            local_map,
            start,
            goals,
            costs=self.costs,
            is_surfing=is_surfing,
            can_cut=can_cut,
            force_goal_passable=force_goal_passable,
            on_debug=_on_debug if debug else None,
            objects_passable=objects_passable,
        )

    def _normal_step_cost(
        self,
        cell_val: int,
        entity_val: int,
        *,
        is_surfing: bool,
        can_cut: bool,
    ) -> Optional[int]:
        return normal_step_cost(
            self.costs,
            cell_val,
            entity_val,
            is_surfing=is_surfing,
            can_cut=can_cut,
        )

    def _shortest_portal_path(
        self,
        local_map: LocalMap,
        source_coords: Iterable[Coord],
        destination_coords: List[Coord],
        is_surfing: bool,
        can_cut: bool,
    ) -> Optional[PathResult]:
        """Portal의 여러 시작 좌표 중 총비용이 가장 작은 경로를 반환."""
        best_result: Optional[PathResult] = None

        for source_coord in source_coords:
            result = self._a_star(
                local_map,
                source_coord,
                destination_coords,
                is_surfing=is_surfing,
                can_cut=can_cut,
                debug=False,
                force_goal_passable=True,
            )

            if result is None:
                continue

            if (
                best_result is None
                or result.cost < best_result.cost
                or (
                    result.cost == best_result.cost
                    and len(result.path) < len(best_result.path)
                )
            ):
                best_result = result

        return best_result

    def _coords_to_actions(
        self,
        start_pos: Coord,
        path: List[Coord],
    ) -> List[str]:
        return coords_to_actions(start_pos, path)

    # ==================================================================
    # Portal 경로 보조
    # ==================================================================
    def _clear_route(self) -> None:
        self.portal_queue.clear()
        self.gate_queue.clear()
        self.action_queue.clear()

    def _ensure_portal_node(self, portal_node: PortalKey) -> None:
        portal = self.global_map.get_portal(portal_node)
        if portal is None:
            return

        self.global_map.gate_graph.add_node(
            portal_node,
            map_id=portal.map_id,
            role=getattr(portal, "role", None),
            direction=portal.direction,
            warp_coords=portal.warp_coords,
            arrival_coords=getattr(portal, "arrival_coords", ()),
            access_coords=getattr(portal, "access_coords", ()),
            label=portal.label,
            kind=portal.kind,
            destination_map_id=portal.destination_map_id,
        )

    def _sync_route_to_map(self, current_map_id: int) -> None:
        """맵 전환 후 이전 맵 Portal들을 큐에서 제거."""
        current_map_id = int(current_map_id)

        if self._active_map_id is None:
            self._active_map_id = current_map_id

        if current_map_id != self._active_map_id:
            old_map_id = self._active_map_id

            while (
                self.portal_queue
                and int(self.portal_queue[0][0]) == old_map_id
            ):
                self.portal_queue.popleft()

            while (
                self.portal_queue
                and int(self.portal_queue[0][0]) != current_map_id
            ):
                self.portal_queue.popleft()

            self.action_queue.clear()
            self._active_map_id = current_map_id

        self._refresh_gate_queue()

    def _current_map_exit_portal(
        self,
        current_map_id: int,
    ) -> Optional[PortalKey]:
        """현재 맵 구간에서 실제 이동 목표가 되는 EXIT를 반환한다."""
        target: Optional[PortalKey] = None

        for portal_node in self.portal_queue:
            if int(portal_node[0]) != int(current_map_id):
                break
            if self._portal_has_role(portal_node, ROLE_EXIT):
                target = portal_node

        return target

    def _next_map_after_current(
        self,
        current_map_id: int,
    ) -> Optional[int]:
        seen_current = False

        for portal_node in self.portal_queue:
            map_id = int(portal_node[0])

            if map_id == int(current_map_id):
                seen_current = True
                continue

            if seen_current:
                return map_id

        return None

    def _refresh_gate_queue(self) -> None:
        """portal_queue에서 맵을 넘기는 EXIT만 골라 gate_queue를 다시 만든다."""
        route = list(self.portal_queue)
        self.gate_queue = deque(
            source
            for source, destination in zip(route, route[1:])
            if int(source[0]) != int(destination[0])
            and self._portal_has_role(source, ROLE_EXIT)
            and self._portal_has_role(destination, ROLE_ENTRY)
        )

    @staticmethod
    def _collapsed_map_route(
        portal_path: List[PortalKey],
    ) -> List[int]:
        route: List[int] = []

        for portal_node in portal_path:
            map_id = int(portal_node[0])
            if not route or route[-1] != map_id:
                route.append(map_id)

        return route

    def _portal_text(self, portal_node: PortalKey) -> str:
        portal = self.global_map.get_portal(portal_node)
        map_id = int(portal_node[0])
        map_name = self.global_map.map_name(map_id)

        if portal is None:
            return f"{map_name}(ID:{map_id}) {portal_node}"

        arrival_coords = list(
            getattr(portal, "arrival_coords", ()) or ()
        )
        arrival_text = (
            f", arrival_coords={arrival_coords}"
            if arrival_coords
            else ""
        )
        access_coords = list(
            getattr(portal, "access_coords", ()) or ()
        )
        access_text = (
            f", access_coords={access_coords}"
            if access_coords
            else ""
        )

        return (
            f"{map_name}(ID:{map_id}) "
            f"role={getattr(portal, 'role', '?')}, "
            f"direction={portal.direction}, "
            f"warp_coords={list(portal.warp_coords)}"
            f"{arrival_text}"
            f"{access_text}, "
            f"label={portal.label!r}"
        )

    @staticmethod
    def _is_portal_key(node: object) -> bool:
        return (
            isinstance(node, tuple)
            and len(node) == 4
            and isinstance(node[0], int)
            and str(node[1]).upper() in (ROLE_EXIT, ROLE_ENTRY)
        )

    def _portal_has_role(self, node: PortalKey, role: str) -> bool:
        """Portal 객체를 우선하고, 없으면 role-aware key로 역할을 확인한다."""
        portal = self.global_map.get_portal(node)
        if portal is not None:
            return str(getattr(portal, "role", "")).upper() == str(role).upper()
        return (
            self._is_portal_key(node)
            and str(node[1]).upper() == str(role).upper()
        )

    def _exit_portal_nodes(self, map_id: int) -> List[PortalKey]:
        getter = getattr(self.global_map, "exit_portal_nodes", None)
        if callable(getter):
            return list(getter(int(map_id)))
        return [
            node
            for node in self.global_map.portal_nodes(int(map_id))
            if self._portal_has_role(node, ROLE_EXIT)
        ]

    def _entry_portal_nodes(self, map_id: int) -> List[PortalKey]:
        getter = getattr(self.global_map, "entry_portal_nodes", None)
        if callable(getter):
            return list(getter(int(map_id)))
        return [
            node
            for node in self.global_map.portal_nodes(int(map_id))
            if self._portal_has_role(node, ROLE_ENTRY)
        ]

    def _remove_virtual_nodes(self, graph: nx.DiGraph) -> None:
        for node in (self.START_NODE, self.END_NODE):
            if graph.has_node(node):
                graph.remove_node(node)

    # ==================================================================
    # 디버그 출력
    # ==================================================================
    def _debug_print_a_star_matrix(
        self,
        local_map: LocalMap,
        matrix,
        entity_matrix,
        origin_x: int,
        origin_y: int,
        start: Coord,
        goals: List[Coord],
        *,
        is_surfing: bool,
        can_cut: bool,
    ) -> None:
        rows, cols = matrix.shape

        print(f"\n[Matrix Debug] A* 탐색 전 맵 상태 ({local_map.map_name})")
        print(f"Start: {start} | Goals: {goals}")
        print(
            f"Ability: surfing={is_surfing}, can_cut={can_cut} | "
            f"Costs: {self.costs.as_dict()}"
        )
        print(
            f"Matrix Size: {rows}x{cols} | "
            f"Origin: ({origin_x}, {origin_y})"
        )

        goal_set = set(goals)

        for row in range(rows):
            row_text = ""
            for col in range(cols):
                value = int(matrix[row, col])
                entity = int(entity_matrix[row, col])
                coord = (col + origin_x, row + origin_y)

                if coord == start:
                    row_text += " S "
                elif coord in goal_set:
                    row_text += " G "
                elif value == LocalMap.MAT_WARP:
                    row_text += " W "
                elif entity == ENTITY_CUT_TREE:
                    row_text += " C " if can_cut else " X "
                elif entity in (ENTITY_NPC, ENTITY_OBJECT):
                    row_text += " X "
                elif value == LocalMap.MAT_WALL:
                    row_text += " X "
                elif value == LocalMap.MAT_UNKNOWN:
                    row_text += " ? "
                else:
                    row_text += f" {value} "

            print(row_text)

        print("==========================================\n")

    def debug_print_grid(
        self,
        local_map: LocalMap,
        start_pos: Coord,
        target_positions: List[Coord],
        path_coords: List[Coord],
    ) -> None:
        matrix, (origin_x, origin_y) = local_map.to_matrix("block")
        height, width = matrix.shape

        print(f"\n[Local Map 시각화: {local_map.map_name}]")
        print(f"출발(S): {start_pos} | 목적지(E): {target_positions}")

        path_marks = {
            coord: str((index + 1) % 10)
            for index, coord in enumerate(path_coords)
        }
        target_set = set(target_positions)

        for row in range(height):
            row_text = ""

            for col in range(width):
                coord = (col + origin_x, row + origin_y)

                if coord == start_pos:
                    row_text += " S "
                elif coord in target_set:
                    row_text += " E "
                elif coord in path_marks:
                    row_text += f" {path_marks[coord]} "
                else:
                    value = int(matrix[row, col])

                    if value in (
                        LocalMap.MAT_FREE,
                        LocalMap.MAT_GRASS,
                    ):
                        row_text += " 0 "
                    elif value == LocalMap.MAT_WATER:
                        row_text += " ~ "
                    elif value == LocalMap.MAT_WARP:
                        row_text += " W "
                    elif value == LocalMap.MAT_LEDGE_D:
                        row_text += " ↓ "
                    elif value == LocalMap.MAT_LEDGE_L:
                        row_text += " ← "
                    elif value == LocalMap.MAT_LEDGE_R:
                        row_text += " → "
                    elif value == LocalMap.MAT_LEDGE_U:
                        row_text += " ↑ "
                    elif value == LocalMap.MAT_UNKNOWN:
                        row_text += " ? "
                    else:
                        row_text += " 1 "

            print(row_text)

        print("-----------------------------------------\n")
