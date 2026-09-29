"""Portal 확장 계층 (facade)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from .Map import GlobalMap as BaseGlobalMap
from .gb_memory import (
    ADDR_GRASS_TILE,
    ADDR_IN_BATTLE,
    ADDR_JOY_IGNORE,
    ADDR_MAP_ID,
    ADDR_PLAYER_FACING,
    ADDR_PLAYER_X,
    ADDR_PLAYER_Y,
    ADDR_SCX,
    ADDR_SCY,
    ADDR_TILESET,
    ADDR_WALK_COUNTER,
    FACING_DELTA,
    MAP_SETTLE_FRAMES,
    read_npcs,
    read_screen_tiles,
)
from .card_key_doors import door_tiles
from .map_types import BlockType
from .portal_local import LocalMap
from .portal_types import (
    ARRIVAL_CONFIRM_TIMEOUT,
    ARRIVAL_STABLE_FRAMES,
    DIR_EAST,
    DIR_INTERNAL,
    DIR_NORTH,
    DIR_SOUTH,
    DIR_WEST,
    FACING_TO_DIRECTION,
    OPPOSITE_DIRECTION,
    ROLE_ENTRY,
    ROLE_EXIT,
    TRANSITION_INFERRED_REVERSE,
    TRANSITION_OBSERVED,
    Cell,
    Coord,
    ObservedTransition,
    Portal,
    PortalKey,
    ReverseExitGroup,
)
from .tile_rules import (
    normalize_screen_tiles,
    scan_screen_cells,
    tileset_rules,
)

# 하위 호환 re-export
__all__ = [
    "ARRIVAL_CONFIRM_TIMEOUT",
    "ARRIVAL_STABLE_FRAMES",
    "Cell",
    "DIR_EAST",
    "DIR_INTERNAL",
    "DIR_NORTH",
    "DIR_SOUTH",
    "DIR_WEST",
    "FACING_TO_DIRECTION",
    "GlobalMap",
    "LocalMap",
    "OPPOSITE_DIRECTION",
    "ObservedTransition",
    "Portal",
    "PortalKey",
    "ROLE_ENTRY",
    "ROLE_EXIT",
    "ReverseExitGroup",
    "TRANSITION_INFERRED_REVERSE",
    "TRANSITION_OBSERVED",
]


class GlobalMap(BaseGlobalMap):
    def __init__(self, map_data_path: Optional[str] = None) -> None:
        # 기본 맵 데이터, 타일/오브젝트/warp 데이터 로드는 원본 Map.py가 담당한다.
        super().__init__(map_data_path)

        # Portal 전역 그래프와 전환 기록.
        self.gate_graph: nx.DiGraph = nx.DiGraph()
        self.observed_transitions: List[ObservedTransition] = []

        # Portal 생성/arrival/access/추론 로그는 기본적으로 숨긴다.
        self._portal_debug: bool = False

    def _reset_tracking(self) -> None:
        """Base 추적 상태 + Portal arrival/스캔 상태를 초기화한다."""
        super()._reset_tracking()

        # Portal 확장형 pending 전환 상태 (base보다 dest_map_id 하나가 더 있다).
        self._pending_map_change: Optional[
            Tuple[int, int, Optional[Coord], Optional[int]]
        ] = None
        self._arrival_candidate: Optional[Coord] = None
        # 동일 좌표가 연속으로 유지된 정렬 프레임 수.
        self._arrival_candidate_frames: int = 0
        # 후보 시작점부터 동일/인접 좌표가 연속으로 이어진 정렬 프레임 수.
        self._arrival_sequence_frames: int = 0
        # 좌표 레지스터의 큰 점프를 감지하기 위한 마지막 정렬 좌표.
        self._arrival_last_pos: Optional[Coord] = None
        self._arrival_wait_frames: int = 0
        # 목적지 맵에서 old_map으로 되돌아가는 reverse EXIT 주변의 유효 arrival 좌표.
        self._pending_arrival_candidates: Tuple[Coord, ...] = ()
        self._pending_arrival_candidate_source: str = ""
        # map ID 쌍으로 찾은 목적지 reverse EXIT 후보들.
        self._pending_reverse_exit_groups: Tuple[ReverseExitGroup, ...] = ()
        self._pending_reverse_exit_group: Optional[ReverseExitGroup] = None
        self._arrival_reject_log_count: int = 0

        # 연속 이동 중 안전한 화면 경계 스캔 상태.
        self._last_scanned_map_id: Optional[int] = None
        self._last_scanned_pos: Optional[Coord] = None
        self._last_scanned_scroll: Optional[Coord] = None

        # 반복 스캔과 실제 방문 횟수를 분리한다.
        self._last_visited_map_id: Optional[int] = None
        self._last_visited_pos: Optional[Coord] = None

    def reset(self) -> None:
        """RL 에피소드 리셋."""
        super().reset()
        self.gate_graph.clear()
        self.observed_transitions.clear()

    def set_portal_debug(self, enabled: bool = True) -> None:
        """Portal 관련 진단 로그 출력을 켜거나 끈다."""
        self._portal_debug = bool(enabled)

    def _portal_log(self, *args, **kwargs) -> None:
        """Portal 전용 조건부 print 래퍼."""
        if self._portal_debug:
            print(*args, **kwargs)

    def _ensure_local(self, map_id: int) -> LocalMap:
        """Portal LocalMap을 확보한다."""
        map_id = int(map_id)
        node = self.graph.nodes.get(map_id)
        if node is not None and "local" in node:
            local = node["local"]
            if isinstance(local, LocalMap):
                return local

            upgraded = LocalMap(map_id, self.map_name(map_id))
            upgraded.grid.update(local.grid)
            upgraded.name_index.update(local.name_index)
            if hasattr(local, "npc_name_index"):
                upgraded.npc_name_index.update(local.npc_name_index)
            if hasattr(local, "npc_seen"):
                upgraded.npc_seen.update(local.npc_seen)
            self.graph.nodes[map_id]["local"] = upgraded
            return upgraded

        name = self.map_name(map_id)
        local = LocalMap(map_id, name)
        if map_id not in self.graph:
            self.graph.add_node(map_id, name=name, local=local)
        else:
            self.graph.nodes[map_id]["name"] = name
            self.graph.nodes[map_id]["local"] = local
        return local

    def _confirm_experienced_warp(
        self,
        local: LocalMap,
        map_id: int,
        px: int,
        py: int,
    ) -> None:
        """밟은 칸이 warps.json에 있으면 JSON 그룹 좌표에 W+EXIT Portal을 찍는다."""
        point = self.warp_at.get(int(map_id), {}).get((int(px), int(py)))
        if point is None:
            return

        coords = self._static_portal_source_coords(
            int(map_id), (int(px), int(py)), point
        )
        dest_id = point.get("dest_id")
        if dest_id is None:
            # LAST_MAP 등은 목적지 확정 전 — JSON 좌표에 W만 표시
            for x, y in coords:
                local.ensure_cell(int(x), int(y)).block_type = BlockType.WARP
            return

        label = self.map_name(int(dest_id))
        portal = local.register_exit_portal(
            self._direction_from_coord(int(map_id), coords[0]),
            list(coords),
            label,
            kind=str(point.get("kind", "warp")),
            destination_map_id=int(dest_id),
        )
        self._ensure_portal_node(portal)

    def _direction_from_coord(
        self,
        map_id: int,
        coord: Tuple[int, int],
        *,
        fallback_facing: Optional[int] = None,
        arriving: bool = False,
    ) -> str:
        """맵 밖 좌표면 경계 방향, 내부 워프면 이동 방향으로 Portal 방향 결정."""
        x, y = int(coord[0]), int(coord[1])
        size = self.map_size_cells.get(int(map_id))
        if size is not None:
            w, h = size
            if y < 0:
                return DIR_NORTH
            if y >= h:
                return DIR_SOUTH
            if x < 0:
                return DIR_WEST
            if x >= w:
                return DIR_EAST

        direction = FACING_TO_DIRECTION.get(fallback_facing, DIR_INTERNAL)
        if arriving:
            return OPPOSITE_DIRECTION.get(direction, DIR_INTERNAL)
        return direction
    def _static_portal_source_coords(
        self,
        map_id: int,
        selected_xy: Coord,
        selected_point: dict,
    ) -> List[Coord]:
        """선택된 JSON point와 같은 논리적 출입구의 raw 좌표를 묶는다."""
        map_id = int(map_id)
        selected = (int(selected_xy[0]), int(selected_xy[1]))
        selected_direction = self._direction_from_coord(map_id, selected)
        selected_kind = str(selected_point.get("kind", "warp"))
        selected_dest = selected_point.get("dest_id")
        selected_dest_const = str(selected_point.get("dest_const", "")).upper()

        compatible: set[Coord] = set()
        for (x, y), point in self.warp_at.get(map_id, {}).items():
            coord = (int(x), int(y))
            if str(point.get("kind", "warp")) != selected_kind:
                continue
            if point.get("dest_id") != selected_dest:
                continue
            if str(point.get("dest_const", "")).upper() != selected_dest_const:
                continue
            if self._direction_from_coord(map_id, coord) != selected_direction:
                continue
            compatible.add(coord)

        if selected not in compatible:
            return [selected]

        component: set[Coord] = set()
        stack = [selected]
        while stack:
            coord = stack.pop()
            if coord in component:
                continue
            component.add(coord)
            x, y = coord
            for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if neighbor in compatible and neighbor not in component:
                    stack.append(neighbor)

        return sorted(component)
    def _static_exit_groups(
        self,
        map_id: int,
        destination_map_id: int,
        *,
        allow_last_map: bool = True,
    ) -> Tuple[ReverseExitGroup, ...]:
        """map ID 쌍으로 설명되는 논리적 JSON EXIT 묶음을 반환한다."""
        map_id = int(map_id)
        destination_map_id = int(destination_map_id)
        points = self.warp_at.get(map_id, {})

        exact: List[Tuple[Coord, dict]] = []
        last_map: List[Tuple[Coord, dict]] = []
        for (x, y), point in points.items():
            coord = (int(x), int(y))
            dest_id = point.get("dest_id")
            if dest_id is not None and int(dest_id) == destination_map_id:
                exact.append((coord, point))
            elif (
                allow_last_map
                and dest_id is None
                and str(point.get("dest_const", "")).upper() == "LAST_MAP"
            ):
                last_map.append((coord, point))

        selected = exact if exact else last_map
        source = (
            "warps.json exact map pair"
            if exact
            else "warps.json LAST_MAP map pair"
        )
        groups: Dict[Tuple[Coord, ...], ReverseExitGroup] = {}
        for coord, point in selected:
            raw_coords = tuple(
                self._static_portal_source_coords(map_id, coord, point)
            )
            if raw_coords in groups:
                continue
            arrival_candidates = self._reverse_exit_group_arrival_candidates(
                map_id,
                destination_map_id,
                raw_coords,
            )
            groups[raw_coords] = ReverseExitGroup(
                map_id=map_id,
                raw_coords=raw_coords,
                direction=self._direction_from_coord(map_id, raw_coords[0]),
                kind=str(point.get("kind", "warp")),
                destination_map_id=destination_map_id,
                arrival_candidates=tuple(sorted(arrival_candidates)),
                source=source,
            )

        return tuple(
            sorted(groups.values(), key=lambda group: group.raw_coords)
        )

    def _source_boundary_lane_offsets(
        self,
        source_map_id: int,
        destination_map_id: int,
        destination_direction: str,
    ) -> Tuple[int, ...]:
        """출발 connection의 평행축 lane 오프셋을 반환한다."""
        source_map_id = int(source_map_id)
        destination_map_id = int(destination_map_id)
        points = self.warp_at.get(source_map_id, {})

        exact: List[Tuple[Coord, dict]] = []
        last_map: List[Tuple[Coord, dict]] = []
        for (x, y), point in points.items():
            coord = (int(x), int(y))
            dest_id = point.get("dest_id")
            if dest_id is not None and int(dest_id) == destination_map_id:
                exact.append((coord, point))
            elif (
                dest_id is None
                and str(point.get("dest_const", "")).upper() == "LAST_MAP"
            ):
                last_map.append((coord, point))

        selected = exact if exact else last_map
        expected_source_direction = OPPOSITE_DIRECTION.get(
            str(destination_direction).upper()
        )
        best_offsets: Tuple[int, ...] = (0,)
        seen_groups: set[Tuple[Coord, ...]] = set()

        for coord, point in selected:
            raw_coords = tuple(
                self._static_portal_source_coords(source_map_id, coord, point)
            )
            if raw_coords in seen_groups or not raw_coords:
                continue
            seen_groups.add(raw_coords)

            source_direction = self._direction_from_coord(
                source_map_id,
                raw_coords[0],
            )
            if (
                expected_source_direction is not None
                and source_direction != expected_source_direction
            ):
                continue

            if source_direction in (DIR_NORTH, DIR_SOUTH):
                axis_values = sorted({int(x) for x, _ in raw_coords})
            elif source_direction in (DIR_WEST, DIR_EAST):
                axis_values = sorted({int(y) for _, y in raw_coords})
            else:
                continue

            if not axis_values:
                continue
            base = axis_values[0]
            offsets = tuple(value - base for value in axis_values)
            if (
                len(offsets) > len(best_offsets)
                or (
                    len(offsets) == len(best_offsets)
                    and offsets[-1] > best_offsets[-1]
                )
            ):
                best_offsets = offsets

        return best_offsets

    def _reverse_exit_group_arrival_candidates(
        self,
        map_id: int,
        source_map_id: int,
        raw_coords: Tuple[Coord, ...],
    ) -> Tuple[Coord, ...]:
        """reverse EXIT 묶음에서 실제 arrival 후보를 생성한다."""
        candidates: set[Coord] = set()
        for raw in raw_coords:
            candidates.update(
                self._raw_reverse_exit_arrival_coords(map_id, raw)
            )

        if not raw_coords:
            return ()

        direction = self._direction_from_coord(map_id, raw_coords[0])
        if direction not in (DIR_NORTH, DIR_SOUTH, DIR_WEST, DIR_EAST):
            return tuple(sorted(candidates))

        lane_offsets = self._source_boundary_lane_offsets(
            source_map_id,
            map_id,
            direction,
        )
        if len(lane_offsets) <= 1:
            return tuple(sorted(candidates))

        if direction in (DIR_NORTH, DIR_SOUTH):
            anchor = min(int(x) for x, _ in raw_coords)
            boundary_y = int(raw_coords[0][1])
            for offset in lane_offsets:
                candidates.update(
                    self._raw_reverse_exit_arrival_coords(
                        map_id,
                        (anchor + int(offset), boundary_y),
                    )
                )
        else:
            boundary_x = int(raw_coords[0][0])
            anchor = min(int(y) for _, y in raw_coords)
            for offset in lane_offsets:
                candidates.update(
                    self._raw_reverse_exit_arrival_coords(
                        map_id,
                        (boundary_x, anchor + int(offset)),
                    )
                )

        return tuple(sorted(candidates))

    @staticmethod
    def _distance_to_group_hint(group: ReverseExitGroup, coord: Coord) -> int:
        """관측 좌표와 logical EXIT의 raw/arrival 후보 간 최소 거리."""
        x, y = int(coord[0]), int(coord[1])
        points = set(group.raw_coords) | set(group.arrival_candidates)
        return min(
            (abs(px - x) + abs(py - y) for px, py in points),
            default=10**9,
        )

    def _select_static_exit_group(
        self,
        map_id: int,
        destination_map_id: int,
        *,
        hint: Optional[Coord],
        allow_last_map: bool = True,
        max_dist: int = 2,
    ) -> Optional[ReverseExitGroup]:
        """map ID 쌍을 우선하고, 여러 문일 때만 좌표 힌트를 사용한다."""
        groups = self._static_exit_groups(
            map_id,
            destination_map_id,
            allow_last_map=allow_last_map,
        )
        if len(groups) == 1:
            return groups[0]
        if not groups or hint is None:
            return None

        # ReverseExitGroup은 순서 비교(ordering)를 지원하지 않는다.
        scored = [
            (self._distance_to_group_hint(group, hint), group)
            for group in groups
        ]
        best_distance = min(distance for distance, _ in scored)
        if best_distance > int(max_dist):
            return None
        best = [
            group
            for distance, group in scored
            if distance == best_distance
        ]
        return best[0] if len(best) == 1 else None

    def _register_static_exit_group(
        self,
        local: LocalMap,
        group: ReverseExitGroup,
        label: str,
        destination_map_id: int,
        *,
        forced_direction: Optional[str] = None,
    ) -> Portal:
        direction = (
            str(forced_direction).upper()
            if forced_direction is not None
            else group.direction
        )
        portal = local.register_exit_portal(
            direction,
            list(group.raw_coords),
            label,
            kind=group.kind,
            destination_map_id=int(destination_map_id),
        )
        self._ensure_portal_node(portal)
        return portal

    def portal_nodes(self, map_id: int) -> List[PortalKey]:
        """특정 map_id에 등록된 Portal 노드 키 목록."""
        local = self.get_local_map(map_id)
        if local is None:
            return []
        return [portal.key for portal in local.iter_portals()]
    def exit_portal_nodes(self, map_id: int) -> List[PortalKey]:
        """특정 맵의 JSON 기반 EXIT 노드만 반환한다."""
        local = self.get_local_map(map_id)
        if local is None:
            return []
        return [p.key for p in local.iter_portals() if p.is_exit]
    def entry_portal_nodes(self, map_id: int) -> List[PortalKey]:
        """특정 맵의 안정화 기반 ENTRY 노드만 반환한다."""
        local = self.get_local_map(map_id)
        if local is None:
            return []
        return [p.key for p in local.iter_portals() if p.is_entry]
    def get_portal(self, node: PortalKey) -> Optional[Portal]:
        """PortalKey로 LocalMap의 Portal 객체를 얻는다."""
        map_id = int(node[0])
        local = self.get_local_map(map_id)
        if local is None:
            return None
        return local.get_portal(node)
    def portal_coords(self, node: PortalKey) -> List[Tuple[int, int]]:
        portal = self.get_portal(node)
        return [] if portal is None else list(portal.warp_coords)
    def update(
        self,
        map_id: int,
        player_pos: Tuple[int, int],
        tiles_20x18: Any,
        tileset_id: int,
        npcs: Optional[Dict[int, Dict[str, int]]] = None,
        grass_tile_id: Optional[int] = None,
        facing: Optional[int] = None,
    ) -> None:
        """1 step마다 호출."""
        map_id = int(map_id)
        px, py = int(player_pos[0]), int(player_pos[1])

        # --- 1) Global 노드 확보 + 화면 20x18 → 이동칸 10x9 셀 갱신 ------
        local: LocalMap = self._ensure_local(map_id)
        local.tileset = int(tileset_id)
        scan_screen_cells(
            local,
            player_pos=(px, py),
            tiles=normalize_screen_tiles(tiles_20x18),
            rules=tileset_rules(tileset_id, grass_tile_id),
            npcs=npcs or {},
            hidden_at=self.hidden_objects.get(map_id, {}),
            hidden_names=getattr(self, "hidden_object_names", {}).get(map_id, {}),
            npc_names=getattr(self, "npc_object_names", {}).get(map_id, {}),
            static_warps=self.warp_at.get(map_id, {}).keys(),
            door_warps=self._door_warp_coords(map_id),
            card_key_doors=getattr(self, "card_key_doors", {}).get(map_id),
            card_key_tiles=door_tiles(map_id),
        )

        # --- 3) 맵 전환 처리 (edge + 워프 칸 이름) -----------------------
        if self.prev_map_id is not None and self.prev_map_id != map_id:
            self._handle_map_change(
                self.prev_map_id,
                map_id,
                px,
                py,
                facing,
                old_facing=self.prev_facing,
            )

        # --- 4) 플레이어 현재 칸 방문 처리 ------------------------------
        # 같은 위치에서 화면을 반복 스캔해도 visited_count가 증가하지 않도록
        # 실제 map_id/좌표가 달라졌을 때만 방문으로 기록한다.
        current_visit_pos = (px, py)
        if (
            self._last_visited_map_id != map_id
            or self._last_visited_pos != current_visit_pos
        ):
            self.last_first_visit = local.ensure_cell(px, py).mark_visited()
            self._last_visited_map_id = map_id
            self._last_visited_pos = current_visit_pos
        else:
            self.last_first_visit = False

        # --- 4b) 밟은 워프 칸을 warps.json으로 확정 (W + Portal) --------
        self._confirm_experienced_warp(local, map_id, px, py)

        # --- 5) 상태 저장 -----------------------------------------------
        self.prev_map_id = map_id
        self.prev_pos = (px, py)
        self.prev_facing = facing
    def _should_scan_current_view(
        self,
        pyboy: Any,
        map_id: int,
        px: int,
        py: int,
    ) -> bool:
        """현재 프레임의 화면을 LocalMap에 저장해도 되는지 판정한다."""
        scx = int(pyboy.memory[ADDR_SCX])
        scy = int(pyboy.memory[ADDR_SCY])
        if scx % 8 != 0 or scy % 8 != 0:
            return False

        current_map_id = int(map_id)
        current_pos = (int(px), int(py))
        walking = int(pyboy.memory[ADDR_WALK_COUNTER]) != 0

        # 새 맵에서 arrival 안정화가 끝난 뒤의 최초 화면.
        if self._last_scanned_map_id != current_map_id:
            return True

        # 연속 이동 중이어도 새 셀 좌표 + 정렬된 화면이면 스캔한다.
        if self._last_scanned_pos != current_pos:
            return True

        # 같은 위치에서 걷는 중이면 화면만 움직이는 중간 프레임일 수 있다.
        return not walking
    def _scan_current_view_from_pyboy(
        self,
        pyboy: Any,
        map_id: int,
        px: int,
        py: int,
        facing: Optional[int],
    ) -> bool:
        """정렬된 현재 화면을 LocalMap에 반영한다."""
        if not self._should_scan_current_view(
            pyboy,
            map_id,
            px,
            py,
        ):
            return False

        tileset_id = int(pyboy.memory[ADDR_TILESET])
        tiles = read_screen_tiles(pyboy)
        npcs = read_npcs(pyboy)
        grass_rt = int(pyboy.memory[ADDR_GRASS_TILE])
        self._refresh_card_key_doors(pyboy)        # 실프 카드키 문 (core.card_key_doors)

        self.update(
            int(map_id),
            (int(px), int(py)),
            tiles,
            tileset_id,
            npcs,
            grass_rt,
            facing=facing,
        )

        self._last_scanned_map_id = int(map_id)
        self._last_scanned_pos = (int(px), int(py))
        self._last_scanned_scroll = (
            int(pyboy.memory[ADDR_SCX]),
            int(pyboy.memory[ADDR_SCY]),
        )
        return True
    def update_from_pyboy(self, pyboy: Any) -> bool:
        """PyBoy에서 읽어 LocalMap 갱신 + 맵 전환 arrival 안정화."""
        if int(pyboy.memory[ADDR_IN_BATTLE]) != 0:
            return False

        # MapID 감지는 화면 정렬보다 먼저 수행한다.
        map_id = int(pyboy.memory[ADDR_MAP_ID])
        px = int(pyboy.memory[ADDR_PLAYER_X])
        py = int(pyboy.memory[ADDR_PLAYER_Y])
        facing = int(pyboy.memory[ADDR_PLAYER_FACING])
        current_pos = (px, py)

        scx = int(pyboy.memory[ADDR_SCX])
        scy = int(pyboy.memory[ADDR_SCY])
        aligned = scx % 8 == 0 and scy % 8 == 0
        walking = int(pyboy.memory[ADDR_WALK_COUNTER]) != 0
        joy_locked = int(pyboy.memory[ADDR_JOY_IGNORE]) != 0

        # ── 맵 전환 감지 ──────────────────────────────────────────────
        # EnterMap 중이라도 MapID가 바뀌었다면 옛 맵의 prev_pos/facing이
        # 남아 있을 때 pending 정보를 먼저 보존한다. VRAM은 아직 읽지 않는다.
        if self.prev_map_id is not None and self.prev_map_id != map_id:
            self._queue_map_change(self.prev_map_id, map_id)
            self.prev_map_id = map_id
            self.prev_pos = current_pos
            self.prev_facing = facing
            self._map_settle = MAP_SETTLE_FRAMES
            return True

        # EnterMap/LoadMapData·대화·컷신 중에는 화면을 신뢰하지 않는다.
        if joy_locked:
            return False

        # settle은 정렬된 셀 경계 수가 아니라 실제 안전 프레임 수로 센다.
        if self._map_settle > 0:
            self._map_settle -= 1
            self.prev_map_id = map_id
            self.prev_pos = current_pos
            self.prev_facing = facing
            return False

        # LocalMap 화면과 arrival 후보는 8픽셀 셀 경계에서만 해석한다.
        if not aligned:
            return False

        # ── 보류 중인 맵 전환 ─────────────────────────────────────────
        if self._pending_map_change is not None:
            old_id, new_id, near_old, old_facing = self._pending_map_change
            self._arrival_wait_frames += 1
            candidate = self._arrival_candidate

            # 첫 후보는 반드시 목적지 맵의 reverse EXIT 주변이어야 한다.
            if candidate is None:
                plausible = self._arrival_candidate_plausible(
                    old_id,
                    new_id,
                    near_old,
                    old_facing,
                    current_pos,
                    allow_timeout_fallback=False,
                )
                if not plausible:
                    self._arrival_candidate_frames = 0
                    self._arrival_sequence_frames = 0
                    self._arrival_last_pos = None
                    self._arrival_reject_log_count += 1

                    if self._arrival_reject_log_count == 1:
                        self._portal_log(
                            "[portal arrival] reverse EXIT 밖의 좌표를 대기합니다:",
                            f"transition={old_id}->{new_id}",
                            f"observed={current_pos}",
                            f"expected={list(self._pending_arrival_candidates)}",
                            f"source={self._pending_arrival_candidate_source}",
                        )

                    if self._arrival_wait_frames >= ARRIVAL_CONFIRM_TIMEOUT:
                        self._portal_log(
                            "[portal arrival warning] 제한 시간 안에 reverse EXIT "
                            "근처의 목적지 좌표를 확인하지 못해 transition 기록을 "
                            "취소합니다:",
                            f"transition={old_id}->{new_id}",
                            f"last_observed={current_pos}",
                            f"expected={list(self._pending_arrival_candidates)}",
                            f"source={self._pending_arrival_candidate_source}",
                        )
                        self._clear_pending_map_change_state()

                    self.prev_map_id = map_id
                    self.prev_pos = current_pos
                    self.prev_facing = facing
                    return False

                matched_group = self._match_pending_reverse_exit_group(current_pos)
                if matched_group is None:
                    self.prev_map_id = map_id
                    self.prev_pos = current_pos
                    self.prev_facing = facing
                    return False
                self._pending_reverse_exit_group = matched_group
                self._pending_arrival_candidates = matched_group.arrival_candidates
                self._pending_arrival_candidate_source = matched_group.source
                self._arrival_candidate = current_pos
                self._arrival_candidate_frames = 1
                self._arrival_sequence_frames = 1
                self._arrival_last_pos = current_pos
                self._arrival_reject_log_count = 0
                self._portal_log(
                    "[portal arrival] reverse EXIT 후보 확인:",
                    f"transition={old_id}->{new_id}",
                    f"candidate={current_pos}",
                    f"expected={list(self._pending_arrival_candidates)}",
                    f"source={self._pending_arrival_candidate_source}",
                    f"canonical={None if self._pending_reverse_exit_group is None else list(self._pending_reverse_exit_group.raw_coords)}",
                )
                self.prev_map_id = map_id
                self.prev_pos = current_pos
                self.prev_facing = facing
                return self._scan_current_view_from_pyboy(
                    pyboy, map_id, px, py, facing
                )

            # 후보 시작 후에도 좌표 자체는 목적지 맵 내부여야 한다.
            if not self._coord_inside_map(new_id, current_pos):
                self._arrival_candidate = None
                self._arrival_candidate_frames = 0
                self._arrival_sequence_frames = 0
                self._arrival_last_pos = None
                self.prev_map_id = map_id
                self.prev_pos = current_pos
                self.prev_facing = facing
                return False

            last_pos = self._arrival_last_pos or candidate
            step_from_last = (
                abs(current_pos[0] - last_pos[0])
                + abs(current_pos[1] - last_pos[1])
            )

            if step_from_last == 0:
                self._arrival_candidate_frames += 1
                self._arrival_sequence_frames += 1
                self._arrival_last_pos = current_pos

                if (
                    not walking
                    and self._arrival_candidate_frames
                    >= ARRIVAL_STABLE_FRAMES
                ):
                    self._handle_map_change(
                        old_id,
                        new_id,
                        px,
                        py,
                        facing,
                        near_old=near_old,
                        old_facing=old_facing,
                        arrival_pos=candidate,
                        reverse_group=self._pending_reverse_exit_group,
                    )
                    self._clear_pending_map_change_state()

                self.prev_map_id = map_id
                self.prev_pos = current_pos
                self.prev_facing = facing
                return self._scan_current_view_from_pyboy(
                    pyboy, map_id, px, py, facing
                )

            if step_from_last == 1:
                # 첫 후보가 reverse EXIT 근처임을 이미 확인했다.
                self._arrival_sequence_frames += 1
                self._arrival_candidate_frames = 1
                self._arrival_last_pos = current_pos

                self._handle_map_change(
                    old_id,
                    new_id,
                    px,
                    py,
                    facing,
                    near_old=near_old,
                    old_facing=old_facing,
                    arrival_pos=candidate,
                    reverse_group=self._pending_reverse_exit_group,
                )
                self._clear_pending_map_change_state()
            else:
                self._portal_log(
                    "[portal arrival] 좌표 레지스터 재정렬 감지:",
                    f"old_candidate={candidate}",
                    f"last={last_pos}",
                    f"new={current_pos}",
                )
                self._arrival_candidate = None
                self._arrival_candidate_frames = 0
                self._arrival_sequence_frames = 0
                self._arrival_last_pos = None

                if self._arrival_candidate_plausible(
                    old_id,
                    new_id,
                    near_old,
                    old_facing,
                    current_pos,
                    allow_timeout_fallback=False,
                ):
                    matched_group = self._match_pending_reverse_exit_group(current_pos)
                    if matched_group is not None:
                        self._pending_reverse_exit_group = matched_group
                        self._pending_arrival_candidates = matched_group.arrival_candidates
                        self._pending_arrival_candidate_source = matched_group.source
                        self._arrival_candidate = current_pos
                        self._arrival_candidate_frames = 1
                        self._arrival_sequence_frames = 1
                        self._arrival_last_pos = current_pos
                        self._arrival_reject_log_count = 0
                        self._portal_log(
                            "[portal arrival] 재정렬 후 reverse EXIT 후보 확인:",
                            f"transition={old_id}->{new_id}",
                            f"candidate={current_pos}",
                            f"canonical={list(matched_group.raw_coords)}",
                        )

            self.prev_map_id = map_id
            self.prev_pos = current_pos
            self.prev_facing = facing

        return self._scan_current_view_from_pyboy(
            pyboy,
            map_id,
            px,
            py,
            facing,
        )
    def _ensure_portal_node(self, portal: Portal) -> PortalKey:
        """Portal을 gate_graph 노드로 등록한다."""
        node = portal.key
        self.gate_graph.add_node(
            node,
            map_id=portal.map_id,
            role=portal.role,
            direction=portal.direction,
            warp_coords=portal.warp_coords,
            arrival_coords=portal.arrival_coords,
            access_coords=portal.access_coords,
            label=portal.label,
            kind=portal.kind,
            destination_map_id=portal.destination_map_id,
        )
        return node
    def _link_portal_transition(
        self,
        source: Portal,
        destination: Portal,
        *,
        confidence: str = TRANSITION_OBSERVED,
        inferred_from: Optional[Tuple[PortalKey, PortalKey]] = None,
    ) -> str:
        """EXIT -> ENTRY transition을 연결하고 관측 신뢰도를 보존한다."""
        if not source.is_exit:
            raise ValueError("transition source must be an EXIT portal")
        if not destination.is_entry:
            raise ValueError("transition destination must be an ENTRY portal")

        confidence = str(confidence)
        if confidence not in (
            TRANSITION_OBSERVED,
            TRANSITION_INFERRED_REVERSE,
        ):
            raise ValueError(f"unsupported transition confidence: {confidence}")

        source.destination_map_id = destination.map_id
        source.destination_portal = destination.key

        source_node = self._ensure_portal_node(source)
        destination_node = self._ensure_portal_node(destination)

        existing = self.gate_graph.get_edge_data(source_node, destination_node)
        previous_confidence: Optional[str] = None
        if existing is not None:
            previous_confidence = str(
                existing.get("confidence") or TRANSITION_OBSERVED
            )

        if previous_confidence == TRANSITION_OBSERVED:
            effective_confidence = TRANSITION_OBSERVED
        elif confidence == TRANSITION_OBSERVED:
            effective_confidence = TRANSITION_OBSERVED
        else:
            effective_confidence = TRANSITION_INFERRED_REVERSE

        if existing is None:
            status = "created"
        elif (
            previous_confidence == TRANSITION_INFERRED_REVERSE
            and effective_confidence == TRANSITION_OBSERVED
        ):
            status = "upgraded"
        else:
            status = "unchanged"

        self.gate_graph.add_edge(
            source_node,
            destination_node,
            weight=1,
            edge_type="transition",
            confidence=effective_confidence,
            observed=(effective_confidence == TRANSITION_OBSERVED),
        )
        edge_data = self.gate_graph[source_node][destination_node]
        if effective_confidence == TRANSITION_INFERRED_REVERSE:
            if inferred_from is not None:
                edge_data["inferred_from"] = inferred_from
        else:
            edge_data.pop("inferred_from", None)

        map_existing = self.graph.get_edge_data(
            int(source.map_id),
            int(destination.map_id),
        )
        map_previous = None
        if map_existing is not None:
            map_previous = str(
                map_existing.get("confidence") or TRANSITION_OBSERVED
            )
        map_confidence = (
            TRANSITION_OBSERVED
            if map_previous == TRANSITION_OBSERVED
            or effective_confidence == TRANSITION_OBSERVED
            else TRANSITION_INFERRED_REVERSE
        )
        self.graph.add_edge(
            int(source.map_id),
            int(destination.map_id),
            weight=1,
            confidence=map_confidence,
        )

        # ENTRY -> EXIT 역간선은 여기서 만들지 않는다.
        return status

    def _register_inferred_entry_portal(
        self,
        local: LocalMap,
        source_group: ReverseExitGroup,
        source_name: str,
        *,
        source_map_id: int,
        entry_direction: str,
        inferred_access: Coord,
    ) -> Portal:
        """실제 역전환 전에도 reciprocal ENTRY를 보수적으로 등록한다."""
        portal = local.register_portal(
            str(entry_direction).upper(),
            list(source_group.raw_coords),
            source_name,
            role=ROLE_ENTRY,
            kind="inferred_reverse",
            destination_map_id=int(source_map_id),
        )
        access = (int(inferred_access[0]), int(inferred_access[1]))
        if self._coord_inside_map(local.map_id, access):
            local.record_portal_access(portal.key, access)
        self._ensure_portal_node(portal)
        return portal

    def _infer_reciprocal_transition(
        self,
        *,
        old_local: LocalMap,
        new_local: LocalMap,
        source_group: ReverseExitGroup,
        reverse_group: ReverseExitGroup,
        source_exit: Portal,
        destination_entry: Portal,
        source_access: Optional[Coord],
        old_name: str,
        new_name: str,
    ) -> None:
        """A.EXIT→B.ENTRY 관측으로 B.EXIT→A.ENTRY를 추론한다."""
        if source_access is None:
            return
        if int(source_group.destination_map_id) != int(new_local.map_id):
            return
        if int(reverse_group.destination_map_id) != int(old_local.map_id):
            return

        reverse_exit = self._register_static_exit_group(
            new_local,
            reverse_group,
            old_name,
            old_local.map_id,
        )
        source_entry = self._register_inferred_entry_portal(
            old_local,
            source_group,
            new_name,
            source_map_id=new_local.map_id,
            entry_direction=OPPOSITE_DIRECTION.get(
                reverse_exit.direction,
                DIR_INTERNAL,
            ),
            inferred_access=source_access,
        )

        status = self._link_portal_transition(
            reverse_exit,
            source_entry,
            confidence=TRANSITION_INFERRED_REVERSE,
            inferred_from=(source_exit.key, destination_entry.key),
        )
        if status == "created":
            self._portal_log(
                "[portal inferred] reciprocal transition 생성:",
                f"{new_local.map_id}->{old_local.map_id}",
                f"EXIT={reverse_exit.key}",
                f"ENTRY={source_entry.key}",
                f"access={list(source_entry.access_coords)}",
            )

    def _mark_inferred_reverse_confirmed(
        self,
        inferred_from: Optional[Tuple[PortalKey, PortalKey]],
    ) -> None:
        """실제 역방향 관측 시 원래 transition history를 확인 상태로 승격."""
        if inferred_from is None:
            return
        original_source, original_destination = inferred_from
        for index, transition in enumerate(self.observed_transitions):
            if (
                transition.source_portal == original_source
                and transition.destination_portal == original_destination
                and not transition.reverse_confirmed
            ):
                self.observed_transitions[index] = replace(
                    transition,
                    reverse_confirmed=True,
                )

    def _queue_map_change(self, old_map_id: int, new_map_id: int) -> None:
        """MapID 쌍으로 destination reverse EXIT를 먼저 확정한다."""
        old_map_id, new_map_id = int(old_map_id), int(new_map_id)
        near_old = self._compute_near_old()
        old_facing = self.prev_facing
        self._ensure_local(new_map_id)

        groups = self._static_exit_groups(
            new_map_id,
            old_map_id,
            allow_last_map=True,
        )
        if not groups:
            self._portal_log(
                "[portal arrival warning] 목적지 reverse EXIT가 없어 transition을 "
                "기록하지 않습니다:",
                f"transition={old_map_id}->{new_map_id}",
            )
            self._clear_pending_map_change_state()
            return

        self._pending_map_change = (
            old_map_id,
            new_map_id,
            near_old,
            old_facing,
        )
        self._pending_reverse_exit_groups = groups
        self._pending_reverse_exit_group = groups[0] if len(groups) == 1 else None

        if self._pending_reverse_exit_group is not None:
            self._pending_arrival_candidates = (
                self._pending_reverse_exit_group.arrival_candidates
            )
            self._pending_arrival_candidate_source = (
                self._pending_reverse_exit_group.source
            )
        else:
            self._pending_arrival_candidates = tuple(sorted({
                coord
                for group in groups
                for coord in group.arrival_candidates
            }))
            self._pending_arrival_candidate_source = (
                f"ambiguous map-pair groups={len(groups)}"
            )

        self._arrival_candidate = None
        self._arrival_candidate_frames = 0
        self._arrival_sequence_frames = 0
        self._arrival_last_pos = None
        self._arrival_wait_frames = 0
        self._arrival_reject_log_count = 0

        self._portal_log(
            "[portal arrival] reverse EXIT 준비:",
            f"transition={old_map_id}->{new_map_id}",
            f"groups={[list(group.raw_coords) for group in groups]}",
            f"selected={None if self._pending_reverse_exit_group is None else list(self._pending_reverse_exit_group.raw_coords)}",
            f"arrival_candidates={list(self._pending_arrival_candidates)}",
        )
    def _clear_pending_map_change_state(self) -> None:
        """맵 전환 pending/arrival 임시 상태를 한 번에 초기화한다."""
        self._pending_map_change = None
        self._arrival_candidate = None
        self._arrival_candidate_frames = 0
        self._arrival_sequence_frames = 0
        self._arrival_last_pos = None
        self._arrival_wait_frames = 0
        self._pending_arrival_candidates = ()
        self._pending_arrival_candidate_source = ""
        self._pending_reverse_exit_groups = ()
        self._pending_reverse_exit_group = None
        self._arrival_reject_log_count = 0
        self._map_settle = 0
    def _pending_source_access_coord(
        self,
        near_old: Optional[Tuple[int, int]],
        old_facing: Optional[int],
    ) -> Optional[Tuple[int, int]]:
        """pending 전환의 옛 맵 실제 출발 위치를 복원한다."""
        if near_old is None or old_facing is None:
            return None
        delta = FACING_DELTA.get(old_facing)
        if delta is None:
            return None
        return (
            int(near_old[0]) - int(delta[0]),
            int(near_old[1]) - int(delta[1]),
        )

    def _valid_exit_access_coords(
        self,
        portal: Portal,
        old_facing: Optional[int],
    ) -> List[Coord]:
        """EXIT Portal을 작동시키기 직전일 수 있는 맵 내부 좌표 목록."""
        if portal is None or not portal.is_exit:
            return []

        direction_delta: Dict[str, Coord] = {
            DIR_NORTH: (0, -1),
            DIR_SOUTH: (0, 1),
            DIR_WEST: (-1, 0),
            DIR_EAST: (1, 0),
        }
        # 경계 Portal은 JSON 방향이 확정 정보이므로 stale facing보다 우선한다.
        delta: Optional[Coord] = direction_delta.get(
            str(portal.direction).upper()
        )

        candidates: set[Coord] = set()
        for raw_x, raw_y in portal.warp_coords:
            x, y = int(raw_x), int(raw_y)
            if delta is not None:
                # 플레이어는 delta 방향으로 raw warp에 진입하므로 바로 전 칸은 raw warp - delta 이다.
                candidates.add((x - delta[0], y - delta[1]))
            else:
                # INTERNAL warp는 전환 순간 facing이 stale일 수 있다.
                candidates.update(
                    {
                        (x, y - 1),
                        (x, y + 1),
                        (x - 1, y),
                        (x + 1, y),
                    }
                )

        return sorted(
            coord
            for coord in candidates
            if not self._cell_off_map(portal.map_id, *coord)
        )

    def _validated_source_access_coord(
        self,
        portal: Portal,
        near_old: Optional[Coord],
        old_facing: Optional[int],
    ) -> Optional[Coord]:
        """관측된 EXIT access를 검사하고 필요한 경우 안전하게 보정한다."""
        valid_coords = self._valid_exit_access_coords(portal, old_facing)
        if not valid_coords:
            portal.access_coords = ()
            return None

        valid_set = set(valid_coords)

        # 이전 왕복에서 잘못 들어간 (예: 문에서 두 칸 떨어진) 좌표도 정리한다.
        portal.access_coords = tuple(
            sorted(
                (int(x), int(y))
                for x, y in portal.access_coords
                if (int(x), int(y)) in valid_set
            )
        )

        provisional = self._pending_source_access_coord(near_old, old_facing)
        if provisional is not None:
            provisional = (int(provisional[0]), int(provisional[1]))
            if provisional in valid_set:
                return provisional

        if near_old is None:
            return None

        reference = (int(near_old[0]), int(near_old[1]))

        def rank(coord: Coord) -> Tuple[int, int, int, int]:
            near_distance = abs(coord[0] - reference[0]) + abs(coord[1] - reference[1])
            if provisional is None:
                provisional_distance = near_distance
            else:
                provisional_distance = (
                    abs(coord[0] - provisional[0])
                    + abs(coord[1] - provisional[1])
                )
            return near_distance, provisional_distance, coord[1], coord[0]

        corrected = min(valid_coords, key=rank)
        corrected_distance = (
            abs(corrected[0] - reference[0])
            + abs(corrected[1] - reference[1])
        )

        # 정상 near_old는 raw warp 자체이거나 access 좌표이므로 유효 후보와의 거리는 0 또는 1이다.
        if corrected_distance > 1:
            self._portal_log(
                "[portal access warning] EXIT access 후보가 전환 위치에서 너무 멀어 "
                "기록하지 않습니다:",
                f"portal={portal.key}",
                f"near_old={reference}",
                f"provisional={provisional}",
                f"valid={valid_coords}",
            )
            return None

        if provisional != corrected:
            self._portal_log(
                "[portal access] 오래된/잘못된 EXIT access를 보정합니다:",
                f"portal={portal.key}",
                f"observed={provisional}",
                f"corrected={corrected}",
            )

        return corrected
    def _coord_inside_map(self, map_id: int, coord: Coord) -> bool:
        """좌표가 해당 맵의 내부 셀인지 확인한다."""
        x, y = int(coord[0]), int(coord[1])
        size = self.map_size_cells.get(int(map_id))
        if size is None:
            return True
        width, height = int(size[0]), int(size[1])
        return 0 <= x < width and 0 <= y < height

    def _raw_reverse_exit_arrival_coords(
        self,
        map_id: int,
        raw_coord: Coord,
    ) -> List[Coord]:
        """reverse EXIT raw 좌표에서 가능한 실제 arrival 좌표를 만든다."""
        x, y = int(raw_coord[0]), int(raw_coord[1])
        size = self.map_size_cells.get(int(map_id))
        if size is None:
            return [(x, y)]

        width, height = int(size[0]), int(size[1])
        candidates: set[Coord] = set()

        if x < 0:
            candidates.add((0, min(max(y, 0), height - 1)))
        elif x >= width:
            candidates.add((width - 1, min(max(y, 0), height - 1)))
        elif y < 0:
            candidates.add((min(max(x, 0), width - 1), 0))
        elif y >= height:
            candidates.add((min(max(x, 0), width - 1), height - 1))
        else:
            candidates.update(
                {
                    (x, y),
                    (x, y - 1),
                    (x, y + 1),
                    (x - 1, y),
                    (x + 1, y),
                }
            )

        return sorted(
            coord
            for coord in candidates
            if self._coord_inside_map(map_id, coord)
        )

    def _match_pending_reverse_exit_group(
        self,
        candidate: Coord,
    ) -> Optional[ReverseExitGroup]:
        """현재 좌표와 일치하는 pending reverse EXIT를 고른다."""
        normalized = (int(candidate[0]), int(candidate[1]))
        if self._pending_reverse_exit_group is not None:
            group = self._pending_reverse_exit_group
            return group if normalized in set(group.arrival_candidates) else None

        matches = [
            group
            for group in self._pending_reverse_exit_groups
            if normalized in set(group.arrival_candidates)
        ]
        if len(matches) > 1:
            # 도로 connection 은 길 사이 나무·울타리로 여러 묶음으로 나뉘고, 묶음마다 도착 후보가 넓게 잡혀 서로 겹친다 (블루시티 남쪽 (16,35) 가 11~14 묶음과…
            def raw_distance(group: ReverseExitGroup) -> int:
                return min(
                    abs(int(x) - normalized[0]) + abs(int(y) - normalized[1])
                    for x, y in group.raw_coords
                )

            best = min(raw_distance(group) for group in matches)
            matches = [group for group in matches if raw_distance(group) == best]
        if len(matches) != 1:
            return None

        group = matches[0]
        self._pending_reverse_exit_group = group
        self._pending_arrival_candidates = group.arrival_candidates
        self._pending_arrival_candidate_source = group.source
        return group
    def _arrival_candidate_plausible(
        self,
        old_map_id: int,
        new_map_id: int,
        near_old: Optional[Coord],
        old_facing: Optional[int],
        candidate: Coord,
        *,
        allow_timeout_fallback: bool = False,
    ) -> bool:
        """첫 arrival 좌표가 선택된 reverse EXIT 후보인지 엄격히 검사한다."""
        del old_map_id, near_old, old_facing, allow_timeout_fallback
        normalized = (int(candidate[0]), int(candidate[1]))
        if not self._coord_inside_map(int(new_map_id), normalized):
            return False
        return self._match_pending_reverse_exit_group(normalized) is not None
    def _register_destination_entry_portal(
        self,
        local: LocalMap,
        arrival_coord: Coord,
        source_exit: Portal,
        source_name: str,
        reverse_group: ReverseExitGroup,
    ) -> Portal:
        """reverse EXIT의 canonical 좌표로 ENTRY를 만들고 실제 arrival를 기록."""
        entry_direction = OPPOSITE_DIRECTION.get(
            source_exit.direction,
            DIR_INTERNAL,
        )
        portal = local.register_entry_portal(
            entry_direction,
            reverse_group.raw_coords,
            source_name,
            source_map_id=source_exit.map_id,
            arrival_coord=(int(arrival_coord[0]), int(arrival_coord[1])),
        )
        self._ensure_portal_node(portal)
        return portal
    def _handle_map_change(
        self,
        old_map_id: int,
        new_map_id: int,
        new_x: int,
        new_y: int,
        new_facing: Optional[int] = None,
        near_old: Optional[Coord] = None,
        old_facing: Optional[int] = None,
        arrival_pos: Optional[Coord] = None,
        reverse_group: Optional[ReverseExitGroup] = None,
    ) -> None:
        """map ID 쌍으로 선택한 canonical EXIT/ENTRY 전환을 기록한다."""
        old_map_id, new_map_id = int(old_map_id), int(new_map_id)
        new_name = self.map_name(new_map_id)
        old_name = self.map_name(old_map_id)

        old_local = self.get_local_map(old_map_id)
        new_local = self._ensure_local(new_map_id)
        if old_local is None:
            return

        if near_old is None:
            near_old = self._compute_near_old()

        actual_arrival = (
            (int(arrival_pos[0]), int(arrival_pos[1]))
            if arrival_pos is not None
            else (int(new_x), int(new_y))
        )

        # 목적지 ENTRY의 canonical 좌표는 new_map에서 old_map으로 돌아가는 reverse EXIT로 결정한다.
        if reverse_group is None:
            reverse_group = self._select_static_exit_group(
                new_map_id,
                old_map_id,
                hint=actual_arrival,
                allow_last_map=True,
                max_dist=2,
            )
        if reverse_group is None:
            self._portal_log(
                "[portal warning] 목적지 reverse EXIT를 확정하지 못해 전환을 "
                "무시합니다:",
                f"old_map={old_map_id}",
                f"new_map={new_map_id}",
                f"arrival={actual_arrival}",
            )
            return
        if actual_arrival not in set(reverse_group.arrival_candidates):
            self._portal_log(
                "[portal warning] 실제 arrival가 선택된 reverse EXIT 후보 밖이라 "
                "전환을 무시합니다:",
                f"transition={old_map_id}->{new_map_id}",
                f"arrival={actual_arrival}",
                f"reverse={list(reverse_group.raw_coords)}",
                f"expected={list(reverse_group.arrival_candidates)}",
            )
            return

        # 출발 EXIT도 map ID 쌍으로 우선 결정한다.
        source_group = self._select_static_exit_group(
            old_map_id,
            new_map_id,
            hint=near_old,
            allow_last_map=True,
            max_dist=2,
        )
        if source_group is None:
            self._portal_log(
                "[portal warning] 출발 JSON EXIT를 확정하지 못해 전환을 "
                "무시합니다:",
                f"old_map={old_map_id}",
                f"new_map={new_map_id}",
                f"near_old={near_old}",
            )
            return

        source_exit = self._register_static_exit_group(
            old_local,
            source_group,
            new_name,
            new_map_id,
        )
        destination_entry = self._register_destination_entry_portal(
            new_local,
            actual_arrival,
            source_exit,
            old_name,
            reverse_group,
        )

        source_access = self._validated_source_access_coord(
            source_exit,
            near_old,
            old_facing,
        )
        if source_access is not None:
            old_local.record_portal_access(source_exit.key, source_access)

        self._ensure_portal_node(source_exit)
        self._ensure_portal_node(destination_entry)

        existing_edge = self.gate_graph.get_edge_data(
            source_exit.key,
            destination_entry.key,
        )
        inferred_from = None
        if existing_edge is not None:
            inferred_from = existing_edge.get("inferred_from")

        transition_status = self._link_portal_transition(
            source_exit,
            destination_entry,
            confidence=TRANSITION_OBSERVED,
        )
        reverse_confirmed = transition_status == "upgraded"
        if reverse_confirmed:
            self._mark_inferred_reverse_confirmed(inferred_from)
            self._portal_log(
                "[portal observed] inferred reverse를 실제 관측으로 승격:",
                f"{old_map_id}->{new_map_id}",
                f"EXIT={source_exit.key}",
                f"ENTRY={destination_entry.key}",
            )

        self.observed_transitions.append(
            ObservedTransition(
                source_map_id=old_map_id,
                source_exit_coord=(
                    int(source_group.raw_coords[0][0]),
                    int(source_group.raw_coords[0][1]),
                ),
                destination_map_id=new_map_id,
                destination_arrival_coord=actual_arrival,
                source_portal=source_exit.key,
                destination_portal=destination_entry.key,
                source_access_coord=source_access,
                reverse_confirmed=reverse_confirmed,
            )
        )

        self._infer_reciprocal_transition(
            old_local=old_local,
            new_local=new_local,
            source_group=source_group,
            reverse_group=reverse_group,
            source_exit=source_exit,
            destination_entry=destination_entry,
            source_access=source_access,
            old_name=old_name,
            new_name=new_name,
        )
