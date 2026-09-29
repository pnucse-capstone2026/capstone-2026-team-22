"""Map.py — 포켓몬 레드 Dual Map (GlobalMap + LocalMap)"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np

from .game_data import (
    load_map_consts,
    data_path,
    load_hidden_object_names,
    load_hidden_objects,
    load_map_names,
    load_npc_object_names,
    load_warps,
)
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
from .card_key_doors import CARD_KEY_ITEM, door_tiles, locked_door_cells, upgrade_seen_doors
from .config import cheat
from .map_types import BlockType, Cell, EntityType
from .tile_rules import normalize_screen_tiles, scan_screen_cells, tileset_rules

Coord = Tuple[int, int]


# ===========================================================================
# LocalMap — 한 map_id의 격자 지도 (Dict 저장, 필요 시 numpy 변환)
# ===========================================================================
class LocalMap:
    """한 map_id의 확장형 격자."""

    # 행렬 인코딩 값 (A*가 그대로 쓰도록 공개 상수로 둠).
    MAT_UNKNOWN = -1   # 아직 관측 안 된 칸
    MAT_FREE = 0       # 일반 통행
    MAT_WALL = 1
    MAT_WATER = 2
    MAT_GRASS = 3      # 전투용 풀
    MAT_LEDGE_U = 4
    MAT_LEDGE_D = 5
    MAT_LEDGE_L = 6
    MAT_LEDGE_R = 7
    MAT_WARP = 8

    def __init__(self, map_id: int, map_name: str) -> None:
        self.map_id = int(map_id)
        self.map_name = map_name
        self.grid: Dict[Coord, Cell] = {}
        # 셀이름 → 좌표 목록. 같은 이름의 워프가 여러 칸일 수 있어(2칸짜리 문) 리스트.
        self.name_index: Dict[str, List[Coord]] = {}
        # NPC 이름 → 좌표 목록 (워프 name_index 와 별도).
        # **살아 있는 뷰**다. 화면을 벗어난 칸은 tile_rules 가 매 스캔
        # clear_npc_name 으로 지우므로 여기서도 사라진다. Local A* 의 좌표원.
        self.npc_name_index: Dict[str, List[Coord]] = {}
        # 이 맵에서 **한 번이라도 관측한** NPC 이름.
        self.npc_seen: Set[str] = set()
        # "이 맵은 더 볼 게 없다" (engine/explore_gate.py 가 갱신).
        self.closed: bool = False
        # 그 닫힘이 **비전머신 능력에 걸려 있었는지**.
        self.closed_gated_by: Set[str] = set()
        # (잠만보·바위처럼 줍거나 치울 수 없는 것.) 사람처럼 이름을 붙여 두어 current_npc.json 과 LLM 의 NPC 목록에 나오고, Navigation 이 목표로 찾아갈…
        self.blocking_objects: Dict[Coord, str] = {}
        self.closed_caps: Optional[Tuple[bool, bool]] = None
        # closed      : 이 맵만 본 얕은 판정 — 탐험을 부를지 정한다 final_done  : 여기에 맵 안에서 관측한 워프 하나하나의 너머 구역까지 끝났는가…
        self.final_done: bool = False
        self.portal_done: Dict[Coord, Tuple[bool, str]] = {}
        # 이 맵을 마지막으로 스캔했을 때의 타일셋(wCurMapTileset).
        self.tileset: Optional[int] = None

    @property
    def closed_gated(self) -> bool:
        """닫힘이 비전머신에 걸려 있었는가 (무엇인지는 ``closed_gated_by``)."""
        return bool(self.closed_gated_by)

    # ------------------------------------------------------------------
    # 셀 접근
    # ------------------------------------------------------------------
    def get_cell(self, x: int, y: int) -> Optional[Cell]:
        """특정 위치의 셀."""
        return self.grid.get((int(x), int(y)))

    def ensure_cell(self, x: int, y: int) -> Cell:
        """셀이 없으면 만들어서 반환."""
        key = (int(x), int(y))
        cell = self.grid.get(key)
        if cell is None:
            cell = Cell()
            self.grid[key] = cell
        return cell

    def set_cell_name(self, x: int, y: int, name: str) -> None:
        """POI 이름 등록."""
        cell = self.ensure_cell(x, y)
        if cell.cell_name is None:
            cell.cell_name = name
            cell.block_type = BlockType.WARP
            self.name_index.setdefault(name, []).append((int(x), int(y)))

    def set_npc_name(self, x: int, y: int, name: str) -> None:
        """NPC 이름 등록/갱신."""
        name = str(name).strip()
        if not name:
            return
        x, y = int(x), int(y)
        cell = self.ensure_cell(x, y)
        old = cell.npc_name
        self.npc_seen.add(name)
        if old == name:
            coords = self.npc_name_index.setdefault(name, [])
            if (x, y) not in coords:
                coords.append((x, y))
            return
        if old:
            self._remove_npc_index(old, x, y)
        cell.npc_name = name
        coords = self.npc_name_index.setdefault(name, [])
        if (x, y) not in coords:
            coords.append((x, y))

    def clear_npc_name(self, x: int, y: int) -> None:
        """칸에서 NPC 이름 제거."""
        x, y = int(x), int(y)
        cell = self.grid.get((x, y))
        if cell is None or not cell.npc_name:
            return
        self._remove_npc_index(cell.npc_name, x, y)
        cell.npc_name = None

    def _remove_npc_index(self, name: str, x: int, y: int) -> None:
        coords = self.npc_name_index.get(name)
        if not coords:
            return
        key = (int(x), int(y))
        self.npc_name_index[name] = [c for c in coords if c != key]
        if not self.npc_name_index[name]:
            del self.npc_name_index[name]

    def list_npc_names(self) -> List[str]:
        """지금 화면에 보이는 NPC 이름."""
        return sorted(self.npc_name_index.keys())

    def list_npc_seen(self) -> List[str]:
        """이 맵에서 한 번이라도 본 NPC 이름."""
        return sorted(self.npc_seen)

    def npc_coords(self, name: str) -> List[Coord]:
        return list(self.npc_name_index.get(str(name).strip(), []))

    def grass_coords(self) -> List[Coord]:
        return [
            (x, y)
            for (x, y), cell in self.grid.items()
            if cell.block_type is BlockType.GRASS
        ]

    def bounds(self) -> Optional[Tuple[int, int, int, int]]:
        """관측된 영역의 (min_x, min_y, max_x, max_y)."""
        if not self.grid:
            return None
        xs = [p[0] for p in self.grid]
        ys = [p[1] for p in self.grid]
        return min(xs), min(ys), max(xs), max(ys)

    # ------------------------------------------------------------------
    # numpy 변환
    # ------------------------------------------------------------------
    @staticmethod
    def cell_to_mat(cell: Cell) -> int:
        """A* 입력용 셀 값 — 통행 가능 여부의 단일 기준."""
        if cell.block_type is BlockType.WARP:
            return LocalMap.MAT_WARP
        if cell.entity in (EntityType.NPC, EntityType.OBJECT):
            return LocalMap.MAT_WALL
        if cell.block_type is None:
            return LocalMap.MAT_FREE
        return int(cell.block_type.value)

    def to_matrix(self, kind: str = "block") -> Tuple[np.ndarray, Coord]:
        """Dict grid → numpy matrix 변환."""
        b = self.bounds()
        if b is None:
            return np.zeros((0, 0), dtype=np.int8), (0, 0)
        min_x, min_y, max_x, max_y = b
        h, w = max_y - min_y + 1, max_x - min_x + 1

        if kind == "visited":
            mat = np.full((h, w), self.MAT_UNKNOWN, dtype=np.int8)
            for (x, y), cell in self.grid.items():
                mat[y - min_y, x - min_x] = 1 if cell.visited else 0
        elif kind == "entity":
            code = {EntityType.NPC: 1, EntityType.OBJECT: 2,
                    EntityType.CUT_TREE: 3}
            mat = np.zeros((h, w), dtype=np.int8)
            for (x, y), cell in self.grid.items():
                if cell.entity is not None:
                    mat[y - min_y, x - min_x] = code[cell.entity]
        else:  # "block"
            mat = np.full((h, w), self.MAT_UNKNOWN, dtype=np.int8)
            for (x, y), cell in self.grid.items():
                mat[y - min_y, x - min_x] = self.cell_to_mat(cell)
        return mat, (min_x, min_y)


# ===========================================================================
# GlobalMap — Directed Graph (관리자 / 진입점)
# ===========================================================================
class GlobalMap:
    """맵 간 연결을 담는 방향 그래프 + step 갱신 관리자."""

    def __init__(self, map_data_path: Optional[str] = None) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        # map_id → 문·계단 좌표 (scan 마다 다시 만들지 않는다)
        self._door_warp_cache: Dict[int, Set[Coord]] = {}
        self._reset_tracking()

        # map_data.json: map_id → 이름. 없으면 맵 이름이 "MAP_xx"로 표시된다.
        path = map_data_path or data_path("map_data.json")
        self.map_names: Dict[int, str] = load_map_names(path) if path else {}

        # hidden_objects.json: 포켓몬센터 PC / 벤치 가이 / 체육관 석상 등 스프라이트 RAM에 없어 read_npcs()로는 안 잡히는 BG 상호작용 대상.
        path = data_path("hidden_objects.json")
        self.hidden_objects: Dict[int, Dict[Coord, EntityType]] = (
            load_hidden_objects(path) if path else {}
        )
        self.hidden_object_names: Dict[int, Dict[Coord, str]] = (
            load_hidden_object_names(path) if path else {}
        )

        # npc_objects.json: 맵 오브젝트 const 이름 (좌표 매칭용)
        path = data_path("npc_objects.json")
        self.npc_object_names: Dict[int, Dict[Coord, str]] = (
            load_npc_object_names(path) if path else {}
        )

        # warps.json: 룩업 테이블만 유지.
        path = data_path("warps.json")
        self.warp_at: Dict[int, Dict[Coord, dict]] = {}
        self.map_size_cells: Dict[int, Coord] = {}
        # map_id → warps.json const ("MT_MOON_1F"). 도로·게이트 판정용 (explore_gate).
        self.map_consts: Dict[int, str] = {}
        if path:
            self.warp_at, self.map_size_cells = load_warps(path)
            self.map_consts = load_map_consts(path)
        # 실프 카드키 문: map_id → 아직 안 열린 문 칸.
        self.card_key_doors: Dict[int, Set[Coord]] = {}
        self._card_key_sig: Optional[tuple] = None

    def _refresh_card_key_doors(self, pyboy: Any) -> None:
        """카드키 · 문 플래그를 읽어 ``card_key_doors`` 를 맞춘다 (``update_from_pyboy`` 가 매 스캔 부른다)."""
        from .gb_memory import has_bag_item

        doors: Dict[int, Set[Coord]] = {}
        if cheat("card_key_doors", True) and has_bag_item(pyboy, CARD_KEY_ITEM):
            doors = locked_door_cells(pyboy.memory)
        sig = tuple(sorted((m, tuple(sorted(c))) for m, c in doors.items()))
        if sig == self._card_key_sig:
            return
        self._card_key_sig = sig
        self.card_key_doors = doors
        fixed = 0
        for mid, cells in doors.items():
            local = self.get_local_map(int(mid))
            if local is not None:
                n = upgrade_seen_doors(local, cells)
                if n:
                    fixed += n
                    local.closed = False           # 문 너머가 새로 이어졌다 — 닫힘 표시를 푼다
                    if hasattr(local, "final_done"):
                        local.final_done = False
        if fixed:
            print(f"[map] 카드키 문 {fixed}칸을 '앞에서 A 로 여는 문' 으로 바꿨습니다")

    @property
    def arrival_pending(self) -> bool:
        """맵이 바뀐 뒤 도착 좌표가 아직 확정되지 않았는가 (그동안 지도 스캔이 막혀 있다)."""
        return self._pending_map_change is not None or int(getattr(self, "_map_settle", 0) or 0) > 0

    def _reset_tracking(self) -> None:
        """맵 전환 추적 상태를 초기값으로 되돌린다."""
        self.prev_map_id: Optional[int] = None
        self.prev_pos: Optional[Coord] = None
        # 직전 관측 프레임에서 플레이어가 바라보던 방향(옛 맵 출구 칸 보정용).
        self.prev_facing: Optional[int] = None
        # 맵 전환 직후 VRAM 안정화 대기 (게이트 통과 프레임 카운트).
        self._map_settle: int = 0
        # 맵 전환 보류: (old_map_id, new_map_id, near_old).
        self._pending_map_change: Optional[
            Tuple[int, int, Optional[Coord]]
        ] = None
        # settle 직후 스폰 좌표. 이 칸에서 한 칸 벗어나면 pending flush.
        self._arrival_pos: Optional[Coord] = None
        # 직전 update에서 첫 방문이 있었는지 (Exploration RL step 보상용).
        self.last_first_visit: bool = False

    # ------------------------------------------------------------------
    # 생명주기 (RL 에피소드용)
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """RL 에피소드 리셋."""
        self.graph.clear()
        self._reset_tracking()

    def reset_visited(self) -> None:
        """지도 구조(벽/물/이름/edge)는 남기고 visited만 초기화."""
        for _, data in self.graph.nodes(data=True):
            for cell in data["local"].grid.values():
                cell.visited = False
                cell.visited_count = 0
        self._reset_tracking()

    # ------------------------------------------------------------------
    # 조회
    # ------------------------------------------------------------------
    def map_name(self, map_id: int) -> str:
        return self.map_names.get(int(map_id), f"MAP_{int(map_id)}")

    def get_local_map(self, map_id: int) -> Optional[LocalMap]:
        if int(map_id) not in self.graph:
            return None
        return self.graph.nodes[int(map_id)].get("local")

    def find_map_id_by_name(self, name: str) -> Union[int, bool]:
        """맵 이름 → map_id."""
        for mid in self.graph.nodes:
            if self.graph.nodes[mid]["name"] == name:
                return mid
        return False

    def get_route(
        self, current_map_id: int, target: Union[int, str]
    ) -> Union[List[int], bool]:
        """현재 맵 → 목표 맵의 경유 map_id 리스트 (다익스트라, edge weight 1)."""
        src = int(current_map_id)
        if isinstance(target, str):
            dst = self.find_map_id_by_name(target)
            if dst is False:
                return False
        else:
            dst = int(target)

        if src not in self.graph or dst not in self.graph:
            return False
        try:
            return nx.dijkstra_path(self.graph, src, dst, weight="weight")
        except nx.NetworkXNoPath:
            return False

    def get_summary_text(self) -> str:
        """관측된 맵·연결·POI 요약 텍스트 (디버그 / summary 명령용)."""
        lines: List[str] = []
        for mid in self.graph.nodes:
            name = self.graph.nodes[mid]["name"]
            local: LocalMap = self.graph.nodes[mid]["local"]
            neighbors = [
                self.graph.nodes[n]["name"] for n in self.graph.successors(mid)
            ]
            pois = sorted(local.name_index.keys())
            lines.append(
                f"- {name} (id={mid}): 관측 {len(local.grid)}칸, "
                f"출구→[{', '.join(neighbors) if neighbors else '없음'}], "
                f"POI=[{', '.join(pois) if pois else '없음'}]"
            )
        return "\n".join(lines) if lines else "(아직 아는 맵 없음)"

    # ------------------------------------------------------------------
    # 노드/워프 확보
    # ------------------------------------------------------------------
    def _ensure_local(self, map_id: int) -> LocalMap:
        """LocalMap을 확보한다."""
        map_id = int(map_id)
        node = self.graph.nodes.get(map_id)
        if node is not None and "local" in node:
            return node["local"]

        name = self.map_name(map_id)
        local = LocalMap(map_id, name)
        if map_id not in self.graph:
            self.graph.add_node(map_id, name=name, local=local)
        else:
            self.graph.nodes[map_id]["name"] = name
            self.graph.nodes[map_id]["local"] = local
        return local

    def _door_warp_coords(self, map_id: int) -> Set[Coord]:
        """이 맵의 **문·계단(kind: warp)** 좌표."""
        cached = self._door_warp_cache.get(int(map_id))
        if cached is None:
            cached = {
                (int(x), int(y))
                for (x, y), point in (self.warp_at.get(int(map_id)) or {}).items()
                if str(point.get("kind", "")) != "connection"
            }
            self._door_warp_cache[int(map_id)] = cached
        return cached

    def _confirm_experienced_warp(
        self, local: LocalMap, map_id: int, px: int, py: int
    ) -> None:
        """플레이어가 밟은 칸이 warps.json에 있으면 그 좌표에 W를 찍는다."""
        point = self.warp_at.get(int(map_id), {}).get((int(px), int(py)))
        if point is None:
            return
        cell = local.ensure_cell(int(px), int(py))
        cell.block_type = BlockType.WARP
        dest_id = point.get("dest_id")
        if dest_id is None or cell.cell_name is not None:
            return
        dest_name = self.map_name(int(dest_id))
        cell.cell_name = dest_name
        positions = local.name_index.setdefault(dest_name, [])
        if (int(px), int(py)) not in positions:
            positions.append((int(px), int(py)))

    # ------------------------------------------------------------------
    # step 갱신 — 메인 진입점
    # ------------------------------------------------------------------
    def update(
        self,
        map_id: int,
        player_pos: Coord,
        tiles_20x18: Any,
        tileset_id: int,
        npcs: Optional[Dict[int, Dict[str, int]]] = None,
        grass_tile_id: Optional[int] = None,
        facing: Optional[int] = None,
    ) -> None:
        """1 step마다 호출."""
        map_id = int(map_id)
        px, py = int(player_pos[0]), int(player_pos[1])

        local = self._ensure_local(map_id)
        local.tileset = int(tileset_id)
        scan_screen_cells(
            local,
            player_pos=(px, py),
            tiles=normalize_screen_tiles(tiles_20x18),
            rules=tileset_rules(tileset_id, grass_tile_id),
            npcs=npcs or {},
            hidden_at=self.hidden_objects.get(map_id, {}),
            hidden_names=self.hidden_object_names.get(map_id, {}),
            npc_names=self.npc_object_names.get(map_id, {}),
            static_warps=self.warp_at.get(map_id, {}).keys(),
            door_warps=self._door_warp_coords(map_id),
            card_key_doors=self.card_key_doors.get(map_id),
            card_key_tiles=door_tiles(map_id),
        )

        if self.prev_map_id is not None and self.prev_map_id != map_id:
            self._handle_map_change(self.prev_map_id, map_id, px, py, facing)

        self.last_first_visit = local.ensure_cell(px, py).mark_visited()
        self._confirm_experienced_warp(local, map_id, px, py)

        self.prev_map_id = map_id
        self.prev_pos = (px, py)
        self.prev_facing = facing

    def update_from_pyboy(self, pyboy: Any) -> bool:
        """PyBoy 인스턴스에서 메모리를 읽어 update() 호출."""
        if int(pyboy.memory[ADDR_IN_BATTLE]) != 0:
            return False
        if int(pyboy.memory[ADDR_JOY_IGNORE]) != 0:
            return False
        if (int(pyboy.memory[ADDR_SCX]) % 8 != 0
                or int(pyboy.memory[ADDR_SCY]) % 8 != 0):
            return False

        map_id = int(pyboy.memory[ADDR_MAP_ID])
        px = int(pyboy.memory[ADDR_PLAYER_X])
        py = int(pyboy.memory[ADDR_PLAYER_Y])
        facing = int(pyboy.memory[ADDR_PLAYER_FACING])

        def remember() -> None:
            self.prev_map_id = map_id
            self.prev_pos = (px, py)
            self.prev_facing = facing

        # ── 맵 전환 감지: 보류만. 라벨/edge/그리드는 안에서 1스텝 후 ───
        if self.prev_map_id is not None and self.prev_map_id != map_id:
            if self._pending_map_change is not None:
                # 도착 칸에서 한 걸음도 안 떼고 곧바로 되돌아 나갔다 (connection 경계 왕복).
                old_id, new_id, near_old = self._pending_map_change
                ax, ay = self._arrival_pos or self.prev_pos or (px, py)
                self._handle_map_change(old_id, new_id, ax, ay, None, near_old=near_old)
                self._pending_map_change = None
            self._queue_map_change(self.prev_map_id, map_id)
            remember()
            self._map_settle = MAP_SETTLE_FRAMES
            self._arrival_pos = None
            return True

        # 전환 직후 몇 프레임은 화면이 안정될 때까지 타일 스캔 보류.
        if self._map_settle > 0:
            self._map_settle -= 1
            remember()
            if self._map_settle == 0 and self._pending_map_change is not None:
                # 스폰 좌표 고정 — 여기서 한 칸 움직이면 flush + 스캔 재개
                self._arrival_pos = (px, py)
            return False

        # 보류 중: 스폰에 서 있으면 타일 스캔도 하지 않음
        if self._pending_map_change is not None:
            if self._arrival_pos is None or (px, py) == self._arrival_pos:
                remember()
                return False
            # 1스텝 이동 → edge/이름 적용 후 아래 update()로 그리드 갱신
            old_id, new_id, near_old = self._pending_map_change
            self._handle_map_change(
                old_id, new_id, px, py, facing, near_old=near_old,
            )
            self._pending_map_change = None
            self._arrival_pos = None

        if int(pyboy.memory[ADDR_WALK_COUNTER]) != 0:
            return False

        self._refresh_card_key_doors(pyboy)
        self.update(
            map_id,
            (px, py),
            read_screen_tiles(pyboy),
            int(pyboy.memory[ADDR_TILESET]),
            read_npcs(pyboy),
            int(pyboy.memory[ADDR_GRASS_TILE]),
            facing=facing,
        )
        return True

    # ------------------------------------------------------------------
    # 내부: 맵 전환
    # ------------------------------------------------------------------
    def _compute_near_old(self) -> Optional[Coord]:
        """전환 직전 위치 + 진행 방향 = 옛 맵 출구 칸 힌트."""
        near_old = self.prev_pos
        if near_old is None:
            return None
        d = FACING_DELTA.get(self.prev_facing)
        return near_old if d is None else (near_old[0] + d[0], near_old[1] + d[1])

    def _queue_map_change(self, old_map_id: int, new_map_id: int) -> None:
        """map_id 변경 즉시: LocalMap 확보 + 라벨/edge 보류 (1스텝 후 flush)."""
        near_old = self._compute_near_old()
        self._ensure_local(new_map_id)
        self._pending_map_change = (old_map_id, new_map_id, near_old)
        self._arrival_pos = None

    def _link_maps_bidirectional(self, a: int, b: int) -> None:
        """맵 전환 한 번으로 A↔B 양방향 edge 등록."""
        a, b = int(a), int(b)
        if a == b:
            return
        self.graph.add_edge(a, b, weight=1)
        self.graph.add_edge(b, a, weight=1)

    def _cell_off_map(self, map_id: int, x: int, y: int) -> bool:
        size = self.map_size_cells.get(int(map_id))
        if size is None:
            return False
        w, h = size
        return not (0 <= int(x) < w and 0 <= int(y) < h)

    def _cell_just_outside(self, map_id: int, x: int, y: int) -> bool:
        """맵 밖이되 경계에서 맨해튼 거리 1칸 (connection/가장자리 출구)."""
        size = self.map_size_cells.get(int(map_id))
        if size is None or not self._cell_off_map(map_id, x, y):
            return False
        w, h = size
        x, y = int(x), int(y)
        cx = min(max(x, 0), w - 1)
        cy = min(max(y, 0), h - 1)
        return abs(cx - x) + abs(cy - y) <= 1

    def _label_near_static(
        self,
        local: LocalMap,
        dest_name: str,
        near: Coord,
        dest_map_id: Optional[int],
        *,
        allow_last_map: bool = False,
        max_dist: int = 2,
    ) -> bool:
        """확정된 JSON 좌표 중 이번 전환 위치와 가까운 칸에 이름을 붙인다."""
        points = self.warp_at.get(local.map_id)
        if not points:
            return False
        nx_, ny_ = int(near[0]), int(near[1])
        hit = False
        for (x, y), p in points.items():
            dest_id = p.get("dest_id")
            if dest_id is not None and dest_map_id is not None:
                match = int(dest_id) == int(dest_map_id)
            else:
                match = allow_last_map and dest_id is None
            if not match or abs(x - nx_) + abs(y - ny_) > max_dist:
                continue
            local.set_cell_name(x, y, dest_name)
            hit = True
        return hit

    def _handle_map_change(
        self,
        old_map_id: int,
        new_map_id: int,
        new_x: int,
        new_y: int,
        new_facing: Optional[int] = None,
        near_old: Optional[Coord] = None,
    ) -> None:
        """map_id 변화 시 — 걷기(문/계단/connection)로 확인된 전환만 연결."""
        new_name = self.map_name(new_map_id)
        old_name = self.map_name(old_map_id)

        old_local = self.get_local_map(old_map_id)
        new_local = self._ensure_local(new_map_id)

        if near_old is None:
            near_old = self._compute_near_old()

        near_new = (new_x, new_y)
        d = FACING_DELTA.get(new_facing) if new_facing is not None else None
        if d is not None:
            near_new = (new_x - d[0], new_y - d[1])

        # 1) 문/계단 — JSON 근처만
        used_old = False
        if old_local is not None and near_old is not None:
            used_old = self._label_near_static(
                old_local, new_name, near_old, new_map_id, allow_last_map=True,
            )
        used_new = self._label_near_static(
            new_local, old_name, near_new, old_map_id, allow_last_map=True,
        )

        # 2) connection — 경계 밖 1칸만
        if (
            not used_old
            and old_local is not None
            and near_old is not None
            and self._cell_just_outside(old_map_id, *near_old)
        ):
            old_local.set_cell_name(near_old[0], near_old[1], new_name)
            used_old = True

        if not used_new and self._cell_just_outside(new_map_id, *near_new):
            new_local.set_cell_name(near_new[0], near_new[1], old_name)
            used_new = True

        if used_old or used_new:
            self._link_maps_bidirectional(old_map_id, new_map_id)
