"""closure.py — "이 맵은 더 볼 게 없다" (기하학적 폐영역) 판정."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Set, Tuple

from .map_types import BlockType, Cell, EntityType
from .rl_obs import LEDGE_PASS_DELTA

Coord = Tuple[int, int]

_DIRS: Tuple[Coord, ...] = ((0, -1), (0, 1), (-1, 0), (1, 0))

# 진입 판정 결과
ENTRY_UNKNOWN = "unknown"    # 미관측 -> 경계가 열려 있다
ENTRY_BLOCK = "block"        # 벽/물/나무/역방향 언덕
ENTRY_STEP = "step"          # 그냥 걸어 들어간다
ENTRY_JUMP = "jump"          # 언덕 — 착지칸까지 한 번에
ENTRY_TERMINAL = "terminal"  # 워프 — 영역에는 넣되 그 너머로 확장하지 않는다

# 능력만 있으면 지나갈 수 있었던 경계 (ClosureResult.gated_by 에 들어가는 값).
# **둘을 구분해 둔다** — 물에 막혀 닫힌 맵은 풀베기를 얻어도 달라지지 않는다.
GATE_CUT_TREE = "cut_tree"   # 풀베기
GATE_WATER = "water"         # 파도타기
GATE_SNORLAX = "snorlax"     # 길을 막고 자는 잠만보 — 포켓몬 피리
GATE_SILPH_SCOPE = "silph_scope"  # 포켓몬 타워 6층 유령 마로왁 — 실프스코프
GATE_LABELS = {GATE_CUT_TREE: "나무", GATE_WATER: "물", GATE_SNORLAX: "잠만보",
               GATE_SILPH_SCOPE: "유령"}
GATE_HM_NAMES = {GATE_CUT_TREE: "풀베기", GATE_WATER: "파도타기", GATE_SNORLAX: "포켓몬 피리",
                 GATE_SILPH_SCOPE: "실프스코프"}
_GATE_ORDER = (GATE_WATER, GATE_CUT_TREE, GATE_SNORLAX, GATE_SILPH_SCOPE)
# 비전머신이 아니라 **물건**으로 여는 경계 — 상태 라벨에 무엇이 필요한지 같이 적는다.
_ITEM_GATES = (GATE_SNORLAX, GATE_SILPH_SCOPE)

# 되돌리므로, 조건이 갖춰지기 전에는 벽으로 친다.
STORY_GATES: dict = {(147, (10, 16)): GATE_SILPH_SCOPE}
_story_gate_open: Optional[Callable[[str], bool]] = None


def set_story_gate_checker(fn: Optional[Callable[[str], bool]]) -> None:
    """경계 이름 → 열렸는가."""
    global _story_gate_open
    _story_gate_open = fn


def story_gate_closed(map_id: Any, coord: Coord) -> Optional[str]:
    """그 칸이 아직 닫힌 스토리 경계면 그 이름."""
    gate = STORY_GATES.get((int(map_id), (int(coord[0]), int(coord[1]))))
    if gate is None or _story_gate_open is None:
        return None
    try:
        return None if _story_gate_open(gate) else gate
    except Exception:
        return None

# 길을 막고 자는 잠만보의 자리 (data/npc_objects.json ROUTE12/16_SNORLAX).
SNORLAX_SPOTS: dict = {23: (10, 62), 27: (26, 10)}   # ROUTE_12, ROUTE_16


def gate_label(gated_by: Any) -> str:
    """{"water"} -> "물", {"cut_tree","water"} -> "물·나무"."""
    got = set(gated_by or ())
    return "·".join(GATE_LABELS[g] for g in _GATE_ORDER if g in got)


def gate_note(gated_by: Any) -> str:
    """상태 라벨 괄호 안에 들어갈 말."""
    label = gate_label(gated_by)
    if not label:
        return ""
    got = set(gated_by or ())
    items = [GATE_HM_NAMES[g] for g in _ITEM_GATES if g in got]
    tail = f" — {' · '.join(items)} 필요" if items else ""
    return f"{label} 너머는 아직{tail}"


def snorlax_blocking(local_map: Any) -> bool:
    """이 맵의 잠만보가 아직 그 자리에 누워 있는가 (관측 기준)."""
    spot = SNORLAX_SPOTS.get(int(getattr(local_map, "map_id", -1) or -1))
    if spot is None:
        return False
    cell = (getattr(local_map, "grid", None) or {}).get(spot)
    return cell is not None and cell.entity is EntityType.OBJECT


@dataclass
class ClosureResult:
    closed: bool
    reason: str
    reachable: Set[Coord] = field(default_factory=set)
    frontier: Set[Coord] = field(default_factory=set)
    warps: Set[Coord] = field(default_factory=set)
    # 맵 밖 좌표까지 포함한 "찾아낸 출구" 전부 (discovered_exits)
    exits: Set[Coord] = field(default_factory=set)
    # **갈 수 있는데** 아직 들어가 보지 않은 문. warps(도달 가능·맵 안) 중
    # block_type 이 아직 WARP 로 확정되지 않은 것. 풀베기 너머처럼 애초에
    # 못 가는 워프는 도달 영역에 없으므로 여기 안 들어온다.
    unresolved: Set[Coord] = field(default_factory=set)
    # 도달 영역에 **붙어 있는** 맵 사각형 밖 출구 칸.
    edge_exits: Set[Coord] = field(default_factory=set)
    # **관측한 맵 안 워프 전부** (집·상점·체육관 문). 도달 영역으로 제한하지
    # 않는다 — 작은 주머니에 서 있으면 문이 하나도 안 들어와 "볼 게 없다 =
    # 완료"가 되어 버린다. 맵 밖(도로 connection)은 여기 안 들어간다.
    observed_warps: Set[Coord] = field(default_factory=set)
    visited: int = 0
    map_size: Optional[Coord] = None
    # 경계에 **능력만 있으면 지나갈 수 있었던 것**이 실제로 있었는가.
    gated_by: Set[str] = field(default_factory=set)

    @property
    def capability_gated(self) -> bool:
        """풀베기/파도타기를 얻으면 다시 열려야 하는 닫힘인가."""
        return bool(self.gated_by)

    @property
    def warp_count(self) -> int:
        return len(self.warps)

    @property
    def exit_count(self) -> int:
        return len(self.exits)


def _is_static_connection(
    static_warps: Optional[Dict[Coord, Dict[str, Any]]], coord: Coord
) -> bool:
    """warps.json 이 그 좌표를 도로 connection 이라고 하는가."""
    if not static_warps:
        return False
    point = static_warps.get((int(coord[0]), int(coord[1])))
    return bool(point) and str(point.get("kind", "")) == "connection"


def _is_static_warp(
    static_warps: Optional[Dict[Coord, Dict[str, Any]]], coord: Coord
) -> bool:
    """warps.json 이 **문·계단(kind: warp)** 이라고 하는 좌표인가."""
    if not static_warps:
        return False
    point = static_warps.get((int(coord[0]), int(coord[1])))
    return bool(point) and str(point.get("kind", "")) != "connection"


def entry_kind(
    cell: Optional[Cell],
    delta: Coord,
    *,
    is_surfing: bool = False,
    can_cut: bool = False,
    npc_blocks: bool = False,
) -> str:
    """``delta`` 방향으로 이 칸에 들어가려 할 때 어떻게 되는가."""
    if cell is None:
        return ENTRY_UNKNOWN

    # 워프가 최우선. 확정(block_type=WARP)이든 관측 후보든 이 맵의 출구다.
    if cell.block_type is BlockType.WARP or cell.warp_candidate:
        return ENTRY_TERMINAL

    if npc_blocks and cell.entity in (EntityType.NPC, EntityType.OBJECT):
        return ENTRY_BLOCK

    # ★ CUT_TREE 를 WALL 보다 먼저 본다.
    if cell.entity is EntityType.CUT_TREE:
        return ENTRY_STEP if can_cut else ENTRY_BLOCK

    if cell.block_type is BlockType.WALL:
        return ENTRY_BLOCK
    if cell.block_type is BlockType.WATER:
        return ENTRY_STEP if is_surfing else ENTRY_BLOCK

    pass_delta = LEDGE_PASS_DELTA.get(cell.block_type)
    if pass_delta is not None:
        # 언덕은 일방통행 — 반대 방향에서는 벽이다.
        return ENTRY_JUMP if tuple(pass_delta) == tuple(delta) else ENTRY_BLOCK

    return ENTRY_STEP


def _gated_reason(
    cell: Optional[Cell], is_surfing: bool, can_cut: bool
) -> Optional[str]:
    """이 칸이 **능력이 없어서만** 막힌 것이면 그 이유."""
    if cell is None:
        return None
    if cell.entity is EntityType.CUT_TREE:
        return None if can_cut else GATE_CUT_TREE
    if cell.block_type is BlockType.WATER:
        return None if is_surfing else GATE_WATER
    return None


def reachable_and_frontier(
    local_map: Any,
    start: Coord,
    map_size: Optional[Coord],
    *,
    is_surfing: bool = False,
    can_cut: bool = False,
    npc_blocks: bool = False,
    static_warps: Optional[Dict[Coord, Dict[str, Any]]] = None,
    offmap_frontier: bool = False,
) -> ClosureResult:
    """시작 칸에서 4방향 BFS."""
    grid: Dict[Coord, Cell] = getattr(local_map, "grid", None) or {}
    if not grid:
        return ClosureResult(False, "no_grid")
    snorlax_spot = SNORLAX_SPOTS.get(int(getattr(local_map, "map_id", -1) or -1))
    map_key = getattr(local_map, "map_id", None)

    # 맵 크기를 모르면 "맵 밖"을 판정할 수 없다.
    if not map_size:
        return ClosureResult(False, "no_map_size")
    w, h = int(map_size[0]), int(map_size[1])
    if w <= 0 or h <= 0:
        return ClosureResult(False, "no_map_size")

    def in_map(p: Coord) -> bool:
        return 0 <= p[0] < w and 0 <= p[1] < h

    start = (int(start[0]), int(start[1]))
    if not in_map(start):
        return ClosureResult(False, "seed_off_map", map_size=(w, h))

    # 물 위에 서 있으면 지금 파도타기 중이다.
    seed = grid.get(start)
    if seed is not None and seed.block_type is BlockType.WATER:
        is_surfing = True

    opts = dict(is_surfing=is_surfing, can_cut=can_cut, npc_blocks=npc_blocks)

    reach: Set[Coord] = {start}
    frontier: Set[Coord] = set()
    gated: Set[str] = set()
    warps: Set[Coord] = set()
    seen: Set[Coord] = {start}
    queue = deque([start])

    if seed is not None and (
        seed.block_type is BlockType.WARP or seed.warp_candidate
    ):
        warps.add(start)

    edge_exits: Set[Coord] = set()
    while queue:
        cur = queue.popleft()
        for delta in _DIRS:
            nxt = (cur[0] + delta[0], cur[1] + delta[1])
            # 맵 밖은 이 맵의 미탐험이 아니다.
            if not in_map(nxt):
                out = grid.get(nxt)
                # out is not None = 그 칸이 화면에 들어온 적 있다(관측됨).
                if out is not None and (
                    out.block_type is BlockType.WARP
                    or out.warp_candidate
                    or _is_static_connection(static_warps, nxt)
                ):
                    edge_exits.add(nxt)
                elif _is_static_warp(static_warps, nxt):
                    # 아직 화면에 안 들어왔어도 warps.json 이 아는 출구다
                    edge_exits.add(nxt)
                if out is None and offmap_frontier:
                    frontier.add(nxt)          # 맵 끝이 모르는 곳과 맞닿았다 = 아직 안 닫혔다
                continue

            if nxt == snorlax_spot and snorlax_blocking(local_map):
                gated.add(GATE_SNORLAX)        # 누워 있는 잠만보 — 피리로 깨워야 지나간다
                continue
            story = story_gate_closed(map_key, nxt) if map_key is not None else None
            if story:
                gated.add(story)               # 스토리 조건 전 — 밟으면 게임이 되돌린다
                continue

            kind = entry_kind(grid.get(nxt), delta, **opts)
            if kind == ENTRY_UNKNOWN:
                frontier.add(nxt)
                continue
            if kind == ENTRY_BLOCK:
                # **warps.json 이 아는 워프 칸은 막혀 보여도 출구로 센다.**
                # 배 타는 곳처럼 문 칸이 벽으로 분류되는 곳이 있어서, 그냥 벽으로
                # 두면 그 맵이 "다 봤다"로 닫혀 버린다.
                if _is_static_warp(static_warps, nxt):
                    warps.add(nxt)
                    continue
                gate = _gated_reason(grid.get(nxt), is_surfing, can_cut)
                if gate:
                    gated.add(gate)
                continue

            reach.add(nxt)
            if kind == ENTRY_TERMINAL:
                # 출구는 영역에 넣되 통과시키지 않는다.
                warps.add(nxt)
                continue

            if kind == ENTRY_JUMP:
                land = (nxt[0] + delta[0], nxt[1] + delta[1])
                if not in_map(land):
                    continue
                land_kind = entry_kind(grid.get(land), delta, **opts)
                if land_kind == ENTRY_UNKNOWN:
                    frontier.add(land)
                    continue
                if land_kind == ENTRY_BLOCK:
                    gate = _gated_reason(grid.get(land), is_surfing, can_cut)
                    if gate:
                        gated.add(gate)
                    continue  # 착지 못 하면 애초에 못 뛴다
                reach.add(land)
                if land_kind == ENTRY_TERMINAL:
                    warps.add(land)
                    continue
                nxt = land

            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)

    visited = sum(
        1
        for p in reach
        if (c := grid.get(p)) is not None and c.visited
    )
    closed = not frontier
    return ClosureResult(
        closed=closed,
        reason="closed" if closed else "frontier",
        reachable=reach,
        frontier=frontier,
        warps=warps,
        edge_exits=edge_exits,
        visited=visited,
        map_size=(w, h),
        gated_by=gated,
    )


def discovered_exits(
    local_map: Any,
    static_warps: Optional[Dict[Coord, Dict[str, Any]]] = None,
) -> Set[Coord]:
    """이 맵에서 찾아낸 출구 좌표 전부."""
    grid: Dict[Coord, Any] = getattr(local_map, "grid", None) or {}
    found: Set[Coord] = set()
    for coord, cell in grid.items():
        if cell.block_type is BlockType.WARP or cell.warp_candidate:
            found.add(coord)
    if static_warps:
        for coord, point in static_warps.items():
            key = (int(coord[0]), int(coord[1]))
            if str(point.get("kind", "")) == "connection" and key in grid:
                found.add(key)
    for coords in (getattr(local_map, "name_index", None) or {}).values():
        for coord in coords:
            found.add(tuple(coord))
    return found


def pick_seed(
    local_map: Any,
    map_size: Optional[Coord],
    *,
    is_surfing: bool = False,
    can_cut: bool = False,
) -> Optional[Coord]:
    """그 맵에서 BFS 를 시작할 칸."""
    grid: Dict[Coord, Cell] = getattr(local_map, "grid", None) or {}
    if not grid or not map_size:
        return None
    w, h = int(map_size[0]), int(map_size[1])
    opts = dict(is_surfing=is_surfing, can_cut=can_cut)
    fallback: Optional[Coord] = None
    for coord, cell in grid.items():
        if not (0 <= coord[0] < w and 0 <= coord[1] < h):
            continue
        # delta 는 언덕 방향 판정용이라 시드에서는 의미가 없다.
        if entry_kind(cell, (0, 0), **opts) != ENTRY_STEP:
            continue
        pos = (int(coord[0]), int(coord[1]))
        if getattr(cell, "visited", False):
            return pos
        if fallback is None:
            fallback = pos
    return fallback


def unresolved_warps(local_map: Any) -> Set[Coord]:
    """**문인 줄은 아는데 아직 들어가 보지 않은** 지점."""
    out: Set[Coord] = set()
    for coord, cell in (getattr(local_map, "grid", None) or {}).items():
        if cell.warp_candidate and cell.block_type is not BlockType.WARP:
            out.add(coord)
    return out


def mark_frontier(local_map: Any, result: ClosureResult) -> None:
    """판정 결과를 Cell 에 남긴다 (지도 출력·다음 판정에서 재사용)."""
    grid: Dict[Coord, Cell] = getattr(local_map, "grid", None) or {}
    touching: Set[Coord] = set()
    for f in result.frontier:
        for delta in _DIRS:
            p = (f[0] - delta[0], f[1] - delta[1])
            if p in result.reachable:
                touching.add(p)
    for coord, cell in grid.items():
        cell.frontier = coord in touching


def observed_in_map_warps(local_map: Any, map_size: Optional[Coord]) -> Set[Coord]:
    """**관측한** 맵 사각형 안의 워프 좌표 전부 (집·상점·체육관 문)."""
    if not map_size:
        return set()
    w, h = int(map_size[0]), int(map_size[1])
    found: Set[Coord] = set()
    for coord, cell in (getattr(local_map, "grid", None) or {}).items():
        if not (cell.block_type is BlockType.WARP or cell.warp_candidate):
            continue
        x, y = int(coord[0]), int(coord[1])
        if 0 <= x < w and 0 <= y < h:
            found.add((x, y))
    return found


_LAST_MAP_CACHE: dict = {}


def warp_dest(gmap: Any, map_id: int, point: dict) -> Optional[int]:
    """warps.json 출구가 이어지는 맵."""
    if str(point.get("dest_const", "")) != "LAST_MAP":
        dest = point.get("dest_id")
        return None if dest is None else int(dest)
    warp_at = getattr(gmap, "warp_at", None) or {}
    table = _LAST_MAP_CACHE.get(id(warp_at))
    if table is None:
        sources: dict = {}
        for src_id, points in warp_at.items():
            for pt in points.values():
                if str(pt.get("kind", "")) == "connection" or pt.get("dest_id") is None:
                    continue
                tgt = [q for q in (warp_at.get(int(pt["dest_id"])) or {}).values()
                       if str(q.get("kind", "")) != "connection"]
                idx = pt.get("dest_warp")
                if isinstance(idx, int) and 1 <= idx <= len(tgt) and \
                        str(tgt[idx - 1].get("dest_const", "")) == "LAST_MAP":
                    sources.setdefault(int(pt["dest_id"]), set()).add(int(src_id))
        table = {m: next(iter(s)) for m, s in sources.items() if len(s) == 1}
        _LAST_MAP_CACHE.clear()
        _LAST_MAP_CACHE[id(warp_at)] = table
    return table.get(int(map_id))


def unpassed_static_exits(gmap: Any, map_id: int, result: "ClosureResult") -> Set[Coord]:
    """warps.json 의 warp 출구 중 **목적지 맵과 아직 안 이어진** 것 (도달 영역에 붙은 것만)."""
    static = (getattr(gmap, "warp_at", None) or {}).get(int(map_id)) or {}
    graph = getattr(gmap, "graph", None)
    out: Set[Coord] = set()
    for coord, point in static.items():
        if str(point.get("kind", "")) == "connection":
            continue
        dest = warp_dest(gmap, map_id, point)
        if dest is None or dest == int(map_id):
            continue
        if graph is not None and (graph.has_edge(int(map_id), dest) or graph.has_edge(dest, int(map_id))):
            continue
        x, y = int(coord[0]), int(coord[1])
        if (x, y) in result.reachable or any((x - dx, y - dy) in result.reachable for dx, dy in _DIRS):
            out.add((x, y))
    return out


# 문을 세지 않는 맵 — **폐영역만 만족하면 다 본 것으로 본다.** 포켓몬센터가 그렇다.
EXIT_FREE_MARKS = ("pokemon center", "pokecenter", "포켓몬센터")


def is_exit_free_map(name: Any) -> bool:
    """이 맵은 출구 조건을 따지지 않는가 (이름으로 본다)."""
    low = str(name or "").lower().replace("_", " ")
    return any(mark in low for mark in EXIT_FREE_MARKS)


def pokecenter_done(local: Any, name: Any = None) -> bool:
    """포켓몬센터는 **들어가서 간호사를 봤으면** 끝이다."""
    if name is None:
        name = getattr(local, "map_name", "")
    if local is None or not is_exit_free_map(name):
        return False
    seen = getattr(local, "npc_seen", None) or ()
    return any("NURSE" in str(n).upper() for n in seen)


def closure_for(
    gmap: Any,
    map_id: int,
    pos: Coord,
    *,
    is_surfing: bool = False,
    can_cut: bool = False,
    npc_blocks: bool = False,
    mark: bool = True,
    offmap_frontier: Optional[bool] = None,
) -> ClosureResult:
    """GlobalMap 편의 래퍼."""
    if offmap_frontier is None:
        from .config import exploration_opt
        offmap_frontier = bool(exploration_opt("close_offmap_frontier", True))
    map_id = int(map_id)
    local = None
    getter = getattr(gmap, "get_local_map", None)
    if getter is not None:
        local = getter(map_id)
    if local is None:
        return ClosureResult(False, "no_local_map")

    size = getattr(gmap, "map_size_cells", {}).get(map_id)
    # warps.json 의 이 맵 항목. 도로 connection 을 관측만으로 출구로 세는 데 쓴다.
    static = (getattr(gmap, "warp_at", None) or {}).get(map_id)
    result = reachable_and_frontier(
        local,
        pos,
        size,
        is_surfing=is_surfing,
        can_cut=can_cut,
        npc_blocks=npc_blocks,
        static_warps=static,
        offmap_frontier=bool(offmap_frontier),
    )
    return _with_exits(local, size, static, result, mark)


def _with_exits(local: Any, size: Optional[Coord], static: Any, result: ClosureResult, mark: bool) -> ClosureResult:
    """BFS 결과에 출구 정보를 붙인다."""
    result.exits = discovered_exits(local, static)
    result.observed_warps = observed_in_map_warps(local, size)
    # 도달 가능한 출구에 한정한다.
    reachable_exits = result.warps | result.edge_exits
    result.unresolved = unresolved_warps(local) & reachable_exits
    if mark:
        mark_frontier(local, result)
    return result


# ---------------------------------------------------------------------------
# 워프 칸 → (목적지 맵, 도착 칸)  — 포탈 단위 종료 판정의 재료
# ---------------------------------------------------------------------------
def warp_index_coord(gmap: Any, dest_map_id: int, index: Any) -> Optional[Coord]:
    """warps.json 의 ``dest_warp`` → 목적지 맵의 그 워프 좌표."""
    points = (getattr(gmap, "warp_at", None) or {}).get(int(dest_map_id)) or {}
    warps = [c for c, p in points.items() if str(p.get("kind", "")) != "connection"]
    try:
        idx = int(index)
    except (TypeError, ValueError):
        return None
    if 1 <= idx <= len(warps):
        return (int(warps[idx - 1][0]), int(warps[idx - 1][1]))
    return None


def static_warp_arrival(gmap: Any, map_id: int, coord: Coord) -> Optional[Tuple[int, Optional[Coord]]]:
    """warps.json 만으로 워프 칸 → (목적지 맵, 도착 칸)."""
    point = ((getattr(gmap, "warp_at", None) or {}).get(int(map_id)) or {}).get(
        (int(coord[0]), int(coord[1]))
    )
    if point is None or str(point.get("kind", "")) == "connection":
        return None
    dest = warp_dest(gmap, map_id, point)
    if dest is None:
        return None
    return int(dest), warp_index_coord(gmap, dest, point.get("dest_warp"))


def experienced_warp_arrival(gmap: Any, map_id: int, coord: Coord) -> Optional[Tuple[int, Coord]]:
    """실제로 지나간 전환(``gmap.observed_transitions``)에서 워프 칸 → (목적지 맵, 도착 칸)."""
    target = (int(coord[0]), int(coord[1]))
    for t in reversed(list(getattr(gmap, "observed_transitions", None) or ())):
        src = getattr(t, "source_portal", None)
        if not src or int(src[0]) != int(map_id):
            continue
        cells = tuple((int(c[0]), int(c[1])) for c in (src[3] or ()))
        if target not in cells:
            continue
        arrival = getattr(t, "destination_arrival_coord", None)
        if arrival is None:
            continue
        return int(t.destination_map_id), (int(arrival[0]), int(arrival[1]))
    return None


def seed_inside_map(coord: Coord, map_size: Coord) -> Coord:
    """맵 밖 좌표(``y == height`` 문 등)를 BFS 시드로 쓸 **바로 안쪽 칸**으로."""
    w, h = int(map_size[0]), int(map_size[1])
    x = min(max(int(coord[0]), 0), w - 1)
    y = min(max(int(coord[1]), 0), h - 1)
    return (x, y)


def region_key(map_id: int, result: ClosureResult) -> Tuple[int, Optional[Coord]]:
    """구역의 정체 — 같은 격자에서 같은 구역이면 같은 값 (memo · 순환 판정용)."""
    return (int(map_id), min(result.reachable) if result.reachable else None)


__all__ = [
    "ClosureResult",
    "is_exit_free_map",
    "gate_note",
    "set_story_gate_checker",
    "story_gate_closed",
    "snorlax_blocking",
    "pokecenter_done",
    "unpassed_static_exits",
    "warp_dest",
    "closure_for",
    "warp_index_coord",
    "static_warp_arrival",
    "experienced_warp_arrival",
    "seed_inside_map",
    "region_key",
    "observed_in_map_warps",
    "discovered_exits",
    "entry_kind",
    "mark_frontier",
    "reachable_and_frontier",
    "_is_static_warp",
    "ENTRY_BLOCK",
    "ENTRY_JUMP",
    "ENTRY_STEP",
    "ENTRY_TERMINAL",
    "ENTRY_UNKNOWN",
]
