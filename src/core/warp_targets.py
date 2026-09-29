"""안 지나간 출구 찾기 — BFS 탐험 · play.py 의 스킬 선택 · SeekWarp 가 같이 쓴다."""

from __future__ import annotations

from typing import Any, List, Set, Tuple

from .closure import closure_for, warp_dest
from .map_types import BlockType

Coord = Tuple[int, int]

_DIRS: Tuple[Tuple[str, Coord], ...] = (
    ("DOWN", (0, 1)), ("UP", (0, -1)), ("LEFT", (-1, 0)), ("RIGHT", (1, 0)),
)


def _static_connection(gmap: Any, map_id: int, coord: Coord) -> bool:
    point = (getattr(gmap, "warp_at", None) or {}).get(int(map_id), {}).get(tuple(coord))
    return bool(point) and str(point.get("kind", "")) == "connection"


def _linked(gmap: Any, a: int, b: int) -> bool:
    graph = getattr(gmap, "graph", None)
    return graph is not None and (graph.has_edge(int(a), int(b)) or graph.has_edge(int(b), int(a)))


def _door_passed(static: dict, grid: dict, coord: Coord) -> bool:
    """이 문을 지나갔는가 — **붙어 있는 같은 목적지 칸** 중 하나라도 밟았으면 지나간 것."""
    def dest_of(info: dict):
        return info.get("dest_id"), str(info.get("dest_const", ""))

    want = dest_of(static.get(coord) or {})
    group, todo = {coord}, [coord]
    while todo:
        cx, cy = todo.pop()
        cell = grid.get((cx, cy))
        if cell is not None and cell.block_type is BlockType.WARP:
            return True
        for nb in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
            info = static.get(nb)
            if nb not in group and info and dest_of(info) == want:
                group.add(nb)
                todo.append(nb)
    return False


def unlinked_exits(gmap: Any, map_id: int) -> List[Coord]:
    """warps.json 이 아는 출구 중 **목적지 맵과 아직 안 이어진** 것."""
    static = (getattr(gmap, "warp_at", None) or {}).get(int(map_id)) or {}
    size = (getattr(gmap, "map_size_cells", None) or {}).get(int(map_id))
    grid = getattr(gmap.get_local_map(int(map_id)), "grid", None) or {}
    out: List[Coord] = []
    for coord, point in static.items():
        key = (int(coord[0]), int(coord[1]))
        in_map = bool(size) and 0 <= key[0] < size[0] and 0 <= key[1] < size[1]
        if in_map and str(point.get("kind", "")) != "connection":
            # **맵 안 출구는 칸으로 본다.** 목적지 맵으로 보면 같은 맵으로 가는 두 번째
            # 출구가 "이미 이어졌다" 로 사라진다 — 돌산터널 남쪽 출구(15,33)는 북쪽 입구와
            # 목적지(10번도로)가 같아서, 한 번도 안 지나갔는데 할 일에서 빠졌다.
            if _door_passed(static, grid, key):
                continue      # 그 문(붙어 있는 칸 포함)을 밟아 봤다
            out.append(key)
            continue
        dest = warp_dest(gmap, map_id, point)
        if dest is None:
            continue
        if _linked(gmap, map_id, dest):
            continue          # 그 문으로 들어왔으면 어디로 이어지는지 이미 안다
        out.append(key)
    return sorted(out)


def _in_map_static_warps(gmap: Any, map_id: int, res: Any) -> Set[Coord]:
    """맵 **안**의 warps.json 워프 중 아직 안 밟아 본 것 (갈 수 있는 것만)."""
    size = (getattr(gmap, "map_size_cells", None) or {}).get(int(map_id))
    static = (getattr(gmap, "warp_at", None) or {}).get(int(map_id)) or {}
    if not static:
        return set()
    local = gmap.get_local_map(int(map_id))
    grid = getattr(local, "grid", None) or {}
    out: Set[Coord] = set()
    for coord, point in static.items():
        if str(point.get("kind", "")) == "connection":
            continue
        if size and not (0 <= coord[0] < size[0] and 0 <= coord[1] < size[1]):
            continue
        key = (int(coord[0]), int(coord[1]))
        cell = grid.get(key)
        if cell is not None and cell.block_type is BlockType.WARP:
            continue                                   # 이미 밟아 본 워프
        if key in res.reachable:
            out.add(key)
    return out


def _off_map_warps(gmap: Any, map_id: int, res: Any) -> Set[Coord]:
    """맵 사각형 밖에 있는 **warp** 출구 중 아직 안 지나간 것 (도달 영역에 붙은 것만)."""
    size = (getattr(gmap, "map_size_cells", None) or {}).get(int(map_id))
    static = (getattr(gmap, "warp_at", None) or {}).get(int(map_id)) or {}
    if not size or not static:
        return set()
    out: Set[Coord] = set()
    for coord, point in static.items():
        if str(point.get("kind", "")) == "connection":
            continue
        if 0 <= coord[0] < size[0] and 0 <= coord[1] < size[1]:
            continue
        dest = warp_dest(gmap, map_id, point)
        if dest is None or _linked(gmap, map_id, dest):
            continue
        if any((coord[0] - dx, coord[1] - dy) in res.reachable for _, (dx, dy) in _DIRS):
            out.add((int(coord[0]), int(coord[1])))
    return out


def seek_targets(
    gmap: Any,
    map_id: int,
    pos: Coord,
    *,
    can_cut: bool = False,
    can_surf: bool = False,
) -> List[Coord]:
    """이 맵에서 **아직 안 통과한** 출구 좌표."""
    res = closure_for(gmap, int(map_id), pos, is_surfing=can_surf, can_cut=can_cut, mark=False)
    local = gmap.get_local_map(int(map_id))
    grid = getattr(local, "grid", None) or {}
    found: Set[Coord] = set(res.unresolved)
    for c in res.edge_exits:
        cell = grid.get(c)
        if cell is None or cell.block_type is BlockType.WARP:
            continue                                  # 이미 지나간 connection
        if _static_connection(gmap, map_id, c):
            found.add(c)
    found |= _in_map_static_warps(gmap, map_id, res)
    found |= _off_map_warps(gmap, map_id, res)
    return sorted(found, key=lambda c: abs(c[0] - pos[0]) + abs(c[1] - pos[1]))


__all__ = ["seek_targets", "unlinked_exits"]
