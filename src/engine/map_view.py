"""LocalMap 장면/POI 터미널 출력 (play.py / test_map.py 공용)."""

from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

from core.Map import GlobalMap, LocalMap
from core.gb_memory import (
    CELL_H,
    CELL_W,
    PLAYER_SCREEN_CELL_X,
    PLAYER_SCREEN_CELL_Y,
)
from core.map_types import BlockType, EntityType

SCENE_LEGEND = (
    "P=플레이어  .=통행  o=통행(방문)  ?=미관측과 맞닿음  #=벽  ~=물  G=전투풀  "
    "U/D/L/R=언덕  W=워프(확정)  w=워프(관측)  T=풀베기나무  N=NPC  O=오브젝트  "
    "(공백)=미관측"
)

_BLOCK_CHAR = {
    BlockType.GRASS: "G",
    BlockType.LEDGE_U: "U",
    BlockType.LEDGE_D: "D",
    BlockType.LEDGE_L: "L",
    BlockType.LEDGE_R: "R",
    BlockType.WALL: "#",
    BlockType.WATER: "~",
    BlockType.WARP: "W",
}


def cell_to_char(cell) -> str:
    if cell is None:
        return " "
    # WARP 최우선 — 오브젝트/NPC와 겹쳐도 W로 표시
    if cell.block_type is BlockType.WARP:
        return "W"
    # 아직 밟지 않았지만 화면 타일로 문/계단이라고 알아본 칸 (폐영역 판정용)
    if cell.warp_candidate:
        return "w"
    if cell.entity is EntityType.NPC:
        return "N"
    if cell.entity is EntityType.OBJECT:
        return "O"
    if cell.entity is EntityType.CUT_TREE:
        return "T"
    if cell.block_type in _BLOCK_CHAR:
        return _BLOCK_CHAR[cell.block_type]
    # 지형 정보를 덮지 않도록 일반 통행칸에만 폐영역 경계를 표시한다.
    if cell.frontier:
        return "?"
    return "o" if cell.visited else "."


def print_scene_matrix(gmap: GlobalMap, map_id: int, px: int, py: int) -> None:
    """현재 화면 범위(10x9 셀)를 LocalMap 기준 문자 매트릭스로 출력."""
    local = gmap.get_local_map(map_id)
    map_name = gmap.map_name(map_id)
    if local is None:
        print(f"  [M] map_id={map_id} name={map_name} LocalMap 없음")
        return

    x0 = px - PLAYER_SCREEN_CELL_X
    y0 = py - PLAYER_SCREEN_CELL_Y

    print(
        f"  [M] map_id={map_id}  name={map_name}  "
        f"player=({px},{py})"
    )
    for cy in range(CELL_H):
        ay = y0 + cy
        row_chars = []
        for cx in range(CELL_W):
            ax = x0 + cx
            if (ax, ay) == (px, py):
                row_chars.append("P")
            else:
                row_chars.append(cell_to_char(local.get_cell(ax, ay)))
        print("      " + "  ".join(row_chars))
    print(f"      ({SCENE_LEGEND})")


def _local_grid_extent(
    local: LocalMap,
) -> Optional[Tuple[int, int, int, int]]:
    """LocalMap.grid + name_index 좌표의 (min_x, min_y, max_x, max_y)."""
    xs: List[int] = []
    ys: List[int] = []
    for x, y in local.grid:
        xs.append(x)
        ys.append(y)
    for coords in local.name_index.values():
        for x, y in coords:
            xs.append(x)
            ys.append(y)
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _iter_poi_lines(local: LocalMap) -> Iterator[str]:
    if not local.name_index:
        yield "  [POI] name_index: (없음)"
        return
    yield f"  [POI] name_index ({len(local.name_index)}종):"
    for name in sorted(local.name_index.keys()):
        for x, y in local.name_index[name]:
            cell = local.get_cell(x, y)
            ch = cell_to_char(cell) if cell is not None else "?"
            yield f'      · "{name}" → ({x},{y}) [{ch}]'


def _iter_matrix_map_lines(
    local: LocalMap,
    min_x: int,
    min_y: int,
    max_x: int,
    max_y: int,
    px: int | None,
    py: int | None,
    chunk_rows: int | None,
) -> Iterator[str]:
    """LocalMap → 2D 문자 행렬 (to_matrix bounds 기준)."""
    yield "      " + " ".join(f"{cx:2d}" for cx in range(min_x, max_x + 1))
    y = min_y
    while y <= max_y:
        if chunk_rows is None:
            y_end = max_y
        else:
            y_end = min(y + chunk_rows - 1, max_y)
        if chunk_rows is not None:
            yield f"  --- y={y:2d} ~ {y_end:2d} ---"
        for cy in range(y, y_end + 1):
            row_chars = []
            for cx in range(min_x, max_x + 1):
                if px is not None and py is not None and (cx, cy) == (px, py):
                    row_chars.append("P")
                else:
                    row_chars.append(cell_to_char(local.get_cell(cx, cy)))
            yield f"  y={cy:3d} " + " ".join(row_chars)
        y = y_end + 1


def iter_full_local_map_lines(
    gmap: GlobalMap,
    map_id: int,
    px: int | None = None,
    py: int | None = None,
    chunk_rows: int | None = None,
) -> Iterator[str]:
    """전체 맵 문자열 라인 생성."""
    local = gmap.get_local_map(map_id)
    if local is None:
        yield "  [F] LocalMap 없음"
        return

    extent = _local_grid_extent(local)
    if extent is None:
        yield f'  [F] "{gmap.map_name(map_id)}" (id={map_id}): 아직 관측된 칸 없음'
        return

    min_x, min_y, max_x, max_y = extent
    n_cells = len(local.grid)
    mat, _ = local.to_matrix("block")
    mat_h, mat_w = mat.shape if mat.size else (0, 0)
    map_name = gmap.map_name(map_id)

    if chunk_rows is None:
        mode = "한 번에"
    else:
        mode = f"y {chunk_rows}줄씩"

    yield (
        f'  [F] "{map_name}" 2D 맵 (id={map_id}, '
        f"bounds ({min_x},{min_y})~({max_x},{max_y}), "
        f"to_matrix {mat_w}x{mat_h}, grid {n_cells}칸, {mode})"
    )
    if px is not None and py is not None:
        yield f"      플레이어 = ({px},{py})"

    yield from _iter_matrix_map_lines(
        local, min_x, min_y, max_x, max_y, px, py, chunk_rows
    )

    yield f"      ({SCENE_LEGEND})"
    yield from _iter_poi_lines(local)


def format_full_local_map_text(
    gmap: GlobalMap,
    map_id: int,
    px: int | None = None,
    py: int | None = None,
    chunk_rows: int | None = None,
) -> str:
    """전체 맵 텍스트 (파일 저장용)."""
    return "\n".join(iter_full_local_map_lines(gmap, map_id, px, py, chunk_rows))


def print_full_local_map(
    gmap: GlobalMap,
    map_id: int,
    px: int | None = None,
    py: int | None = None,
    chunk_rows: int | None = None,
) -> None:
    """현재 map_id LocalMap 전체를 문자 매트릭스로 출력."""
    for line in iter_full_local_map_lines(gmap, map_id, px, py, chunk_rows):
        print(line)
