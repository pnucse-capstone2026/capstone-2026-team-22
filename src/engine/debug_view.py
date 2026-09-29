"""view 플래그와 화면/LocalMap 디버그 매트릭스 출력."""

from __future__ import annotations

import numpy as np

from core.gb_memory import (
    CELL_H,
    CELL_W,
    PLAYER_SCREEN_CELL_X,
    PLAYER_SCREEN_CELL_Y,
    SCREEN_TILE_H,
    SCREEN_TILE_W,
)
from core.portal import GlobalMap

view_flags: dict[str, bool] = {
    "scene": False,
    "block": False,
    "name": False,
    "tiles": False,
}
VIEW_KEYS = frozenset(view_flags)

WAIT_FRAMES = 16


def _print_view_status() -> None:
    parts = [f"{k}={'ON' if v else 'off'}" for k, v in view_flags.items()]
    print("[view] " + " | ".join(parts))


def _set_view_flag(key: str, on: bool) -> None:
    view_flags[key] = on
    print(f"[view] {key} → {'ON' if on else 'OFF'}")


def handle_view_command(line: str) -> None:
    parts = line.strip().lower().split()
    if len(parts) == 1:
        _print_view_status()
        return
    if parts[1] == "all" and len(parts) == 3 and parts[2] in ("on", "off"):
        for k in view_flags:
            view_flags[k] = parts[2] == "on"
        print(f"[view] all → {parts[2].upper()}")
        _print_view_status()
        return
    if len(parts) == 3 and parts[1] in VIEW_KEYS and parts[2] in ("on", "off"):
        _set_view_flag(parts[1], parts[2] == "on")
        return
    print("[view] 사용: view | view scene|block|name|tiles on|off | view all on|off")


def print_block_matrix(gmap: GlobalMap, map_id: int, px: int, py: int) -> None:
    """현재 화면 범위의 to_matrix('block') 숫자 매트릭스."""
    local = gmap.get_local_map(map_id)
    if local is None:
        print("  [M] LocalMap 없음")
        return

    x0 = px - PLAYER_SCREEN_CELL_X
    y0 = py - PLAYER_SCREEN_CELL_Y
    mat, (ox, oy) = local.to_matrix("block")
    print("  [M] to_matrix('block') 화면 영역 "
          f"(플레이어 = ({px},{py}), "
          "-1=미관측 0=통행 1=벽 2=물 3=전투풀 4=U 5=D 6=L 7=R 8=워프):")
    header = "      " + " ".join(f"{x0 + cx:2d}" for cx in range(CELL_W))
    print(header)
    for cy in range(CELL_H):
        ay = y0 + cy
        parts = []
        for cx in range(CELL_W):
            ax = x0 + cx
            if (ax, ay) == (px, py):
                parts.append(" P")
            else:
                mr, mc = ay - oy, ax - ox
                if 0 <= mr < mat.shape[0] and 0 <= mc < mat.shape[1]:
                    parts.append(f"{int(mat[mr, mc]):2d}")
                else:
                    parts.append("-1")
        print(f"  y={ay:3d} " + " ".join(parts))


def print_tile_id_matrix(tiles: np.ndarray, px: int, py: int) -> None:
    """VRAM에서 읽은 BG 타일 ID 20×18."""
    stand_r = PLAYER_SCREEN_CELL_Y * 2 + 1
    stand_c = PLAYER_SCREEN_CELL_X * 2
    print(f"  [T] VRAM 타일 ID {SCREEN_TILE_W}x{SCREEN_TILE_H} "
          f"(hex, * = 플레이어 셀 좌하단 tiles[{stand_r},{stand_c}])")
    print("      " + " ".join(f"{tx:2d}" for tx in range(SCREEN_TILE_W)))
    for ty in range(SCREEN_TILE_H):
        parts = []
        for tx in range(SCREEN_TILE_W):
            tid = int(tiles[ty, tx])
            if ty == stand_r and tx == stand_c:
                parts.append(f"*{tid:02X}")
            else:
                parts.append(f" {tid:02X}")
        print(f"  t={ty:2d} " + " ".join(parts))
    x0 = px - PLAYER_SCREEN_CELL_X
    y0 = py - PLAYER_SCREEN_CELL_Y
    print(f"      맵 셀 좌상단=({x0},{y0}), 플레이어 맵좌표=({px},{py})")


def print_local_grid_coords(gmap: GlobalMap, map_id: int, px: int, py: int) -> None:
    """LocalMap.grid Dict 에 존재하는 (x, y) 좌표 목록 출력."""
    local = gmap.get_local_map(map_id)
    if local is None:
        print("  [G] LocalMap 없음")
        return

    if not local.grid:
        name = gmap.map_name(map_id)
        print(f'  [G] "{name}" (id={map_id}): grid 비어 있음 (아직 관측된 칸 없음)')
        return

    coords = sorted(local.grid.keys(), key=lambda p: (p[1], p[0]))
    xs = [x for x, _ in coords]
    ys = [y for _, y in coords]
    x0 = px - PLAYER_SCREEN_CELL_X
    y0 = py - PLAYER_SCREEN_CELL_Y
    x1 = x0 + CELL_W - 1
    y1 = y0 + CELL_H - 1

    print(
        f'  [G] LocalMap.grid 좌표 {len(coords)}칸 '
        f'(map="{local.map_name}" id={map_id})'
    )
    print(
        f"      bounds: x=[{min(xs)},{max(xs)}] y=[{min(ys)},{max(ys)}]  "
        f"플레이어=({px},{py})  "
        f"현재 화면=[({x0},{y0})~({x1},{y1})]"
    )
    print("      (*=플레이어 위치  +=현재 화면 10×9 안)")

    by_y: dict[int, list[int]] = {}
    for x, y in coords:
        by_y.setdefault(y, []).append(x)

    for y in sorted(by_y):
        parts = []
        for x in by_y[y]:
            if (x, y) == (px, py):
                parts.append(f"({x},{y})*")
            elif x0 <= x <= x1 and y0 <= y <= y1:
                parts.append(f"({x},{y})+")
            else:
                parts.append(f"({x},{y})")
        print(f"  y={y:3d}  " + " ".join(parts))


def print_cell_name_matrix(gmap: GlobalMap, map_id: int, px: int, py: int) -> None:
    """현재 화면 범위의 cell_name(POI)만 표시."""
    local = gmap.get_local_map(map_id)
    if local is None:
        print("  [N] LocalMap 없음")
        return
    x0 = px - PLAYER_SCREEN_CELL_X
    y0 = py - PLAYER_SCREEN_CELL_Y
    print(f"  [N] cell_name 매트릭스 (좌상단 = ({x0},{y0}), "
          f"플레이어 = ({px},{py}), 빈칸=이름 없음)")
    header = "      " + " ".join(f"{x0 + cx:2d}" for cx in range(CELL_W))
    print(header)
    for cy in range(CELL_H):
        ay = y0 + cy
        row = []
        for cx in range(CELL_W):
            ax = x0 + cx
            if (ax, ay) == (px, py):
                row.append("P   ")
            else:
                cell = local.get_cell(ax, ay)
                if cell is None or not cell.cell_name:
                    row.append("....")
                else:
                    row.append(cell.cell_name[:4].ljust(4))
        print(f"  y={ay:3d} " + "  ".join(row))

