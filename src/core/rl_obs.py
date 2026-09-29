"""rl_obs.py — Exploration 정책의 관측 생성 (학습/추론 공용)."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Set, Tuple

import numpy as np

from core.map_types import BlockType, EntityType

Coord = Tuple[int, int]

# 관측 창 (이동칸 단위). 화면 = 20x18 타일 = 10x9 이동칸, 플레이어는 (4,4).
WINDOW: Tuple[int, int] = (10, 9)       # (가로 칸 수, 세로 칸 수)
PLAYER_CELL: Tuple[int, int] = (4, 4)   # 창 안에서 플레이어가 서는 (col, row)
# 옛 정사각 창. 호출자는 없지만 grid_size= 로 넘기면 아직 된다.
GRID_SIZE = 21

OBS_VERSION = 4
# 4 = 직사각 화면 창(10x9) · warp_candidate 도 워프로 · 8채널 one-hot(build_channels) 1 = 예전 화면 CNN 관측 2 = 격자 관측…

# 지형 코드 우선순위는 engine/map_view.py 의 cell_to_char 와 **같다**.
T_UNKNOWN = 0    # grid 에 키 없음 (아직 화면에 들어온 적 없는 칸)
T_FREE = 1
T_WALL = 2
T_WATER = 3
T_GRASS = 4
T_LEDGE_U = 5
T_LEDGE_D = 6
T_LEDGE_L = 7
T_LEDGE_R = 8
T_WARP = 9
T_NPC = 10
T_OBJECT = 11
T_CUT_TREE = 12

N_TERRAIN = 13

# 디버그 출력용 — SCENE_LEGEND 와 같은 문자
TERRAIN_CHAR = {
    T_UNKNOWN: " ",
    T_FREE: ".",
    T_WALL: "#",
    T_WATER: "~",
    T_GRASS: "G",
    T_LEDGE_U: "U",
    T_LEDGE_D: "D",
    T_LEDGE_L: "L",
    T_LEDGE_R: "R",
    T_WARP: "W",
    T_NPC: "N",
    T_OBJECT: "O",
    T_CUT_TREE: "T",
}

_BLOCK_TERRAIN = {
    BlockType.WALL: T_WALL,
    BlockType.WATER: T_WATER,
    BlockType.GRASS: T_GRASS,
    BlockType.LEDGE_U: T_LEDGE_U,
    BlockType.LEDGE_D: T_LEDGE_D,
    BlockType.LEDGE_L: T_LEDGE_L,
    BlockType.LEDGE_R: T_LEDGE_R,
}

# 언덕은 일방통행이다. 이 방향으로 들어갈 때만 통과.
_LEDGE_PASS_DELTA = {
    BlockType.LEDGE_U: (0, -1),
    BlockType.LEDGE_D: (0, 1),
    BlockType.LEDGE_L: (-1, 0),
    BlockType.LEDGE_R: (1, 0),
}
# 공개 별칭 — core/closure.py 가 같은 방향 정의를 쓴다.
LEDGE_PASS_DELTA = _LEDGE_PASS_DELTA

# 스칼라 관측의 one-hot 폭.
N_MAP_ROLE = 3


def terrain_code(cell: Any) -> int:
    """Cell → 지형 코드."""
    if cell is None:
        return T_UNKNOWN
    # 워프 최우선 — 오브젝트/NPC 와 겹쳐도 워프로 본다.
    if cell.block_type is BlockType.WARP or getattr(cell, "warp_candidate", False):
        return T_WARP
    if cell.entity is EntityType.NPC:
        return T_NPC
    if cell.entity is EntityType.OBJECT:
        return T_OBJECT
    if cell.entity is EntityType.CUT_TREE:
        return T_CUT_TREE
    block = _BLOCK_TERRAIN.get(cell.block_type)
    if block is not None:
        return block
    return T_FREE


def structural_block(cell: Any, dx: int, dy: int) -> bool:
    """(dx, dy) 로 이 칸에 들어가려 할 때 **구조물**이 막는가."""
    if cell is None:
        return False
    if cell.block_type is BlockType.WARP:
        return False
    if cell.block_type in (BlockType.WALL, BlockType.WATER):
        return True
    pass_delta = _LEDGE_PASS_DELTA.get(cell.block_type)
    if pass_delta is not None:
        return (dx, dy) != pass_delta
    return False


def _window_shape(
    window: Tuple[int, int],
    player_cell: Tuple[int, int],
    grid_size: Optional[int] = None,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """(w, h), (player_col, player_row)."""
    if grid_size is not None:
        g = int(grid_size)
        if g % 2 == 0:
            raise ValueError(f"grid_size 는 홀수여야 한다 (플레이어 중앙 정렬): {g}")
        return (g, g), (g // 2, g // 2)
    w, h = int(window[0]), int(window[1])
    cx, cy = int(player_cell[0]), int(player_cell[1])
    if w <= 0 or h <= 0 or not (0 <= cx < w and 0 <= cy < h):
        raise ValueError(f"창 {window} 안에 플레이어 칸 {player_cell} 이 없다")
    return (w, h), (cx, cy)


def window_origin(
    px: int,
    py: int,
    *,
    window: Tuple[int, int] = WINDOW,
    player_cell: Tuple[int, int] = PLAYER_CELL,
    grid_size: Optional[int] = None,
) -> Tuple[int, int, int, int]:
    """창의 좌상단 맵 좌표와 크기 → (x0, y0, w, h)."""
    (w, h), (cx, cy) = _window_shape(window, player_cell, grid_size)
    return int(px) - cx, int(py) - cy, w, h


def build_grids(
    local_map: Any,
    px: int,
    py: int,
    *,
    visited: Optional[Set[Coord]] = None,
    window: Tuple[int, int] = WINDOW,
    player_cell: Tuple[int, int] = PLAYER_CELL,
    grid_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """플레이어 기준 창 두 장을 만든다."""
    x0, y0, w, h = window_origin(px, py, window=window, player_cell=player_cell, grid_size=grid_size)
    terrain = np.zeros((h, w), dtype=np.uint8)
    seen = np.zeros((h, w), dtype=np.uint8)

    grid = getattr(local_map, "grid", None) if local_map is not None else None
    if grid is None:
        return terrain, seen

    marks = visited if visited is not None else frozenset()
    for row in range(h):
        y = y0 + row
        for col in range(w):
            x = x0 + col
            cell = grid.get((x, y))
            if cell is None:
                continue
            terrain[row, col] = terrain_code(cell)
            if (x, y) in marks:
                seen[row, col] = 1
    return terrain, seen


# 8채널 one-hot — Coverage / SeekWarp 정책 입력 순서를 바꾸면 학습된 체크포인트가 무효가 된다.
CH_UNKNOWN = 0   # 화면에 들어온 적 없는 칸
CH_FREE = 1      # 통행 가능 (풀·언덕 포함 — 언덕의 방향은 화면이 보여 준다)
CH_WALL = 2
CH_WATER = 3
CH_TREE = 4      # 풀베기 나무
CH_WARP = 5      # 확정 워프 + 화면으로 본 문
CH_ACTOR = 6     # NPC · 오브젝트 (비켜 줄 수 있는 것)
CH_VISITED = 7   # 이번 에피소드에 밟은 칸
N_CHANNELS = 8

_TERRAIN_TO_CH = {
    T_UNKNOWN: CH_UNKNOWN,
    T_FREE: CH_FREE,
    T_GRASS: CH_FREE,
    T_LEDGE_U: CH_FREE,
    T_LEDGE_D: CH_FREE,
    T_LEDGE_L: CH_FREE,
    T_LEDGE_R: CH_FREE,
    T_WALL: CH_WALL,
    T_WATER: CH_WATER,
    T_CUT_TREE: CH_TREE,
    T_WARP: CH_WARP,
    T_NPC: CH_ACTOR,
    T_OBJECT: CH_ACTOR,
}


def build_channels(
    local_map: Any,
    px: int,
    py: int,
    *,
    visited: Optional[Set[Coord]] = None,
    window: Tuple[int, int] = WINDOW,
    player_cell: Tuple[int, int] = PLAYER_CELL,
    grid_size: Optional[int] = None,
) -> np.ndarray:
    """(N_CHANNELS, h, w) uint8 one-hot."""
    terrain, seen = build_grids(
        local_map, px, py, visited=visited,
        window=window, player_cell=player_cell, grid_size=grid_size,
    )
    h, w = terrain.shape
    out = np.zeros((N_CHANNELS, h, w), dtype=np.uint8)
    for code, ch in _TERRAIN_TO_CH.items():
        out[ch][terrain == code] = 1
    out[CH_VISITED] = seen
    return out


def upscale2(arr: np.ndarray) -> np.ndarray:
    """이동칸 격자를 2배로 늘려 화면 타일 격자(18x20)와 맞춘다."""
    return np.repeat(np.repeat(arr, 2, axis=-2), 2, axis=-1)


def build_obs(
    local_map: Any,
    px: int,
    py: int,
    *,
    visited: Optional[Set[Coord]] = None,
    map_role: int = 0,
    progress: float = 0.0,
    stall: float = 0.0,
    window: Tuple[int, int] = WINDOW,
    player_cell: Tuple[int, int] = PLAYER_CELL,
    grid_size: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """정책 입력 dict (지형 코드 + 스칼라)."""
    terrain, seen = build_grids(
        local_map, px, py, visited=visited,
        window=window, player_cell=player_cell, grid_size=grid_size,
    )
    return {
        "terrain": terrain,
        "visited": seen,
        "map_role": np.array([int(map_role) % N_MAP_ROLE], dtype=np.uint8),
        "progress": np.array([float(progress)], dtype=np.float32),
        "stall": np.array([float(stall)], dtype=np.float32),
    }


def obs_spec(
    *,
    window: Tuple[int, int] = WINDOW,
    player_cell: Tuple[int, int] = PLAYER_CELL,
    grid_size: Optional[int] = None,
) -> Dict[str, Any]:
    """체크포인트에 같이 저장할 관측 스펙."""
    (w, h), (cx, cy) = _window_shape(window, player_cell, grid_size)
    return {
        "obs_version": OBS_VERSION,
        "window": (w, h),
        "player_cell": (cx, cy),
        "channels": ((N_CHANNELS, h, w), "uint8"),
        "terrain": ((h, w), "uint8"),
        "visited": ((h, w), "uint8"),
        "map_role": ((1,), "uint8"),
        "progress": ((1,), "float32"),
        "stall": ((1,), "float32"),
    }


def render_terrain(
    terrain: np.ndarray,
    visited: Optional[np.ndarray] = None,
    *,
    mark_player: bool = True,
    player_cell: Optional[Tuple[int, int]] = None,
) -> str:
    """관측 격자를 사람이 읽는 문자 그림으로."""
    rows = []
    h, w = terrain.shape
    if player_cell is None:
        cy, cx = h // 2, w // 2
    else:
        cx, cy = int(player_cell[0]), int(player_cell[1])
    for r in range(h):
        line = []
        for c in range(w):
            code = int(terrain[r, c])
            ch = TERRAIN_CHAR.get(code, "?")
            if code == T_FREE and visited is not None and visited[r, c]:
                ch = "o"
            if mark_player and r == cy and c == cx:
                ch = "P"
            line.append(ch)
        rows.append("".join(line))
    return "\n".join(rows)


def coverage(
    local_map: Any,
    visited: Iterable[Coord],
    *,
    map_size: Optional[Coord] = None,
) -> Tuple[int, int, float]:
    """맵 커버리지 = (방문 칸, 통행 가능하다고 관측된 칸, 비율)."""
    grid = getattr(local_map, "grid", None) if local_map is not None else None
    if not grid:
        return 0, 0, 0.0

    def in_map(x: int, y: int) -> bool:
        if map_size is None:
            return True
        w, h = map_size
        return 0 <= x < int(w) and 0 <= y < int(h)

    # 통행 가능 = 벽도 물도 아닌 칸.
    blocked = (BlockType.WALL, BlockType.WATER)
    walkable = {
        (x, y)
        for (x, y), cell in grid.items()
        if in_map(x, y) and cell.block_type not in blocked
    }
    # 실제로 밟은 칸은 정의상 통행 가능하다 — 관측이 늦어 아직 grid 에 반영되지 않았을 수 있으므로 합집합으로 분모를 잡는다.
    marks = {tuple(c) for c in visited if in_map(*c)}
    total = len(walkable | marks)
    hit = len(marks)
    return hit, total, (hit / total if total else 0.0)


# 화면 — 2비트(4단계) 명암, 2배 다운샘플.
SCREEN_SHAPE = (1, 72, 80)
_PIXEL_LEVELS = np.array([0, 85, 153, 255], dtype=np.uint8)


def quantize_screen(gray: np.ndarray) -> np.ndarray:
    """PyBoy 화면 한 채널(144x160 uint8) → (1, 72, 80) uint8, 값 0..3."""
    g = np.asarray(gray)[::2, ::2]
    q = np.digitize(g, _PIXEL_LEVELS, right=True).astype(np.uint8)
    return np.clip(q, 0, 3)[None, :, :]


# Coverage / SeekWarp 정책이 받는 관측. (모양, 최댓값)
GRID_UP_SHAPE = (N_CHANNELS, 2 * WINDOW[1], 2 * WINDOW[0])   # (8, 18, 20)
POLICY_OBS_SPEC = {
    "screen": (SCREEN_SHAPE, 3),
    "grid": (GRID_UP_SHAPE, 1),
}


def policy_obs(
    local_map: Any,
    screen_gray: Any,
    px: int,
    py: int,
    *,
    visited: Optional[Set[Coord]] = None,
) -> Dict[str, np.ndarray]:
    """정책 입력 = 화면 + 화면 크기 격자 8채널(2배 확대)."""
    return {
        "screen": quantize_screen(screen_gray),
        "grid": upscale2(build_channels(local_map, px, py, visited=visited)),
    }
