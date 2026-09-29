"""tileset_data.py — pokered 타일셋 데이터 테이블."""

from __future__ import annotations

from typing import Dict, Optional

# 1.
_WALK_OVERWORLD = frozenset({0x00, 0x10, 0x1B, 0x20, 0x21, 0x23, 0x2C, 0x2D,
                             0x2E, 0x30, 0x31, 0x33, 0x39, 0x3C, 0x3E, 0x52,
                             0x54, 0x58, 0x5B})
_WALK_REDS_HOUSE = frozenset({0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x14, 0x1C, 0x1A})
_WALK_MART_PKC = frozenset({0x11, 0x1A, 0x1C, 0x3C, 0x5E})
_WALK_FOREST = frozenset({0x1E, 0x20, 0x2E, 0x30, 0x34, 0x37, 0x39, 0x3A,
                          0x40, 0x51, 0x52, 0x5A, 0x5C, 0x5E, 0x5F})
_WALK_DOJO_GYM = frozenset({0x11, 0x16, 0x19, 0x2B, 0x3C, 0x3D, 0x3F, 0x4A,
                            0x4C, 0x4D, 0x03})
_WALK_HOUSE = frozenset({0x01, 0x12, 0x14, 0x28, 0x32, 0x37, 0x44, 0x54, 0x5C})
_WALK_GATE = frozenset({0x01, 0x12, 0x14, 0x1A, 0x1C, 0x37, 0x38, 0x3B, 0x3C, 0x5E})
_WALK_UNDERGROUND = frozenset({0x0B, 0x0C, 0x13, 0x15, 0x18})
_WALK_SHIP = frozenset({0x04, 0x0D, 0x17, 0x1D, 0x1E, 0x23, 0x34, 0x37, 0x39, 0x4A})
_WALK_SHIP_PORT = frozenset({0x0A, 0x1A, 0x32, 0x3B})
_WALK_CEMETERY = frozenset({0x01, 0x10, 0x13, 0x1B, 0x22, 0x42, 0x52})
_WALK_INTERIOR = frozenset({0x04, 0x0F, 0x15, 0x1F, 0x3B, 0x45, 0x47, 0x55, 0x56})
# $22(구멍)은 통과 화이트리스트에서 제외 — 괴력 바위 구멍은 이동 불가.
_WALK_CAVERN = frozenset({0x05, 0x15, 0x16, 0x18, 0x1A, 0x20, 0x21, 0x2A, 0x2D, 0x30})
_WALK_LOBBY = frozenset({0x14, 0x17, 0x1A, 0x1C, 0x20, 0x38, 0x45})
_WALK_MANSION = frozenset({0x01, 0x05, 0x11, 0x12, 0x14, 0x1A, 0x1C, 0x2C, 0x53})
_WALK_LAB = frozenset({0x0C, 0x26, 0x16, 0x1E, 0x34, 0x37})
_WALK_CLUB = frozenset({0x0F, 0x1A, 0x1F, 0x26, 0x28, 0x29, 0x2C, 0x2D, 0x2E, 0x2F, 0x41})
# $11(구멍)은 통과 화이트리스트에서 제외.
_WALK_FACILITY = frozenset({0x01, 0x10, 0x13, 0x1B, 0x20, 0x21, 0x22,
                            0x30, 0x31, 0x32, 0x42, 0x43, 0x48, 0x52, 0x55,
                            0x58, 0x5E})
_WALK_PLATEAU = frozenset({0x1B, 0x23, 0x2C, 0x2D, 0x3B, 0x45})

# 타일셋 ID는 constants/tileset_constants.asm 순서를 따른다.
TILESET_WALKABLE: Dict[int, frozenset] = {
    0: _WALK_OVERWORLD,
    1: _WALK_REDS_HOUSE, 4: _WALK_REDS_HOUSE,
    2: _WALK_MART_PKC, 6: _WALK_MART_PKC,
    3: _WALK_FOREST,
    5: _WALK_DOJO_GYM, 7: _WALK_DOJO_GYM,
    8: _WALK_HOUSE,
    9: _WALK_GATE, 10: _WALK_GATE, 12: _WALK_GATE,
    11: _WALK_UNDERGROUND,
    13: _WALK_SHIP,
    14: _WALK_SHIP_PORT,
    15: _WALK_CEMETERY,
    16: _WALK_INTERIOR,
    17: _WALK_CAVERN,
    18: _WALK_LOBBY,
    19: _WALK_MANSION,
    20: _WALK_LAB,
    21: _WALK_CLUB,
    22: _WALK_FACILITY,
    23: _WALK_PLATEAU,
}

# 2.
WATER_TILE = 0x14
# 물이 존재하는 타일셋 (data/tilesets/water_tilesets.asm)
WATER_TILESETS = frozenset({0, 3, 5, 7, 13, 14, 17, 22, 23})

# 풀 타일 (data/tilesets/tileset_headers.asm 의 grass tile 컬럼, -1 제외) → 야생 인카운터(전투)용 풀.
TILESET_GRASS_TILE: Dict[int, int] = {
    0: 0x52,    # OVERWORLD
    3: 0x20,    # FOREST
    23: 0x45,   # PLATEAU
}

# 풀베기(Cut) 대상 나무 타일 (engine/overworld/cut.asm UsedCut 판정 그대로)
TILESET_CUT_TREE_TILE: Dict[int, int] = {
    0: 0x3D,    # OVERWORLD cut tree
    7: 0x50,    # GYM cut tree
}

# 괴력 바위 구멍 등 (data/tilesets/warp_pad_hole_tile_ids.asm, value=2 hole) 게임에서는 낙하 워프가 될 수 있으나, 지도/내비에서는…
TILESET_HOLE_TILE: Dict[int, int] = {
    17: 0x22,   # CAVERN
    22: 0x11,   # FACILITY
}

# 워프(문·계단·사다리) 타일 — data/tilesets/warp_tile_ids.asm 그대로.
TILESET_WARP_TILE: Dict[int, frozenset] = {
    0: frozenset({0x1B, 0x58}),              # OVERWORLD (문 매트)
    1: frozenset({0x1A, 0x1C}),              # REDS_HOUSE_1
    2: frozenset({0x5E}),                    # MART
    3: frozenset({0x5A, 0x5C, 0x3A}),        # FOREST
    4: frozenset({0x1A, 0x1C}),              # REDS_HOUSE_2
    5: frozenset({0x4A}),                    # DOJO
    6: frozenset({0x5E}),                    # POKECENTER
    7: frozenset({0x4A}),                    # GYM
    8: frozenset({0x54, 0x5C, 0x32}),        # HOUSE
    9: frozenset({0x3B, 0x1A, 0x1C}),        # FOREST_GATE (fallthrough)
    10: frozenset({0x3B, 0x1A, 0x1C}),       # MUSEUM      (fallthrough)
    11: frozenset({0x13}),                   # UNDERGROUND
    12: frozenset({0x3B, 0x1A, 0x1C}),       # GATE        (fallthrough)
    13: frozenset({0x37, 0x39, 0x1E, 0x4A}),  # SHIP
    14: frozenset(),                         # SHIP_PORT (없음)
    15: frozenset({0x1B, 0x13}),             # CEMETERY    (fallthrough)
    16: frozenset({0x15, 0x55, 0x04}),       # INTERIOR
    17: frozenset({0x18, 0x1A, 0x22}),       # CAVERN (사다리 + $22 구멍)
    18: frozenset({0x1A, 0x1C, 0x38}),       # LOBBY
    19: frozenset({0x1A, 0x1C, 0x53}),       # MANSION
    20: frozenset({0x34}),                   # LAB
    21: frozenset(),                         # CLUB (없음)
    22: frozenset({0x43, 0x58, 0x20, 0x1B, 0x13}),  # FACILITY (fallthrough)
    23: frozenset({0x1B, 0x3B}),             # PLATEAU
}

# 3.
CAVERN_EDGE_WALL_TILESETS = frozenset({17})  # CAVERN

# 위 변 단위 규칙에서 제외할 "밟을 수 있는" 워프 타일 (사다리 등, 구멍 제외).
TILESET_STANDABLE_WARP_TILE: Dict[int, frozenset] = {
    17: frozenset({0x18, 0x1A}),  # CAVERN ladders
}

# 4.
LEDGE_TILESET = 0
LEDGE_TILE_DOWN = frozenset({0x37, 0x36})
LEDGE_TILE_LEFT = frozenset({0x27})
LEDGE_TILE_RIGHT = frozenset({0x0D, 0x1D})
LEDGE_TILE_UP: frozenset = frozenset()   # Gen1 OVERWORLD에는 위로 뛰는 ledge가 없음
ALL_LEDGE_TILES = (
    LEDGE_TILE_DOWN | LEDGE_TILE_LEFT | LEDGE_TILE_RIGHT | LEDGE_TILE_UP
)


def grass_tile_for(tileset_id: int, runtime_value: Optional[int] = None) -> Optional[int]:
    """현재 타일셋의 전투 풀 타일 ID."""
    if runtime_value is not None:
        return None if int(runtime_value) in (0xFF, 0x00) else int(runtime_value)
    return TILESET_GRASS_TILE.get(int(tileset_id))
