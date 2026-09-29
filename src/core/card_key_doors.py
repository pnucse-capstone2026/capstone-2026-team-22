"""실프주식회사 카드키 문 — 카드키가 있으면 벽이 아니라 "앞에서 A 로 여는 오브젝트" 로 본다."""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, Iterable, Optional, Set, Tuple

Coord = Tuple[int, int]

CARD_KEY_ITEM = 0x30          # constants/item_constants.asm CARD_KEY

SILPH_CO_11F = 235

# 문 타일 (bottom-left 타일 = 우리 bl_tile, 게임의 wTileInFrontOfPlayer 와 같은 자리)
FACILITY_DOOR_TILES: FrozenSet[int] = frozenset({0x18, 0x24})
SILPH_11F_DOOR_TILES: FrozenSet[int] = frozenset({0x5E})

# map_id → [(이벤트 이름, (주소, 비트), 문 칸들)]. 이름·비트는 data/events.json 과 같다.
SILPH_DOORS: Dict[int, Tuple[Tuple[str, Tuple[int, int], Tuple[Coord, ...]], ...]] = {
    207: (  # SILPH_CO_2F — 블록 $54 (2,2) (2,5)
        ("Silph Co 2 Unlocked Door1", (0xD826, 5), ((4, 4), (5, 4))),
        ("Silph Co 2 Unlocked Door2", (0xD826, 6), ((4, 10), (5, 10))),
    ),
    208: (  # SILPH_CO_3F — 블록 $5F (4,4) (8,4)
        ("Silph Co 3 Unlocked Door1", (0xD828, 0), ((9, 8), (9, 9))),
        ("Silph Co 3 Unlocked Door2", (0xD828, 1), ((17, 8), (17, 9))),
    ),
    209: (  # SILPH_CO_4F — 블록 $54 (2,6) (6,4)
        ("Silph Co 4 Unlocked Door1", (0xD82A, 0), ((4, 12), (5, 12))),
        ("Silph Co 4 Unlocked Door2", (0xD82A, 1), ((12, 8), (13, 8))),
    ),
    210: (  # SILPH_CO_5F — 블록 $5F (3,2) (3,6) (7,5)
        ("Silph Co 5 Unlocked Door1", (0xD82C, 0), ((7, 4), (7, 5))),
        ("Silph Co 5 Unlocked Door2", (0xD82C, 1), ((7, 12), (7, 13))),
        ("Silph Co 5 Unlocked Door3", (0xD82C, 2), ((15, 10), (15, 11))),
    ),
    211: (  # SILPH_CO_6F — 블록 $5F (2,6)
        ("Silph Co 6 Unlocked Door", (0xD82E, 7), ((5, 12), (5, 13))),
    ),
    212: (  # SILPH_CO_7F — 블록 $54 (5,3) (10,2) (10,6)
        ("Silph Co 7 Unlocked Door1", (0xD830, 4), ((10, 6), (11, 6))),
        ("Silph Co 7 Unlocked Door2", (0xD830, 5), ((20, 4), (21, 4))),
        ("Silph Co 7 Unlocked Door3", (0xD830, 6), ((20, 12), (21, 12))),
    ),
    213: (  # SILPH_CO_8F — 블록 $5F (3,4)
        ("Silph Co 8 Unlocked Door", (0xD832, 0), ((7, 8), (7, 9))),
    ),
    233: (
        ("Silph Co 9 Unlocked Door1", (0xD834, 0), ((3, 8), (3, 9))),
        ("Silph Co 9 Unlocked Door2", (0xD834, 1), ((18, 4), (19, 4))),
        ("Silph Co 9 Unlocked Door3", (0xD834, 2), ((18, 10), (19, 10))),
        ("Silph Co 9 Unlocked Door4", (0xD834, 3), ((11, 12), (11, 13))),
    ),
    234: (  # SILPH_CO_10F — 블록 $54 (5,4)
        ("Silph Co 10 Unlocked Door", (0xD836, 0), ((10, 8), (11, 8))),
    ),
    235: (
        ("Silph Co 11 Unlocked Door", (0xD838, 0), ((6, 13), (7, 13))),
    ),
}


def door_tiles(map_id: int) -> FrozenSet[int]:
    """그 맵에서 카드키 문으로 보는 타일 ID (실프 층이 아니면 빈 집합)."""
    if int(map_id) not in SILPH_DOORS:
        return frozenset()
    return SILPH_11F_DOOR_TILES if int(map_id) == SILPH_CO_11F else FACILITY_DOOR_TILES


def _flag_set(memory: Any, addr_bit: Tuple[int, int]) -> bool:
    addr, bit = addr_bit
    try:
        return bool((int(memory[int(addr)]) >> int(bit)) & 1)
    except Exception:
        return False


def locked_door_cells(memory: Any, map_ids: Optional[Iterable[int]] = None) -> Dict[int, Set[Coord]]:
    """아직 안 열린 문 칸 (map_id → 칸들)."""
    out: Dict[int, Set[Coord]] = {}
    for mid in (map_ids if map_ids is not None else SILPH_DOORS):
        for _name, addr_bit, cells in SILPH_DOORS.get(int(mid), ()):
            if not _flag_set(memory, addr_bit):
                out.setdefault(int(mid), set()).update(cells)
    return out


def mark_door_cell(cell: Any) -> bool:
    """문 칸을 "통행 가능 + OBJECT" 로."""
    from .map_types import BlockType, EntityType

    if cell is None or cell.block_type is BlockType.WARP:
        return False
    changed = cell.block_type is not None or cell.entity is not EntityType.OBJECT
    cell.block_type = None
    cell.entity = EntityType.OBJECT
    return changed


def upgrade_seen_doors(local: Any, cells: Iterable[Coord]) -> int:
    """카드키를 얻기 전에 **벽으로 찍혀 저장된** 문 칸을 고친다."""
    grid = getattr(local, "grid", None)
    if not grid:
        return 0
    n = 0
    for coord in cells:
        if mark_door_cell(grid.get(tuple(coord))):
            n += 1
    return n


__all__ = [
    "CARD_KEY_ITEM",
    "SILPH_DOORS",
    "door_tiles",
    "locked_door_cells",
    "mark_door_cell",
    "upgrade_seen_doors",
]
