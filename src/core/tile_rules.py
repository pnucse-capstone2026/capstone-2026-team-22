"""tile_rules.py — 화면 BG 타일 → 셀 의미(BlockType / EntityType) 판정."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection, Dict, Optional, Tuple

import numpy as np

from .config import cheat
from .gb_memory import (
    CELL_H,
    CELL_W,
    FIRST_STILL_SPRITE_ID,
    SPRITE_SNORLAX,
    PLAYER_SCREEN_CELL_X,
    PLAYER_SCREEN_CELL_Y,
    SCREEN_TILE_H,
    SCREEN_TILE_W,
)
from .map_types import BlockType, Cell, EntityType
from .tileset_data import (
    ALL_LEDGE_TILES,
    CAVERN_EDGE_WALL_TILESETS,
    LEDGE_TILE_DOWN,
    LEDGE_TILE_LEFT,
    LEDGE_TILE_RIGHT,
    LEDGE_TILE_UP,
    LEDGE_TILESET,
    TILESET_CUT_TREE_TILE,
    TILESET_HOLE_TILE,
    TILESET_STANDABLE_WARP_TILE,
    TILESET_WALKABLE,
    TILESET_WARP_TILE,
    WATER_TILE,
    WATER_TILESETS,
    grass_tile_for,
)

Coord = Tuple[int, int]

# 이동칸(2x2) 안에서 각 변을 이루는 (dy, dx) 오프셋.
_EDGE_OFFSETS: Dict[str, Tuple[Coord, Coord]] = {
    "top": ((0, 0), (0, 1)),
    "bottom": ((1, 0), (1, 1)),
    "left": ((0, 0), (1, 0)),
    "right": ((0, 1), (1, 1)),
}

# 한 변 두 칸이 모두 해당 ledge 타일이면 그 방향 언덕으로 확정.
_LEDGE_EDGE_RULES = (
    ("bottom", LEDGE_TILE_DOWN, BlockType.LEDGE_D),
    ("left", LEDGE_TILE_LEFT, BlockType.LEDGE_L),
    ("right", LEDGE_TILE_RIGHT, BlockType.LEDGE_R),
    ("top", LEDGE_TILE_UP, BlockType.LEDGE_U),
)

# hidden_objects.json 의 자판기 라벨(TEXT_CELADONMARTROOF_VENDING_MACHINE1~3)과, 그것들에 붙일 이름.
VENDING_MACHINE_MARK = "VENDING_MACHINE"
VENDING_MACHINE_NAME = "CELADONMARTROOF_VENDING_MACHINE"


# ---------------------------------------------------------------------------
# 2x2 타일 접근자
# ---------------------------------------------------------------------------
def cell_tiles(tiles: np.ndarray, cy: int, cx: int) -> Tuple[int, int, int, int]:
    """이동칸 (cy, cx)의 (TL, TR, BL, BR) 타일 ID."""
    ty, tx = cy * 2, cx * 2
    return (
        int(tiles[ty, tx]),
        int(tiles[ty, tx + 1]),
        int(tiles[ty + 1, tx]),
        int(tiles[ty + 1, tx + 1]),
    )


def edge_tiles(tiles: np.ndarray, cy: int, cx: int, edge: str) -> Tuple[int, int]:
    """이동칸 한 변(top/bottom/left/right) 두 칸의 타일 ID."""
    ty, tx = cy * 2, cx * 2
    a, b = _EDGE_OFFSETS[edge]
    return int(tiles[ty + a[0], tx + a[1]]), int(tiles[ty + b[0], tx + b[1]])


def bl_tile(tiles: np.ndarray, cy: int, cx: int) -> int:
    """게임 CheckTilePassable이 보는 좌하단 타일."""
    return int(tiles[cy * 2 + 1, cx * 2])


# ---------------------------------------------------------------------------
# 타일셋별 판정 테이블
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TilesetRules:
    """한 타일셋의 판정에 필요한 값 묶음."""

    tileset_id: int
    walkable: frozenset
    grass_tile: Optional[int]
    cut_tree_tile: Optional[int]
    hole_tile: Optional[int]
    standable_warp: frozenset
    warp_tiles: frozenset
    has_water: bool
    is_ledge_tileset: bool
    cavern_edge_wall: bool


def tileset_rules(
    tileset_id: int, grass_tile_id: Optional[int] = None
) -> TilesetRules:
    """타일셋 ID(+ wGrassTile 실측값)로 판정 테이블을 만든다."""
    tid = int(tileset_id)
    return TilesetRules(
        tileset_id=tid,
        walkable=TILESET_WALKABLE.get(tid, frozenset()),
        grass_tile=grass_tile_for(tid, grass_tile_id),
        cut_tree_tile=TILESET_CUT_TREE_TILE.get(tid),
        hole_tile=TILESET_HOLE_TILE.get(tid),
        standable_warp=TILESET_STANDABLE_WARP_TILE.get(tid, frozenset()),
        warp_tiles=TILESET_WARP_TILE.get(tid, frozenset()),
        has_water=tid in WATER_TILESETS,
        is_ledge_tileset=tid == LEDGE_TILESET,
        cavern_edge_wall=tid in CAVERN_EDGE_WALL_TILESETS,
    )


# ---------------------------------------------------------------------------
# 개별 판정
# ---------------------------------------------------------------------------
def classify_ledge(tiles: np.ndarray, cy: int, cx: int) -> Optional[BlockType]:
    """2x2 언덕 타일 배치로 점프 방향 판정 (OVERWORLD 전용)."""
    quad = cell_tiles(tiles, cy, cx)
    if sum(1 for t in quad if t in ALL_LEDGE_TILES) >= 3:
        return BlockType.WALL

    for edge, ledge_ids, block in _LEDGE_EDGE_RULES:
        if ledge_ids and all(
            t in ledge_ids for t in edge_tiles(tiles, cy, cx, edge)
        ):
            return block

    bl = quad[2]
    for _edge, ledge_ids, block in _LEDGE_EDGE_RULES:
        if bl in ledge_ids:
            return block
    return None


def cell_has_tile(tiles: np.ndarray, cy: int, cx: int, tile_id: int) -> bool:
    """이동칸 2x2 중 하나라도 해당 타일 ID면 True (해안·파도·풀 보완)."""
    return tile_id in cell_tiles(tiles, cy, cx)


def cavern_edge_looks_like_wall(
    tiles: np.ndarray, cy: int, cx: int, walkable: frozenset
) -> bool:
    """동굴 2x2에서 한 변(위/아래/왼/오)이 통째로 비통과면 벽으로 본다."""
    return any(
        all(t not in walkable for t in edge_tiles(tiles, cy, cx, edge))
        for edge in _EDGE_OFFSETS
    )


def classify_cell_block_type(
    rules: TilesetRules, tiles: np.ndarray, cy: int, cx: int
) -> Optional[BlockType]:
    """화면 셀 (cy, cx)의 block_type을 판정한다 (None=일반 통행)."""
    tile = bl_tile(tiles, cy, cx)

    if rules.has_water and cell_has_tile(tiles, cy, cx, WATER_TILE):
        return BlockType.WATER

    if rules.is_ledge_tileset:
        ledge = classify_ledge(tiles, cy, cx)
        if ledge is not None:
            return ledge

    if tile == rules.cut_tree_tile or tile == rules.hole_tile:
        return BlockType.WALL

    if rules.grass_tile is not None and cell_has_tile(
        tiles, cy, cx, rules.grass_tile
    ):
        return BlockType.GRASS

    # 동굴: 2x2 한 변이 모두 벽이면 BL이 통과 타일이어도 벽.
    if (
        rules.cavern_edge_wall
        and tile not in rules.standable_warp
        and cavern_edge_looks_like_wall(tiles, cy, cx, rules.walkable)
    ):
        return BlockType.WALL

    return None if tile in rules.walkable else BlockType.WALL


# 실내 남쪽 문은 warps.json 이 y=height 이고 화면에 보이는 발판은 그 안쪽 1칸.
_WARP_CANDIDATE_MAX_DIST = 1


def near_static_warp(
    coord: Coord,
    static_warps: Optional[Collection[Coord]],
    *,
    max_dist: int = _WARP_CANDIDATE_MAX_DIST,
) -> bool:
    """이 칸이 warps.json 워프(또는 바로 옆칸)인가."""
    if not static_warps:
        return False
    x, y = int(coord[0]), int(coord[1])
    for wx, wy in static_warps:
        if abs(int(wx) - x) + abs(int(wy) - y) <= max_dist:
            return True
    return False


# ---------------------------------------------------------------------------
# 화면 전체 스캔
# ---------------------------------------------------------------------------
def normalize_screen_tiles(tiles_20x18: Any) -> np.ndarray:
    """화면 타일 입력을 (18, 20) 배열로 맞춘다."""
    tiles = np.asarray(tiles_20x18)
    if tiles.shape == (SCREEN_TILE_H, SCREEN_TILE_W):
        return tiles
    if tiles.shape == (SCREEN_TILE_W, SCREEN_TILE_H):
        return tiles.T
    raise ValueError(f"tiles shape must be (18, 20), got {tiles.shape}")


def scan_screen_cells(
    local: Any,
    *,
    player_pos: Coord,
    tiles: np.ndarray,
    rules: TilesetRules,
    npcs: Dict[int, Dict[str, int]],
    hidden_at: Dict[Coord, EntityType],
    hidden_names: Optional[Dict[Coord, str]] = None,
    npc_names: Optional[Dict[Coord, str]] = None,
    static_warps: Optional[Collection[Coord]] = None,
    door_warps: Optional[Collection[Coord]] = None,
    card_key_doors: Optional[Collection[Coord]] = None,
    card_key_tiles: Collection[int] = (),
) -> None:
    """화면 10x9 이동칸을 LocalMap에 반영한다 (block_type + entity)."""
    px, py = int(player_pos[0]), int(player_pos[1])
    hidden_names = hidden_names or {}
    npc_names = npc_names or {}

    # NPC를 (x, y) → slot 으로 뒤집어 셀 갱신 때 조회
    npc_at: Dict[Coord, int] = {
        (info["x"], info["y"]): slot for slot, info in npcs.items()
    }

    for cy in range(CELL_H):
        for cx in range(CELL_W):
            ax = px + (cx - PLAYER_SCREEN_CELL_X)
            ay = py + (cy - PLAYER_SCREEN_CELL_Y)
            cell: Cell = local.ensure_cell(ax, ay)

            # 경험으로 확정된 W는 화면 타일 재판정으로 덮지 않는다.
            if cell.block_type is not BlockType.WARP:
                cell.block_type = classify_cell_block_type(rules, tiles, cy, cx)

            # 카드키 문 — 타일이 아직 문일 때만 (열리면 바닥 타일이라 평소 판정으로 돌아간다)
            card_door = bool(
                card_key_doors
                and (ax, ay) in card_key_doors
                and bl_tile(tiles, cy, cx) in card_key_tiles
                and cell.block_type is not BlockType.WARP
            )
            if card_door:
                cell.block_type = None

            # 관측만으로 알 수 있는 "여기는 문/계단/사다리다" 표시.
            if door_warps and (ax, ay) in door_warps:
                cell.warp_candidate = True

            # NPC가 떠난 자리는 마지막 else에서 None으로 지워져 자가 수정된다.
            slot = npc_at.get((ax, ay))
            hidden = hidden_at.get((ax, ay))
            if slot is not None:
                cell.entity = (
                    EntityType.OBJECT
                    if npcs[slot]["picture"] >= FIRST_STILL_SPRITE_ID
                    else EntityType.NPC
                )
                if cell.entity is EntityType.NPC:
                    # 우선순위: npc_objects const → hidden 라벨 → picture
                    label = (
                        npc_names.get((ax, ay))
                        or hidden_names.get((ax, ay))
                        or f"NPC_{npcs[slot]['picture']}"
                    )
                    if hasattr(local, "set_npc_name"):
                        local.set_npc_name(ax, ay, label)
                else:
                    # 오브젝트.
                    keep = (getattr(local, "blocking_objects", None) or {}).get((ax, ay))
                    # 잠만보는 앞에서 눌러 보기 전에도 이름을 붙인다.
                    if (
                        not keep
                        and npcs[slot]["picture"] == SPRITE_SNORLAX
                        and cheat("snorlax_as_npc", True)
                    ):
                        keep = npc_names.get((ax, ay))
                    if keep and hasattr(local, "set_npc_name"):
                        local.set_npc_name(ax, ay, keep)
                    elif hasattr(local, "clear_npc_name"):
                        local.clear_npc_name(ax, ay)
            elif hidden is not None:
                # hidden event는 정적 데이터라 관측과 무관하게 항상 유지
                cell.entity = hidden
                if hidden is EntityType.NPC:
                    label = (
                        npc_names.get((ax, ay))
                        or hidden_names.get((ax, ay))
                        or f"HiddenNPC_{ax}_{ay}"
                    )
                    if hasattr(local, "set_npc_name"):
                        local.set_npc_name(ax, ay, label)
                elif (
                    VENDING_MACHINE_MARK in hidden_names.get((ax, ay), "")
                    and cheat("buy_saffron_drink", False)
                    and hasattr(local, "set_npc_name")
                ):
                    # 자판기는 사람이 아니라 이름이 없다 — 그러면 플래너가 `목표 NPC:` 로 지목할 수단이 없고, 음료를 못 사니 노랑시티 게이트가 영영 안 열린다.
                    local.set_npc_name(ax, ay, VENDING_MACHINE_NAME)
            elif card_door:
                cell.entity = EntityType.OBJECT
                if hasattr(local, "clear_npc_name"):
                    local.clear_npc_name(ax, ay)
            elif (
                rules.cut_tree_tile is not None
                and bl_tile(tiles, cy, cx) == rules.cut_tree_tile
            ):
                cell.entity = EntityType.CUT_TREE
                if hasattr(local, "clear_npc_name"):
                    local.clear_npc_name(ax, ay)
            else:
                cell.entity = None
                if hasattr(local, "clear_npc_name"):
                    local.clear_npc_name(ax, ay)
                # 막던 오브젝트가 사라졌다 (깨운 잠만보 등) — 기록도 지운다
                (getattr(local, "blocking_objects", None) or {}).pop((ax, ay), None)
