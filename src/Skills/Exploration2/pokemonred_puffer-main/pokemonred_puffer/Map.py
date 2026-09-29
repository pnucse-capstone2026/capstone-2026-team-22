"""
Map.py — 포켓몬 레드 Dual Map (GlobalMap + LocalMap + Cell)

구조
----------------------------------------------------------------
GlobalMap (nx.DiGraph, 관리자 겸 진입점)
  └─ node[map_id] = { "name": str, "local": LocalMap }
       └─ LocalMap.grid: Dict[(x, y), Cell]   # 이동칸(2x2 타일) 단위

데이터 흐름 (1 step 마다)
----------------------------------------------------------------
PyBoy 메모리
  ├─ BG 타일 20x18   : VRAM 0x9800 + SCX/SCY      (mainSprite.py 방식)
  ├─ 플레이어 좌표    : 0xD362(X), 0xD361(Y)       (이동칸 단위)
  ├─ map_id          : 0xD35E
  ├─ 타일셋 ID       : 0xD367 (wCurMapTileset)
  └─ NPC/오브젝트    : WRAM 0xC100 / 0xC200 슬롯   (test_memory.py 방식)
        ↓
GlobalMap.update_from_pyboy(pyboy)  또는  GlobalMap.update(...)
        ↓
20x18 타일 → 10x9 이동칸(셀) 변환 → Cell(block_type / entity / visited) 갱신
map_id 변화 → Global edge 추가 + 경계 셀에 cell_name(POI) 기록

타일 ID 해석은 pokered-master 디스어셈블 데이터 기반:
  - data/tilesets/collision_tile_ids.asm  → 타일셋별 "통과 가능" 타일 화이트리스트
  - engine/items/item_effects.asm 등       → 물 타일은 전 타일셋 공통 $14
  - data/tilesets/tileset_headers.asm      → 풀 타일 (Overworld $52 등)

이 파일은 '지도' 자체만 담당한다.
A* (Navigation Skill) / Exploration RL / LLM 플래너는 여기서 구현하지 않고,
필요한 정보·함수가 있으면 [첨가] 주석으로 표시했다.

주석 범례
----------------------------------------------------------------
[첨가]      A*/RL/LLM 쪽에서 쓰라고 미리 넣어둔 정보/함수 + 이유
[추후 ...]  실기 검증이 필요하거나, 지금 코드만으로는 동작하지 않는 부분
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np


def _resolve_packaged_data_file(filename: str) -> Optional[Path]:
    """패키지 옆 JSON → 없으면 maze RL 루트 ``data/`` (HRL과 동일 정본)."""
    local = Path(__file__).resolve().parent / filename
    if local.is_file():
        return local
    project_data = Path(__file__).resolve().parents[4] / "data" / filename
    if project_data.is_file():
        return project_data
    return None


# ===========================================================================
# 1. PyBoy / PokeRed 메모리 주소  (mainSprite.py, game_data/test_memory.py 참조)
# ===========================================================================
ADDR_PLAYER_X = 0xD362        # 플레이어 X (이동칸 단위, 맵 로컬 좌표)
ADDR_PLAYER_Y = 0xD361        # 플레이어 Y
ADDR_MAP_ID = 0xD35E          # 현재 map_id
ADDR_TILESET = 0xD367         # wCurMapTileset (test_map.py [A]에서 sym 대조 PASS)

ADDR_SCY = 0xFF42             # BG 스크롤 Y
ADDR_SCX = 0xFF43             # BG 스크롤 X
VRAM_BG_BASE = 0x9800         # BG 타일맵 시작 주소

ADDR_IN_BATTLE = 0xD057       # 0이 아니면 배틀 중 → 지도 갱신 스킵
ADDR_GRASS_TILE = 0xD535      # wGrassTile — 현재 타일셋의 전투 풀 타일 ID (0xFF=없음)
ADDR_JOY_IGNORE = 0xCD6B      # wJoyIgnore — EnterMap/대화/스크립트 중 non-zero
# EnterMap은 LoadMapData 동안 입력을 잠그므로, 이때 VRAM을 읽으면 페이드·로딩
# 중간 화면이 지도에 오염된다. update_from_pyboy 게이트로 사용.
ADDR_WALK_COUNTER = 0xCFC5    # wWalkCounter — 0=정지/걸음 종료, 1~8=걸음 애니 중
# 걸음 애니 중에는 SCX/SCY만 먼저 바뀌고 px/py는 마지막 프레임에 갱신된다.
# !=0 이면 grid 스캔 스kip → 가장자리 셀에 '다음 칸' 타일이 잘못 기록되는 것 방지.

# 맵 전환 감지 직후, 게이트를 통과한 프레임을 이만큼 더 기다린 뒤에야
# 화면 타일을 읽는다. (wCurMap은 페이드보다 먼저 바뀌는 구간 보완)
MAP_SETTLE_FRAMES = 3

SPRITE_STATE1 = 0xC100        # wSpriteStateData1 (슬롯당 0x10바이트, +0 = picture ID)
SPRITE_STATE2 = 0xC200        # wSpriteStateData2 (+4 = mapY, +5 = mapX)
SPRITE_SLOT_SIZE = 0x10
SPRITE_MAP_COORD_OFFSET = 4   # pokered object_event가 좌표를 +4 해서 저장함
# (검증 완료) macros/scripts/maps.asm의 object_event 매크로가 `db \2 + 4` 로
# 좌표를 +4 해서 저장 + 실기에서 NPC 표시 위치가 화면과 일치 확인.
# 주의: 슬롯 0(플레이어)의 C204/C205에는 map 좌표가 들어있지 않음(0) —
#       플레이어 좌표는 반드시 0xD362/0xD361을 쓸 것.

# picture ID 경계: 이 값(FIRST_STILL_SPRITE=$3D) 이상이면 사람이 아닌 오브젝트
# (몬스터볼, 바위, 화석, 잠자는 잠만보 등) — constants/sprite_constants.asm
FIRST_STILL_SPRITE_ID = 0x3D

# 플레이어가 바라보는 방향 (wSpritePlayerStateData1FacingDirection = C109)
ADDR_PLAYER_FACING = 0xC109
FACING_DOWN = 0x00
FACING_UP = 0x04
FACING_LEFT = 0x08
FACING_RIGHT = 0x0C
# 방향 → 앞으로 한 칸 이동 (dx, dy). 맵 전환 시 출구/복귀 칸 추정에 사용.
FACING_DELTA: Dict[int, Tuple[int, int]] = {
    FACING_DOWN: (0, 1),
    FACING_UP: (0, -1),
    FACING_LEFT: (-1, 0),
    FACING_RIGHT: (1, 0),
}

# ===========================================================================
# 2. 화면/좌표 상수
# ===========================================================================
SCREEN_TILE_W = 20            # 화면 타일 가로
SCREEN_TILE_H = 18            # 화면 타일 세로
CELL_W = SCREEN_TILE_W // 2   # 이동칸 가로 = 10
CELL_H = SCREEN_TILE_H // 2   # 이동칸 세로 = 9

# 플레이어가 화면에서 차지하는 이동칸 (10x9 격자 기준)
PLAYER_SCREEN_CELL_X = 4
PLAYER_SCREEN_CELL_Y = 4
# [검증됨] Gen1은 플레이어가 화면 고정 위치에 그려짐. 통상 (4,4)로 알려져 있으나
#            실기에서 mainSprite.py 출력으로 플레이어 스프라이트 타일 위치
#            (타일 (8,8)~(9,9) = 셀 (4,4)) 실측 후 틀리면 이 값만 고치면 됨.

# ===========================================================================
# 3. pokered 타일 데이터
#    (pokered-master/data/tilesets/collision_tile_ids.asm 을 그대로 옮김)
#    각 set은 해당 타일셋에서 "통과 가능한" 타일 ID 화이트리스트.
#    여기 없는 타일 = 벽/장애물.
# ===========================================================================
# 타일셋 ID (constants/tileset_constants.asm)
TILESET_NAMES = {
    0: "OVERWORLD", 1: "REDS_HOUSE_1", 2: "MART", 3: "FOREST",
    4: "REDS_HOUSE_2", 5: "DOJO", 6: "POKECENTER", 7: "GYM",
    8: "HOUSE", 9: "FOREST_GATE", 10: "MUSEUM", 11: "UNDERGROUND",
    12: "GATE", 13: "SHIP", 14: "SHIP_PORT", 15: "CEMETERY",
    16: "INTERIOR", 17: "CAVERN", 18: "LOBBY", 19: "MANSION",
    20: "LAB", 21: "CLUB", 22: "FACILITY", 23: "PLATEAU",
}

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
_WALK_CAVERN = frozenset({0x05, 0x15, 0x18, 0x1A, 0x20, 0x21, 0x22, 0x2A, 0x2D, 0x30})
_WALK_LOBBY = frozenset({0x14, 0x17, 0x1A, 0x1C, 0x20, 0x38, 0x45})
_WALK_MANSION = frozenset({0x01, 0x05, 0x11, 0x12, 0x14, 0x1A, 0x1C, 0x2C, 0x53})
_WALK_LAB = frozenset({0x0C, 0x26, 0x16, 0x1E, 0x34, 0x37})
_WALK_CLUB = frozenset({0x0F, 0x1A, 0x1F, 0x26, 0x28, 0x29, 0x2C, 0x2D, 0x2E, 0x2F, 0x41})
_WALK_FACILITY = frozenset({0x01, 0x10, 0x11, 0x13, 0x1B, 0x20, 0x21, 0x22,
                            0x30, 0x31, 0x32, 0x42, 0x43, 0x48, 0x52, 0x55,
                            0x58, 0x5E})
_WALK_PLATEAU = frozenset({0x1B, 0x23, 0x2C, 0x2D, 0x3B, 0x45})

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

# 물 타일: 물이 있는 모든 타일셋에서 공통으로 $14
# (engine/battle/wild_encounters.asm: "in all tilesets with a water tile, this is its id")
WATER_TILE = 0x14
# 물이 존재하는 타일셋 (data/tilesets/water_tilesets.asm)
WATER_TILESETS = frozenset({0, 3, 5, 7, 13, 14, 17, 22, 23})

# 풀 타일 (data/tilesets/tileset_headers.asm 의 grass tile 컬럼, -1 제외)
# → 야생 인카운터(전투)용 풀. BlockType.GRASS 로 표시 (통행 가능).
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

# 언덕(ledge) — data/tilesets/ledge_tiles.asm 타일 ID + 2x2 배치
# OVERWORLD(0)만. 3칸↑ ledge→WALL; 한 변 두 칸 전부 ledge→LEDGE_*; BL만 ledge→방향 fallback.
LEDGE_TILESET = 0
LEDGE_TILE_DOWN = frozenset({0x37, 0x36})
LEDGE_TILE_LEFT = frozenset({0x27})
LEDGE_TILE_RIGHT = frozenset({0x0D, 0x1D})
LEDGE_TILE_UP: frozenset = frozenset()
ALL_LEDGE_TILES = (
    LEDGE_TILE_DOWN | LEDGE_TILE_LEFT | LEDGE_TILE_RIGHT | LEDGE_TILE_UP
)


def _cell_ledge_positions(
    tiles: np.ndarray, cy: int, cx: int
) -> Tuple[Tuple[int, int], ...]:
    ty0, ty1 = cy * 2, cy * 2 + 1
    tx0, tx1 = cx * 2, cx * 2 + 1
    return ((ty0, tx0), (ty0, tx1), (ty1, tx0), (ty1, tx1))


def _cell_edge_all_ledge_tiles(
    tiles: np.ndarray,
    cy: int,
    cx: int,
    edge: str,
    ledge_ids: frozenset,
) -> bool:
    """2x2 셀 한 변 두 칸 모두 ledge 타일 ID이면 True."""
    if not ledge_ids:
        return False
    ty0, ty1 = cy * 2, cy * 2 + 1
    tx0, tx1 = cx * 2, cx * 2 + 1
    if edge == "top":
        positions = ((ty0, tx0), (ty0, tx1))
    elif edge == "bottom":
        positions = ((ty1, tx0), (ty1, tx1))
    elif edge == "left":
        positions = ((ty0, tx0), (ty1, tx0))
    elif edge == "right":
        positions = ((ty0, tx1), (ty1, tx1))
    else:
        return False
    return all(int(tiles[ty, tx]) in ledge_ids for ty, tx in positions)


def _ledge_from_bl_tile(bl: int) -> Optional[BlockType]:
    """좌하단(BL) 타일 ID만으로 언덕 방향 추정 (게임 충돌과 동일 타일)."""
    if bl in LEDGE_TILE_DOWN:
        return BlockType.LEDGE_D
    if bl in LEDGE_TILE_LEFT:
        return BlockType.LEDGE_L
    if bl in LEDGE_TILE_RIGHT:
        return BlockType.LEDGE_R
    if bl in LEDGE_TILE_UP:
        return BlockType.LEDGE_U
    return None


def classify_ledge(
    tileset_id: int,
    tiles: np.ndarray,
    cy: int,
    cx: int,
) -> Optional[BlockType]:
    """2x2 이동칸에서 언덕 타일 배치로 점프 방향 판정.

    - ledge 타일이 3칸 이상이면 WALL (절벽/장식 혼합)
    - 아래/왼/오/위 '두 칸' 변이 각각 전부 ledge ID일 때 LEDGE_*
    - 그 외 BL(좌하단)만 ledge ID이면 해당 방향 LEDGE_* (fallback)
    """
    if int(tileset_id) != LEDGE_TILESET:
        return None
    ledge_count = sum(
        1
        for ty, tx in _cell_ledge_positions(tiles, cy, cx)
        if int(tiles[ty, tx]) in ALL_LEDGE_TILES
    )
    if ledge_count >= 3:
        return BlockType.WALL
    if _cell_edge_all_ledge_tiles(tiles, cy, cx, "bottom", LEDGE_TILE_DOWN):
        return BlockType.LEDGE_D
    if _cell_edge_all_ledge_tiles(tiles, cy, cx, "left", LEDGE_TILE_LEFT):
        return BlockType.LEDGE_L
    if _cell_edge_all_ledge_tiles(tiles, cy, cx, "right", LEDGE_TILE_RIGHT):
        return BlockType.LEDGE_R
    if _cell_edge_all_ledge_tiles(tiles, cy, cx, "top", LEDGE_TILE_UP):
        return BlockType.LEDGE_U
    bl = int(tiles[cy * 2 + 1, cx * 2])
    return _ledge_from_bl_tile(bl)


def cell_has_water_tile(
    tiles: np.ndarray, cy: int, cx: int, water_tile: int
) -> bool:
    """셀 2x2 타일 중 하나라도 water_tile 이면 True (해안·파도 보완)."""
    for ty in range(cy * 2, cy * 2 + 2):
        for tx in range(cx * 2, cx * 2 + 2):
            if int(tiles[ty, tx]) == water_tile:
                return True
    return False


def cell_has_grass_tile(
    tiles: np.ndarray, cy: int, cx: int, grass_tile: int
) -> bool:
    """셀 2x2 타일 중 하나라도 grass_tile 이면 True (wild_encounters 보완)."""
    for ty in range(cy * 2, cy * 2 + 2):
        for tx in range(cx * 2, cx * 2 + 2):
            if int(tiles[ty, tx]) == grass_tile:
                return True
    return False


# ===========================================================================
# 4. Cell — 이동칸(2x2 타일) 하나
# ===========================================================================
class BlockType(Enum):
    """셀 지형/통행 분류. None=일반 통행 가능.

    to_matrix('block') 인코딩과 .value 가 동일 (0은 None=FREE 로만 사용).
    """
    WALL = 1       # 통행 불가 (벽, 건물, 베기 전 나무 등)
    WATER = 2      # 물 (파도타기 시만 통행)
    GRASS = 3      # 전투용 풀 (통행 가능, 야생 인카운터 구역)
    LEDGE_U = 4    # 언덕 — 위로 점프 진입 [추후 타일 실측]
    LEDGE_D = 5    # 언덕 — 아래로 점프 (플레이어가 북쪽에서 ↓)
    LEDGE_L = 6    # 언덕 — 왼쪽으로 점프
    LEDGE_R = 7    # 언덕 — 오른쪽으로 점프
    WARP = 8       # 워프 셀


class EntityType(Enum):
    """셀 위 상호작용 대상 (NPC·오브젝트·풀베기 나무)."""
    NPC = "npc"            # 사람 NPC (트레이너 포함)
    OBJECT = "object"      # 상호작용 오브젝트 (몬스터볼, 바위, 표지판 스프라이트 등)
    CUT_TREE = "cut_tree"  # 풀베기(Cut) 나무 (block_type=WALL + entity)


@dataclass
class Cell:
    """LocalMap 격자 한 칸 (= 이동칸 1개 = 화면 타일 2x2).

    cell_name  : POI 이름. map_id가 바뀌는 순간 경계(워프/출입구) 셀에
                 map_data.json 의 목적지 맵 이름을 넣고 이후 고정.
                 나머지 셀은 None. (LLM이 목표를 이름으로 던져줄 때 검색용)
    block_type : None=일반 통행 / WALL / WATER / GRASS(전투풀) /
                 LEDGE_U·D·L·R(방향 언덕, 일방 점프)
    entity     : None 또는 EntityType (NPC, OBJECT, CUT_TREE)
    visited    : 에이전트가 실제로 밟았으면 True (기본 False)
    """
    cell_name: Optional[str] = None
    block_type: Optional[BlockType] = None
    entity: Optional[EntityType] = None
    visited: bool = False

    # [첨가] visited_count: Exploration RL이 "새 칸 방문 보상"뿐 아니라
    #        재방문 페널티/카운트 기반 보상을 쓸 수 있게 방문 횟수도 기록.
    visited_count: int = 0

    def is_passable(self, surfing: bool = False) -> bool:
        """이 셀을 지나갈 수 있는가.

        통행 가능: None(일반 바닥), GRASS(전투 풀), WARP(워프 위를 밟아야 함).
        WATER는 파도타기 중에만. WALL·언덕은 False (언덕은 점프로만).
        NPC/OBJECT가 서 있는 칸은 실제로 막혀 있으므로 통행 불가.
        (CUT_TREE는 block_type=WALL 이라 아래에서 자동으로 걸러짐)
        """
        if self.entity in (EntityType.NPC, EntityType.OBJECT):
            return False
        if self.block_type is None or self.block_type in (
            BlockType.GRASS,
            BlockType.WARP,
        ):
            return True
        if self.block_type is BlockType.WATER:
            return surfing
        return False

    def mark_visited(self) -> bool:
        """
        나중에 에이전트가 직접 경험하게 되면서 방문을 추가하도록 함.
        굳이 필요없는데 그냥 셀에다가 기능을 추가함.
        """
        first = not self.visited
        self.visited = True
        self.visited_count += 1
        return first


# ===========================================================================
# 5. LocalMap — 한 map_id의 격자 지도 (Dict 저장, 필요 시 numpy 변환)
# ===========================================================================
class LocalMap:
    """한 map_id의 확장형 격자.

    grid는 Dict[(x, y), Cell] — 관측한 칸만 존재하므로 지도 크기를
    미리 몰라도 되고, 에이전트가 이동하며 자연스럽게 넓어진다.
    A* 등 행렬이 필요한 시점에만 to_matrix()로 numpy 변환.

    좌표계: 게임 WRAM의 맵 로컬 좌표(0xD362/0xD361, 이동칸 단위) 그대로 사용.
    """

    def __init__(self, map_id: int, map_name: str) -> None:
        self.map_id = int(map_id)
        self.map_name = map_name
        self.grid: Dict[Tuple[int, int], Cell] = {}
        # 현재 존재하는 셀이름 목록: {cell_name: (x, y)}
        # 같은 이름의 워프가 여러 칸일 수 있으므로(2칸짜리 문 등) 값은 리스트.
        self.name_index: Dict[str, List[Tuple[int, int]]] = {}

    # ------------------------------------------------------------------
    # 셀 접근
    # ------------------------------------------------------------------
    def get_cell(self, x: int, y: int) -> Optional[Cell]:
        """(요구사항 1) 특정 위치의 셀을 넘긴다. 미관측이면 None."""
        return self.grid.get((int(x), int(y)))

    def ensure_cell(self, x: int, y: int) -> Cell:
        """셀이 없으면 만들어서 반환."""
        key = (int(x), int(y))
        if key not in self.grid:
            self.grid[key] = Cell()
        return self.grid[key]

    def set_cell_name(self, x: int, y: int, name: str) -> None:
        """POI 이름 등록. 이미 이름이 있으면 고정(덮어쓰지 않음)."""
        cell = self.ensure_cell(x, y)
        if cell.cell_name is None:
            cell.cell_name = name
            cell.block_type = BlockType.WARP
            self.name_index.setdefault(name, []).append((int(x), int(y)))

    # ------------------------------------------------------------------
    # 셀이름 검색 (격자 내에서)
    # ------------------------------------------------------------------
    def find_name(self, name: str) -> Union[Tuple[int, int], bool]:
        """(요구사항 3) 셀이름 검색. 있으면 (x, y), 없으면 False.

        같은 이름이 여러 칸이면 첫 번째 등록 칸을 반환.
        (LLM이 "Viridian City로 가"라고 하면 이 함수로 목표 좌표를 얻는다)
        """
        positions = self.name_index.get(name)
        if not positions:
            return False
        return positions[0]

    # ------------------------------------------------------------------
    # numpy 변환용 범위 리턴
    # ------------------------------------------------------------------
    def bounds(self) -> Optional[Tuple[int, int, int, int]]:
        """관측된 영역의 (min_x, min_y, max_x, max_y). 비어 있으면 None."""
        if not self.grid:
            return None
        xs = [p[0] for p in self.grid]
        ys = [p[1] for p in self.grid]
        return min(xs), min(ys), max(xs), max(ys)

    # 행렬 인코딩 값 (A*가 그대로 쓰도록 공개 상수로 둠)
    MAT_UNKNOWN = -1   # 아직 관측 안 된 칸
    MAT_FREE = 0       # 일반 통행
    MAT_WALL = 1
    MAT_WATER = 2
    MAT_GRASS = 3      # 전투용 풀
    MAT_LEDGE_U = 4
    MAT_LEDGE_D = 5
    MAT_LEDGE_L = 6
    MAT_LEDGE_R = 7
    MAT_WARP = 8       # BlockType.WARP.value 와 동일

    @staticmethod
    def block_type_to_mat(bt: Optional[BlockType]) -> int:
        """Cell.block_type → to_matrix('block') 숫자."""
        if bt is None:
            return LocalMap.MAT_FREE
        return int(bt.value)

    @staticmethod
    def cell_to_mat(cell: "Cell") -> int:
        """A* 입력용 셀 값. NPC/OBJECT가 있으면 통행 불가(벽)로 인코딩.

        block_type만 보면 NPC가 서 있는 통행칸이 '통과 가능'으로 나와서
        경로가 NPC를 뚫고 지나가므로, 여기서 벽으로 덮어준다.
        """
        if cell.entity in (EntityType.NPC, EntityType.OBJECT):
            return LocalMap.MAT_WALL
        return LocalMap.block_type_to_mat(cell.block_type)

    def to_matrix(self, kind: str = "block") -> Tuple[np.ndarray, Tuple[int, int]]:
        """(요구사항 2) Dict grid → numpy matrix 변환.

        Parameters
        ----------
        kind : "block"   → 지형 행렬
               -1=미관측 / 0=통행 / 1=벽 / 2=물 / 3=전투풀
               / 4=언덕U / 5=언덕D / 6=언덕L / 7=언덕R / 8=워프
               (NPC·OBJECT가 있는 칸은 통행 불가라 1=벽으로 인코딩)
               "visited" → 방문 행렬 (미관측=-1 / 미방문=0 / 방문=1)
               [첨가] "entity" → 엔티티 행렬 (없음=0 / NPC=1 / OBJECT=2 / CUT_TREE=3)

        Returns
        -------
        (matrix, (origin_x, origin_y))
            matrix[row, col] 은 맵 좌표 (origin_x + col, origin_y + row) 의 값.
            A*가 결과 경로를 다시 맵 좌표로 되돌릴 때 origin을 쓰면 된다.
        """
        b = self.bounds()
        if b is None:
            return np.zeros((0, 0), dtype=np.int8), (0, 0)
        min_x, min_y, max_x, max_y = b
        h = max_y - min_y + 1
        w = max_x - min_x + 1

        if kind == "visited":
            mat = np.full((h, w), self.MAT_UNKNOWN, dtype=np.int8)
            for (x, y), cell in self.grid.items():
                mat[y - min_y, x - min_x] = 1 if cell.visited else 0
        elif kind == "entity":
            entity_code = {EntityType.NPC: 1, EntityType.OBJECT: 2,
                           EntityType.CUT_TREE: 3}
            mat = np.zeros((h, w), dtype=np.int8)
            for (x, y), cell in self.grid.items():
                if cell.entity is not None:
                    mat[y - min_y, x - min_x] = entity_code[cell.entity]
        elif kind == "name":
            names: List[List[str]] = [[""] * w for _ in range(h)]
            for (x, y), cell in self.grid.items():
                if cell.cell_name:
                    names[y - min_y][x - min_x] = cell.cell_name
            return names, (min_x, min_y)
        else:  # "block"
            mat = np.full((h, w), self.MAT_UNKNOWN, dtype=np.int8)
            for (x, y), cell in self.grid.items():
                mat[y - min_y, x - min_x] = self.cell_to_mat(cell)
        return mat, (min_x, min_y)

# ===========================================================================
# 6. GlobalMap — Directed Graph (관리자 / 진입점)
# ===========================================================================
class GlobalMap:
    """맵 간 연결을 담는 방향 그래프 + step 갱신 관리자.

    - node[map_id] = { "name": 맵 이름, "local": LocalMap }
    - edge(u, v) / edge(v, u) = 문·계단·connection으로 걸어 이동한 경우에만
                     양방향 연결. Fly/텔레포트/블랙아웃 등은 연결하지 않음.
    """

    def __init__(self, map_data_path: Optional[str] = None) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()

        # 맵 전환 감지용
        self.prev_map_id: Optional[int] = None
        self.prev_pos: Optional[Tuple[int, int]] = None
        # 직전 관측 프레임에서 플레이어가 바라보던 방향(옛 맵 출구 칸 보정용).
        self.prev_facing: Optional[int] = None
        # 맵 전환 직후 VRAM 안정화 대기 (게이트 통과 프레임 카운트).
        self._map_settle: int = 0
        # 맵 전환 보류: (old_map_id, new_map_id, near_old).
        # 새 맵에서 1스텝 이동한 뒤에야 edge/name_index 갱신.
        self._pending_map_change: Optional[
            Tuple[int, int, Optional[Tuple[int, int]]]
        ] = None
        # settle 직후 스폰 좌표. 이 칸에서 한 칸 벗어나면 pending flush.
        self._arrival_pos: Optional[Tuple[int, int]] = None

        # [첨가] 직전 update에서 첫 방문이 있었는지.
        #        Exploration RL이 step 보상으로 바로 읽어가기 위함.
        self.last_first_visit: bool = False

        # map_data.json: map_id(str) → 이름
        self.map_names: Dict[int, str] = {}
        if map_data_path is None:
            resolved = _resolve_packaged_data_file("map_data.json")
            map_data_path = str(resolved) if resolved is not None else None
        if map_data_path is not None:
            self._load_map_data(map_data_path)
        # [추후] map_data.json이 없으면 맵 이름이 "MAP_xx"로 표시됨.
        #        LLM이 이름으로 목표를 주려면 map_data.json 경로가 반드시 필요.

        # hidden_objects.json: 포켓몬센터 PC / 벤치 가이 / 체육관 석상 등
        # BG 타일 상호작용 대상 (pokered data/events/hidden_events.asm 추출).
        # 스프라이트 RAM에 없어서 read_npcs()로는 절대 안 잡히는 것들.
        # {map_id: {(x, y): EntityType}}
        # 생성: python tools/extract_hidden_objects.py
        self.hidden_objects: Dict[int, Dict[Tuple[int, int], EntityType]] = {}
        hidden_path = _resolve_packaged_data_file("hidden_objects.json")
        if hidden_path is not None:
            self._load_hidden_objects(str(hidden_path))

        # warps.json: 타일셋별 워프 타일 ID + 문/계단 (x,y).
        # connection 은 JSON 시드 안 함 — 맵 전환 시 facing 휴리스틱(예전에 잘 되던 방식).
        # 문/계단 이름만 JSON 근처 칸 + 실제 전환 후.
        # 생성: python tools/extract_warps.py
        self.warp_tiles_by_tileset: Dict[int, frozenset] = {}
        # map_id → {(x,y): point_dict}  (warp_event 만, connection 제외)
        self.warp_at: Dict[int, Dict[Tuple[int, int], dict]] = {}
        # map_id → (width_cells, height_cells)  connection 오프맵 판정용
        self.map_size_cells: Dict[int, Tuple[int, int]] = {}
        warps_path = Path(__file__).resolve().parent / "warps.json"
        if warps_path.exists():
            self._load_warps(str(warps_path))

    def _load_warps(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for ts_str, entry in data.get("tilesets", {}).items():
            try:
                ts_id = int(ts_str)
            except ValueError:
                continue
            self.warp_tiles_by_tileset[ts_id] = frozenset(
                int(t) for t in entry.get("warp_tiles", [])
            )
        for mid_str, entry in data.get("maps", {}).items():
            try:
                mid = int(mid_str)
            except ValueError:
                continue
            w = entry.get("width_cells")
            h = entry.get("height_cells")
            if w is not None and h is not None:
                self.map_size_cells[mid] = (int(w), int(h))
            by_xy: Dict[Tuple[int, int], dict] = {}
            for p in entry.get("points", []):
                # connection 은 런타임 휴리스틱으로만 — JSON 좌표 미리 안 씀
                if p.get("kind") == "connection":
                    continue
                by_xy[(int(p["x"]), int(p["y"]))] = p
            if by_xy:
                self.warp_at[mid] = by_xy

    def _ensure_local(self, map_id: int) -> LocalMap:
        """LocalMap 확보. connection/문 미리 시드하지 않음."""
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

    def _cell_off_map(self, map_id: int, x: int, y: int) -> bool:
        size = self.map_size_cells.get(int(map_id))
        if size is None:
            return False
        w, h = size
        return not (0 <= int(x) < w and 0 <= int(y) < h)

    def _edge_exit_target(
        self, map_id: int, x: int, y: int
    ) -> Tuple[int, int]:
        """맵 경계에 붙은 출구 매트/카펫 → A* 목표를 '한 칸 밖'으로.

        실내 빨간 발판(남쪽 y=h-1 등)은 그 칸에 서면 A*가 도착 처리만 하고
        한 걸음 더 안 나가서 전환이 안 된다. 경계 밖 칸을 목표로 두면
        매트를 밟고 밖으로 나가려다 워프가 걸린다.
        맵 안쪽 계단·워프패드는 좌표 그대로.
        """
        size = self.map_size_cells.get(int(map_id))
        if size is None:
            return int(x), int(y)
        w, h = size
        x, y = int(x), int(y)
        if y == h - 1:
            return x, y + 1
        if y == 0:
            return x, y - 1
        if x == 0:
            return x - 1, y
        if x == w - 1:
            return x + 1, y
        return x, y

    def _confirm_warp_at(
        self,
        local: LocalMap,
        x: int,
        y: int,
        tile: int,
        tileset_id: int,
    ) -> bool:
        """JSON 문/계단 좌표 ∩ 워프 타일이면 WARP 표시 (경계 매트는 한 칸 밖)."""
        p = self.warp_at.get(local.map_id, {}).get((int(x), int(y)))
        if p is None:
            return False
        warp_ids = self.warp_tiles_by_tileset.get(int(tileset_id), frozenset())
        if int(tile) not in warp_ids:
            return False
        tx, ty = self._edge_exit_target(local.map_id, x, y)
        cell = local.ensure_cell(tx, ty)
        cell.block_type = BlockType.WARP
        return True

    def _label_near_static(
        self,
        local: LocalMap,
        dest_name: str,
        near: Tuple[int, int],
        dest_map_id: Optional[int],
        *,
        allow_last_map: bool = False,
        max_dist: int = 2,
    ) -> bool:
        """문/계단 JSON 칸 중, 이번에 넘긴 위치 근처에만 이름.

        경계 출구 매트는 이름/WARP 를 한 칸 밖에 둔다 (A*가 매트에서 멈추지 않게).
        """
        points = self.warp_at.get(local.map_id)
        if not points:
            return False
        nx_, ny_ = int(near[0]), int(near[1])
        hit = False
        for (x, y), p in points.items():
            dest_id = p.get("dest_id")
            match = False
            if dest_id is not None and dest_map_id is not None:
                if int(dest_id) == int(dest_map_id):
                    match = True
            elif allow_last_map and dest_id is None:
                match = True
            if not match:
                continue
            # near 는 발판 칸이거나 그 한 칸 밖일 수 있음
            tx, ty = self._edge_exit_target(local.map_id, x, y)
            if min(
                abs(x - nx_) + abs(y - ny_),
                abs(tx - nx_) + abs(ty - ny_),
            ) > max_dist:
                continue
            local.set_cell_name(tx, ty, dest_name)
            hit = True
        return hit

    def _map_has_static_warps(self, map_id: int) -> bool:
        return int(map_id) in self.warp_at and bool(self.warp_at[int(map_id)])

    def _load_hidden_objects(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for mid_str, entry in data.items():
            per_map: Dict[Tuple[int, int], EntityType] = {}
            for x, y, kind, _func in entry["events"]:
                per_map[(int(x), int(y))] = (
                    EntityType.NPC if kind == "npc" else EntityType.OBJECT
                )
            self.hidden_objects[int(mid_str)] = per_map

    def _load_map_data(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for region in data.get("regions", []):
            try:
                mid = int(region["id"])
            except (KeyError, ValueError):
                continue
            if mid >= 0:
                self.map_names[mid] = str(region.get("name", f"MAP_{mid}"))

    def map_name(self, map_id: int) -> str:
        return self.map_names.get(int(map_id), f"MAP_{int(map_id)}")

    # ------------------------------------------------------------------
    # 생명주기 (RL 에피소드용)
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """RL 에피소드 리셋. 쌓은 지도를 전부 비운다.

        [첨가] 지도를 유지한 채 방문 기록만 지우고 싶으면(에피소드 간
        지도 재사용) reset_visited()를 쓰면 됨 — 아래에 별도 구현.
        """
        self.graph.clear()
        self.prev_map_id = None
        self.prev_pos = None
        self.prev_facing = None
        self._map_settle = 0
        self._pending_map_change = None
        self._arrival_pos = None
        self.last_first_visit = False

    def reset_visited(self) -> None:
        """[첨가] 지도 구조(벽/물/이름/edge)는 남기고 visited만 초기화.

        이유: Exploration RL 학습 시 '지도는 이어 쓰되 에피소드마다
        방문 보상은 새로 시작'하는 세팅이 흔해서 미리 분리해 둠.
        """
        for _, data in self.graph.nodes(data=True):
            local: LocalMap = data["local"]
            for cell in local.grid.values():
                cell.visited = False
                cell.visited_count = 0
        self.prev_map_id = None
        self.prev_pos = None
        self.prev_facing = None
        self._map_settle = 0
        self._pending_map_change = None
        self._arrival_pos = None
        self.last_first_visit = False

    # ------------------------------------------------------------------
    # 조회
    # ------------------------------------------------------------------
    def get_local_map(self, map_id: int) -> Optional[LocalMap]:
        if int(map_id) not in self.graph:
            return None
        return self.graph.nodes[int(map_id)].get("local")

    def has_map(self, map_id: int) -> bool:
        return int(map_id) in self.graph

    def find_map_id_by_name(self, name: str) -> Union[int, bool]:
        """맵 이름 → map_id. 알고 있는(방문한) 맵 중에서만 찾고, 없으면 False.

        LLM이 "Viridian City"처럼 이름으로 목표를 주면:
          1) 이 함수로 map_id를 얻고 (False면 → 미지의 맵 → RL 탐험 담당)
          2) get_route()로 경유 리스트를 뽑는다.
        """
        for mid in self.graph.nodes:
            if self.graph.nodes[mid]["name"] == name:
                return mid
        return False

    # ------------------------------------------------------------------
    # 다익스트라 행선지 리스트
    # ------------------------------------------------------------------
    def get_route(
        self, current_map_id: int, target: Union[int, str]
    ) -> Union[List[int], bool]:
        """현재 맵 → 목표 맵의 경유 map_id 리스트 (다익스트라, edge weight 전부 1).

        target은 map_id(int) 또는 맵 이름(str).
        그래프에 목표가 없거나 경로가 없으면 False.

        반환 예: [0, 12, 1]  (팔레트타운 → 1번도로 → 상록시티)
        각 구간의 실제 이동은 LocalMap에서 다음 맵 이름의 셀을
        find_name()으로 찾아 A*로 가면 된다. (A*는 여기서 구현하지 않음)
        """
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

    # ------------------------------------------------------------------
    # step 갱신 — 메인 진입점
    # ------------------------------------------------------------------
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
        """1 step마다 호출. 화면 관측으로 지도를 넓힌다.

        Parameters
        ----------
        map_id      : 0xD35E 값
        player_pos  : (x, y) — 0xD362, 0xD361 값 (이동칸 단위 맵 로컬 좌표)
        tiles_20x18 : 화면 BG 타일 ID 행렬, shape (18, 20) — [row(y)][col(x)]
        tileset_id  : 0xD367 값 (통과/물/풀 판정용)
        npcs        : {slot: {"picture": int, "x": int, "y": int}}
                      x, y는 이동칸 단위 맵 로컬 좌표 (+4 보정 완료된 값)
        grass_tile_id : wGrassTile(0xD535). None이면 TILESET_GRASS_TILE 룩업.
        facing      : 0xC109 값(0=아래 4=위 8=왼쪽 0xC=오른쪽). 워프 칸 보정용.

        PyBoy를 직접 쓸 거면 update_from_pyboy()가 이걸 대신 호출한다.
        (맵 전환 처리도 update_from_pyboy 쪽에서 RAM만으로 먼저 끝내며,
         여기의 전환 처리는 update()를 직접 호출하는 경우의 폴백이다)
        """
        map_id = int(map_id)
        px, py = int(player_pos[0]), int(player_pos[1])
        tiles = np.asarray(tiles_20x18)
        if tiles.shape != (SCREEN_TILE_H, SCREEN_TILE_W):
            # (20, 18)로 들어오면 전치해서 맞춘다
            if tiles.shape == (SCREEN_TILE_W, SCREEN_TILE_H):
                tiles = tiles.T
            else:
                raise ValueError(f"tiles shape must be (18, 20), got {tiles.shape}")
        npcs = npcs or {}

        # --- 1) Global 노드 확보 ---------------------------------------
        local: LocalMap = self._ensure_local(map_id)

        # --- 2) 화면 20x18 → 이동칸 10x9 로 셀 갱신 ---------------------
        walkable = TILESET_WALKABLE.get(int(tileset_id), frozenset())
        if grass_tile_id is not None:
            grass_tile = None if int(grass_tile_id) in (0xFF, 0x00) else int(grass_tile_id)
        else:
            grass_tile = TILESET_GRASS_TILE.get(int(tileset_id))
        cut_tree_tile = TILESET_CUT_TREE_TILE.get(int(tileset_id))
        has_water = int(tileset_id) in WATER_TILESETS

        # NPC를 (x, y) → slot 으로 뒤집어 셀 갱신 때 조회
        npc_at: Dict[Tuple[int, int], int] = {
            (info["x"], info["y"]): slot for slot, info in npcs.items()
        }
        # hidden object (PC/벤치가이/석상 등 BG 타일 상호작용 대상)
        hidden_at = self.hidden_objects.get(map_id, {})

        for cy in range(CELL_H):
            for cx in range(CELL_W):
                # 화면 셀 → 절대 맵 좌표
                ax = px + (cx - PLAYER_SCREEN_CELL_X)
                ay = py + (cy - PLAYER_SCREEN_CELL_Y)
                cell = local.ensure_cell(ax, ay)

                # 통행 판정 타일: 이동칸(2x2)의 '왼쪽 아래' 타일
                #   (home/overworld.asm CheckTilePassable 계열이 보는 타일.
                #    test_map.py [C]에서 wTilePlayerStandingOn과 일치 확인됨)
                # 풀·물 판정은 cell_has_*()로 셀 2x2 전체를 본다.
                tile = int(tiles[cy * 2 + 1, cx * 2])

                # block_type 판정
                # ── 워프 확정: warps.json (x,y) ∩ 관측 워프 타일 ID ────────
                # 타일만 맞거나 좌표만 맞으면 안 됨. 둘 다일 때만 WARP/이름.
                self._confirm_warp_at(local, ax, ay, tile, tileset_id)

                # WARP sticky: 확정된 워프 칸은 타일 재판정으로 덮지 않음
                if cell.block_type is BlockType.WARP:
                    pass
                elif has_water and cell_has_water_tile(tiles, cy, cx, WATER_TILE):
                    cell.block_type = BlockType.WATER
                else:
                    ledge = classify_ledge(tileset_id, tiles, cy, cx)
                    if ledge is not None:
                        cell.block_type = ledge
                    elif cut_tree_tile is not None and tile == cut_tree_tile:
                        cell.block_type = BlockType.WALL
                    elif grass_tile is not None and cell_has_grass_tile(
                        tiles, cy, cx, grass_tile
                    ):
                        cell.block_type = BlockType.GRASS
                    elif tile in walkable:
                        cell.block_type = None
                    else:
                        cell.block_type = BlockType.WALL

                # entity 판정 — NPC / 오브젝트 / 풀베기 나무 / hidden object
                # NPC가 떠난 자리는 else에서 None으로 지워져 자가 수정됨.
                slot = npc_at.get((ax, ay))
                hidden = hidden_at.get((ax, ay))
                if slot is not None:
                    picture = npcs[slot]["picture"]
                    if picture >= FIRST_STILL_SPRITE_ID:
                        cell.entity = EntityType.OBJECT
                    else:
                        cell.entity = EntityType.NPC
                elif hidden is not None:
                    # hidden event는 정적 데이터라 관측과 무관하게 항상 유지
                    cell.entity = hidden
                elif cut_tree_tile is not None and tile == cut_tree_tile:
                    cell.entity = EntityType.CUT_TREE
                else:
                    cell.entity = None
                # 화면에 텍스트박스/메뉴가 떠 있으면 BG 타일이 UI로 덮일 수 있음.
                # update_from_pyboy는 wJoyIgnore!=0 일 때 스킵하므로 대화·스크립트
                # 구간은 걸러진다. (EnterMap 로드와 동일 주소)

        # --- 3) 맵 전환 처리 (edge + 워프 칸 이름) -----------------------
        if self.prev_map_id is not None and self.prev_map_id != map_id:
            self._handle_map_change(self.prev_map_id, map_id, px, py, facing)

        # --- 4) 플레이어 현재 칸 방문 처리 ------------------------------
        self.last_first_visit = local.ensure_cell(px, py).mark_visited()

        # --- 5) 상태 저장 -----------------------------------------------
        self.prev_map_id = map_id
        self.prev_pos = (px, py)
        self.prev_facing = facing

    # ------------------------------------------------------------------
    # PyBoy에서 직접 읽어서 update까지
    # ------------------------------------------------------------------
    def update_from_pyboy(self, pyboy: Any) -> bool:
        """PyBoy 인스턴스에서 메모리를 읽어 update() 호출.

        스킵 조건 (VRAM/지도 오염 방지):
        - 배틀 중 (배틀 UI)
        - wJoyIgnore != 0 (EnterMap 로드·대화·스크립트 — 화면 전환 구간)
        - 스크롤 비정렬 (걷는 중 반 칸 밀림)
        - wWalkCounter != 0 (걸음 애니 중 — px/py와 화면 타일 불일치)
        - 맵 전환 직후 MAP_SETTLE_FRAMES 동안 (wCurMap이 페이드보다 먼저 바뀜)
        - 맵 전환 후 스폰에서 1스텝 전 (edge/이름 + LocalMap 타일 스캔 모두 보류)

        맵 전환(edge·워프 이름·그리드): 새 맵 스폰 후 안에서 1스텝 이동한
        뒤에만 처리한다. (전환 직후 잔존/불안정 좌표로 LocalMap 오염 방지)
        """
        if int(pyboy.memory[ADDR_IN_BATTLE]) != 0:
            return False

        # EnterMap/LoadMapData·대화·컷신 등 입력이 잠긴 동안은 VRAM을 믿지 않음.
        if int(pyboy.memory[ADDR_JOY_IGNORE]) != 0:
            return False

        if (int(pyboy.memory[ADDR_SCX]) % 8 != 0
                or int(pyboy.memory[ADDR_SCY]) % 8 != 0):
            return False

        map_id = int(pyboy.memory[ADDR_MAP_ID])
        px = int(pyboy.memory[ADDR_PLAYER_X])
        py = int(pyboy.memory[ADDR_PLAYER_Y])
        facing = int(pyboy.memory[ADDR_PLAYER_FACING])

        # ── 맵 전환 감지: 보류만. 라벨/edge/그리드는 안에서 1스텝 후 ───
        if self.prev_map_id is not None and self.prev_map_id != map_id:
            self._queue_map_change(self.prev_map_id, map_id)
            self.prev_map_id = map_id
            self.prev_pos = (px, py)
            self.prev_facing = facing
            self._map_settle = MAP_SETTLE_FRAMES
            self._arrival_pos = None
            return True

        # 전환 직후 몇 프레임은 화면이 안정될 때까지 타일 스캔 보류.
        if self._map_settle > 0:
            self._map_settle -= 1
            self.prev_map_id = map_id
            self.prev_pos = (px, py)
            self.prev_facing = facing
            if self._map_settle == 0 and self._pending_map_change is not None:
                # 스폰 좌표 고정 — 여기서 한 칸 움직이면 flush + 스캔 재개
                self._arrival_pos = (px, py)
            return False

        # 보류 중: 스폰에 서 있으면 타일 스캔도 하지 않음
        if self._pending_map_change is not None:
            if self._arrival_pos is None or (px, py) == self._arrival_pos:
                self.prev_map_id = map_id
                self.prev_pos = (px, py)
                self.prev_facing = facing
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

        tileset_id = int(pyboy.memory[ADDR_TILESET])
        tiles = read_screen_tiles(pyboy)
        npcs = read_npcs(pyboy)
        grass_rt = int(pyboy.memory[ADDR_GRASS_TILE])
        self.update(map_id, (px, py), tiles, tileset_id, npcs, grass_rt,
                    facing=facing)
        return True

    # ------------------------------------------------------------------
    # 내부: 맵 전환
    # ------------------------------------------------------------------
    def _compute_near_old(self) -> Optional[Tuple[int, int]]:
        """전환 직전 위치 + 진행 방향 = 옛 맵 출구 칸 힌트."""
        near_old: Optional[Tuple[int, int]] = self.prev_pos
        if near_old is not None:
            d = FACING_DELTA.get(self.prev_facing)
            if d is not None:
                near_old = (near_old[0] + d[0], near_old[1] + d[1])
        return near_old

    def _link_maps_bidirectional(self, a: int, b: int) -> None:
        """맵 전환 한 번으로 A↔B 양방향 edge 등록."""
        a, b = int(a), int(b)
        if a == b:
            return
        self.graph.add_edge(a, b, weight=1)
        self.graph.add_edge(b, a, weight=1)

    def _queue_map_change(self, old_map_id: int, new_map_id: int) -> None:
        """map_id 변경 즉시: LocalMap 확보 + 라벨/edge 보류 (1스텝 후 flush)."""
        near_old = self._compute_near_old()
        self._ensure_local(new_map_id)
        self._pending_map_change = (old_map_id, new_map_id, near_old)
        self._arrival_pos = None

    def _cell_just_outside(self, map_id: int, x: int, y: int) -> bool:
        """맵 밖이되 경계에서 맨해튼 거리 1칸 (connection/가장자리 출구)."""
        if not self._cell_off_map(map_id, x, y):
            return False
        size = self.map_size_cells.get(int(map_id))
        if size is None:
            return False
        w, h = size
        x, y = int(x), int(y)
        cx = min(max(x, 0), w - 1)
        cy = min(max(y, 0), h - 1)
        return abs(cx - x) + abs(cy - y) <= 1

    def _handle_map_change(
        self,
        old_map_id: int,
        new_map_id: int,
        new_x: int,
        new_y: int,
        new_facing: Optional[int] = None,
        near_old: Optional[Tuple[int, int]] = None,
    ) -> None:
        """map_id 변화 시 — 걷기(문/계단/connection)로 확인된 전환만 연결.

        Fly · 텔레포트 · 블랙아웃 · Escape Rope 등은 출구/connection 증거가
        없으므로 edge·name_index를 만들지 않는다.
        """
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
        used_static_old = False
        if old_local is not None and near_old is not None:
            used_static_old = self._label_near_static(
                old_local, new_name, near_old, new_map_id,
                allow_last_map=True, max_dist=2,
            )
        used_static_new = self._label_near_static(
            new_local, old_name, near_new, old_map_id,
            allow_last_map=True, max_dist=2,
        )

        # 2) connection — 경계 밖 1칸만
        if (
            not used_static_old
            and old_local is not None
            and near_old is not None
            and self._cell_just_outside(old_map_id, near_old[0], near_old[1])
        ):
            old_local.set_cell_name(near_old[0], near_old[1], new_name)
            used_static_old = True

        if (
            not used_static_new
            and self._cell_just_outside(new_map_id, near_new[0], near_new[1])
        ):
            new_local.set_cell_name(near_new[0], near_new[1], old_name)
            used_static_new = True

        if used_static_old or used_static_new:
            self._link_maps_bidirectional(old_map_id, new_map_id)

    # ------------------------------------------------------------------
    # [첨가] 관찰/요약 (RL·LLM 소비용 — 로직은 없고 데이터만 꺼내줌)
    # ------------------------------------------------------------------
    def get_local_obs(
        self, map_id: int, center: Tuple[int, int], radius: int = 4
    ) -> Dict[str, np.ndarray]:
        """[첨가] 현재 위치 주변 (2r+1)x(2r+1) 고정 크기 관측.

        이유: Exploration RL 네트워크 입력은 고정 shape이어야 하는데
        LocalMap은 가변 크기 dict라서, 지도가 잘라서 주는 게 맞음.
        반환: {"block": ..., "visited": ...} 각각 (2r+1, 2r+1) int8.
        """
        local = self.get_local_map(map_id)
        size = 2 * radius + 1
        block = np.full((size, size), LocalMap.MAT_UNKNOWN, dtype=np.int8)
        visited = np.full((size, size), LocalMap.MAT_UNKNOWN, dtype=np.int8)
        if local is None:
            return {"block": block, "visited": visited}
        cx, cy = int(center[0]), int(center[1])
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                cell = local.get_cell(cx + dx, cy + dy)
                if cell is None:
                    continue
                r, c = dy + radius, dx + radius
                block[r, c] = LocalMap.cell_to_mat(cell)
                visited[r, c] = 1 if cell.visited else 0
        return {"block": block, "visited": visited}

    def get_summary_text(self) -> str:
        """[첨가] LLM 프롬프트용 지도 요약 텍스트.

        이유: LLM 플래너가 '지금까지 아는 맵과 연결'을 보고 목표 맵 이름을
        뱉으려면 텍스트 요약이 필요함. (LLM 호출 자체는 여기서 안 함)
        """
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


# ===========================================================================
# 7. PyBoy 메모리 리더 (mainSprite.py / test_memory.py의 로직을 함수화)
# ===========================================================================
def read_screen_tiles(pyboy: Any) -> np.ndarray:
    """화면 BG 타일 ID 20x18 을 읽어 (18, 20) 배열로 반환.

    mainSprite.py 방식: VRAM 0x9800 + (SCY, SCX 스크롤 보정, 32칸 랩어라운드)
    """
    scy = int(pyboy.memory[ADDR_SCY]) // 8
    scx = int(pyboy.memory[ADDR_SCX]) // 8
    tiles = np.zeros((SCREEN_TILE_H, SCREEN_TILE_W), dtype=np.int16)
    for y in range(SCREEN_TILE_H):
        map_y = (y + scy) % 32
        row_base = VRAM_BG_BASE + map_y * 32
        for x in range(SCREEN_TILE_W):
            map_x = (x + scx) % 32
            tiles[y, x] = int(pyboy.memory[row_base + map_x])
    return tiles
    # (검증 완료) BG맵 인덱스 = 타일셋 로컬 타일 ID($00~$5F) 전제로 통행 판정.
    # 스크롤 중(반 칸 밀림) 오염은 update_from_pyboy의 SCX/SCY 정렬 게이트가 막음.


def read_npcs(pyboy: Any) -> Dict[int, Dict[str, int]]:
    """WRAM 스프라이트 슬롯(1~15)에서 NPC/오브젝트를 읽는다.

    슬롯 구조 (pokered ram/wram.asm 주석 기준):
      - 0xC100 + slot*0x10 + 0 → picture ID (0이면 빈 슬롯)
      - 0xC100 + slot*0x10 + 1 → movement status (0=미초기화)
      - 0xC100 + slot*0x10 + 2 → image index ($FF면 화면 밖/숨김)
      - 0xC200 + slot*0x10 + 4 → mapY (+4 보정 저장)
      - 0xC200 + slot*0x10 + 5 → mapX (+4 보정 저장)

    미초기화/화면 밖 스프라이트는 제외한다 — 이걸 안 거르면
    화면에 없는 NPC의 오래된 좌표가 지도에 유령 NPC로 찍힘.

    반환: {slot: {"picture": int, "x": int, "y": int}}  (x, y는 -4 보정 완료)
    """
    npcs: Dict[int, Dict[str, int]] = {}
    for slot in range(1, 16):
        base1 = SPRITE_STATE1 + slot * SPRITE_SLOT_SIZE
        picture = int(pyboy.memory[base1])
        if picture == 0:
            continue
        movement_status = int(pyboy.memory[base1 + 1])
        if movement_status == 0:
            continue  # 미초기화 슬롯 (쓰레기 값일 수 있음)
        image_index = int(pyboy.memory[base1 + 2])
        if image_index == 0xFF:
            continue  # 화면 밖/숨김 스프라이트 (sprite_oam.asm: don't draw)
        base2 = SPRITE_STATE2 + slot * SPRITE_SLOT_SIZE
        raw_y = int(pyboy.memory[base2 + 4])
        raw_x = int(pyboy.memory[base2 + 5])
        npcs[slot] = {
            "picture": picture,
            "x": raw_x - SPRITE_MAP_COORD_OFFSET,
            "y": raw_y - SPRITE_MAP_COORD_OFFSET,
        }
    return npcs