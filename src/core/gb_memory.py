"""gb_memory.py — Game Boy / pokered 메모리 주소와 화면 기하 상수."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

# ===========================================================================
# 1. WRAM — 플레이어 / 맵 상태
# ===========================================================================
ADDR_PLAYER_X = 0xD362        # 플레이어 X (이동칸 단위, 맵 로컬 좌표)
ADDR_PLAYER_Y = 0xD361        # 플레이어 Y
ADDR_MAP_ID = 0xD35E          # 현재 map_id
ADDR_TILESET = 0xD367
ADDR_GRASS_TILE = 0xD535

# ===========================================================================
# 2. VRAM / LCD — 화면 BG
# ===========================================================================
ADDR_SCY = 0xFF42             # BG 스크롤 Y
ADDR_SCX = 0xFF43             # BG 스크롤 X
VRAM_BG_BASE = 0x9800         # BG 타일맵 시작 주소

# ===========================================================================
# 3. 지도 갱신 게이트 — 이 값들이 non-zero면 VRAM/좌표를 믿을 수 없다
# ===========================================================================
ADDR_IN_BATTLE = 0xD057       # 0이 아니면 배틀 중 → 지도 갱신 스킵
ADDR_JOY_IGNORE = 0xCD6B      # wJoyIgnore — EnterMap/대화/스크립트 중 non-zero
# EnterMap은 LoadMapData 동안 입력을 잠그므로, 이때 VRAM을 읽으면 페이드·로딩 중간 화면이 지도에 오염된다.
ADDR_FONT_LOADED = 0xCFC4
ADDR_WALK_COUNTER = 0xCFC5    # wWalkCounter — 0=정지/걸음 종료, 1~8=걸음 애니 중
# 걸음 애니 중에는 SCX/SCY만 먼저 바뀌고 px/py는 마지막 프레임에 갱신된다.

# 3-1.
ADDR_BGP = 0xFF47                 # rBGP — 배경 팔레트. 페이드 동안 몇 프레임마다 바뀐다
ADDR_SIMULATED_JOYPAD = 0xCD38
ADDR_STATUS_FLAGS5 = 0xD730       # wStatusFlags5
ADDR_MOVEMENT_FLAGS = 0xD736      # wMovementFlags
# wStatusFlags5: bit0 스크립트 NPC 이동 · bit1/2 워프 처리 중(실측) · bit5 입력 막음 · bit7 스크립트 이동.
STATUS5_SCENE_MASK = 0xA7
# wMovementFlags: bit1 문에서 나오는 중 · bit6 언덕 점프/낚시 · bit7 회전 타일.
MOVEMENT_SCENE_MASK = 0xC2

# 맵 전환 감지 직후, 게이트를 통과한 프레임을 이만큼 더 기다린 뒤에야 화면 타일을 읽는다.
MAP_SETTLE_FRAMES = 3

# 3-1.
BATTLE_NONE = 0x00
BATTLE_WILD = 0x01
BATTLE_TRAINER = 0x02
BATTLE_BLACKOUT = 0xFF

ADDR_BATTLE_TYPE = 0xD05A
BATTLE_TYPE_NORMAL = 0
BATTLE_TYPE_OLD_MAN = 1
BATTLE_TYPE_SAFARI = 2

ADDR_BATTLE_RESULT = 0xCF0B   # wBattleResult — 0=승/미정 1=패 2=도주|포획
                              # (wram.asm 주석의 "draw"는 싱글에선 도주/포획이다)
ADDR_CUR_OPPONENT = 0xD059    # wCurOpponent — 0이 아니면 다음 스텝에 전투가 예약됨
ADDR_FORCE_CHOOSE_MON = 0xD11F  # wForcePlayerToChooseMon — 기절 후 강제 교체
ADDR_OUT_OF_BATTLE_BLACKOUT = 0xD12D  # 독 데미지 전멸 (wIsInBattle 은 계속 0)
ADDR_NUM_SAFARI_BALLS = 0xDA47

# 고스트 전투는 전용 RAM 플래그가 없다.
ADDR_NUM_BAG_ITEMS = 0xD31D
ADDR_BAG_ITEMS = 0xD31E       # (item_id, qty) 쌍, 0xFF 로 끝남
BAG_MAX_ITEMS = 20
ITEM_SILPH_SCOPE = 0x48
MAP_POKEMON_TOWER_1F = 0x8E
MAP_POKEMON_TOWER_7F = 0x94

# battle_kind() 가 돌려주는 값 중 Battle PPO 가 실제로 조작할 수 있는 것.
PLAYABLE_BATTLE_KINDS = frozenset({"WILD", "TRAINER"})

# 3-2.
ADDR_PARTY_COUNT = 0xD163     # wPartyCount
ADDR_PARTY_MONS = 0xD16B      # wPartyMons
PARTYMON_STRUCT_LENGTH = 0x2C  # 44
MON_SPECIES = 0
MON_HP = 1                    # u16 big-endian
MON_STATUS = 4
MON_MOVES = 8                 # 4바이트
MON_PP = 29                   # 4바이트. 하위 6비트가 PP, 상위 2비트는 PP Up 횟수
MON_LEVEL = 33
MON_MAX_HP = 34               # u16 big-endian
MAX_PARTY = 6
PP_MASK = 0x3F

# ===========================================================================
# 4. 스프라이트 슬롯 (NPC / 오브젝트)
# ===========================================================================
SPRITE_STATE1 = 0xC100
SPRITE_STATE2 = 0xC200        # wSpriteStateData2 (+4 = mapY, +5 = mapX)
SPRITE_SLOT_SIZE = 0x10
SPRITE_MAP_COORD_OFFSET = 4   # pokered object_event가 좌표를 +4 해서 저장함

# picture ID 경계: 이 값(FIRST_STILL_SPRITE=$3D) 이상이면 사람이 아닌 오브젝트 (몬스터볼, 바위, 화석, 잠자는 잠만보 등)…
FIRST_STILL_SPRITE_ID = 0x3D
# 길을 막고 자는 잠만보 (12·16번도로). 오브젝트지만 **스토리의 목표**라 이름을 붙인다.
SPRITE_SNORLAX = 0x43

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
# 5. 화면/좌표 기하
# ===========================================================================
SCREEN_TILE_W = 20            # 화면 타일 가로
SCREEN_TILE_H = 18            # 화면 타일 세로
CELL_W = SCREEN_TILE_W // 2   # 이동칸 가로 = 10
CELL_H = SCREEN_TILE_H // 2   # 이동칸 세로 = 9

# 플레이어가 화면에서 차지하는 이동칸 (10x9 격자 기준)
PLAYER_SCREEN_CELL_X = 4
PLAYER_SCREEN_CELL_Y = 4


# ===========================================================================
# 6. 리더 (mainSprite.py 로직을 함수화)
# ===========================================================================
def read_screen_tiles(pyboy: Any) -> np.ndarray:
    """화면 BG 타일 ID 20x18 을 읽어 (18, 20) 배열로 반환."""
    scy = int(pyboy.memory[ADDR_SCY]) // 8
    scx = int(pyboy.memory[ADDR_SCX]) // 8
    tiles = np.zeros((SCREEN_TILE_H, SCREEN_TILE_W), dtype=np.int16)
    for y in range(SCREEN_TILE_H):
        row_base = VRAM_BG_BASE + ((y + scy) % 32) * 32
        for x in range(SCREEN_TILE_W):
            tiles[y, x] = int(pyboy.memory[row_base + (x + scx) % 32])
    return tiles


def read_npcs(pyboy: Any) -> Dict[int, Dict[str, int]]:
    """WRAM 스프라이트 슬롯(1~15)에서 NPC/오브젝트를 읽는다."""
    npcs: Dict[int, Dict[str, int]] = {}
    for slot in range(1, 16):
        base1 = SPRITE_STATE1 + slot * SPRITE_SLOT_SIZE
        picture = int(pyboy.memory[base1])
        if picture == 0:
            continue
        if int(pyboy.memory[base1 + 1]) == 0:
            continue  # 미초기화 슬롯 (쓰레기 값일 수 있음)
        if int(pyboy.memory[base1 + 2]) == 0xFF:
            continue
        base2 = SPRITE_STATE2 + slot * SPRITE_SLOT_SIZE
        npcs[slot] = {
            "picture": picture,
            "x": int(pyboy.memory[base2 + 5]) - SPRITE_MAP_COORD_OFFSET,
            "y": int(pyboy.memory[base2 + 4]) - SPRITE_MAP_COORD_OFFSET,
        }
    return npcs



def battle_phase(pyboy: Any) -> int:
    """wIsInBattle 원값."""
    try:
        return int(pyboy.memory[ADDR_IN_BATTLE]) & 0xFF
    except Exception:
        return BATTLE_NONE


def is_in_battle(pyboy: Any) -> bool:
    """실제로 전투 중인지."""
    return battle_phase(pyboy) in (BATTLE_WILD, BATTLE_TRAINER)


def has_bag_item(pyboy: Any, item_id: int) -> bool:
    """가방에 특정 아이템이 있는지 (home/item.asm IsItemInBag 과 동일)."""
    try:
        count = int(pyboy.memory[ADDR_NUM_BAG_ITEMS])
        if not 0 < count <= BAG_MAX_ITEMS:
            return False
        for i in range(count):
            entry = int(pyboy.memory[ADDR_BAG_ITEMS + i * 2])
            if entry == 0xFF:
                break
            if entry == int(item_id):
                return True
    except Exception:
        return False
    return False


def give_bag_item(pyboy: Any, item_id: int, quantity: int = 1) -> bool:
    """가방에 아이템을 넣는다 (home/item.asm GiveItem 과 같은 배치)."""
    try:
        count = int(pyboy.memory[ADDR_NUM_BAG_ITEMS])
        if not 0 <= count <= BAG_MAX_ITEMS:
            return False
        for i in range(count):
            base = ADDR_BAG_ITEMS + i * 2
            if int(pyboy.memory[base]) == int(item_id):
                qty = int(pyboy.memory[base + 1])
                pyboy.memory[base + 1] = min(99, qty + int(quantity))
                return True
        if count >= BAG_MAX_ITEMS:
            return False
        base = ADDR_BAG_ITEMS + count * 2
        pyboy.memory[base] = int(item_id) & 0xFF
        pyboy.memory[base + 1] = max(1, min(99, int(quantity)))
        pyboy.memory[base + 2] = 0xFF
        pyboy.memory[ADDR_NUM_BAG_ITEMS] = count + 1
        return True
    except Exception:
        return False


def battle_kind(pyboy: Any) -> str:
    """전투의 종류를 판정한다."""
    raw = battle_phase(pyboy)
    if raw == BATTLE_NONE:
        return "NONE"
    if raw == BATTLE_BLACKOUT:
        return "BLACKOUT"
    if raw == BATTLE_TRAINER:
        return "TRAINER"
    if raw != BATTLE_WILD:
        return "NONE"

    # 여기부터 wIsInBattle == 1. 사파리/올드맨은 둘 다 야생 취급이다.
    try:
        btype = int(pyboy.memory[ADDR_BATTLE_TYPE]) & 0xFF
    except Exception:
        btype = BATTLE_TYPE_NORMAL
    if btype == BATTLE_TYPE_SAFARI:
        return "SAFARI"
    if btype == BATTLE_TYPE_OLD_MAN:
        return "OLD_MAN"

    try:
        cur_map = int(pyboy.memory[ADDR_MAP_ID]) & 0xFF
    except Exception:
        return "WILD"
    if (
        MAP_POKEMON_TOWER_1F <= cur_map <= MAP_POKEMON_TOWER_7F
        and not has_bag_item(pyboy, ITEM_SILPH_SCOPE)
    ):
        return "GHOST"
    return "WILD"


def is_textbox_open(pyboy: Any) -> bool:
    """화면에 텍스트박스/메뉴가 실제로 떠 있는지 (wFontLoaded bit0 만 본다)."""
    try:
        return bool(int(pyboy.memory[ADDR_FONT_LOADED]) & 0x01)
    except Exception:
        return False


def read_party(pyboy: Any) -> list:
    """파티 전체 상태를 읽는다."""
    out: list = []
    try:
        count = int(pyboy.memory[ADDR_PARTY_COUNT])
    except Exception:
        return out
    if not 0 < count <= MAX_PARTY:
        return out

    for slot in range(count):
        base = ADDR_PARTY_MONS + slot * PARTYMON_STRUCT_LENGTH
        try:
            species = int(pyboy.memory[base + MON_SPECIES])
            if species == 0:
                continue
            hp = (int(pyboy.memory[base + MON_HP]) << 8) | int(
                pyboy.memory[base + MON_HP + 1]
            )
            max_hp = (int(pyboy.memory[base + MON_MAX_HP]) << 8) | int(
                pyboy.memory[base + MON_MAX_HP + 1]
            )
            moves = [
                int(pyboy.memory[base + MON_MOVES + i]) for i in range(4)
            ]
            pp = [
                int(pyboy.memory[base + MON_PP + i]) & PP_MASK
                for i in range(4)
            ]
            out.append(
                {
                    "slot": slot,
                    "species_id": species,
                    "level": int(pyboy.memory[base + MON_LEVEL]),
                    "hp": hp,
                    "max_hp": max_hp,
                    "status": int(pyboy.memory[base + MON_STATUS]),
                    "move_ids": moves,
                    "pp": pp,
                }
            )
        except Exception:
            break
    return out


def scene_busy_reason(pyboy: Any) -> str:
    """지금 게임이 연출(대화·스크립트 이동·워프·걸음·전투) 중이면 그 이유, 아니면 빈 문자열."""
    m = pyboy.memory
    if int(m[ADDR_IN_BATTLE]) != 0:
        return "전투"
    if int(m[ADDR_FONT_LOADED]) & 0x01:
        return "대화창"
    if int(m[ADDR_JOY_IGNORE]) != 0:
        return "입력 잠김"
    if int(m[ADDR_SIMULATED_JOYPAD]) != 0:
        return "스크립트 이동"
    if int(m[ADDR_STATUS_FLAGS5]) & STATUS5_SCENE_MASK:
        return "스크립트 진행"
    if int(m[ADDR_MOVEMENT_FLAGS]) & MOVEMENT_SCENE_MASK:
        return "이동 연출"
    if int(m[ADDR_WALK_COUNTER]) != 0:
        return "걸음 중"
    return ""


def is_dialogue_active(pyboy: Any) -> bool:
    """텍스트박스/스크립트 대화 중인지."""
    try:
        font = int(pyboy.memory[ADDR_FONT_LOADED]) & 0x01
        joy = int(pyboy.memory[ADDR_JOY_IGNORE]) != 0
    except Exception:
        return False
    return bool(font or joy)
