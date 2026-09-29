"""디버그 치트 — PyBoy WRAM 직접 조작 (Map.py 관측과 무관)."""

from __future__ import annotations

from core.config import cheat as _cfg_cheat
from core.closure import SNORLAX_SPOTS
from core.gb_memory import (
    ADDR_IN_BATTLE,
    ADDR_MAP_ID,
    ADDR_PLAYER_X,
    ADDR_PLAYER_Y,
    SPRITE_SLOT_SIZE,
    SPRITE_STATE1,
    give_bag_item,
    has_bag_item,
    is_textbox_open,
)
from core.tileset_data import TILESET_CUT_TREE_TILE, TILESET_WALKABLE

# 야생 인카운터 확률 (pokered.sym). 0이면 TryDoWildEncounter 가 항상 실패.
ADDR_GRASS_RATE = 0xD887   # wGrassRate
ADDR_WATER_RATE = 0xD8A4   # wWaterRate
# 타일셋 "통과 가능" 타일 리스트 포인터 (CheckTilePassable 가 참조).
ADDR_TILESET_COLLISION_PTR = 0xD530  # wTilesetCollisionPtr (2바이트 LE)
ADDR_CUR_MAP_TILESET = 0xD367  # wCurMapTileset
ADDR_SIMULATED_JOYPAD_INDEX = 0xCD38  # wSimulatedJoypadStatesIndex
ADDR_STATUS_FLAGS1 = 0xD728  # wStatusFlags1
ADDR_STATUS_FLAGS5 = 0xD730  # wStatusFlags5
BIT_STRENGTH_ACTIVE = 0
BIT_SCRIPTED_MOVEMENT_STATE = 7
# noclip 통과 테이블 = tileset_data.TILESET_WALKABLE (중복 상수 제거) overworld 중에 거의 안 쓰는 박스 슬롯을 통과 타일 테이블 버퍼로…
ADDR_NOCLIP_PASSABLE_BUF = 0xDA80  # wBoxDataStart

# 달맞이산 B2F 화석 (TOGGLE_MT_MOON_B2F_FOSSIL_1/2 = 0x6D/0x6E)
ADDR_MISSABLE_OBJECT_FLAGS = 0xD5A6  # wMissableObjectFlags
ADDR_MT_MOON_FOSSIL_FLAGS = ADDR_MISSABLE_OBJECT_FLAGS + (0x6D // 8)  # 0xD5B3
MT_MOON_FOSSIL_FLAG_MASK = (1 << (0x6D % 8)) | (1 << (0x6E % 8))  # bits 5|6
MAP_ID_MT_MOON_B2F = 61
MT_MOON_FOSSIL_SPRITE_SLOTS = (6, 7)  # npc_objects: DOME / HELIX
SPRITE_IMAGE_INDEX_OFFSET = 2
SPRITE_HIDDEN = 0xFF

# 갈색시티(Vermilion) 체육관 쓰레기통 스위치 퍼즐.
ADDR_VERMILION_GYM_LOCKS = 0xD773
VERMILION_GYM_LOCKS_MASK = (1 << 0) | (1 << 1)
MAP_ID_VERMILION_GYM = 92
# wCurrentMapScriptFlags.
ADDR_CUR_MAP_SCRIPT_FLAGS = 0xD126
BIT_CUR_MAP_LOADED_2 = 6

# 빌의 집(BILLS_HOUSE, map 88) — 포켓몬이 된 빌을 캡슐에서 되돌리는 PC 조작.
MAP_ID_BILLS_HOUSE = 88
ADDR_BILL_EVENTS = 0xD7F2
BIT_BILL_SAID_USE_CELL_SEPARATOR = 6
BIT_USED_CELL_SEPARATOR_ON_BILL = 3
# TOGGLE_BILL_POKEMON (data/npc_objects.json toggle_index 97). 비트가 켜지면 숨김.
ADDR_BILL_POKEMON_TOGGLE = ADDR_MISSABLE_OBJECT_FLAGS + (97 // 8)
BIT_BILL_POKEMON_TOGGLE = 97 % 8
ADDR_JOY_IGNORE = 0xCD6B                 # wJoyIgnore
ADDR_BILLS_HOUSE_CUR_SCRIPT = 0xD661     # wBillsHouseCurScript
SCRIPT_BILLSHOUSE_BILL_EXITS_MACHINE = 3
SCRIPT_BILLSHOUSE_CLEANUP = 4
BIT_GOT_SS_TICKET = 4                    # 0xD7F2 bit 4 = Got Ss Ticket
BIT_MET_BILL_2 = 5                       # 0xD7F2 bit 5 = Met Bill 2
ADDR_MET_BILL = 0xD7F1                   # bit 0 = Met Bill
BIT_MET_BILL = 0
TOGGLE_BILL_1 = 98                       # 캡슐에서 나온 빌 (기본 숨김)
ITEM_SS_TICKET = 0x3F
# 티켓을 받으면 게임이 블루시티 경비원 둘을 바꿔 세운다 (도둑맞은 집 앞 길이 열린다).
TOGGLE_CERULEAN_GUARD_1 = 7
TOGGLE_CERULEAN_GUARD_2 = 9

# 길을 막고 자는 잠만보 (12번도로 · 16번도로).
ITEM_POKE_FLUTE = 0x49
SNORLAX_EVENTS = {
    # map_id: (이벤트 주소, Fight 비트, Beat 비트, 잠만보 토글 번호)
    23: (0xD7D8, 6, 7, 0x1D),   # ROUTE_12 · TOGGLE_ROUTE_12_SNORLAX
    27: (0xD7E0, 0, 1, 0x21),   # ROUTE_16 · TOGGLE_ROUTE_16_SNORLAX
}

# 디버그 치트 (Map.py 무관 — PyBoy WRAM만 조작) 기본값은 settings/config.yaml → cheats 값은 settings/config.yaml 이…
cheat_flags: dict[str, bool] = {
    "noclip": _cfg_cheat("noclip"),
    "disable_wild_encounters": _cfg_cheat("disable_wild_encounters"),
    "skip_mt_moon_fossils": _cfg_cheat("skip_mt_moon_fossils"),
    "auto_win_battle": _cfg_cheat("auto_win_battle"),
    "skip_vermilion_gym_puzzle": _cfg_cheat("skip_vermilion_gym_puzzle"),
    "skip_bill_cell_separator": _cfg_cheat("skip_bill_cell_separator"),
    "wake_snorlax": _cfg_cheat("wake_snorlax"),
    "skip_game_corner_poster": _cfg_cheat("skip_game_corner_poster"),
    "ghost_marowak_needs_scope": _cfg_cheat("ghost_marowak_needs_scope"),
    "ghost_flee": _cfg_cheat("ghost_flee"),
    "buy_saffron_drink": _cfg_cheat("buy_saffron_drink"),
}
_sym_cache: dict[str, int] = {}
_noclip_real_collision_ptr: int | None = None
_noclip_ptr_map_id: int | None = None


# 치트가 직접 켠 이벤트 플래그 (addr, bit).
CHEAT_EVENT_BITS: set[tuple[int, int]] = set()


def _cheat_event(addr: int, *bits: int) -> None:
    for bit in bits:
        CHEAT_EVENT_BITS.add((int(addr), int(bit)))


def _sym(pyboy, name: str) -> int | None:
    """pokered.sym 심볼 주소."""
    addr = _sym_cache.get(name)
    if addr is not None:
        return addr
    try:
        addr = int(pyboy.symbol_lookup(name)[1])
    except Exception:
        return None
    _sym_cache[name] = addr
    return addr


MAP_ID_POKEMON_TOWER_6F = 147
SPECIES_MAROWAK = 0x91
ITEM_SILPH_SCOPE = 0x48


def _unbeatable_ghost_marowak(pyboy) -> bool:
    """지금 전투가 실프스코프 없이 만난 6층 유령 마로왁인가 (scripts/PokemonTower6F.asm)."""
    try:
        if int(pyboy.memory[ADDR_MAP_ID]) != MAP_ID_POKEMON_TOWER_6F:
            return False
    except Exception:
        return False
    if has_bag_item(pyboy, ITEM_SILPH_SCOPE):
        return False                     # 스코프가 있으면 정상 전투 — 이겨도 된다
    for name in ("wEnemyMonSpecies", "wCurOpponent"):
        addr = _sym(pyboy, name)
        if addr is not None and int(pyboy.memory[addr]) == SPECIES_MAROWAK:
            return True
    return False


def is_unbeatable_ghost(pyboy) -> bool:
    """자동 승리 치트가 일부러 안 이기는 전투인가 (play.py 가 도망 경로로 넘긴다)."""
    try:
        if int(pyboy.memory[ADDR_IN_BATTLE]) in (0, 0xFF):
            return False
    except Exception:
        return False
    if not cheat_flags.get("ghost_marowak_needs_scope", True):
        return False
    return _unbeatable_ghost_marowak(pyboy)


def apply_auto_win_battle(pyboy) -> bool:
    """전투 중이면 **상대 HP 를 0 으로** 만들어 그대로 이긴다."""
    if not cheat_flags.get("auto_win_battle", False):
        return False
    try:
        state = int(pyboy.memory[ADDR_IN_BATTLE])
    except Exception:
        return False
    if state == 0 or state == 0xFF:      # 0xFF 는 전멸 후 블랙아웃 연출이다
        return False
    if cheat_flags.get("ghost_marowak_needs_scope", True) and _unbeatable_ghost_marowak(pyboy):
        # **실프스코프 없는 유령 마로왁은 이기게 두지 않는다.** 게임에서는 못 이기는
        # 전투라 7층 계단이 막혀 있어야 하는데, 여기서 이기면 "Beat Ghost Marowak" 이
        # 켜져 스토리 순서를 건너뛴다. 전투는 게임·Battle 스킬이 평소대로 치른다(도망).
        return False

    touched = False
    result = _sym(pyboy, "wBattleResult")
    if result is not None:
        pyboy.memory[result] = 0          # 0 = 승리
    now = _sym(pyboy, "wEnemyMonHP")
    if now is not None:
        pyboy.memory[now] = 0
        pyboy.memory[now + 1] = 0
        touched = True
    count = _sym(pyboy, "wEnemyPartyCount")
    n = int(pyboy.memory[count]) if count is not None else 0
    for i in range(min(6, max(0, n))):
        slot = _sym(pyboy, f"wEnemyMon{i + 1}HP")
        if slot is None:
            continue
        pyboy.memory[slot] = 0
        pyboy.memory[slot + 1] = 0
        touched = True
    return touched


def disable_wild_encounters(pyboy) -> None:
    """풀/물 야생 전투 비활성."""
    if not cheat_flags.get("disable_wild_encounters", True):
        return
    pyboy.memory[ADDR_GRASS_RATE] = 0
    pyboy.memory[ADDR_WATER_RATE] = 0


def apply_skip_mt_moon_fossils(pyboy) -> None:
    """달맞이산 B2F 화석 숨김."""
    if not cheat_flags.get("skip_mt_moon_fossils", False):
        return
    flags = int(pyboy.memory[ADDR_MT_MOON_FOSSIL_FLAGS])
    if (flags & MT_MOON_FOSSIL_FLAG_MASK) != MT_MOON_FOSSIL_FLAG_MASK:
        pyboy.memory[ADDR_MT_MOON_FOSSIL_FLAGS] = flags | MT_MOON_FOSSIL_FLAG_MASK
    if int(pyboy.memory[ADDR_MAP_ID]) != MAP_ID_MT_MOON_B2F:
        return
    for slot in MT_MOON_FOSSIL_SPRITE_SLOTS:
        base = SPRITE_STATE1 + slot * SPRITE_SLOT_SIZE
        if int(pyboy.memory[base]) == 0:
            continue
        pyboy.memory[base + SPRITE_IMAGE_INDEX_OFFSET] = SPRITE_HIDDEN


def apply_skip_vermilion_gym_puzzle(pyboy) -> bool:
    """갈색시티 체육관 쓰레기통 퍼즐을 풀린 것으로 둔다."""
    if not cheat_flags.get("skip_vermilion_gym_puzzle", False):
        return False
    locks = int(pyboy.memory[ADDR_VERMILION_GYM_LOCKS])
    if (locks & VERMILION_GYM_LOCKS_MASK) == VERMILION_GYM_LOCKS_MASK:
        return False
    _cheat_event(ADDR_VERMILION_GYM_LOCKS, 0, 1)
    pyboy.memory[ADDR_VERMILION_GYM_LOCKS] = locks | VERMILION_GYM_LOCKS_MASK
    if int(pyboy.memory[ADDR_MAP_ID]) == MAP_ID_VERMILION_GYM:
        flags = int(pyboy.memory[ADDR_CUR_MAP_SCRIPT_FLAGS])
        pyboy.memory[ADDR_CUR_MAP_SCRIPT_FLAGS] = flags | (1 << BIT_CUR_MAP_LOADED_2)
    print("[cheat] 갈색시티 체육관 퍼즐을 풀린 것으로 둡니다 (전기 문 열림)")
    return True


def apply_skip_bill_cell_separator(pyboy) -> bool:
    """빌 이벤트의 **PC(캡슐) 조작만** 대신한다."""
    if not cheat_flags.get("skip_bill_cell_separator", False):
        return False
    if _give_ss_ticket(pyboy):
        return True
    events = int(pyboy.memory[ADDR_BILL_EVENTS])
    if not events & (1 << BIT_BILL_SAID_USE_CELL_SEPARATOR):
        return False                     # 1) 이 아직이다 — 빌에게 먼저 말을 걸어야 한다
    if events & (1 << BIT_USED_CELL_SEPARATOR_ON_BILL):
        return False                     # 이미 돌려놨다
    toggles = int(pyboy.memory[ADDR_BILL_POKEMON_TOGGLE])
    if not toggles & (1 << BIT_BILL_POKEMON_TOGGLE):
        return False                     # 포켓몬 빌이 아직 캡슐로 안 들어갔다
    if int(pyboy.memory[ADDR_JOY_IGNORE]) != 0:
        return False                     # 연출 중이다
    events |= 1 << BIT_USED_CELL_SEPARATOR_ON_BILL
    _cheat_event(ADDR_BILL_EVENTS, BIT_USED_CELL_SEPARATOR_ON_BILL)
    if int(pyboy.memory[ADDR_MAP_ID]) == MAP_ID_BILLS_HOUSE:
        # 집 안이면 빌이 캡슐에서 걸어 나오는 연출은 게임에 맡긴다
        pyboy.memory[ADDR_BILL_EVENTS] = events
        script = int(pyboy.memory[ADDR_BILLS_HOUSE_CUR_SCRIPT])
        if script not in (SCRIPT_BILLSHOUSE_BILL_EXITS_MACHINE, SCRIPT_BILLSHOUSE_CLEANUP):
            pyboy.memory[ADDR_BILLS_HOUSE_CUR_SCRIPT] = SCRIPT_BILLSHOUSE_BILL_EXITS_MACHINE
        print("[cheat] 빌의 캡슐 PC 조작을 대신했습니다 — 빌이 사람으로 돌아옵니다")
        return True
    # 집을 이미 나왔다 — 연출을 볼 수 없으니 그 결과를 바로 둔다.
    _cheat_event(ADDR_BILL_EVENTS, BIT_MET_BILL_2)
    _cheat_event(ADDR_MET_BILL, BIT_MET_BILL)
    pyboy.memory[ADDR_BILL_EVENTS] = events | (1 << BIT_MET_BILL_2)
    pyboy.memory[ADDR_MET_BILL] = int(pyboy.memory[ADDR_MET_BILL]) | (1 << BIT_MET_BILL)
    _set_toggle(pyboy, TOGGLE_BILL_1, hidden=False)
    print("[cheat] 빌의 캡슐 PC 조작을 대신했습니다 (집 밖) — 빌이 사람으로 돌아왔습니다")
    return True


def apply_wake_snorlax(pyboy) -> bool:
    """지금 맵에 자는 잠만보가 있고 포켓몬 피리가 있으면 깨운다."""
    global _snorlax_woke_on
    if not cheat_flags.get("wake_snorlax", False):
        return False
    map_id = int(pyboy.memory[ADDR_MAP_ID])
    if _snorlax_woke_on is not None and _snorlax_woke_on != map_id:
        _snorlax_woke_on = None          # 그 도로를 떠났다 — 돌아오면 다시 본다
    if _snorlax_woke_on == map_id:
        # **이 도로에서 이미 한 번 깨웠다.** 게임은 Fight 를 끈 뒤 대사를 띄우고,
        # 대사가 닫힌 뒤 몇 프레임 지나서야 잠만보를 숨긴다. 그 틈에는 RAM 만으로
        # "아직 누워 있다" 와 구별이 안 돼서 Fight 를 또 켜게 된다 (실측).
        return False
    spec = SNORLAX_EVENTS.get(map_id)
    if spec is None:
        return False
    addr, fight_bit, beat_bit, toggle = spec
    flags = int(pyboy.memory[addr])
    if flags & ((1 << fight_bit) | (1 << beat_bit)):
        return False                     # 이미 깨웠거나 이미 이겼다
    if _toggle_hidden(pyboy, toggle):
        # **누워 있는 잠만보가 없다.** 게임은 Fight 를 끄자마자 잠만보를 숨기고
        # 전투로 넘어간다 — 이 확인이 없으면 그 사이에 Fight 를 또 켜서, 전투에서
        # 지고 돌아왔을 때 보이지도 않는 잠만보와 다시 싸우게 된다.
        return False
    if int(pyboy.memory[ADDR_IN_BATTLE]) != 0:
        return False
    if int(pyboy.memory[ADDR_JOY_IGNORE]) != 0 or is_textbox_open(pyboy):
        # 연출·대화 중이다.
        return False
    if not has_bag_item(pyboy, ITEM_POKE_FLUTE):
        return False                     # 피리를 아직 못 받았다
    spot = SNORLAX_SPOTS.get(map_id)
    if spot is not None and _cfg_cheat("wake_snorlax_adjacent_only", True):
        px, py = int(pyboy.memory[ADDR_PLAYER_X]), int(pyboy.memory[ADDR_PLAYER_Y])
        if abs(px - spot[0]) + abs(py - spot[1]) != 1:
            return False                 # 잠만보 옆 칸이 아니다 — 피리가 안 들린다
    _cheat_event(addr, fight_bit)
    pyboy.memory[addr] = flags | (1 << fight_bit)
    _snorlax_woke_on = map_id
    print("[cheat] 포켓몬 피리로 잠만보를 깨웁니다 — 전투가 시작됩니다")
    return True


# 무지개 백화점 옥상 (CELADON_MART_ROOF, map 126) — 자판기.
MAP_ID_CELADON_MART_ROOF = 126
VENDING_MACHINE_SPOTS = ((10, 1), (11, 1), (12, 2))     # data/maps/objects/CeladonMartRoof.asm
GUARD_DRINKS = (0x3C, 0x3D, 0x3E)                        # FRESH_WATER · SODA_POP · LEMONADE
ITEM_FRESH_WATER = 0x3C
ADDR_STATUS_FLAGS1 = 0xD728
BIT_GAVE_SAFFRON_GUARDS_DRINK = 6
_vending_talking = False


def apply_buy_saffron_drink(pyboy) -> bool:
    """옥상 자판기에 말을 걸었는데 음료가 안 생겼으면 맛있는물을 하나 넣어 준다."""
    global _vending_talking
    if not cheat_flags.get("buy_saffron_drink", False):
        return False
    if int(pyboy.memory[ADDR_MAP_ID]) != MAP_ID_CELADON_MART_ROOF:
        _vending_talking = False
        return False
    px, py = int(pyboy.memory[ADDR_PLAYER_X]), int(pyboy.memory[ADDR_PLAYER_Y])
    beside = any(abs(px - x) + abs(py - y) == 1 for x, y in VENDING_MACHINE_SPOTS)
    if beside and is_textbox_open(pyboy):
        _vending_talking = True          # 자판기 옆에서 대화창이 떴다 — 끝나기를 기다린다
        return False
    if not _vending_talking:
        return False
    _vending_talking = False             # 대화가 끝났다 (또는 자리를 떴다)
    if int(pyboy.memory[ADDR_STATUS_FLAGS1]) & (1 << BIT_GAVE_SAFFRON_GUARDS_DRINK):
        return False                     # 경비원에게 이미 줬다
    if any(has_bag_item(pyboy, item) for item in GUARD_DRINKS):
        return False                     # 게임이 팔았다 — 할 일 없다
    if not give_bag_item(pyboy, ITEM_FRESH_WATER, 1):
        return False                     # 가방이 찼다
    print("[cheat] 자판기에서 음료가 안 나와서 맛있는물을 넣어 줍니다")
    return True


# 무지개시티 게임코너 (GAME_CORNER, map 135) — 포스터 뒤 스위치 (scripts/GameCorner.asm).
MAP_ID_GAME_CORNER = 135
ADDR_FOUND_ROCKET_HIDEOUT = 0xD77E           # data/events.json "Found Rocket Hideout"
BIT_FOUND_ROCKET_HIDEOUT = 1
SCRIPT_GAMECORNER_ROCKET_BATTLE = 1
GAME_CORNER_STAIRS_BLOCK = 0x43
GAME_CORNER_STAIRS_ROW, GAME_CORNER_STAIRS_COL = 2, 8
_game_corner_battle_seen = False


def apply_skip_game_corner_poster(pyboy) -> bool:
    """포스터 앞 로켓단원을 **이기면** 포스터 스위치까지 눌러 준다."""
    global _game_corner_battle_seen
    if not cheat_flags.get("skip_game_corner_poster", False):
        return False
    try:
        m = pyboy.memory
        if int(m[ADDR_MAP_ID]) != MAP_ID_GAME_CORNER:
            _game_corner_battle_seen = False
            return False
        if int(m[ADDR_FOUND_ROCKET_HIDEOUT]) & (1 << BIT_FOUND_ROCKET_HIDEOUT):
            return False                 # 이미 찾았다
        script_addr = _sym(pyboy, "wGameCornerCurScript")
        if script_addr is None or int(m[script_addr]) != SCRIPT_GAMECORNER_ROCKET_BATTLE:
            _game_corner_battle_seen = False
            return False                 # 로켓단원과 싸우는 단계가 아니다
        battle = int(m[ADDR_IN_BATTLE])
        if battle != 0:
            # 말을 건 뒤 전투 중. 0xFF(전멸) 이면 게임이 스크립트를 되돌린다.
            _game_corner_battle_seen = battle != 0xFF
            return False
        if not _game_corner_battle_seen:
            return False                 # 말은 걸었는데 아직 전투 전이다
        base = _sym(pyboy, "wOverworldMap")
        width_addr = _sym(pyboy, "wCurMapWidth")
        if base is None or width_addr is None:
            return False
        stride = int(m[width_addr]) + 6              # 맵 폭 + 테두리 3블록씩
        addr = (base + stride * 3 + 3
                + GAME_CORNER_STAIRS_ROW * stride + GAME_CORNER_STAIRS_COL)
        _cheat_event(ADDR_FOUND_ROCKET_HIDEOUT, BIT_FOUND_ROCKET_HIDEOUT)
        m[ADDR_FOUND_ROCKET_HIDEOUT] = int(m[ADDR_FOUND_ROCKET_HIDEOUT]) | (1 << BIT_FOUND_ROCKET_HIDEOUT)
        m[addr] = GAME_CORNER_STAIRS_BLOCK
    except Exception:
        return False
    _game_corner_battle_seen = False
    print("[cheat] 로켓단원을 이겼습니다 — 포스터 뒤 스위치를 대신 눌러 지하 계단을 엽니다")
    return True


# 잠만보를 깨운 도로 (map_id). 그 도로를 떠나면 지운다.
_snorlax_woke_on: int | None = None


def _toggle_hidden(pyboy, index: int) -> bool:
    addr = ADDR_MISSABLE_OBJECT_FLAGS + index // 8
    return bool(int(pyboy.memory[addr]) & (1 << (index % 8)))


def _set_toggle(pyboy, index: int, hidden: bool) -> None:
    addr = ADDR_MISSABLE_OBJECT_FLAGS + index // 8
    bit = 1 << (index % 8)
    value = int(pyboy.memory[addr])
    pyboy.memory[addr] = (value | bit) if hidden else (value & ~bit & 0xFF)


def _give_ss_ticket(pyboy) -> bool:
    """캡슐을 쓴 뒤 아직 티켓이 없으면 빌 대사가 하는 일을 대신한다."""
    events = int(pyboy.memory[ADDR_BILL_EVENTS])
    if not events & (1 << BIT_USED_CELL_SEPARATOR_ON_BILL):
        return False                     # 2) 가 아직이다
    if events & (1 << BIT_GOT_SS_TICKET):
        return False                     # 이미 받았다
    if int(pyboy.memory[ADDR_JOY_IGNORE]) != 0:
        return False                     # 연출 중이다
    if int(pyboy.memory[ADDR_MAP_ID]) == MAP_ID_BILLS_HOUSE and int(
        pyboy.memory[ADDR_BILLS_HOUSE_CUR_SCRIPT]
    ) in (SCRIPT_BILLSHOUSE_BILL_EXITS_MACHINE, SCRIPT_BILLSHOUSE_CLEANUP):
        return False                     # 빌이 캡슐에서 나오는 중이다
    if not give_bag_item(pyboy, ITEM_SS_TICKET, 1):
        print("[cheat] 가방이 가득 차 승선 티켓을 넣지 못했습니다 — 가방을 비우면 다시 시도합니다")
        return False
    _cheat_event(ADDR_BILL_EVENTS, BIT_GOT_SS_TICKET)
    pyboy.memory[ADDR_BILL_EVENTS] = events | (1 << BIT_GOT_SS_TICKET)
    _set_toggle(pyboy, TOGGLE_CERULEAN_GUARD_1, hidden=False)
    _set_toggle(pyboy, TOGGLE_CERULEAN_GUARD_2, hidden=True)
    print("[cheat] 빌에게서 승선 티켓을 받았습니다 (Got Ss Ticket)")
    return True


def _read_collision_ptr(pyboy) -> int:
    lo = int(pyboy.memory[ADDR_TILESET_COLLISION_PTR])
    hi = int(pyboy.memory[ADDR_TILESET_COLLISION_PTR + 1])
    return lo | (hi << 8)


def _write_collision_ptr(pyboy, ptr: int) -> None:
    pyboy.memory[ADDR_TILESET_COLLISION_PTR] = ptr & 0xFF
    pyboy.memory[ADDR_TILESET_COLLISION_PTR + 1] = (ptr >> 8) & 0xFF


def _rebuild_cut_passable_table(pyboy) -> bool:
    """고정 통과 타일 + (현재 타일셋의) 풀베기 나무만 합쳐 WRAM 버퍼에 쓴다."""
    tileset = int(pyboy.memory[ADDR_CUR_MAP_TILESET])
    base_tiles = TILESET_WALKABLE.get(tileset)
    if base_tiles is None:
        return False

    merged: list[int] = []
    seen: set[int] = set()
    for t in base_tiles:
        t = int(t) & 0xFF
        if t == 0xFF or t in seen:
            continue
        seen.add(t)
        merged.append(t)

    # UsedCut 과 동일: OVERWORLD->0x3D, GYM->0x50 만
    cut = TILESET_CUT_TREE_TILE.get(tileset)
    if cut is not None:
        cut = int(cut) & 0xFF
        if cut != 0xFF and cut not in seen:
            merged.append(cut)

    buf = ADDR_NOCLIP_PASSABLE_BUF
    for i, t in enumerate(merged):
        pyboy.memory[buf + i] = t
    pyboy.memory[buf + len(merged)] = 0xFF
    return True


def _capture_real_collision_ptr(pyboy) -> None:
    """현재 포인터가 우리 버퍼가 아닐 때만 원본으로 저장."""
    global _noclip_real_collision_ptr, _noclip_ptr_map_id
    ptr = _read_collision_ptr(pyboy)
    if ptr == ADDR_NOCLIP_PASSABLE_BUF:
        return
    _noclip_real_collision_ptr = ptr
    _noclip_ptr_map_id = int(pyboy.memory[ADDR_MAP_ID])


def _enable_strength_for_push(pyboy) -> None:
    """괴력 바위는 통과하지 않고, Strength 활성만 켜서 밀 수 있게 한다."""
    pyboy.memory[ADDR_STATUS_FLAGS1] = int(pyboy.memory[ADDR_STATUS_FLAGS1]) | (
        1 << BIT_STRENGTH_ACTIVE
    )


def apply_noclip_cheat(pyboy) -> None:
    """풀베기 나무만 통과."""
    global _noclip_ptr_map_id, _noclip_real_collision_ptr
    if not cheat_flags["noclip"]:
        if _noclip_real_collision_ptr is not None:
            _write_collision_ptr(pyboy, _noclip_real_collision_ptr)
            _noclip_real_collision_ptr = None
            _noclip_ptr_map_id = None
        return

    cur_map = int(pyboy.memory[ADDR_MAP_ID])
    if _noclip_ptr_map_id is not None and cur_map != _noclip_ptr_map_id:
        _noclip_ptr_map_id = cur_map
        _capture_real_collision_ptr(pyboy)
        return

    if _noclip_real_collision_ptr is None:
        _capture_real_collision_ptr(pyboy)
    if _noclip_real_collision_ptr is None:
        return

    if int(pyboy.memory[ADDR_SIMULATED_JOYPAD_INDEX]) != 0:
        flags5 = int(pyboy.memory[ADDR_STATUS_FLAGS5])
        if not (flags5 & (1 << BIT_SCRIPTED_MOVEMENT_STATE)):
            pyboy.memory[ADDR_SIMULATED_JOYPAD_INDEX] = 0

    _enable_strength_for_push(pyboy)

    if not _rebuild_cut_passable_table(pyboy):
        _write_collision_ptr(pyboy, _noclip_real_collision_ptr)
        return

    _write_collision_ptr(pyboy, ADDR_NOCLIP_PASSABLE_BUF)


def handle_noclip_command(line: str) -> None:
    """noclip | noclip on | noclip off"""
    global _noclip_real_collision_ptr, _noclip_ptr_map_id
    parts = line.strip().lower().split()
    if len(parts) == 1:
        state = "ON" if cheat_flags["noclip"] else "off"
        print(f"[noclip] 현재 {state}  (사용: noclip on|off)")
        return
    if len(parts) >= 2 and parts[1] in ("on", "off"):
        turning_on = parts[1] == "on"
        cheat_flags["noclip"] = turning_on
        if turning_on:
            _noclip_ptr_map_id = None
            _noclip_real_collision_ptr = None
        state = "ON" if turning_on else "OFF"
        print(f"[noclip] -> {state}")
        if turning_on:
            print(
                "  풀베기 나무는 통과, 괴력 바위는 밀기 가능. "
                "그 외 벽/NPC는 막힘."
            )
        else:
            print("  꺼짐. 충돌 포인터 복구 중.")
        return
    print("[noclip] 사용: noclip | noclip on|off")
