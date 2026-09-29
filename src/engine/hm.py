"""hm.py — 비전머신(풀베기 / 파도타기)을 실제로 사용한다."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core import runlog
from core.config import cheat_opt as _cfg_cheat_opt
from core.gb_memory import (
    ADDR_PARTY_COUNT,
    ADDR_TILESET,
    ADDR_WALK_COUNTER,
    is_dialogue_active,
    is_in_battle,
    is_textbox_open,
    read_party,
)
from core.tileset_data import TILESET_CUT_TREE_TILE, WATER_TILE, WATER_TILESETS

# --- 배지 (constants/ram_constants.asm 의 const_def 순서) -------------------
ADDR_OBTAINED_BADGES = 0xD356       # wObtainedBadges
BIT_BOULDERBADGE = 0
BIT_CASCADEBADGE = 1                # 풀베기
BIT_THUNDERBADGE = 2
BIT_RAINBOWBADGE = 3
BIT_SOULBADGE = 4                   # 파도타기
BIT_MARSHBADGE = 5
BIT_VOLCANOBADGE = 6
BIT_EARTHBADGE = 7

ADDR_STATUS_FLAGS1 = 0xD728         # wStatusFlags1
BIT_SURF_ALLOWED = 1

# 배지는 "쓸 수 있는가"만 정한다 (start_sub_menus.asm 의 .cut/.surf 는 wObtainedBadges 만 본다).
EVENT_GOT_HM01 = (0xD803, 0)        # 풀베기
EVENT_GOT_HM03 = (0xD857, 0)        # 파도타기

# --- 이동 상태 --------------------------------------------------------------
ADDR_WALK_BIKE_SURF_STATE = 0xD700  # 0=걷기 1=자전거 2=파도타기
SURF_STATE = 2
ADDR_TILE_IN_FRONT = 0xCFC6         # wTileInFrontOfPlayer
ADDR_ACTION_RESULT = 0xCD6A         # wActionResultOrTookBattleTurn (Cut 성공=1)

# --- 메뉴 -------------------------------------------------------------------
ADDR_TOP_MENU_Y = 0xCC24
ADDR_TOP_MENU_X = 0xCC25
ADDR_CURRENT_MENU_ITEM = 0xCC26
ADDR_MAX_MENU_ITEM = 0xCC28
ADDR_FIELD_MOVES = 0xCD3D           # wFieldMoves (4바이트, FieldMoveNames 인덱스)
ADDR_NUM_FIELD_MOVES = 0xCD41       # 표시 직전에만 유효 — 감지에 쓰지 말 것
ADDR_WHICH_POKEMON = 0xCF92
ADDR_PARTY_COUNT = 0xD163

# 메뉴 기하 (실측 + asm 대조) 스타트 메뉴  : draw_start_menu.asm — topY=2, topX=$0B, max=7(도감有)/6(無) 파티 메뉴    :…
START_MENU_TOP_Y = 2
START_MENU_TOP_X = 0x0B
START_MAX_WITH_DEX = 7
START_MAX_WITHOUT_DEX = 6
PARTY_MENU_TOP_Y = 1
PARTY_MENU_TOP_X = 0
FIELD_MENU_BASE_MAX = 2
FIELD_MENU_BASE_Y = 12

# data/moves/field_moves.asm 의 "FieldMoveNames index"
FIELD_MOVE_CUT = 1
FIELD_MOVE_SURF = 4

# 기술 ID (engine/move_learn.py HM_MOVE_IDS 와 같은 출처)
MOVE_CUT = 0x0F
MOVE_SURF = 0x39

BUTTON_HOLD_FRAMES = 6
BUTTON_GAP_FRAMES = 10
MENU_WAIT_FRAMES = 120
ACTION_WAIT_FRAMES = 420
# 메뉴가 "떠 있다"고 판정된 뒤에도 HandleMenuInput 이 입력을 받을 준비가 될 때까지 몇 프레임 걸린다.
MENU_SETTLE_FRAMES = 24
MENU_ADVANCE_TRIES = 4

# 능력을 쓸지 정하는 설정 값
MODE_STORY = "story"                # 배지 + HM 획득 이벤트 (게임 진행 그대로)
MODE_BADGE = "badge"                # 배지만
MODE_ALWAYS = "always"              # 항상
MODE_OFF = "off"


def _u8(pyboy: Any, addr: int) -> int:
    try:
        return int(pyboy.memory[addr]) & 0xFF
    except Exception:
        return 0


def _mode(name: str) -> str:
    raw = str(_cfg_cheat_opt(name, MODE_OFF)).strip().lower()
    if raw in ("true", "on", "1", "yes"):
        return MODE_ALWAYS
    if raw in ("false", "0", "no", "none"):
        return MODE_OFF
    if raw in (MODE_STORY, MODE_BADGE, MODE_ALWAYS, MODE_OFF):
        return raw
    return MODE_OFF


# ---------------------------------------------------------------------------
# 능력 판정
# ---------------------------------------------------------------------------
def has_badge(pyboy: Any, bit: int) -> bool:
    return bool(_u8(pyboy, ADDR_OBTAINED_BADGES) & (1 << bit))


def badge_count(pyboy: Any) -> int:
    """딴 배지 개수 — 단조 증가하는 진행도 지표."""
    return bin(_u8(pyboy, ADDR_OBTAINED_BADGES)).count("1")


def party_slot_with_move(pyboy: Any, move_id: int) -> Optional[int]:
    for mon in read_party(pyboy):
        if move_id in mon["move_ids"]:
            return int(mon["slot"])
    return None


def has_event(pyboy: Any, flag: tuple) -> bool:
    addr, bit = flag
    return bool(_u8(pyboy, addr) & (1 << bit))


def _unlocked(pyboy: Any, key: str, bit: int, event: tuple) -> bool:
    """그 비전머신을 쓸 자격이 됐는지."""
    mode = _mode(key)
    if mode == MODE_OFF:
        return False
    if mode == MODE_ALWAYS:
        return True
    if not has_badge(pyboy, bit):
        return False
    if mode == MODE_STORY and not has_event(pyboy, event):
        return False
    return True


def cut_unlocked(pyboy: Any) -> bool:
    return _unlocked(pyboy, "hm_cut", BIT_CASCADEBADGE, EVENT_GOT_HM01)


def surf_unlocked(pyboy: Any) -> bool:
    return _unlocked(pyboy, "hm_surf", BIT_SOULBADGE, EVENT_GOT_HM03)


def field_move_status(pyboy: Any) -> List[str]:
    """비전머신을 지금 쓸 수 있는지, 못 쓰면 **무엇이 없어서인지** 한 줄씩."""
    out: List[str] = []
    for label, key, bit, badge, event in (
        ("풀베기", "hm_cut", BIT_CASCADEBADGE, "CASCADE", EVENT_GOT_HM01),
        ("파도타기", "hm_surf", BIT_SOULBADGE, "SOUL", EVENT_GOT_HM03),
    ):
        mode = _mode(key)
        if mode == MODE_OFF:
            continue
        why = []
        if mode != MODE_ALWAYS:
            if not has_badge(pyboy, bit):
                why.append(f"{badge} 배지 없음")
            if mode == MODE_STORY and not has_event(pyboy, event):
                why.append("기술머신 아직 못 받음")
        out.append(
            f"{label}: 쓸 수 있음" if not why else f"{label}: 못 씀 — {' · '.join(why)}"
        )
    return out


def can_cut(pyboy: Any) -> bool:
    if not cut_unlocked(pyboy):
        return False
    return party_slot_with_move(pyboy, MOVE_CUT) is not None


def can_surf(pyboy: Any) -> bool:
    # 몬이 없어도 상태 세팅 폴백으로 건널 수 있다.
    return surf_unlocked(pyboy)


def is_surfing(pyboy: Any) -> bool:
    return _u8(pyboy, ADDR_WALK_BIKE_SURF_STATE) == SURF_STATE


def walk_done(pyboy: Any) -> bool:
    """걸음 애니메이션이 끝났는지 (wWalkCounter == 0)."""
    return _u8(pyboy, ADDR_WALK_COUNTER) == 0


def facing_water(pyboy: Any) -> bool:
    if _u8(pyboy, ADDR_TILESET) not in WATER_TILESETS:
        return False
    return _u8(pyboy, ADDR_TILE_IN_FRONT) == WATER_TILE


def facing_cut_tree(pyboy: Any) -> bool:
    tree = TILESET_CUT_TREE_TILE.get(_u8(pyboy, ADDR_TILESET))
    return tree is not None and _u8(pyboy, ADDR_TILE_IN_FRONT) == tree


# ---------------------------------------------------------------------------
# 능력 지급 (여기까지만 치트)
# ---------------------------------------------------------------------------
def grant_badge(pyboy: Any, bit: int) -> bool:
    """배지 비트를 켠다."""
    cur = _u8(pyboy, ADDR_OBTAINED_BADGES)
    if cur & (1 << bit):
        return False
    pyboy.memory[ADDR_OBTAINED_BADGES] = cur | (1 << bit)
    return True


def allow_surf(pyboy: Any) -> None:
    """wStatusFlags1 의 SURF_ALLOWED 를 켠다 (IsSurfingAllowed 우회)."""
    pyboy.memory[ADDR_STATUS_FLAGS1] = _u8(pyboy, ADDR_STATUS_FLAGS1) | (
        1 << BIT_SURF_ALLOWED
    )


# --- 비전머신 운반용 포켓몬 -------------------------------------------------
# 기존 파티 몬의 기술 슬롯을 뺏지 않는다. 풀베기와 파도타기를 **둘 다** 배우는
# 종을 한 마리 파티 끝에 넣고, 나중에 얻는 기술도 같은 개체에게 추가한다.
HM_MOVE_MAX_PP = {MOVE_CUT: 30, MOVE_SURF: 15}   # data/moves/moves.asm
DEFAULT_CARRIER = "tentacool"

# 마지막 실패 이유. sync 가 같은 말을 반복하지 않도록 한 번만 읽어 찍는다.
_last_error: str = ""


def _carrier_cfg() -> Dict[str, Any]:
    cfg = _cfg_cheat_opt("hm_carrier", None)
    return dict(cfg) if isinstance(cfg, dict) else {}


def _carrier_species_id() -> Optional[int]:
    cfg = _carrier_cfg()
    if not cfg.get("enabled", True):
        return None
    from engine.party_cheat import _resolve_species

    try:
        return int(_resolve_species(cfg.get("species") or DEFAULT_CARRIER)["id"])
    except Exception as exc:
        print(f"[hm] 운반용 종을 알 수 없습니다: {exc}")
        return None


def carrier_slot(pyboy: Any) -> Optional[int]:
    """운반용 몬이 파티에 있으면 그 슬롯 (0-based)."""
    species_id = _carrier_species_id()
    if species_id is None:
        return None
    for mon in read_party(pyboy):
        if int(mon["species_id"]) == species_id:
            return int(mon["slot"])
    return None


def _party_writable(pyboy: Any) -> bool:
    """파티 구조를 건드려도 안전한 시점인가."""
    if is_in_battle(pyboy) or is_textbox_open(pyboy) or is_dialogue_active(pyboy):
        return False
    return bool(read_party(pyboy))


def _teach_to_slot(pyboy: Any, slot: int, move_id: int) -> bool:
    """그 슬롯의 **빈** 기술 칸에 넣는다."""
    from core.gb_memory import (
        ADDR_PARTY_MONS,
        MON_MOVES,
        MON_PP,
        PARTYMON_STRUCT_LENGTH,
    )

    base = ADDR_PARTY_MONS + int(slot) * PARTYMON_STRUCT_LENGTH
    for i in range(4):
        if int(pyboy.memory[base + MON_MOVES + i]) == move_id:
            return True
    for i in range(4):
        if int(pyboy.memory[base + MON_MOVES + i]) == 0:
            pyboy.memory[base + MON_MOVES + i] = move_id
            pyboy.memory[base + MON_PP + i] = HM_MOVE_MAX_PP.get(move_id, 20)
            return True
    return False


def ensure_carrier_move(pyboy: Any, move_id: int) -> Optional[int]:
    """그 기술을 쓸 수 있는 파티 슬롯을 확보한다."""
    global _last_error
    _last_error = ""
    slot = party_slot_with_move(pyboy, move_id)
    if slot is not None:
        return slot
    if _carrier_species_id() is None:
        _last_error = "운반용 포켓몬이 꺼져 있습니다 (cheats.hm_carrier.enabled)"
        return None
    if not _party_writable(pyboy):
        # 전투·대화 중이거나 파티가 아직 없다. 실패가 아니라 "지금은 때가 아님".
        return None

    slot = carrier_slot(pyboy)
    if slot is not None:
        if _teach_to_slot(pyboy, slot, move_id):
            return slot
        _last_error = "운반용 포켓몬의 기술 4칸이 모두 찼습니다"
        return None

    from engine.party_cheat import give_party_mon

    cfg = _carrier_cfg()
    msg = give_party_mon(
        pyboy,
        {
            "enabled": True,
            "species": cfg.get("species") or DEFAULT_CARRIER,
            "level": int(cfg.get("level", 20)),
            "nickname": cfg.get("nickname") or "HMMON",
            "skip_if_have": True,
            "moves": [move_id],
            "pps": [HM_MOVE_MAX_PP.get(move_id, 20)],
        },
    )
    slot = carrier_slot(pyboy)
    if slot is None:
        _last_error = f"운반용 포켓몬을 파티에 넣지 못했습니다 ({msg})"
    return slot


def last_error() -> str:
    """마지막 ``ensure_carrier_move`` 실패 이유."""
    return _last_error


def grant_cut(pyboy: Any) -> bool:
    grant_badge(pyboy, BIT_CASCADEBADGE)
    return ensure_carrier_move(pyboy, MOVE_CUT) is not None


def grant_surf(pyboy: Any) -> bool:
    grant_badge(pyboy, BIT_SOULBADGE)
    allow_surf(pyboy)
    return ensure_carrier_move(pyboy, MOVE_SURF) is not None


# ---------------------------------------------------------------------------
# 사용 — 스타트 메뉴 매크로
# ---------------------------------------------------------------------------
def _tick(pyboy: Any, n: int = 1) -> None:
    for _ in range(n):
        pyboy.tick()


def bump(pyboy: Any, direction: str) -> None:
    """그 방향으로 한 번 밀어 본다."""
    _tap(pyboy, direction)
    for _ in range(MENU_WAIT_FRAMES):
        if walk_done(pyboy):
            break
        _tick(pyboy, 1)


def _tap(pyboy: Any, button: str) -> None:
    try:
        pyboy.button_press(button)
        _tick(pyboy, BUTTON_HOLD_FRAMES)
    finally:
        try:
            pyboy.button_release(button)
        except Exception:
            pass
    _tick(pyboy, BUTTON_GAP_FRAMES)


def _field_moves(pyboy: Any) -> List[int]:
    return [_u8(pyboy, ADDR_FIELD_MOVES + i) for i in range(4)]


def _wait(pyboy: Any, predicate, frames: int = MENU_WAIT_FRAMES) -> bool:
    for _ in range(frames):
        if predicate():
            return True
        _tick(pyboy, 1)
    return predicate()


def _advance(pyboy: Any, done, *, before=None, tries: int = MENU_ADVANCE_TRIES) -> bool:
    """다음 메뉴로 넘어갈 때까지 A 를 누른다."""
    _tick(pyboy, MENU_SETTLE_FRAMES)
    for _ in range(tries):
        if done():
            return True
        if before is not None:
            before()
        _tap(pyboy, "a")
        if _wait(pyboy, done, MENU_WAIT_FRAMES):
            return True
    return done()


def _bail(pyboy: Any, why: str) -> bool:
    """메뉴에서 빠져나온다."""
    print(f"[hm] {why} - 메뉴를 닫습니다")
    for _ in range(4):
        _tap(pyboy, "b")
    return False


def start_menu_open(pyboy: Any) -> bool:
    return (
        _u8(pyboy, ADDR_TOP_MENU_Y) == START_MENU_TOP_Y
        and _u8(pyboy, ADDR_TOP_MENU_X) == START_MENU_TOP_X
        and _u8(pyboy, ADDR_MAX_MENU_ITEM)
        in (START_MAX_WITHOUT_DEX, START_MAX_WITH_DEX)
    )


def party_menu_open(pyboy: Any) -> bool:
    count = _u8(pyboy, ADDR_PARTY_COUNT)
    return (
        count > 0
        and _u8(pyboy, ADDR_TOP_MENU_Y) == PARTY_MENU_TOP_Y
        and _u8(pyboy, ADDR_TOP_MENU_X) == PARTY_MENU_TOP_X
        and _u8(pyboy, ADDR_MAX_MENU_ITEM) == count - 1
    )


def field_move_count(pyboy: Any) -> int:
    """필드기술 메뉴가 떠 있으면 그 기술 개수, 아니면 0."""
    max_item = _u8(pyboy, ADDR_MAX_MENU_ITEM)
    n = max_item - FIELD_MENU_BASE_MAX
    if not 1 <= n <= 4:
        return 0
    if _u8(pyboy, ADDR_TOP_MENU_Y) != FIELD_MENU_BASE_Y - 2 * n:
        return 0
    return n


def use_field_move(pyboy: Any, field_move: int, move_id: int) -> bool:
    """START -> POKéMON -> 몬 -> 필드기술 을 눌러 실제로 사용한다."""
    if is_in_battle(pyboy):
        return False

    slot = party_slot_with_move(pyboy, move_id)
    if slot is None:
        print(f"[hm] 기술 0x{move_id:02X} 를 아는 파티 몬이 없습니다")
        return False

    # 대화/스크립트 중에는 START 가 안 먹는다.
    if not _wait(pyboy, lambda: not is_dialogue_active(pyboy)):
        return False

    # 걸음 애니메이션 중이면 START 가 씹힌다. 몇 번 다시 눌러 본다.
    opened = False
    for _ in range(MENU_ADVANCE_TRIES):
        _tap(pyboy, "start")
        if _wait(pyboy, lambda: start_menu_open(pyboy)):
            opened = True
            break
    if not opened:
        return _bail(pyboy, "스타트 메뉴가 안 열림")

    # 도감이 있으면 POKéMON 이 1번, 없으면 0번이다 (home/start_menu.asm .displayMenuItem 이 도감 없을 때 +1 보정한다).
    def _pick_pokemon() -> None:
        pyboy.memory[ADDR_CURRENT_MENU_ITEM] = (
            1 if _u8(pyboy, ADDR_MAX_MENU_ITEM) == START_MAX_WITH_DEX else 0
        )

    if not _advance(pyboy, lambda: party_menu_open(pyboy), before=_pick_pokemon):
        return _bail(pyboy, "파티 메뉴가 안 열림")

    def _pick_slot() -> None:
        pyboy.memory[ADDR_CURRENT_MENU_ITEM] = slot

    if not _advance(
        pyboy, lambda: field_move_count(pyboy) > 0, before=_pick_slot
    ):
        return _bail(pyboy, "필드기술 메뉴가 안 열림")
    n = field_move_count(pyboy)
    # 이 몬이 실제로 쓸 수 있는 것만 본다 — 뒤쪽은 지난 번 값이 남아 있다.
    moves = _field_moves(pyboy)[:n]
    if field_move not in moves:
        return _bail(pyboy, f"필드기술 목록에 없음 {moves}")

    _tick(pyboy, MENU_SETTLE_FRAMES)
    pyboy.memory[ADDR_CURRENT_MENU_ITEM] = moves.index(field_move)
    _tap(pyboy, "a")
    return True


def _clear_text(pyboy: Any, taps: int = 12) -> None:
    """열려 있는 텍스트박스를 A 로 넘긴다."""
    for _ in range(taps):
        if not is_textbox_open(pyboy):
            return
        _tap(pyboy, "a")


def do_cut(pyboy: Any) -> bool:
    """앞의 나무를 벤다."""
    if not facing_cut_tree(pyboy):
        return False
    pyboy.memory[ADDR_ACTION_RESULT] = 0
    if not use_field_move(pyboy, FIELD_MOVE_CUT, MOVE_CUT):
        return False

    # cut.asm:26-29 — UsedCut 이 벨 수 있는 타일이라고 인정하면 wActionResultOrTookBattleTurn 이 1 이 된다.
    if not _wait(
        pyboy,
        lambda: _u8(pyboy, ADDR_ACTION_RESULT) == 1,
        ACTION_WAIT_FRAMES,
    ):
        return _bail(pyboy, "풀베기가 받아들여지지 않음")

    # "<MON> used CUT!" 텍스트가 A 를 기다린다.
    _clear_text(pyboy)
    print("[hm] 풀베기 성공")
    runlog.log("hm", 사용="풀베기")
    return True


def do_surf(pyboy: Any) -> bool:
    """물로 들어간다."""
    if is_surfing(pyboy) or not facing_water(pyboy):
        return False

    if party_slot_with_move(pyboy, MOVE_SURF) is not None:
        allow_surf(pyboy)
        if use_field_move(pyboy, FIELD_MOVE_SURF, MOVE_SURF):
            if _wait(pyboy, lambda: is_surfing(pyboy), ACTION_WAIT_FRAMES):
                # 파도타기 상태가 됐어도 메뉴/텍스트가 남아 있으면 이동 입력이 먹지 않는다 (wFontLoaded bit0 가 계속 1 로 남는다 — 실측).
                _clear_text(pyboy)
                _wait(
                    pyboy,
                    lambda: walk_done(pyboy) and not is_dialogue_active(pyboy),
                    ACTION_WAIT_FRAMES,
                )
                print("[hm] 파도타기 시작")
                runlog.log("hm", 사용="파도타기")
                return True

    # 폴백 — 게임을 파도타기 상태로 만든다.
    pyboy.memory[ADDR_WALK_BIKE_SURF_STATE] = SURF_STATE
    print("[hm] 파도타기 상태로 전환 (기술 없이 폴백)")
    return is_surfing(pyboy)


# key -> 마지막으로 알린 실패 이유. 같은 말을 두 번 찍지 않기 위한 것.
_warned: Dict[str, str] = {}
# 마지막으로 처리를 끝낸 입력 상태. 이게 그대로면 sync 는 아무것도 하지 않는다.
_sync_sig: Optional[tuple] = None


def _sync_signature(pyboy: Any) -> tuple:
    """``sync`` 의 결과를 바꿀 수 있는 입력 전부."""
    m = pyboy.memory
    return (
        int(m[ADDR_OBTAINED_BADGES]),
        int(m[EVENT_GOT_HM01[0]]),
        int(m[EVENT_GOT_HM03[0]]),
        int(m[ADDR_PARTY_COUNT]),
    )


def sync(pyboy: Any) -> None:
    """설정에 맞춰 비전머신 능력을 맞춰 준다."""
    global _sync_sig
    sig = _sync_signature(pyboy)
    if sig == _sync_sig:
        return

    # 전투·대화 중이라 파티를 못 건드린 것은 "실패"가 아니라 "아직"이다.
    pending = False
    for key, bit, event, move_id, label in (
        ("hm_cut", BIT_CASCADEBADGE, EVENT_GOT_HM01, MOVE_CUT, "풀베기"),
        ("hm_surf", BIT_SOULBADGE, EVENT_GOT_HM03, MOVE_SURF, "파도타기"),
    ):
        if _mode(key) == MODE_ALWAYS and grant_badge(pyboy, bit):
            print(f"[hm] {label} 배지 지급 (always)")
        if not _unlocked(pyboy, key, bit, event):
            continue
        if move_id == MOVE_SURF:
            allow_surf(pyboy)
        if party_slot_with_move(pyboy, move_id) is not None:
            _warned.pop(key, None)
            continue

        slot = ensure_carrier_move(pyboy, move_id)
        if slot is not None:
            _warned.pop(key, None)
            print(f"[hm] {label} 를 파티 {slot + 1}번에게 주었습니다")
            runlog.log("hm", 기술=label, 파티슬롯=slot + 1, 모드=_mode(key))
            continue

        why = last_error()
        if not why:
            pending = True   # 지금은 때가 아님 — 조용히 다음 프레임에 다시
            continue
        if _warned.get(key) != why:
            _warned[key] = why
            print(f"[hm] {label} 를 줄 수 없습니다 — {why}")
            runlog.log("hm", 기술=label, 실패=why, 모드=_mode(key))
            print("     (파티를 한 자리 비우거나 cheats.hm_carrier 를 확인하세요)")

    if not pending:
        _sync_sig = sig


def use_for_direction(pyboy: Any, direction: str) -> str:
    """그 방향이 나무/물이면 알맞은 HM 을 쓴다."""
    if is_in_battle(pyboy):
        return ""
    bump(pyboy, direction)
    if facing_cut_tree(pyboy):
        if can_cut(pyboy) and do_cut(pyboy):
            return "cut"
        return ""
    if facing_water(pyboy) and not is_surfing(pyboy):
        if can_surf(pyboy) and do_surf(pyboy):
            return "surf"
    return ""


__all__ = [
    "BIT_CASCADEBADGE",
    "BIT_SOULBADGE",
    "MOVE_CUT",
    "MOVE_SURF",
    "badge_count",
    "can_cut",
    "can_surf",
    "do_cut",
    "do_surf",
    "facing_cut_tree",
    "facing_water",
    "field_move_count",
    "field_move_status",
    "party_menu_open",
    "start_menu_open",
    "carrier_slot",
    "cut_unlocked",
    "ensure_carrier_move",
    "grant_cut",
    "grant_surf",
    "has_event",
    "surf_unlocked",
    "has_badge",
    "is_surfing",
    "party_slot_with_move",
    "sync",
    "use_field_move",
    "use_for_direction",
    "walk_done",
]
