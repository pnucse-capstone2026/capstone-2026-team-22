"""레벨업 기술 교체를 자동으로 해결한다 — "가장 오래된 기술을 지운다"."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.config import ROOT

# ---------------------------------------------------------------------------
# 주소 — assets/roms/pokered.sym 대조 완료
# ---------------------------------------------------------------------------
ADDR_CURRENT_MENU_ITEM = 0xCC26   # wCurrentMenuItem — 교체 메뉴의 슬롯 인덱스
ADDR_MENU_CURSOR_LO = 0xCC30      # wMenuCursorLocation
ADDR_MENU_CURSOR_HI = 0xCC31
ADDR_MOVE_NUM = 0xD0E0            # wMoveNum — 새로 배우는 기술 ID
ADDR_PARTY_MONS = 0xD16B          # wPartyMons / wPartyMon1
ADDR_TOP_MENU_Y = 0xCC24          # wTopMenuItemY
ADDR_TOP_MENU_X = 0xCC25          # wTopMenuItemX
ADDR_MAX_MENU_ITEM = 0xCC28       # wMaxMenuItem
ADDR_MENU_WATCHED_KEYS = 0xCC29   # wMenuWatchedKeys
ADDR_NUM_MOVES_MINUS_ONE = 0xCD6C  # wNumMovesMinusOne
ADDR_MOVES_BUFFER = 0xD0DC        # wMoves — TryingToLearn 이 복사해 둔 현재 기술
ADDR_WHICH_POKEMON = 0xCF92       # wWhichPokemon
ADDR_PARTY_COUNT = 0xD163         # wPartyCount

PARTYMON_STRUCT_LENGTH = 0x2C
MON_MOVES = 8                     # wPartyMon1Moves = 0xD173
MON_OTID = 12
MON_DVS = 27

# learn_move.asm 의 hlcoord(5,8) = wTileMap(0xC3A0) + 8*20 + 5 = 0xC445 이고, BIT_DOUBLE_SPACED_MENU 가…
MOVE_LEARN_CURSOR_PAIRS: Tuple[Tuple[int, int], ...] = (
    (0x45, 0xC4),
    (0x59, 0xC4),
    (0x6D, 0xC4),
    (0x81, 0xC4),
)

# data/moves/hm_moves.asm — CUT / FLY / SURF / STRENGTH / FLASH.
HM_MOVE_IDS = frozenset({0x0F, 0x13, 0x39, 0x46, 0x94})

# 메뉴 입력 주기. DialogueSkipper 의 A 연타와 같은 값으로 맞춘다.
BUTTON_HOLD_FRAMES = 6
BUTTON_GAP_FRAMES = 10
# A 를 누른 뒤 "1, 2 and...
WRITE_TIMEOUT_FRAMES = 900

LEDGER_PATH = ROOT / "runtime" / "move_ages.json"

POLICIES = ("manual", "oldest", "slot0", "weakest")


def _u8(pyboy: Any, addr: int) -> int:
    return int(pyboy.memory[addr]) & 0xFF


def _party_base(party_index: int) -> int:
    return ADDR_PARTY_MONS + int(party_index) * PARTYMON_STRUCT_LENGTH


def read_party_moves(pyboy: Any, party_index: int) -> List[int]:
    base = _party_base(party_index) + MON_MOVES
    return [_u8(pyboy, base + i) for i in range(4)]


def mon_key(pyboy: Any, party_index: int) -> str:
    """개체 식별자."""
    base = _party_base(party_index)
    otid = (_u8(pyboy, base + MON_OTID) << 8) | _u8(pyboy, base + MON_OTID + 1)
    dvs = (_u8(pyboy, base + MON_DVS) << 8) | _u8(pyboy, base + MON_DVS + 1)
    return f"{otid:04X}-{dvs:04X}"


class MoveAgeLedger:
    """개체별 "슬롯 k 의 기술을 언제 배웠는가" 를 기록한다."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path is not None else LEDGER_PATH
        self.data: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        try:
            with io.open(self.path, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
            if isinstance(raw, dict):
                self.data = {
                    str(k): v for k, v in raw.items() if isinstance(v, dict)
                }
        except FileNotFoundError:
            self.data = {}
        except Exception:
            # 손상된 장부는 버린다. 슬롯 순서로 재초기화될 뿐이다.
            self.data = {}

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with io.open(self.path, "w", encoding="utf-8") as fp:
                json.dump(self.data, fp, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def entry(self, key: str, moves: Sequence[int]) -> Dict[str, Any]:
        """장부 항목을 얻는다."""
        moves = [int(m) & 0xFF for m in moves]
        found = self.data.get(key)
        if (
            isinstance(found, dict)
            and list(found.get("moves") or []) == moves
            and len(found.get("age") or []) == 4
        ):
            return found
        fresh: Dict[str, Any] = {"moves": moves, "age": [0, 1, 2, 3], "seq": 4}
        self.data[key] = fresh
        return fresh

    def oldest_slot(
        self,
        key: str,
        moves: Sequence[int],
        *,
        exclude: Iterable[int] = (),
    ) -> Optional[int]:
        entry = self.entry(key, moves)
        blocked = set(int(i) for i in exclude)
        candidates = [i for i in range(4) if i not in blocked]
        if not candidates:
            return None
        ages = entry["age"]
        return min(candidates, key=lambda i: (int(ages[i]), i))

    def record(
        self,
        key: str,
        slot: int,
        new_move_id: int,
        moves: Sequence[int],
    ) -> None:
        entry = self.entry(key, moves)
        updated = list(entry["moves"])
        updated[int(slot)] = int(new_move_id) & 0xFF
        entry["moves"] = updated
        seq = int(entry.get("seq", 4))
        entry["age"][int(slot)] = seq
        entry["seq"] = seq + 1
        self.save()


class MoveLearnResolver:
    """교체 메뉴에서 슬롯을 골라 확정한다."""

    def __init__(
        self,
        policy: str = "manual",
        *,
        ledger: Optional[MoveAgeLedger] = None,
        debug: bool = False,
    ) -> None:
        key = str(policy or "manual").strip().lower()
        if key not in POLICIES:
            raise ValueError(
                f"알 수 없는 on_move_learn: {policy!r} ({' | '.join(POLICIES)})"
            )
        self.policy = key
        self.debug = bool(debug)
        self.ledger = ledger if ledger is not None else MoveAgeLedger()
        self._move_power: Optional[Dict[int, int]] = None

    @property
    def enabled(self) -> bool:
        return self.policy != "manual"

    # ------------------------------------------------------------------
    # 슬롯 선택
    # ------------------------------------------------------------------
    def _power_table(self) -> Dict[int, int]:
        if self._move_power is None:
            table: Dict[int, int] = {}
            try:
                path = ROOT / "Skills" / "Battle" / "data" / "moves.json"
                with io.open(path, "r", encoding="utf-8") as fp:
                    raw = json.load(fp)
                for move_id, info in (raw or {}).items():
                    try:
                        table[int(move_id)] = int(
                            (info or {}).get("power", 0) or 0
                        )
                    except Exception:
                        continue
            except Exception:
                table = {}
            self._move_power = table
        return self._move_power

    def choose_slot(
        self,
        pyboy: Any,
        party_index: int,
        moves: Sequence[int],
    ) -> Optional[int]:
        # HM 은 게임이 삭제를 거부한다 (learn_move.asm .hm). 후보에서 뺀다.
        exclude = {i for i, m in enumerate(moves) if int(m) in HM_MOVE_IDS}
        if len(exclude) >= 4:
            return None

        if self.policy == "slot0":
            for i in range(4):
                if i not in exclude:
                    return i
            return None

        if self.policy == "weakest":
            power = self._power_table()
            candidates = [i for i in range(4) if i not in exclude]
            return min(
                candidates,
                key=lambda i: (power.get(int(moves[i]), 0), i),
            )

        return self.ledger.oldest_slot(
            mon_key(pyboy, party_index), moves, exclude=exclude
        )

    # ------------------------------------------------------------------
    # 메뉴 조작
    # ------------------------------------------------------------------
    def _press(self, pyboy: Any, button: str) -> None:
        try:
            pyboy.button_press(button)
            for _ in range(BUTTON_HOLD_FRAMES):
                pyboy.tick()
        finally:
            try:
                pyboy.button_release(button)
            except Exception:
                pass
        for _ in range(BUTTON_GAP_FRAMES):
            pyboy.tick()

    def _cursor_pair(self, pyboy: Any) -> Tuple[int, int]:
        return (_u8(pyboy, ADDR_MENU_CURSOR_LO), _u8(pyboy, ADDR_MENU_CURSOR_HI))

    def _move_cursor_to(self, pyboy: Any, target: int) -> bool:
        """wCurrentMenuItem 을 target 으로 옮긴다."""
        for _ in range(8):
            current = _u8(pyboy, ADDR_CURRENT_MENU_ITEM)
            if current == target:
                return self._cursor_pair(pyboy) == MOVE_LEARN_CURSOR_PAIRS[target]
            if not 0 <= current < 4:
                return False
            self._press(pyboy, "down" if current < target else "up")
            if _u8(pyboy, ADDR_CURRENT_MENU_ITEM) == current:
                return False        # 커서가 안 움직였다 = 메뉴가 아니다
        return False

    def resolve(self, pyboy: Any, info: Dict[str, Any]) -> bool:
        """교체 메뉴를 확정한다."""
        if not self.enabled:
            return False
        try:
            party_index = int(info["party_index"])
            new_move_id = int(info["new_move_id"]) & 0xFF
            moves = [int(m) & 0xFF for m in info["current_move_ids"]]
            selected = int(info["selected_slot"])
        except Exception:
            return False
        if len(moves) != 4 or not 0 <= selected < 4:
            return False

        # battle_macro 의 커서 기하가 우리 것과 같은지 확인한다.
        if tuple(info.get("cursor_pair") or ()) != MOVE_LEARN_CURSOR_PAIRS[selected]:
            print(
                "[move-learn] 커서 기하 불일치 - 수동 위임 "
                f"got={tuple(info.get('cursor_pair') or ())}"
            )
            return False

        target = self.choose_slot(pyboy, party_index, moves)
        if target is None:
            print("[move-learn] 지울 수 있는 슬롯이 없음(전부 HM) - 수동 위임")
            return False

        print(
            f"[move-learn] {self.policy}: party={party_index} "
            f"slot{target}(move={moves[target]}) -> {new_move_id}"
        )

        if not self._move_cursor_to(pyboy, target):
            print("[move-learn] 커서 이동 실패 - 수동 위임")
            return False

        self._press(pyboy, "a")

        # learn_move.asm 은 "Poof!
        waited = 0
        while waited < WRITE_TIMEOUT_FRAMES:
            if read_party_moves(pyboy, party_index)[target] == new_move_id:
                key = mon_key(pyboy, party_index)
                self.ledger.record(key, target, new_move_id, moves)
                if self.debug:
                    print(f"[move-learn] 기록 {key} slot{target} seq 갱신")
                return True
            self._press(pyboy, "a")
            waited += BUTTON_HOLD_FRAMES + BUTTON_GAP_FRAMES

        print("[move-learn] 기술이 써지지 않음(timeout) - 수동 위임")
        return False


# ---------------------------------------------------------------------------
# 배틀 밖에서도 쓰는 메뉴 감지
# ---------------------------------------------------------------------------
def move_learn_menu_open(pyboy: Any) -> Optional[Dict[str, Any]]:
    """레벨업 기술 교체 메뉴가 떠 있으면 handoff 와 같은 dict 를 돌려준다."""
    try:
        party_index = _u8(pyboy, ADDR_WHICH_POKEMON)
        party_count = _u8(pyboy, ADDR_PARTY_COUNT)
        if not 0 <= party_index < party_count:
            return None

        selected = _u8(pyboy, ADDR_CURRENT_MENU_ITEM)
        if not 0 <= selected < 4:
            return None

        if not (
            _u8(pyboy, ADDR_TOP_MENU_Y) == 8
            and _u8(pyboy, ADDR_TOP_MENU_X) == 5
            and _u8(pyboy, ADDR_MAX_MENU_ITEM) == 3
            and _u8(pyboy, ADDR_NUM_MOVES_MINUS_ONE) == 3
            and _u8(pyboy, ADDR_MENU_WATCHED_KEYS) == 0x03
        ):
            return None

        cursor_pair = (
            _u8(pyboy, ADDR_MENU_CURSOR_LO),
            _u8(pyboy, ADDR_MENU_CURSOR_HI),
        )
        if cursor_pair != MOVE_LEARN_CURSOR_PAIRS[selected]:
            return None

        new_move_id = _u8(pyboy, ADDR_MOVE_NUM)
        if new_move_id == 0:
            return None

        copied = [_u8(pyboy, ADDR_MOVES_BUFFER + i) for i in range(4)]
        moves = read_party_moves(pyboy, party_index)
        if (
            any(m == 0 for m in moves)
            or copied != moves
            or new_move_id in moves
        ):
            return None
    except Exception:
        return None

    return {
        "party_index": party_index,
        "new_move_id": new_move_id,
        "current_move_ids": tuple(moves),
        "selected_slot": selected,
        "cursor_pair": cursor_pair,
    }


# 세션 기본 리졸버 — DialogueSkipper 4곳에 생성자 인자를 뚫지 않기 위한 슬롯.
_DEFAULT_RESOLVER: Optional[MoveLearnResolver] = None


def set_default_resolver(resolver: Optional[MoveLearnResolver]) -> None:
    global _DEFAULT_RESOLVER
    _DEFAULT_RESOLVER = resolver


def default_resolver() -> Optional[MoveLearnResolver]:
    return _DEFAULT_RESOLVER


__all__ = [
    "MoveAgeLedger",
    "MoveLearnResolver",
    "POLICIES",
    "HM_MOVE_IDS",
    "MOVE_LEARN_CURSOR_PAIRS",
    "mon_key",
    "read_party_moves",
    "move_learn_menu_open",
    "set_default_resolver",
    "default_resolver",
]
