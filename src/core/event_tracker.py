"""게임 wEventFlags 비트 → data/events.json 이름으로 감지."""

from __future__ import annotations

import re
from typing import Any, Callable, Collection, Dict, List, Optional, Tuple

from core.config import event_opt
from core.game_data import EventFlagMap, event_flag_catalog
from core.gb_memory import has_bag_item
from core.LLM import IGNORED_EVENT_NAMES, ONCE_EVENT_NAMES, reset_current_events, record_event

# events.json 자리표시자 이름 (LLM 노이즈) — 건너뜀
_PLACEHOLDER_RE = re.compile(
    r"^(?:[0-9A-Fa-f]{1,3}|Event\s+\d+)$",
    re.IGNORECASE,
)


# 일반 트레이너 승리. events.json 에서 "Trainer" 가 든 이름 317개는 전부 이 꼴이다.
_TRAINER_RE = re.compile(r"^Beat .* Trainer \d+$")

# 몬스터볼을 주워서 얻는 핵심 아이템 — 게임에 이벤트 플래그가 **없다** (pokered: PickUpItemText, 기록은 가방과 오브젝트 숨김 플래그뿐).
KEY_ITEM_EVENTS: Dict[int, str] = {
    0x4A: "Got Lift Key",        # 로켓단 아지트 B4F
    0x48: "Got Silph Scope",     # 로켓단 아지트 B4F (비주기를 이긴 뒤)
    0x30: "Got Card Key",        # 실프주식회사 5F
    0x40: "Got Gold Teeth",      # 사파리존
    0x2B: "Got Secret Key",      # 포켓몬 맨션 B1F
    # 자판기에서 **사는** 음료 (무지개 백화점 옥상) — 노랑시티 게이트 경비원에게 준다.
    0x3C: "Got Fresh Water",
}

# wEventFlags **밖**에 있는 스토리 비트.
STATUS_FLAG_EVENTS: Dict[Tuple[int, int], str] = {
    (0xD728, 6): "Gave Saffron Guards Drink",
}


def is_placeholder_event_name(name: str) -> bool:
    return bool(_PLACEHOLDER_RE.match(str(name).strip()))


def is_trainer_event_name(name: str) -> bool:
    """일반 트레이너 승리(`Beat … Trainer N`)."""
    return bool(_TRAINER_RE.match(str(name).strip()))


def is_ignored_event_name(name: str) -> bool:
    """current_events / auto replan 에 넣지 않을 이벤트."""
    text = str(name).strip()
    return (
        text in IGNORED_EVENT_NAMES
        or is_placeholder_event_name(text)
        or is_trainer_event_name(text)
    )


class EventWatcher:
    """PyBoy 메모리의 이벤트 플래그 상승 엣지를 추적한다."""

    def __init__(
        self,
        catalog: Optional[EventFlagMap] = None,
        *,
        cheat_bits: Optional[Callable[[], Collection[Tuple[int, int]]]] = None,
        key_items: Optional[bool] = None,
        status_flags: Optional[bool] = None,
    ) -> None:
        self.catalog: EventFlagMap = dict(catalog if catalog is not None else event_flag_catalog())
        # wEventFlags 밖의 스토리 비트도 같은 목록에 얹는다 (events.status_flag_events)
        if bool(event_opt("status_flag_events", True)) if status_flags is None else status_flags:
            self.catalog.update(STATUS_FLAG_EVENTS)
        # 치트가 켠 (addr, bit) 을 돌려주는 함수.
        self._cheat_bits = cheat_bits
        # 직전 poll 에서 **치트가 켠** 이벤트 이름.
        self.cheated: List[str] = []
        # (addr, bit) → 직전 프레임 값 0/1
        self._prev: Dict[Tuple[int, int], int] = {}
        # ONCE_EVENT_NAMES 중 이번 세션에 이미 알린 것
        self._once_seen: set = set()
        # 플래그 없는 핵심 아이템을 가방에서 볼지 (events.key_item_events)
        self._key_items: Dict[int, str] = (
            KEY_ITEM_EVENTS
            if (bool(event_opt("key_item_events", True)) if key_items is None else key_items)
            else {}
        )
        # KEY_ITEM_EVENTS 중 직전 poll 에 가방에 있던 아이템
        self._had_items: set = set()
        self._addrs: List[int] = sorted({addr for addr, _ in self.catalog})

    def reset_session(self, pyboy: Any, *, seed_already_set: bool = True) -> List[str]:
        """play.py 시작 시 호출."""
        reset_current_events()
        self._prev.clear()
        self._once_seen = set()

        seeded: List[str] = []
        for (addr, bit), name in self.catalog.items():
            on = self._read_bit(pyboy, addr, bit)
            self._prev[(addr, bit)] = on
            if seed_already_set and on and not is_ignored_event_name(name):
                record_event(name)
                seeded.append(name)
                if name in ONCE_EVENT_NAMES:
                    self._once_seen.add(name)
        self._had_items = set()
        for item_id, name in self._key_items.items():
            if has_bag_item(pyboy, item_id):
                self._had_items.add(item_id)
                if seed_already_set:
                    record_event(name)
                    seeded.append(name)
        return seeded

    def poll(self, pyboy: Any) -> Tuple[List[str], List[str]]:
        """한 프레임."""
        mem = pyboy.memory
        byte_cache: Dict[int, int] = {}
        for addr in self._addrs:
            try:
                byte_cache[addr] = int(mem[addr])
            except Exception:
                byte_cache[addr] = 0

        newly: List[str] = []
        skipped: List[str] = []
        cheated = set(self._cheat_bits() or ()) if self._cheat_bits is not None else set()
        self.cheated = []
        for (addr, bit), name in self.catalog.items():
            on = (byte_cache.get(addr, 0) >> bit) & 1
            prev = self._prev.get((addr, bit), 0)
            self._prev[(addr, bit)] = on
            if prev == 0 and on == 1:
                if is_placeholder_event_name(name) or name in IGNORED_EVENT_NAMES:
                    continue            # 없는 것으로 친다 — 기록도 재계획도 안 한다
                if is_trainer_event_name(name):
                    skipped.append(name)
                    continue
                if name in ONCE_EVENT_NAMES:
                    if name in self._once_seen:
                        continue        # 이미 한 번 받았다 — 다시 켜져도 새 일이 아니다
                    self._once_seen.add(name)
                if (addr, bit) in cheated:
                    skipped.append(f"{name} (치트)")
                    self.cheated.append(name)
                    continue
                record_event(name)
                newly.append(name)
        # 플래그 없는 핵심 아이템 — 가방에 새로 들어온 것만. 쓰고 없어져도 되돌리지 않는다.
        for item_id, name in self._key_items.items():
            if item_id not in self._had_items and has_bag_item(pyboy, item_id):
                self._had_items.add(item_id)
                record_event(name)
                newly.append(name)
        return newly, skipped

    def _read_bit(self, pyboy: Any, addr: int, bit: int) -> int:
        try:
            byte = int(pyboy.memory[int(addr)])
        except Exception:
            return 0
        return (byte >> int(bit)) & 1
