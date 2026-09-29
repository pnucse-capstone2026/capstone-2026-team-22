"""파티 체력/PP 를 플래너 프롬프트용 한 줄씩으로 만든다."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import ROOT
from core.gb_memory import read_party

# wObtainedBadges — engine/hm.py 와 같은 주소를 본다(그쪽은 치트용, 여기는 읽기용).
ADDR_OBTAINED_BADGES = 0xD356

_DATA_DIR = ROOT / "Skills" / "Battle" / "data"

# 젠1 상태이상 비트 (constants/status_constants.asm)
_STATUS_BITS = (
    (0x07, "SLP"),   # bits 0-2 = 남은 턴 수
    (0x08, "PSN"),
    (0x10, "BRN"),
    (0x20, "FRZ"),
    (0x40, "PAR"),
)

_cache: Dict[str, Dict[str, Any]] = {}


def _load(name: str) -> Dict[str, Any]:
    if name in _cache:
        return _cache[name]
    data: Dict[str, Any] = {}
    try:
        raw = json.loads((_DATA_DIR / name).read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            data = raw
    except Exception:
        data = {}
    _cache[name] = data
    return data


def _species_name(species_id: int) -> str:
    meta = _load("species.json").get(str(int(species_id)))
    if isinstance(meta, dict) and meta.get("name"):
        return str(meta["name"])
    return f"MON#{int(species_id)}"


def _move_meta(move_id: int) -> Dict[str, Any]:
    meta = _load("moves.json").get(str(int(move_id)))
    return meta if isinstance(meta, dict) else {}


def _status_label(raw: int) -> str:
    raw = int(raw) & 0xFF
    if raw == 0:
        return ""
    for mask, name in _STATUS_BITS:
        if raw & mask:
            return name
    return "STATUS"


def format_party_lines(pyboy: Any) -> List[str]:
    """프롬프트에 그대로 넣을 한 줄짜리 문자열 리스트."""
    lines: List[str] = []
    for mon in read_party(pyboy):
        moves: List[str] = []
        for move_id, pp in zip(mon["move_ids"], mon["pp"]):
            if not move_id:
                continue
            meta = _move_meta(move_id)
            name = str(meta.get("name") or f"MOVE#{move_id}")
            max_pp = meta.get("pp")
            moves.append(f"{name} {pp}/{max_pp}" if max_pp else f"{name} {pp}")

        status = _status_label(mon["status"])
        head = (
            f"{mon['slot'] + 1} {_species_name(mon['species_id'])} "
            f"Lv{mon['level']} HP {mon['hp']}/{mon['max_hp']}"
        )
        if mon["hp"] <= 0:
            head += " (기절)"
        elif status:
            head += f" ({status})"
        lines.append(f"{head} · " + (", ".join(moves) or "기술 없음"))
    return lines


# 배지 이름 — 게임이 트레이너 카드에 보여 주는 그대로.
BADGE_NAMES = (
    "BOULDER", "CASCADE", "THUNDER", "RAINBOW",
    "SOUL", "MARSH", "VOLCANO", "EARTH",
)


def format_badges(pyboy: Any) -> List[str]:
    """가진 배지 이름 목록."""
    try:
        raw = int(pyboy.memory[ADDR_OBTAINED_BADGES])
    except Exception:
        return []
    return [name for i, name in enumerate(BADGE_NAMES) if raw & (1 << i)]


def badges_from_ref(ref: Optional[dict] = None) -> List[str]:
    if ref is None:
        from engine.state import pyboy_ref as ref  # 지연 import (순환 방지)
    pyboy = ref.get("pyboy")
    return format_badges(pyboy) if pyboy is not None else []


def party_lines_from_ref(ref: Optional[dict] = None) -> List[str]:
    """engine.state.pyboy_ref 처럼 핸들을 담은 dict 에서 바로 만든다."""
    if ref is None:
        from engine.state import pyboy_ref as ref  # 지연 import (순환 방지)
    pyboy = ref.get("pyboy")
    if pyboy is None:
        return []
    return format_party_lines(pyboy)


__all__ = [
    "BADGE_NAMES",
    "badges_from_ref",
    "format_badges",
    "format_party_lines",
    "party_lines_from_ref",
]
