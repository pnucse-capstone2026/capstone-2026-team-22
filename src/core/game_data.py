"""game_data.py — 프로젝트 JSON 데이터 로더."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

from .map_types import EntityType

Coord = Tuple[int, int]
# (addr, bit) → display name
EventFlagMap = Dict[Tuple[int, int], str]

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

_EVENT_KEY_RE = re.compile(r"^(0x[0-9A-Fa-f]+)-([0-7])$")


def data_path(filename: str) -> Optional[str]:
    """data/ 아래 데이터 파일 경로."""
    path = DATA_DIR / filename
    return str(path) if path.exists() else None


def load_map_names(path: str) -> Dict[int, str]:
    """map_data.json → {map_id: 이름}."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    names: Dict[int, str] = {}
    for region in data.get("regions", []):
        try:
            mid = int(region["id"])
        except (KeyError, ValueError):
            continue
        if mid >= 0:
            names[mid] = str(region.get("name", f"MAP_{mid}"))
    return names


def load_hidden_objects(path: str) -> Dict[int, Dict[Coord, EntityType]]:
    """hidden_objects.json → {map_id: {(x, y): EntityType}}."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result: Dict[int, Dict[Coord, EntityType]] = {}
    for mid_str, entry in data.items():
        result[int(mid_str)] = {
            (int(x), int(y)): (
                EntityType.NPC if kind == "npc" else EntityType.OBJECT
            )
            for x, y, kind, _func in entry["events"]
        }
    return result


def load_hidden_object_names(path: str) -> Dict[int, Dict[Coord, str]]:
    """hidden_objects.json → {map_id: {(x, y): 함수/라벨 이름}}."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result: Dict[int, Dict[Coord, str]] = {}
    for mid_str, entry in data.items():
        labels: Dict[Coord, str] = {}
        for x, y, _kind, func in entry["events"]:
            name = str(func).strip()
            if name:
                labels[(int(x), int(y))] = name
        if labels:
            result[int(mid_str)] = labels
    return result


def load_npc_object_names(path: str) -> Dict[int, Dict[Coord, str]]:
    """npc_objects.json → {map_id: {(x, y): object const 이름}}."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    maps = data.get("maps", data)
    result: Dict[int, Dict[Coord, str]] = {}
    if not isinstance(maps, dict):
        return result

    for mid_str, entry in maps.items():
        if not isinstance(entry, dict):
            continue
        try:
            mid = int(mid_str)
        except (TypeError, ValueError):
            continue
        labels: Dict[Coord, str] = {}
        for obj in entry.get("objects", []):
            if not isinstance(obj, dict):
                continue
            # item 볼 등은 entity=OBJECT 쪽이라 스킵해도 되지만, 좌표에 사람이 있으면 이름으로 쓸 수 있게 npc/trainer만.
            kind = str(obj.get("kind") or "npc").lower()
            if kind not in ("npc", "trainer"):
                continue
            name = str(obj.get("const") or "").strip()
            if not name:
                continue
            try:
                x, y = int(obj["x"]), int(obj["y"])
            except (KeyError, TypeError, ValueError):
                continue
            labels[(x, y)] = name
        if labels:
            result[mid] = labels
    return result


def load_map_consts(path: str) -> Dict[int, str]:
    """warps.json → {map_id: const} (예: 59 → "MT_MOON_1F")."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    consts: Dict[int, str] = {}
    for mid_str, entry in data.get("maps", {}).items():
        try:
            mid = int(mid_str)
        except ValueError:
            continue
        const = entry.get("const")
        if const:
            consts[mid] = str(const)
    return consts


def load_warps(
    path: str,
) -> Tuple[Dict[int, Dict[Coord, dict]], Dict[int, Coord]]:
    """warps.json → ({map_id: {(x, y): point}}, {map_id: (width, height)})."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    warp_at: Dict[int, Dict[Coord, dict]] = {}
    map_size_cells: Dict[int, Coord] = {}

    for mid_str, entry in data.get("maps", {}).items():
        try:
            mid = int(mid_str)
        except ValueError:
            continue

        width = entry.get("width_cells")
        height = entry.get("height_cells")
        if width is not None and height is not None:
            map_size_cells[mid] = (int(width), int(height))

        points = {
            (int(p["x"]), int(p["y"])): dict(p)
            for p in entry.get("points", [])
        }
        if points:
            warp_at[mid] = points

    return warp_at, map_size_cells


def load_event_flags(path: Optional[str] = None) -> EventFlagMap:
    """events.json → {(addr, bit): name}."""
    resolved = path or data_path("events.json")
    if not resolved:
        return {}
    with open(resolved, "r", encoding="utf-8") as f:
        raw = json.load(f)

    out: EventFlagMap = {}
    if not isinstance(raw, dict):
        return out
    for key, name in raw.items():
        m = _EVENT_KEY_RE.match(str(key).strip())
        if not m:
            continue
        addr = int(m.group(1), 16)
        bit = int(m.group(2))
        out[(addr, bit)] = str(name)
    return out


@lru_cache(maxsize=1)
def event_flag_catalog() -> EventFlagMap:
    """캐시된 data/events.json 카탈로그."""
    return load_event_flags()
