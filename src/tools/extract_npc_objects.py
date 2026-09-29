"""pokered map objects → 정적 NPC JSON (이벤트 NPC 구분용 스케치)."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
POKERED = ROOT / "third_party" / "pokered-master"
OUT = ROOT / "data" / "npc_objects.json"
MAP_DATA = ROOT / "data" / "map_data.json"

OBJ_EVENT_RE = re.compile(
    r"^\s*object_event\s+"
    r"(-?\d+)\s*,\s*(-?\d+)\s*,\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)"
    r"(?:\s*,\s*(\w+)\s*(?:,\s*(\w+))?)?"
)
CONST_EXPORT_RE = re.compile(r"^\s*const_export\s+(\w+)")
WARPS_TO_RE = re.compile(r"^\s*def_warps_to\s+(\w+)", re.MULTILINE)
TOGGLE_FOR_RE = re.compile(r"^\s*toggle_consts_for\s+(\w+)")
TOGGLE_CONST_RE = re.compile(r"^\s*const\s+(TOGGLE_\w+)")
TOGGLE_MAP_RE = re.compile(r"^\s*toggleable_objects_for\s+(\w+)")
TOGGLE_STATE_RE = re.compile(r"^\s*toggle_object_state\s+(\w+)\s*,\s*(ON|OFF)")
EVENT_RE = re.compile(r"\b(EVENT_[A-Z0-9_]+)\b")
TOGGLE_USE_RE = re.compile(r"\b(TOGGLE_[A-Z0-9_]+)\b")


def parse_map_ids() -> Dict[str, int]:
    ids: Dict[str, int] = {}
    idx = 0
    pat = re.compile(r"^\s*map_const\s+(\w+),")
    for line in (POKERED / "constants" / "map_constants.asm").read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        m = pat.match(line)
        if m:
            ids[m.group(1)] = idx
            idx += 1
    return ids


def parse_display_names() -> Dict[int, str]:
    if not MAP_DATA.is_file():
        return {}
    data = json.loads(MAP_DATA.read_text(encoding="utf-8"))
    out: Dict[int, str] = {}
    for region in data.get("regions", []):
        try:
            mid = int(region["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if mid < 0:
            continue
        name = str(region.get("name", "")).strip()
        if name:
            out[mid] = name
    return out


def _classify_kind(
    sprite: str, extra1: Optional[str], extra2: Optional[str]
) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    """return (kind, trainer_info|None, item_id|None)."""
    if extra1 and extra2:
        return (
            "trainer",
            {"class_or_mon": extra1, "number_or_level": extra2},
            None,
        )
    if extra1 and not extra2:
        # item ball style: 7 args
        return "item", None, extra1
    if sprite in {"SPRITE_POKE_BALL", "SPRITE_FOSSIL"}:
        return "item", None, None
    return "npc", None, None


def parse_map_objects() -> Dict[str, Dict[str, Any]]:
    """MAP_CONST → {object_const → object dict, ordered list}."""
    result: Dict[str, Dict[str, Any]] = {}
    objects_dir = POKERED / "data" / "maps" / "objects"
    for asm in sorted(objects_dir.glob("*.asm")):
        text = asm.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()

        const_names: List[str] = []
        in_const = False
        for line in lines:
            if re.match(r"^\s*object_const_def\b", line):
                in_const = True
                const_names = []
                continue
            if in_const:
                m = CONST_EXPORT_RE.match(line)
                if m:
                    const_names.append(m.group(1))
                    continue
                if line.strip() and not line.strip().startswith(";"):
                    # 다음 섹션
                    if not line.strip().startswith("const"):
                        in_const = False

        warps = WARPS_TO_RE.search(text)
        if not warps:
            continue
        map_const = warps.group(1)

        objs: List[Dict[str, Any]] = []
        obj_i = 0
        for line in lines:
            m = OBJ_EVENT_RE.match(line)
            if not m:
                continue
            x, y = int(m.group(1)), int(m.group(2))
            sprite, movement, facing = m.group(3), m.group(4), m.group(5)
            text_id = m.group(6)
            extra1, extra2 = m.group(7), m.group(8)
            kind, trainer, item = _classify_kind(sprite, extra1, extra2)
            name = (
                const_names[obj_i]
                if obj_i < len(const_names)
                else f"OBJECT_{obj_i + 1}"
            )
            objs.append(
                {
                    "object_id": obj_i + 1,  # pokered object_const_def starts at 1
                    "const": name,
                    "x": x,
                    "y": y,
                    "sprite": sprite,
                    "movement": movement,
                    "facing_or_range": facing,
                    "text_id": text_id,
                    "kind": kind,
                    "trainer": trainer,
                    "item": item,
                }
            )
            obj_i += 1

        result[map_const] = {
            "source_file": asm.name,
            "objects": objs,
            "by_const": {o["const"]: o for o in objs},
        }
    return result


def parse_toggles() -> Tuple[
    Dict[str, Dict[str, Any]], Dict[str, str]
]:
    """Returns: toggle_by_object_const: OBJECT_CONST → {toggle_const, toggle_index, default_state…"""
    # 1) TOGGLE_* index order + map
    toggle_index: Dict[str, int] = {}
    toggle_map: Dict[str, str] = {}  # TOGGLE_* → MAP_CONST
    idx = 0
    current_map: Optional[str] = None
    for line in (POKERED / "constants" / "toggle_constants.asm").read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        m = TOGGLE_FOR_RE.match(line)
        if m:
            current_map = m.group(1)
            continue
        m = TOGGLE_CONST_RE.match(line)
        if m and current_map is not None:
            name = m.group(1)
            toggle_index[name] = idx
            toggle_map[name] = current_map
            idx += 1

    # 2) object_const ↔ default ON/OFF (order matches TOGGLE_* within map)
    by_object: Dict[str, Dict[str, Any]] = {}
    toggle_to_object: Dict[str, str] = {}
    current_map = None
    map_toggle_queue: List[str] = []

    # rebuild per-map ordered TOGGLE_* list
    toggles_in_map: Dict[str, List[str]] = defaultdict(list)
    for tname, mid in sorted(toggle_index.items(), key=lambda kv: kv[1]):
        toggles_in_map[toggle_map[tname]].append(tname)

    for line in (POKERED / "data" / "maps" / "toggleable_objects.asm").read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        m = TOGGLE_MAP_RE.match(line)
        if m:
            current_map = m.group(1)
            map_toggle_queue = list(toggles_in_map.get(current_map, []))
            continue
        m = TOGGLE_STATE_RE.match(line)
        if not m or current_map is None:
            continue
        obj_const, state = m.group(1), m.group(2)
        if not map_toggle_queue:
            continue
        tconst = map_toggle_queue.pop(0)
        info = {
            "toggle_const": tconst,
            "toggle_index": toggle_index.get(tconst),
            "default_state": state,
            "map_const": current_map,
        }
        by_object[obj_const] = info
        toggle_to_object[tconst] = obj_const

    return by_object, toggle_to_object


def parse_related_events(
    toggle_to_object: Dict[str, str],
) -> Dict[str, List[str]]:
    """TOGGLE_* → 근처 스크립트의 EVENT_* 목록 (휴리스틱)."""
    related: Dict[str, set] = defaultdict(set)
    scripts = POKERED / "scripts"
    window = 40
    for asm in sorted(scripts.glob("*.asm")):
        lines = asm.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, line in enumerate(lines):
            for tconst in TOGGLE_USE_RE.findall(line):
                if tconst not in toggle_to_object:
                    continue
                lo = max(0, i - window)
                hi = min(len(lines), i + window + 1)
                chunk = "\n".join(lines[lo:hi])
                for ev in EVENT_RE.findall(chunk):
                    related[tconst].add(ev)
    return {k: sorted(v) for k, v in related.items()}


def main() -> None:
    map_ids = parse_map_ids()
    display = parse_display_names()
    maps = parse_map_objects()
    toggle_by_obj, toggle_to_obj = parse_toggles()
    related = parse_related_events(toggle_to_obj)

    out_maps: Dict[str, Any] = {}
    n_objects = 0
    n_eventish = 0

    for map_const, payload in sorted(maps.items(), key=lambda kv: map_ids.get(kv[0], 10**9)):
        if map_const not in map_ids:
            continue
        mid = map_ids[map_const]
        objects_out: List[Dict[str, Any]] = []
        for obj in payload["objects"]:
            tinfo = toggle_by_obj.get(obj["const"])
            tconst = tinfo["toggle_const"] if tinfo else None
            rel = related.get(tconst, []) if tconst else []
            is_event_npc = bool(tinfo) or bool(rel)
            entry = {
                **obj,
                "is_toggleable": bool(tinfo),
                "is_event_npc": is_event_npc,
                "toggle": tinfo,
                "related_events": rel,
            }
            objects_out.append(entry)
            n_objects += 1
            if is_event_npc:
                n_eventish += 1

        out_maps[str(mid)] = {
            "const": map_const,
            "name": display.get(mid, map_const),
            "source_file": payload["source_file"],
            "objects": objects_out,
        }

    doc = {
        "schema": "npc_objects_v0",
        "note": (
            "정적 NPC/오브젝트 목록. play·Map 미연결. "
            "is_event_npc ≈ toggleable(스토리 show/hide) 또는 스크립트에서 "
            "TOGGLE과 같이 보이는 EVENT_* 가 있는 경우. 완전 정확하진 않음."
        ),
        "counts": {
            "maps": len(out_maps),
            "objects": n_objects,
            "event_npc_candidates": n_eventish,
        },
        "maps": out_maps,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"maps={len(out_maps)} objects={n_objects} "
        f"event_npc~={n_eventish} -> {OUT}"
    )


if __name__ == "__main__":
    main()
