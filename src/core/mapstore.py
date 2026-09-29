"""runtime/current_map.json · current_npc.json — **디버깅용 덤프**."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT / "runtime"
MAP_FILE = RUNTIME_DIR / "current_map.json"
NPC_FILE = RUNTIME_DIR / "current_npc.json"
# 탐험 스킬이 "여기는 못 지나간다 / 이 목표는 접었다" 고 기억하는 것.
BLOCKED_FILE = RUNTIME_DIR / "current_blocked.json"
# 길을 막은 NPC (다가가면 대화가 뜨고 끝나면 밀려나는 것이 되풀이되는 경우).
BLOCKED_NPC_FILE = RUNTIME_DIR / "current_blocked_npc.json"
# 최근 계획과 결과.
PAST_PLAN_FILE = RUNTIME_DIR / "past_plan.json"


def _load(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save(path: Path, data: Dict[str, Any]) -> None:
    try:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass          # 기록 실패가 실행을 막지는 않는다


def load_maps() -> Dict[str, Dict[str, Any]]:
    """맵 이름 -> {map_id, cells, exits, pois}."""
    out: Dict[str, Dict[str, Any]] = {}
    for name, info in _load(MAP_FILE).items():
        if isinstance(info, dict):
            out[str(name)] = info
    return out


def load_npcs() -> Dict[str, List[str]]:
    """맵 이름 -> NPC 이름 목록."""
    out: Dict[str, List[str]] = {}
    for name, names in _load(NPC_FILE).items():
        if isinstance(names, list):
            out[str(name)] = [str(n) for n in names]
    return out


def load_blocked() -> Dict[str, Dict[str, Any]]:
    """맵 이름 -> {map_id, blocked, given_up}."""
    out: Dict[str, Dict[str, Any]] = {}
    for name, info in _load(BLOCKED_FILE).items():
        if isinstance(info, dict):
            out[str(name)] = info
    return out


def save_blocked(data: Dict[str, Dict[str, Any]]) -> None:
    """탐험 스킬의 막힌 칸 기억을 파일로 덮어쓴다 (디버깅용)."""
    _save(BLOCKED_FILE, dict(data or {}))


def load_past_plans() -> List[Dict[str, Any]]:
    """최근 계획 기록."""
    data = _load(PAST_PLAN_FILE)
    rows = data.get("계획") if isinstance(data, dict) else None
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []


def save_past_plans(rows: List[Dict[str, Any]]) -> None:
    """최근 계획 기록을 덮어쓴다."""
    _save(PAST_PLAN_FILE, {"계획": list(rows or [])})


def load_blocked_npcs() -> Dict[str, List[Dict[str, Any]]]:
    """맵 이름 -> 길을 막은 NPC 목록."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for name, rows in _load(BLOCKED_NPC_FILE).items():
        if isinstance(rows, list):
            out[str(name)] = [r for r in rows if isinstance(r, dict)]
    return out


def save_blocked_npcs(data: Dict[str, List[Dict[str, Any]]]) -> None:
    """길을 막은 NPC 기록을 파일로 덮어쓴다 (디버깅용)."""
    _save(BLOCKED_NPC_FILE, dict(data or {}))


def snapshot(gmap: Any) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]]]:
    """지금 GlobalMap 이 아는 것을 두 dict 로."""
    maps: Dict[str, Dict[str, Any]] = {}
    npcs: Dict[str, List[str]] = {}
    try:
        nodes = list(gmap.graph.nodes)
    except Exception:
        return maps, npcs

    for mid in nodes:
        mid = int(mid)
        name = gmap.map_name(mid)
        local = gmap.get_local_map(mid)
        exits = []
        try:
            exits = sorted(gmap.map_name(int(n)) for n in gmap.graph.successors(mid))
        except Exception:
            pass
        pois: List[str] = []
        seen: List[str] = []
        cells = 0
        if local is not None:
            cells = len(getattr(local, "grid", None) or {})
            pois = sorted((getattr(local, "name_index", None) or {}).keys())
            getter = getattr(local, "list_npc_seen", None) or getattr(
                local, "list_npc_names", None
            )
            if getter is not None:
                try:
                    seen = list(getter())
                except Exception:
                    seen = []
        maps[name] = {"map_id": mid, "cells": cells, "exits": exits, "pois": pois}
        npcs[name] = seen
    return maps, npcs


def dump(gmap: Any) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]]]:
    """지금 GlobalMap 상태를 두 파일에 덮어쓴다."""
    maps, npcs = snapshot(gmap)
    _save(MAP_FILE, maps)
    _save(NPC_FILE, npcs)
    return maps, npcs


def clear() -> None:
    """덤프를 비운다."""
    _save(MAP_FILE, {})
    _save(NPC_FILE, {})
    _save(BLOCKED_FILE, {})
    _save(BLOCKED_NPC_FILE, {})
    _save(PAST_PLAN_FILE, {"계획": []})


__all__ = [
    "BLOCKED_FILE",
    "BLOCKED_NPC_FILE",
    "PAST_PLAN_FILE",
    "MAP_FILE",
    "NPC_FILE",
    "clear",
    "dump",
    "load_blocked",
    "load_blocked_npcs",
    "load_past_plans",
    "load_maps",
    "load_npcs",
    "save_blocked",
    "save_blocked_npcs",
    "save_past_plans",
    "snapshot",
]
