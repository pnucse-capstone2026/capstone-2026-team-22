"""settings/config.yaml 로드."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
SETTINGS_DIR = ROOT / "settings"
CONFIG_PATH = SETTINGS_DIR / "config.yaml"
ENV_PATH = SETTINGS_DIR / ".env"

# 설정의 정본은 settings/config.yaml 이다.
_DEFAULTS: Dict[str, Any] = {
    "emulation_speed": 3,
    "pyboy": {
        "window": "SDL2",
        "scale": 3,
        "sound_emulated": False,
        "sound_volume": 0,
        "no_input": False,
        "log_level": "WARNING",
        "symbols": "",
        "title_status": False,
    },
    "cheats": {
        "disable_wild_encounters": False,
        "auto_win_battle": False,
        "noclip": False,
        "hm_cut": "story",
        "hm_surf": "badge",
        "hm_carrier": {
            "enabled": True,
            "species": "tentacool",
            "level": 20,
            "nickname": "HMMON",
        },
        "skip_mt_moon_fossils": True,
        "skip_vermilion_gym_puzzle": False,
        "skip_bill_cell_separator": False,
        "wake_snorlax": False,
        "wake_snorlax_adjacent_only": True,
        "snorlax_as_npc": True,
        "buy_saffron_drink": True,
        "skip_game_corner_poster": False,
        "ghost_marowak_needs_scope": True,
        "ghost_flee": True,
        "blocking_object_interact": True,
        "card_key_doors": True,
        "blocking_object_record": True,
        "bike_clerk_with_voucher": True,
        "party_mon": {
            "enabled": False,
            "species": "mewtwo",
            "level": 100,
            "nickname": "MEWTWO",
            "skip_if_have": False,
        },
    },
    "events": {
        "poll_every_frames": 60,
        "seed_already_set": True,
        "key_item_events": True,
        "status_flag_events": True,
    },
    "exploration": {
        "skill": 1,
        "auto_split": False,
        "max_steps": 900,
        "max_steps_manual": 0,
        "model": "",
        "model2": "",
        "model_coverage": "",
        "model_seekwarp": "",
        "seekwarp_max_steps": 300,
        "bfs_cross_warps": True,
        "bfs_travel": True,
        "bfs_home_map_only": True,
        "bfs_free_roam_tilesets": [17],
        "bfs_fail_limit": 2,
        "bfs_travel_max_hops": 2,
        "device": "cpu",
        "greedy": False,
        "cheat_profile": "off",
        "close_portals": True,
        "close_on_map_closed": True,
        "close_stop_explore": True,
        "close_check_every": 25,
        "close_min_warps": 1,
        "close_require_all_warps": True,
        "close_max_depth": 4,
        "close_no_recurse_marks": ["route"],
        "close_skip_tilesets": [],
        "close_npc_blocks": False,
        "close_offmap_frontier": True,
        "close_debug": False,
    },
    "battle": {
        "enabled": True,
        "model": "Skills/Battle/models/model.zip",
        "sym": "assets/roms/pokered.sym",
        "device": "cpu",
        "deterministic": True,
        "max_turns": 200,
        "prepare_timeout_frames": 2400,
        "on_failure": "amash",
        "on_move_learn": "oldest",
        "pp_stuck_limit": 100,
        "debug": False,
        "macro_debug": False,
    },
    "auto": {
        "max_cycles": 500,
        "max_steps": 0,
        "settle_frames": 60,
    },
    "recording": {
        "start_on_launch": False,
        "speed": 8,
        "fps": 60,
        "scale": 3,
        "overlay": True,
        "plan_hold_sec": 1.0,
        "out_dir": "recordings",
        "font": "",
    },
    "planner": {
        "backend": "claude",
        "blocked_cells": True,
        "open_gates_after_drink": True,
        "history_fail_reason": True,
        "claude_cli": "claude",
        "claude_model": "",
        "claude_effort": "",
        "timeout_sec": 120,
    },
}

_cache: Dict[str, Any] | None = None


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, val in overlay.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_config(*, reload: bool = False) -> Dict[str, Any]:
    global _cache
    if _cache is not None and not reload:
        return _cache

    cfg = deepcopy(_DEFAULTS)

    if CONFIG_PATH.is_file():
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML 필요: pip install pyyaml") from exc
        data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"config.yaml 최상위는 mapping 이어야 함: {CONFIG_PATH}")
        cfg = _deep_merge(cfg, data)

    _cache = cfg
    return cfg


def _section_opt(section: str, name: str, default: Any = None) -> Any:
    """settings/config.yaml 의 <section>.<name>."""
    return load_config().get(section, {}).get(name, default)


def pyboy_opt(name: str, default: Any = None) -> Any:
    """pyboy 섹션의 값."""
    return _section_opt("pyboy", name, default)


def cheat(name: str, default: bool = False) -> bool:
    return bool(_section_opt("cheats", name, default))


def cheat_opt(name: str, default: Any = None) -> Any:
    """cheats 섹션의 원값."""
    return _section_opt("cheats", name, default)


def event_opt(name: str, default: Any = None) -> Any:
    return _section_opt("events", name, default)


def planner_opt(name: str, default: Any = None) -> Any:
    return _section_opt("planner", name, default)


def auto_opt(name: str, default: Any = None) -> Any:
    return _section_opt("auto", name, default)


def recording_opt(name: str, default: Any = None) -> Any:
    return _section_opt("recording", name, default)


def exploration_opt(name: str, default: Any = None) -> Any:
    return _section_opt("exploration", name, default)


def battle_opt(name: str, default: Any = None) -> Any:
    return _section_opt("battle", name, default)
