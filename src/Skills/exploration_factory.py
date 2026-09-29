"""settings/config.yaml 의 ``exploration.skill`` 로 Exploration 구현을 고른다."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.config import exploration_opt

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 설정에 적을 수 있는 값 → 정규화된 선택지
_ALIASES = {
    "1": "1", "exploration": "1", "legacy": "1", "screen": "1",
    "2": "2", "exploration2": "2", "tracking": "2",
    "coverage": "coverage", "cover": "coverage",
    "seekwarp": "seekwarp", "seek": "seekwarp", "warp": "seekwarp",
    "bfs": "bfs", "rule": "bfs", "frontier": "bfs",
}

_LABEL = {
    "1": "Skills/Exploration",
    "2": "Skills/Exploration2",
    "coverage": "Skills/Coverage",
    "seekwarp": "Skills/SeekWarp",
    "bfs": "Skills/BFS",
}
_MODEL_KEY = {"1": "model", "2": "model2", "coverage": "model_coverage", "seekwarp": "model_seekwarp", "bfs": "model_bfs"}
_FOLDER = {"1": "Exploration", "2": "Exploration2", "coverage": "Coverage", "seekwarp": "SeekWarp", "bfs": "BFS"}
# 모델 없이 규칙으로 도는 스킬. play.py 가 비전머신 능력(capabilities)을 넘겨 준다.
RULE_BASED = ("seekwarp", "bfs")


def normalize_choice(raw: Any) -> str:
    key = str(raw if raw is not None else 1).strip().lower()
    choice = _ALIASES.get(key)
    if choice is None:
        raise ValueError(
            f"exploration.skill 값이 이상합니다: {raw!r}\n"
            f"  쓸 수 있는 값: " + ", ".join(f"{k} (={v})" for k, v in _LABEL.items())
        )
    return choice


def exploration_skill_choice() -> str:
    """settings/config.yaml 의 ``exploration.skill`` 을 정규화한다."""
    return normalize_choice(exploration_opt("skill", 1))


def exploration_skill_label(choice: str | None = None) -> str:
    """로그용 한 줄 설명."""
    c = choice or exploration_skill_choice()
    return f"{c} — {_LABEL[c]}"


def configured_model_path(choice: str | None = None) -> Path | None:
    """선택한 스킬의 모델 설정을 실제 경로로 바꾼다."""
    selected = choice or exploration_skill_choice()
    raw_value = str(exploration_opt(_MODEL_KEY[selected], "") or "").strip()
    if not raw_value:
        return None
    raw = Path(raw_value).expanduser()
    if raw.is_absolute():
        return raw
    if len(raw.parts) == 1:
        return (_PROJECT_ROOT / "Skills" / _FOLDER[selected] / raw).resolve()
    return (_PROJECT_ROOT / raw).resolve()


def _import_skill_class(choice: str):
    if choice == "2":
        from Skills.Exploration2 import ExplorationSkill
    elif choice == "coverage":
        from Skills.Coverage import ExplorationSkill
    elif choice == "seekwarp":
        from Skills.SeekWarp import ExplorationSkill
    elif choice == "bfs":
        from Skills.BFS import ExplorationSkill
    else:
        from Skills.Exploration import ExplorationSkill
    return ExplorationSkill


def _int_list(raw: Any) -> list[int]:
    """설정값 → 정수 목록."""
    if raw is None or isinstance(raw, bool):
        return []
    if isinstance(raw, (int, float)):
        raw = [raw]
    try:
        return [int(v) for v in raw]
    except (TypeError, ValueError):
        return []


def make_exploration_skill(choice: str | None = None, **kwargs: Any) -> Any:
    """설정(또는 ``choice``)이 고른 구현을 모델·공통 추론 옵션으로 만든다."""
    selected = normalize_choice(choice) if choice is not None else exploration_skill_choice()
    cls = _import_skill_class(selected)

    options: dict[str, Any] = {
        "device": str(exploration_opt("device", "cpu") or "cpu"),
        "greedy": bool(exploration_opt("greedy", False)),
        "cheat_profile": str(exploration_opt("cheat_profile", "off") or "off"),
    }
    model_path = configured_model_path(selected)
    if model_path is not None:
        options["model_path"] = model_path
    if selected == "seekwarp":
        options["max_steps"] = int(exploration_opt("seekwarp_max_steps", 300))
    if selected == "bfs":
        options["max_steps"] = int(exploration_opt("max_steps", 900))
        options["cross_warps"] = bool(exploration_opt("bfs_cross_warps", True))
        options["travel"] = bool(exploration_opt("bfs_travel", True))
        options["fail_limit"] = int(exploration_opt("bfs_fail_limit", 2))
        options["travel_max_hops"] = int(exploration_opt("bfs_travel_max_hops", 2))
        options["home_map_only"] = bool(exploration_opt("bfs_home_map_only", False))
        options["free_roam_tilesets"] = _int_list(exploration_opt("bfs_free_roam_tilesets", [17]))
    # 직접 호출자가 준 값은 config 보다 우선한다.
    options.update(kwargs)
    return cls(**options)
