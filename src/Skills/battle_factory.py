"""settings/config.yaml의 battle 설정으로 BattleSkill을 만든다."""

from __future__ import annotations

from pathlib import Path

from core.config import ROOT, battle_opt


def _project_path(value: str, fallback: str) -> Path:
    raw = str(value or fallback).strip()
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def make_battle_skill(pyboy):
    """무거운 PPO import/load는 첫 배틀의 ``start``까지 미룬다."""
    from Skills.Battle import BattleSkill

    skill = BattleSkill(
        model_path=_project_path(
            battle_opt("model", ""),
            "Skills/Battle/models/model.zip",
        ),
        sym_path=_project_path(
            battle_opt("sym", ""),
            "assets/roms/pokered.sym",
        ),
        device=str(battle_opt("device", "cpu")),
        deterministic=bool(battle_opt("deterministic", True)),
        max_turns=int(battle_opt("max_turns", 200)),
        prepare_timeout_frames=int(
            battle_opt("prepare_timeout_frames", 2400)
        ),
        debug=bool(battle_opt("debug", False)),
        macro_debug=bool(battle_opt("macro_debug", False)),
    )
    skill.bind_pyboy(pyboy)
    return skill


def flee_battle(pyboy, battle_skill=None, *, tries: int = 6) -> bool:
    """전투 메뉴에서 RUN 을 골라 도망친다."""
    from Skills.Battle.battle_macro import BattleMacro, BattleManualHandoff

    macro = getattr(battle_skill, "battle_macro", None)
    if macro is None:
        from Skills.Battle.gb_memory import BattleRAMReader

        ram = BattleRAMReader(
            pyboy,
            sym_path=_project_path(battle_opt("sym", ""), "assets/roms/pokered.sym"),
        )
        macro = BattleMacro(pyboy, battle_ram=ram)
    macro.reset_battle()
    for _ in range(max(1, int(tries))):
        if macro.in_battle() is False:
            return True
        try:
            if not macro.prepare_decision_menu():
                continue
            if macro.run_away():
                return True
        except BattleManualHandoff:
            return False
        except Exception as exc:          # 매크로가 모르는 화면 — 사람에게 넘긴다
            print(f"[battle] 도망 중 오류: {type(exc).__name__}: {exc}")
            return False
    return macro.in_battle() is False


__all__ = ["flee_battle", "make_battle_skill"]
