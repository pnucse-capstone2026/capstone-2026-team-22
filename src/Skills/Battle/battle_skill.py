"""play.py에서 재사용하는 Battle RL 런타임."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .battle_macro import BattleMacro, BattleManualHandoff
from .gb_memory import BattleRAMReader, DEFAULT_SYM_PATH


class BattleSkillError(RuntimeError):
    """Battle Skill 준비 또는 한 턴 실행 실패."""


def _manual_handoff_from(exc: BaseException) -> Optional[BattleManualHandoff]:
    """BattleAgentError에 감싸진 수동 위임 예외를 찾아낸다."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, BattleManualHandoff):
            return current
        current = current.__cause__ or current.__context__
    return None


class BattleSkill:
    """PPO 로드와 PyBoy battle macro를 한 배틀 단위로 관리한다."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        sym_path: str | Path = DEFAULT_SYM_PATH,
        device: str = "cpu",
        deterministic: bool = True,
        max_turns: int = 200,
        prepare_timeout_frames: int = 2400,
        debug: bool = False,
        macro_debug: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.sym_path = Path(sym_path)
        self.device = str(device)
        self.deterministic = bool(deterministic)
        self.max_turns = max(1, int(max_turns))
        self.prepare_timeout_frames = max(1, int(prepare_timeout_frames))
        self.debug = bool(debug)
        self.macro_debug = bool(macro_debug)

        self.pyboy = None
        self.battle_ram: BattleRAMReader | None = None
        self.battle_macro: BattleMacro | None = None
        self.agent = None
        self.active = False
        self.turns = 0
        self.reason = ""

    @property
    def loaded(self) -> bool:
        return self.agent is not None

    def bind_pyboy(self, pyboy) -> None:
        if self.pyboy is pyboy and self.battle_ram is not None:
            return
        self.pyboy = pyboy
        self.battle_ram = BattleRAMReader(pyboy, sym_path=self.sym_path)
        self.battle_macro = BattleMacro(
            pyboy,
            battle_ram=self.battle_ram,
            debug=self.macro_debug,
        )
        self.agent = None

    def ensure_loaded(self) -> None:
        if self.agent is not None:
            return
        if self.battle_ram is None or self.battle_macro is None:
            raise BattleSkillError("BattleSkill.bind_pyboy()가 먼저 필요합니다.")

        # Torch/SB3는 첫 배틀에서만 import/load한다.
        from .battle_agent import BattleAgent

        self.agent = BattleAgent(
            battle_ram=self.battle_ram,
            battle_macro=self.battle_macro,
            model_path=self.model_path,
            device=self.device,
            deterministic=self.deterministic,
            debug=self.debug,
        )

    def start(self) -> bool:
        if self.battle_ram is None or self.battle_macro is None:
            raise BattleSkillError("BattleSkill에 PyBoy가 연결되지 않았습니다.")
        if not self.battle_ram.is_in_battle():
            self.active = False
            self.reason = "not_in_battle"
            return False

        self.ensure_loaded()
        self.battle_ram.reset_battle()
        self.battle_macro.reset_battle()
        self.agent.reset_battle()
        self.turns = 0
        self.reason = ""

        ready = self._prepare_decision()
        if not ready:
            self.active = False
            self.reason = "battle_ended_during_prepare"
            return False

        self.active = True
        return True

    def step(self):
        if not self.active:
            return None
        assert self.battle_ram is not None
        assert self.agent is not None

        if not self.battle_ram.is_in_battle():
            self.active = False
            self.reason = "battle_end"
            return None
        if self.turns >= self.max_turns:
            self.active = False
            self.reason = f"max_turns:{self.max_turns}"
            raise BattleSkillError(f"배틀 최대 턴 {self.max_turns} 초과")

        state = self.battle_ram.read()
        try:
            result = self.agent.act(
                state,
                deterministic=self.deterministic,
            )
        except Exception as exc:
            handoff = _manual_handoff_from(exc)
            if handoff is not None:
                self.active = False
                self.reason = f"manual_handoff:{handoff.reason}"
                raise handoff
            self.active = False
            self.reason = f"error:{type(exc).__name__}"
            raise

        self.turns += 1
        print(
            f"[battle] TURN {self.turns} "
            f"action={result.decision.rl_action_name}"
        )

        if not result.battle_still_active:
            self.active = False
            self.reason = "battle_end"
        elif not result.next_decision_ready:
            self.active = False
            self.reason = "next_decision_not_ready"
            raise BattleSkillError(
                "배틀은 계속 중이지만 다음 PPO decision menu가 준비되지 않았습니다."
            )
        return result

    def stop(self, reason: str) -> None:
        self.active = False
        self.reason = str(reason)
        if self.battle_macro is not None:
            self.battle_macro.reset_battle()

    def _prepare_decision(self) -> bool:
        """배틀 어느 시점에서 시작해도 첫 PPO decision menu까지 진행한다."""
        assert self.pyboy is not None
        assert self.battle_ram is not None
        assert self.battle_macro is not None

        if not self.battle_ram.is_in_battle():
            return False

        seen_one = self.battle_macro.first_mon_value() == 1
        flag = self.battle_macro.first_mon_value()
        # MOVE/PARTY cursor bytes are reused during intro text, so they
        # must not skip the first-mon send-out wait.
        recognized_menu = self.battle_macro.menu_context() in {
            "INITIAL",
            "MAIN",
        }
        if flag == 0 and recognized_menu:
            seen_one = True

        if not (seen_one and flag == 0):
            b_pressed = False
            hold_left = 0
            release_left = 0
            try:
                for _ in range(self.prepare_timeout_frames):
                    if not self.battle_ram.is_in_battle():
                        return False
                    flag = self.battle_macro.first_mon_value()
                    if flag == 1:
                        seen_one = True
                    if seen_one and flag == 0:
                        if b_pressed:
                            self.pyboy.button_release("b")
                            b_pressed = False
                        return bool(
                            self.pyboy.tick()
                            and self.battle_macro.prepare_decision_menu(
                                timeout_frames=self.prepare_timeout_frames
                            )
                        )

                    if b_pressed:
                        hold_left -= 1
                        if hold_left <= 0:
                            self.pyboy.button_release("b")
                            b_pressed = False
                            release_left = 6
                    elif release_left > 0:
                        release_left -= 1
                    else:
                        self.pyboy.button_press("b")
                        b_pressed = True
                        hold_left = 4

                    if not self.pyboy.tick():
                        return False
            finally:
                if b_pressed:
                    self.pyboy.button_release("b")
            raise BattleSkillError("첫 포켓몬 등장 연출 준비 timeout")

        return bool(
            self.battle_macro.prepare_decision_menu(
                timeout_frames=self.prepare_timeout_frames
            )
        )


__all__ = ["BattleSkill", "BattleSkillError"]
