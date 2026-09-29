"""스킬 진행 중 끼어든 텍스트박스를 A 연타로 넘긴다."""

from __future__ import annotations

from typing import Any, Callable, Optional

from core import runlog
from core.gb_memory import (
    ADDR_MAP_ID,
    ADDR_PLAYER_X,
    ADDR_PLAYER_Y,
    is_dialogue_active,
    is_in_battle,
    is_textbox_open,
)
from engine.move_learn import default_resolver, move_learn_menu_open

# 배틀 A 연타와 같은 주기 (누르는 프레임 / 떼는 프레임)
DIALOGUE_A_HOLD = 6
DIALOGUE_A_GAP = 10
# 텍스트박스가 꺼진 뒤 이만큼 조용해야 대화가 끝난 것으로 본다.
DIALOGUE_CLEAR_FRAMES = 20
# 안전장치 — 이만큼 눌러도 안 끝나면 포기하고 스킬을 계속 굴린다.
DIALOGUE_MAX_FRAMES = 1800
# A 로 안 끝나면 B 로 빠져나온다 — 자판기/구매 메뉴처럼 A 가 오히려 메뉴를 파고드는 화면이 있다.
DIALOGUE_ESCAPE_TAPS = 4
# wJoyIgnore 만 켜진 상태(텍스트박스 없음)는 워프·스크립트 입력 잠금이다.
DIALOGUE_JOY_GRACE_FRAMES = 60


class DialogueSkipper:
    """텍스트박스를 A로 넘기는 작은 상태 기계."""

    def __init__(
        self,
        pyboy: Any,
        *,
        tag: str = "skill",
        on_start: Optional[Callable[[], None]] = None,
        is_active: Optional[Callable[[], bool]] = None,
        what: str = "대화",
        move_learn: Any = None,
        verbose: bool = True,
    ) -> None:
        self.pyboy = pyboy
        self.tag = tag                # 로그 접두사 ("nav" / "explore")
        # 대화 감지를 찍을지.
        self.verbose = bool(verbose)
        self.on_start = on_start      # 대화 시작 시 호출자가 버튼/큐를 정리하도록
        # 무엇을 넘기는가.
        self._is_active = is_active
        self.what = what
        # 레벨업 기술 교체 화면에서 A 를 누르지 않기 위한 정책.
        self._move_learn = move_learn
        self._move_learn_warned = False
        self.active = False
        self._a_t = 0
        self._btn: Optional[str] = None
        self._clear = 0
        self._frames = 0
        self._finished = False
        self._gave_up = False
        self._joy_frames = 0
        self._escaping = False
        self._escape_taps = 0
        self._escape_t = 0

    # -- 조회 ---------------------------------------------------------------
    def _detect(self, *, advance: bool) -> bool:
        """지금 A 로 넘겨야 하는 상태인가."""
        if self._is_active is not None:
            return bool(self._is_active())
        if is_textbox_open(self.pyboy):
            if advance:
                self._joy_frames = 0
            return True
        # 텍스트박스는 없는데 입력만 잠긴 상태 = 워프/스크립트 이동.
        if not advance:
            return self._joy_frames >= DIALOGUE_JOY_GRACE_FRAMES
        if is_dialogue_active(self.pyboy):
            self._joy_frames += 1
            return self._joy_frames >= DIALOGUE_JOY_GRACE_FRAMES
        self._joy_frames = 0
        return False

    def pending(self) -> bool:
        """상태를 진행시키지 않고 '지금 처리해야 하는가'만 본다."""
        if self.active:
            return True
        if self._gave_up:
            return False
        return self._detect(advance=False)

    def take_finished(self) -> bool:
        """대화가 방금 끝났으면 True 를 한 번만 돌려준다."""
        if self._finished:
            self._finished = False
            return True
        return False

    # -- 진행 ---------------------------------------------------------------
    def busy(self) -> bool:
        """한 프레임 진행."""
        if self._detect(advance=True):
            if self._gave_up:
                return False
            if not self.active:
                self._begin()
            self._clear = 0
            self._frames += 1
            if self._frames > DIALOGUE_MAX_FRAMES:
                if self._try_escape():
                    return True
                print(
                    f"[{self.tag}] {self.what}가 {DIALOGUE_MAX_FRAMES}프레임 넘게 "
                    "안 끝남 — A 연타 포기, 스킬 계속"
                )
                self._gave_up = True
                self.active = False
                self._escaping = False
                self.release()
                return False
            self._mash()
            return True

        if self._gave_up:
            # 포기했던 것이 끝났다 — 다음에는 다시 반응한다
            self._gave_up = False
            self._escaping = False
            self._escape_taps = 0
            self._escape_t = 0
            return False
        if not self.active:
            return False

        # 텍스트박스가 꺼졌다.
        self.release()
        self._clear += 1
        self._frames += 1
        if self._clear < DIALOGUE_CLEAR_FRAMES:
            return True
        self._end()
        return False

    # -- 정리 ---------------------------------------------------------------
    def release(self) -> None:
        """누르고 있던 A 를 뗀다."""
        if self._btn is not None:
            try:
                self.pyboy.button_release(self._btn)
            except Exception:
                pass
            self._btn = None

    def reset(self) -> None:
        """스킬 재시작 시 상태 초기화."""
        self.release()
        self.active = False
        self._a_t = 0
        self._clear = 0
        self._frames = 0
        self._finished = False
        self._gave_up = False
        self._joy_frames = 0
        self._escaping = False
        self._escape_taps = 0
        self._escape_t = 0

    # -- 내부 ---------------------------------------------------------------
    def _begin(self) -> None:
        self.active = True
        self._a_t = 0
        self._clear = 0
        self._frames = 0
        self._escaping = False
        self._escape_taps = 0
        self._escape_t = 0
        self.release()
        if self.verbose:
            print(f"[{self.tag}] {self.what} 감지 — A 로 넘기는 중 (끝나면 재개)")
        if self.on_start is not None:
            self.on_start()

    def _end(self) -> None:
        self.active = False
        self.release()
        self._joy_frames = 0
        if self._escaping:
            print(f"[{self.tag}] B 로 {self.what} 탈출 성공 ({self._escape_taps + 1}회)")
            self._log_escape("탈출", self._escape_taps + 1)
        self._escaping = False
        self._escape_taps = 0
        self._escape_t = 0
        self._finished = True

    def _guard_move_learn(self) -> bool:
        """레벨업 기술 교체 메뉴에서는 A 를 누르지 않는다."""
        info = move_learn_menu_open(self.pyboy)
        if info is None:
            self._move_learn_warned = False
            return False

        self.release()
        resolver = self._move_learn or default_resolver()
        if resolver is not None and getattr(resolver, "enabled", False):
            try:
                resolver.resolve(self.pyboy, info)
            except Exception as exc:
                print(
                    f"[{self.tag}] 기술 교체 처리 실패: "
                    f"{type(exc).__name__}: {exc}"
                )
            return True

        if not self._move_learn_warned:
            self._move_learn_warned = True
            print(
                f"[{self.tag}] 레벨업 기술 교체 화면 — A 를 멈춥니다. "
                "지울 기술을 직접 고르세요"
            )
        return True

    def _where(self) -> str:
        """탈출이 어디서 일어났는지 — 빈도와 위치를 모르니 관측이 먼저다."""
        try:
            m = int(self.pyboy.memory[ADDR_MAP_ID])
            x = int(self.pyboy.memory[ADDR_PLAYER_X])
            y = int(self.pyboy.memory[ADDR_PLAYER_Y])
        except Exception:
            return "?"
        return f"map {m} ({x},{y})"

    def _log_escape(self, result: str, taps: int) -> None:
        try:
            runlog.log(
                "menu_escape",
                태그=self.tag,
                위치=self._where(),
                결과=result,
                B누름=taps,
            )
        except Exception:
            pass

    def _try_escape(self) -> bool:
        """A 로 안 끝나는 메뉴를 B 로 닫는다."""
        if self._guard_move_learn():
            # 기술 교체 화면은 A 도 B 도 누르면 안 된다. 리졸버에 맡긴다.
            return True

        if is_in_battle(self.pyboy):
            # 전투 중 B 는 명령 취소이고, 진화 연출도 B 로 끊긴다.
            return False

        if self._escape_taps >= DIALOGUE_ESCAPE_TAPS:
            self._log_escape("실패", self._escape_taps)
            return False

        if not self._escaping:
            self._escaping = True
            self._escape_t = 0
            self.release()
            print(
                f"[{self.tag}] {self.what}가 A 로 안 끝남 — "
                f"B 로 탈출 시도 (최대 {DIALOGUE_ESCAPE_TAPS}회, {self._where()})"
            )

        self._escape_t += 1
        if self._escape_t == 1:
            try:
                self.pyboy.button_press("b")
                self._btn = "b"
            except Exception:
                pass
        elif self._escape_t == DIALOGUE_A_HOLD:
            self.release()
        elif self._escape_t >= DIALOGUE_A_HOLD + DIALOGUE_A_GAP:
            self._escape_t = 0
            self._escape_taps += 1
        return True

    def _mash(self) -> None:
        if self._guard_move_learn():
            self._a_t = 0
            return
        self._a_t += 1
        if self._a_t == 1:
            try:
                self.pyboy.button_press("a")
                self._btn = "a"
            except Exception:
                pass
        elif self._a_t == DIALOGUE_A_HOLD:
            self.release()
        elif self._a_t >= DIALOGUE_A_HOLD + DIALOGUE_A_GAP:
            self._a_t = 0
