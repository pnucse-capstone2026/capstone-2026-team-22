"""학습/play 공통 액션 실행기.

RedGymEnv.run_action_on_emulator 와 play explore 가 동일한
press → release@8 → tick(action_freq) → dialog mash 경로를 쓴다.

에이전트 액션은 방향키 4개만. 대화 잠금 해제용 A mash는 env 측 자동화다.
Press/release 순서는 actions.ACTIONS 이름에서만 파생한다.
"""

from __future__ import annotations

from typing import Any

from pyboy.utils import WindowEvent

from pokemonred_puffer.actions import ACTION_NAMES, ACTIONS

# Name → PyBoy events. Order always follows ACTIONS, never a parallel list.
_PRESS_BY_NAME: dict[str, WindowEvent] = {
    "DOWN": WindowEvent.PRESS_ARROW_DOWN,
    "LEFT": WindowEvent.PRESS_ARROW_LEFT,
    "RIGHT": WindowEvent.PRESS_ARROW_RIGHT,
    "UP": WindowEvent.PRESS_ARROW_UP,
}
_RELEASE_BY_NAME: dict[str, WindowEvent] = {
    "DOWN": WindowEvent.RELEASE_ARROW_DOWN,
    "LEFT": WindowEvent.RELEASE_ARROW_LEFT,
    "RIGHT": WindowEvent.RELEASE_ARROW_RIGHT,
    "UP": WindowEvent.RELEASE_ARROW_UP,
}

VALID_ACTIONS = [_PRESS_BY_NAME[spec.name] for spec in ACTIONS]
VALID_RELEASE_ACTIONS = [_RELEASE_BY_NAME[spec.name] for spec in ACTIONS]

ACTION_NAME_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(ACTION_NAMES)}

DEFAULT_ACTION_FREQ = 24
RELEASE_DELAY_FRAMES = 8


def resolve_action_index(action: int | str) -> int:
    if isinstance(action, str):
        key = str(action).strip().upper()
        if key not in ACTION_NAME_TO_INDEX:
            raise ValueError(f"Unknown action {action!r}; expected one of {ACTION_NAMES}")
        idx = ACTION_NAME_TO_INDEX[key]
    else:
        # numpy.int64 등은 isinstance(..., int) 가 False 일 수 있음
        idx = int(action)
    if not 0 <= idx < len(VALID_ACTIONS):
        raise ValueError(f"Action index out of range: {idx}")
    return idx


def _read_joy_ignore(pyboy: Any) -> int:
    try:
        return int(pyboy.memory[pyboy.symbol_lookup("wJoyIgnore")[1]])
    except Exception:
        return 0


def _tick_frames(pyboy: Any, count: int, *, render: bool, realtime: bool) -> None:
    """realtime 이면 1프레임씩 tick 해서 emulation_speed 스로틀이 먹게 한다."""
    n = max(0, int(count))
    if n <= 0:
        return
    if realtime:
        for _ in range(n):
            pyboy.tick(1, render=True)
        return
    pyboy.tick(n, render=render)


def execute_rl_action(
    pyboy: Any,
    action: int | str,
    *,
    action_freq: int = DEFAULT_ACTION_FREQ,
    disable_ai_actions: bool = False,
    release_delay: int = RELEASE_DELAY_FRAMES,
    realtime: bool = False,
) -> int:
    """학습과 동일한 한 스텝 입력 실행. 반환: action index.

    realtime=False (학습 기본): 중간 tick 은 render=False 배치, 마지막 1프레임만 True.
    realtime=True (play): 1프레임씩 render=True — 창 갱신 + emulation_speed 정상화.
    """
    idx = resolve_action_index(action)
    freq = max(1, int(action_freq))
    render = bool(realtime)

    if not disable_ai_actions:
        pyboy.send_input(VALID_ACTIONS[idx])
        pyboy.send_input(VALID_RELEASE_ACTIONS[idx], delay=int(release_delay))
    _tick_frames(pyboy, freq - 1, render=render, realtime=realtime)

    # 대화/스크립트 잠금 중 A mash (에이전트 액션이 아님)
    for _ in range(1000):
        if not _read_joy_ignore(pyboy):
            break
        pyboy.button("a", 8)
        _tick_frames(pyboy, freq, render=render, realtime=realtime)

    pyboy.tick(1, render=True)
    return idx
