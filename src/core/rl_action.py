"""rl_action.py — 방향키 액션 하나를 PyBoy 에 실행하는 공용 경로 (학습·play 공유)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from pyboy.utils import WindowEvent


@dataclass(frozen=True)
class ActionSpec:
    name: str
    delta: Tuple[int, int]   # 이동에 성공했을 때의 (dx, dy)


# 이 순서가 정책 logits 의 순서다. 바꾸면 체크포인트가 무효가 된다.
ACTIONS: Tuple[ActionSpec, ...] = (
    ActionSpec("DOWN", (0, 1)),
    ActionSpec("LEFT", (-1, 0)),
    ActionSpec("RIGHT", (1, 0)),
    ActionSpec("UP", (0, -1)),
)
ACTION_NAMES: Tuple[str, ...] = tuple(a.name for a in ACTIONS)
N_ACTIONS = len(ACTIONS)
DIRECTION_DELTA: Dict[int, Tuple[int, int]] = {i: a.delta for i, a in enumerate(ACTIONS)}
ACTION_NAME_TO_INDEX: Dict[str, int] = {n: i for i, n in enumerate(ACTION_NAMES)}

_PRESS = {
    "DOWN": WindowEvent.PRESS_ARROW_DOWN,
    "LEFT": WindowEvent.PRESS_ARROW_LEFT,
    "RIGHT": WindowEvent.PRESS_ARROW_RIGHT,
    "UP": WindowEvent.PRESS_ARROW_UP,
}
_RELEASE = {
    "DOWN": WindowEvent.RELEASE_ARROW_DOWN,
    "LEFT": WindowEvent.RELEASE_ARROW_LEFT,
    "RIGHT": WindowEvent.RELEASE_ARROW_RIGHT,
    "UP": WindowEvent.RELEASE_ARROW_UP,
}
VALID_ACTIONS = [_PRESS[a.name] for a in ACTIONS]
VALID_RELEASE_ACTIONS = [_RELEASE[a.name] for a in ACTIONS]

# 한 액션이 차지하는 프레임 수. 캐릭터가 정확히 한 칸 움직이는 길이다.
DEFAULT_ACTION_FREQ = 24
RELEASE_DELAY_FRAMES = 8
# 대화 잠금(wJoyIgnore) 이 풀릴 때까지 A 를 몇 번까지 누를지
_JOY_MASH_LIMIT = 1000


def resolve_action_index(action: int | str) -> int:
    if isinstance(action, str):
        key = action.strip().upper()
        if key not in ACTION_NAME_TO_INDEX:
            raise ValueError(f"Unknown action {action!r}; expected one of {ACTION_NAMES}")
        return ACTION_NAME_TO_INDEX[key]
    idx = int(action)   # numpy.int64 등
    if not 0 <= idx < N_ACTIONS:
        raise ValueError(f"Action index out of range: {idx}")
    return idx


def _read_joy_ignore(pyboy: Any) -> int:
    try:
        return int(pyboy.memory[pyboy.symbol_lookup("wJoyIgnore")[1]])
    except Exception:
        return 0


def _tick(pyboy: Any, count: int, *, render: bool, realtime: bool) -> None:
    """realtime 이면 1프레임씩 돌려 emulation_speed 스로틀이 먹게 한다."""
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
    """한 스텝 실행."""
    idx = resolve_action_index(action)
    freq = max(1, int(action_freq))
    render = bool(realtime)

    if not disable_ai_actions:
        pyboy.send_input(VALID_ACTIONS[idx])
        pyboy.send_input(VALID_RELEASE_ACTIONS[idx], delay=int(release_delay))
    _tick(pyboy, freq - 1, render=render, realtime=realtime)

    # 대화/스크립트 잠금 중 A 연타 — 에이전트 액션이 아니다
    for _ in range(_JOY_MASH_LIMIT):
        if not _read_joy_ignore(pyboy):
            break
        pyboy.button("a", 8)
        _tick(pyboy, freq, render=render, realtime=realtime)

    pyboy.tick(1, render=True)
    return idx
