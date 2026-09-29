"""play.py / test_map.py 공용 세션 코드."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

from core.config import load_config, pyboy_opt
from core.gb_memory import (
    ADDR_IN_BATTLE,
    ADDR_MAP_ID,
    ADDR_PLAYER_X,
    ADDR_PLAYER_Y,
    ADDR_SCX,
    ADDR_SCY,
    read_screen_tiles,
)
from core.portal import GlobalMap

from engine import explore_gate
from engine.state import pyboy_ref

from engine.debug_view import (
    WAIT_FRAMES,
    print_block_matrix,
    print_cell_name_matrix,
    print_local_grid_coords,
    print_tile_id_matrix,
    view_flags,
)
from engine.map_view import (
    format_full_local_map_text,
    print_full_local_map,
    print_scene_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
ROM_PATH = ROOT / "assets" / "roms" / "pokered.gb"
STATE_PATH = ROOT / "assets" / "states" / "init.state"
# play.py 종료 시점 스냅샷 (Ctrl+C 포함). 다음 세션 시작 로드 경로와는 별개.
PLAY_STATE_DIR = ROOT / "play_state"
PLAY_STATE_PATH = PLAY_STATE_DIR / "init.state"
MAP_DUMP_DIR = ROOT / "runtime" / "map_dump"

_SEP = "=" * 60


# ---------------------------------------------------------------------------
# 부팅
# ---------------------------------------------------------------------------
_PYBOY_WINDOWS = ("SDL2", "OpenGL", "GLFW", "null")
_PYBOY_LOG_LEVELS = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")


def _pyboy_kwargs() -> dict:
    """settings/config.yaml 의 ``pyboy:`` 를 PyBoy 생성자 kwargs 로."""
    # yaml 의 `window: null` / `~` 은 파서가 None 으로 준다.
    raw_window = pyboy_opt("window", "SDL2")
    window = "null" if raw_window is None else str(raw_window).strip()
    if window.lower() in ("none", "null", "headless"):
        window = "null"
    elif not window:
        # 빈 문자열은 "비워 뒀다"로 본다.
        window = "SDL2"
    if window not in _PYBOY_WINDOWS:
        print(
            f"[pyboy] window 값이 이상합니다: {window!r} — SDL2 로 돌립니다 "
            f"(쓸 수 있는 값: {', '.join(_PYBOY_WINDOWS)})"
        )
        window = "SDL2"

    level = str(pyboy_opt("log_level", "WARNING") or "WARNING").strip().upper()
    if level not in _PYBOY_LOG_LEVELS:
        print(f"[pyboy] log_level 값이 이상합니다: {level!r} — WARNING 으로 돌립니다")
        level = "WARNING"

    kwargs: dict = {
        "window": window,
        "sound_emulated": bool(pyboy_opt("sound_emulated", False)),
        "no_input": bool(pyboy_opt("no_input", False)),
        "log_level": level,
        "title_status": bool(pyboy_opt("title_status", False)),
    }
    # scale 은 창이 있을 때만 의미가 있다.
    if window != "null":
        try:
            kwargs["scale"] = max(1, int(pyboy_opt("scale", 3)))
        except (TypeError, ValueError):
            kwargs["scale"] = 3
    if kwargs["sound_emulated"]:
        try:
            kwargs["sound_volume"] = min(100, max(0, int(pyboy_opt("sound_volume", 0))))
        except (TypeError, ValueError):
            kwargs["sound_volume"] = 0

    raw_sym = str(pyboy_opt("symbols", "") or "").strip()
    if raw_sym:
        sym = Path(raw_sym)
        if not sym.is_absolute():
            sym = ROOT / sym
        if sym.is_file():
            kwargs["symbols"] = str(sym)
        else:
            print(f"[pyboy] symbols 파일이 없습니다: {sym} — 넘기지 않습니다")
    return kwargs


def open_emulator(*, emulation_speed: Optional[int] = None) -> Any:
    """ROM/PyBoy 확인 → PyBoy 생성 → init.state 로드까지."""
    if emulation_speed is None:
        emulation_speed = load_config().get("emulation_speed", 3)
    if not ROM_PATH.exists():
        print(f"ROM이 없습니다: {ROM_PATH}")
        sys.exit(1)

    try:
        from pyboy import PyBoy
    except ModuleNotFoundError:
        print("pyboy 미설치. pip install pyboy")
        sys.exit(1)

    kwargs = _pyboy_kwargs()
    pyboy = PyBoy(str(ROM_PATH), **kwargs)
    pyboy.set_emulation_speed(int(emulation_speed))
    if kwargs.get("window") == "null":
        print("[pyboy] 헤드리스(window: null) — 창이 없어 직접 조작할 수 없습니다")

    if STATE_PATH.exists():
        with open(STATE_PATH, "rb") as f:
            pyboy.load_state(f)
        print(f"상태 로드: {STATE_PATH.name}")

    return pyboy


def save_play_state(pyboy: Any) -> Optional[Path]:
    """현재 에뮬레이터 상태를 play_state/init.state 에 덮어쓴다."""
    if pyboy is None:
        return None
    try:
        PLAY_STATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(PLAY_STATE_PATH, "wb") as f:
            pyboy.save_state(f)
    except Exception as exc:
        print(f"[state] 저장 실패: {exc}")
        return None
    print(f"[state] 저장: {PLAY_STATE_PATH}")
    return PLAY_STATE_PATH


def read_player(pyboy: Any) -> tuple[int, int, int]:
    """(map_id, x, y) 를 한 번에 읽는다."""
    return (
        int(pyboy.memory[ADDR_MAP_ID]),
        int(pyboy.memory[ADDR_PLAYER_X]),
        int(pyboy.memory[ADDR_PLAYER_Y]),
    )


# ---------------------------------------------------------------------------
# LocalMap 출력 / local 계열 명령
# ---------------------------------------------------------------------------
def print_local_map_block(
    gmap: GlobalMap,
    map_id: int,
    x: int,
    y: int,
    *,
    chunk_rows: Optional[int] = None,
) -> None:
    """구분선으로 감싼 LocalMap 전체 출력."""
    print("\n" + _SEP)
    print_full_local_map(gmap, map_id, x, y, chunk_rows=chunk_rows)
    print(_SEP + "\n")


def _print_closure(
    gmap: GlobalMap,
    map_id: int,
    x: int,
    y: int,
    *,
    before: bool = False,
) -> None:
    """Exploration 최종 종료 조건(폐영역 + 포탈 너머 구역) 요약을 지도 아래에 붙인다."""
    pyboy = pyboy_ref.get("pyboy")
    if pyboy is None:
        return
    try:
        lines = explore_gate.describe(gmap, pyboy, int(map_id), (int(x), int(y)))
    except Exception as exc:
        if not before:
            print(f"[closure] 판정 실패: {type(exc).__name__}: {exc}")
        return
    if before:
        return
    for line in lines:
        print(line)
    print()


def handle_local_action(
    action: str,
    gmap: GlobalMap,
    map_id: int,
    x: int,
    y: int,
) -> bool:
    """local / local split / local file / local coords 처리."""
    if action == "local":
        _print_closure(gmap, map_id, x, y, before=True)
        print_local_map_block(gmap, map_id, x, y)
        _print_closure(gmap, map_id, x, y)
    elif action == "local_split":
        _print_closure(gmap, map_id, x, y, before=True)
        print_local_map_block(gmap, map_id, x, y, chunk_rows=10)
        _print_closure(gmap, map_id, x, y)
    elif action == "local_file":
        MAP_DUMP_DIR.mkdir(parents=True, exist_ok=True)
        dump_path = MAP_DUMP_DIR / f"map_{map_id}.txt"
        dump_path.write_text(
            format_full_local_map_text(gmap, map_id, x, y),
            encoding="utf-8",
        )
        print(f"\n[local file] {dump_path}\n")
    elif action == "local_coords":
        print("\n" + _SEP)
        print_local_grid_coords(gmap, map_id, x, y)
        print(_SEP + "\n")
    else:
        return False
    return True


# ---------------------------------------------------------------------------
# view 출력 타이밍
# ---------------------------------------------------------------------------
class ViewGate:
    """`view ..."""

    def __init__(self, pyboy: Any) -> None:
        self.last_map, self.last_x, self.last_y = read_player(pyboy)
        self.waiting = False
        self.counter = 0

    def ready(self, pyboy: Any, map_id: int, x: int, y: int) -> bool:
        if not any(view_flags.values()):
            return False

        moved = (map_id, x, y) != (self.last_map, self.last_x, self.last_y)
        if not self.waiting:
            if not moved:
                return False
            self.last_map, self.last_x, self.last_y = map_id, x, y
            self.waiting = True
            self.counter = WAIT_FRAMES * 3
            return False

        self.counter -= 1
        scx = int(pyboy.memory[ADDR_SCX])
        scy = int(pyboy.memory[ADDR_SCY])
        aligned = scx % 8 == 0 and scy % 8 == 0
        if not aligned and self.counter > 0:
            return False
        self.waiting = False

        return int(pyboy.memory[ADDR_IN_BATTLE]) == 0


def render_views(gmap: GlobalMap, pyboy: Any, map_id: int, x: int, y: int) -> None:
    """켜져 있는 view 플래그만 출력."""
    if view_flags["scene"]:
        print_scene_matrix(gmap, map_id, x, y)
    if view_flags["block"]:
        print_block_matrix(gmap, map_id, x, y)
    if view_flags["name"]:
        print_cell_name_matrix(gmap, map_id, x, y)
    if view_flags["tiles"]:
        print_tile_id_matrix(read_screen_tiles(pyboy), x, y)
