"""Convert a Game Boy battery save (.sav) into a PyBoy snapshot (.state)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Map.py / main.py 와 동일한 주소
ADDR_MAP_ID = 0xD35E
ADDR_PLAYER_X = 0xD362
ADDR_PLAYER_Y = 0xD361
ADDR_PARTY_COUNT = 0xD163  # wPartyCount — 0이면 아직 타이틀/메뉴

# MBC3+RAM+BATT (Pokemon Red) — 4 banks x 8 KiB
EXPECTED_SAV_SIZE = 32 * 1024


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert .sav to PyBoy .state")
    parser.add_argument(
        "sav",
        type=Path,
        help="Input battery save (.sav)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .state path (default: <sav_stem>.state next to input)",
    )
    parser.add_argument(
        "--rom",
        type=Path,
        default=ROOT / "assets" / "roms" / "pokered.gb",
        help="Matching ROM file (default: assets/roms/pokered.gb)",
    )
    parser.add_argument(
        "--window",
        default="null",
        choices=("null", "SDL2", "OpenGL", "GLFW"),
        help="PyBoy window type (default: null — no GUI)",
    )
    parser.add_argument(
        "--no-boot",
        action="store_true",
        help="Skip title-screen automation; only load .sav and tick boot frames",
    )
    parser.add_argument(
        "--boot-frames",
        type=int,
        default=1200,
        help="Frames to wait after ROM boot before title input (default: 1200)",
    )
    parser.add_argument(
        "--start-presses",
        type=int,
        default=3,
        help="How many times to press START on the title screen (default: 3)",
    )
    parser.add_argument(
        "--ingame-timeout",
        type=int,
        default=1200,
        help="Max frames to wait for overworld after Continue (default: 1200)",
    )
    return parser.parse_args()


def _validate_sav(sav_path: Path) -> None:
    size = sav_path.stat().st_size
    if size != EXPECTED_SAV_SIZE:
        print(
            f"[경고] .sav 크기가 {size} bytes 입니다. "
            f"Pokemon Red/Blue 표준은 {EXPECTED_SAV_SIZE} bytes (32 KiB)입니다.",
            file=sys.stderr,
        )


def _press(pyboy, window_event, name: str, hold: int = 30, wait: int = 30) -> None:
    pyboy.send_input(getattr(window_event, f"PRESS_{name}"))
    for _ in range(hold):
        pyboy.tick()
    pyboy.send_input(getattr(window_event, f"RELEASE_{name}"))
    for _ in range(wait):
        pyboy.tick()


def _is_save_loaded(pyboy) -> bool:
    """Continue 이후 세이브가 로드됐는지 wPartyCount로 판별."""
    return int(pyboy.memory[ADDR_PARTY_COUNT]) > 0


def _boot_pokemon_red(
    pyboy,
    window_event,
    boot_frames: int,
    start_presses: int,
    ingame_timeout: int,
) -> bool:
    """Title screen -> Continue -> wait until party is loaded from .sav."""
    for _ in range(boot_frames):
        pyboy.tick()

    for _ in range(start_presses):
        _press(pyboy, window_event, "BUTTON_START", hold=15, wait=60)

    _press(pyboy, window_event, "BUTTON_A", hold=15, wait=60)

    for _ in range(ingame_timeout):
        pyboy.tick()
        if _is_save_loaded(pyboy):
            return True
    return _is_save_loaded(pyboy)


def convert_sav_to_state(
    sav_path: Path,
    rom_path: Path,
    output_path: Path,
    *,
    window: str = "null",
    auto_boot: bool = True,
    boot_frames: int = 1200,
    start_presses: int = 3,
    ingame_timeout: int = 1200,
) -> None:
    try:
        from pyboy import PyBoy
        from pyboy.utils import WindowEvent
    except ModuleNotFoundError as exc:
        raise SystemExit("pyboy가 설치되어 있지 않습니다. pip install pyboy") from exc

    if not rom_path.is_file():
        raise SystemExit(f"ROM을 찾을 수 없습니다: {rom_path}")
    if not sav_path.is_file():
        raise SystemExit(f".sav 파일을 찾을 수 없습니다: {sav_path}")

    _validate_sav(sav_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with sav_path.open("rb") as sav_file:
        pyboy = PyBoy(
            str(rom_path),
            window=window,
            ram_file=sav_file,
            sound_emulated=False,
        )

    try:
        if auto_boot:
            ok = _boot_pokemon_red(
                pyboy,
                WindowEvent,
                boot_frames,
                start_presses,
                ingame_timeout,
            )
            if not ok:
                print(
                    "[경고] Continue 후에도 wPartyCount가 0입니다. "
                    "빈 세이브이거나 ROM/세이브가 맞지 않을 수 있습니다.",
                    file=sys.stderr,
                )
        else:
            for _ in range(boot_frames):
                pyboy.tick()

        map_id = int(pyboy.memory[ADDR_MAP_ID])
        px = int(pyboy.memory[ADDR_PLAYER_X])
        py = int(pyboy.memory[ADDR_PLAYER_Y])
        party = int(pyboy.memory[ADDR_PARTY_COUNT])
        print(
            f"스냅샷 직전 상태: map_id={map_id} (0x{map_id:02X}), "
            f"player=({px}, {py}), party={party}"
        )

        with output_path.open("wb") as out:
            pyboy.save_state(out)

        print(f"저장 완료: {output_path} ({output_path.stat().st_size} bytes)")
    finally:
        pyboy.stop(save=False)


def main() -> None:
    args = _parse_args()
    sav_path = args.sav.expanduser().resolve()
    rom_path = args.rom.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else sav_path.with_suffix(".state")
    )

    convert_sav_to_state(
        sav_path,
        rom_path,
        output_path,
        window=args.window,
        auto_boot=not args.no_boot,
        boot_frames=args.boot_frames,
        start_presses=args.start_presses,
        ingame_timeout=args.ingame_timeout,
    )


if __name__ == "__main__":
    main()
