"""PyBoy construction and shared WRAM access helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pyboy import PyBoy

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SYMBOLS_PATH = PROJECT_ROOT / "assets" / "roms" / "pokered.sym"


def create_pyboy(gb_path, *, headless: bool):
    return PyBoy(
        str(gb_path),
        debug=False,
        no_input=False,
        window="null" if headless else "SDL2",
        log_level="CRITICAL",
        symbols=str(SYMBOLS_PATH),
        sound_emulated=False,
    )


def symbol_address(holder: Any, symbol: str) -> int:
    """Cache WRAM/ROM symbol addresses on ``holder._symbol_addresses``."""
    cache = getattr(holder, "_symbol_addresses", None)
    if cache is None:
        cache = {}
        holder._symbol_addresses = cache
    try:
        return cache[symbol]
    except KeyError:
        address = int(holder.pyboy.symbol_lookup(symbol)[1])
        cache[symbol] = address
        return address


def read_m(holder: Any, addr: str | int) -> int:
    if isinstance(addr, str):
        return int(holder.pyboy.memory[symbol_address(holder, addr)])
    return int(holder.pyboy.memory[int(addr)])


def get_game_coords(holder: Any) -> tuple[int, int, int]:
    return (
        read_m(holder, "wXCoord"),
        read_m(holder, "wYCoord"),
        read_m(holder, "wCurMap"),
    )
