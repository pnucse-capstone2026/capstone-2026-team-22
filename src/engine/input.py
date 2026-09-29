"""PyBoy 버튼 헬퍼."""

from __future__ import annotations

from typing import Any, Optional


def release_button(pyboy: Any, button: Optional[str]) -> None:
    if not button:
        return
    try:
        pyboy.button_release(button)
    except Exception:
        pass
