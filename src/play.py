"""루트 진입점 — scripts/play.py 실행."""

from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "play.py"), run_name="__main__")