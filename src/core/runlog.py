"""runlog.py — 터미널에 찍히는 것을 runtime/logging.json 에도 남긴다."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT / "runtime"
LOG_FILE = RUNTIME_DIR / "logging.json"

# 한 파일이 무한정 커지지 않도록. 넘으면 오래된 것부터 버린다.
MAX_ENTRIES = 2000


def reset(note: str = "") -> None:
    """기록을 비운다."""
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    try:
        LOG_FILE.write_text("{}\n", encoding="utf-8")
    except OSError:
        return
    if note:
        log("session", note=note)


def _load() -> Dict[str, Any]:
    if not LOG_FILE.is_file():
        return {}
    try:
        data = json.loads(LOG_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def log(kind: str, **fields: Any) -> int:
    """한 사건을 append 하고 그 번호를 돌려준다."""
    try:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        data = _load()
        # **가장 큰 번호 + 1.** 예전에는 "비어 있는 가장 작은 번호" 를 썼는데, 상한을 넘겨
        # 오래된 1번이 지워지면 새 항목이 1번을 받고, 곧바로 "가장 오래된 것" 으로 다시
        # 지워졌다 — 2000개를 채운 뒤로는 아무것도 기록되지 않았다 (실측: 21:11 에서 멈춤).
        n = max((int(k) for k in data if str(k).isdigit()), default=0) + 1
        entry: Dict[str, Any] = {
            "kind": str(kind),
            "time": datetime.now().strftime("%H:%M:%S"),
        }
        for key, value in fields.items():
            if value is None:
                continue
            entry[key] = value
        data[str(n)] = entry

        if len(data) > MAX_ENTRIES:
            for old in sorted(data, key=lambda k: int(k))[: len(data) - MAX_ENTRIES]:
                data.pop(old, None)

        LOG_FILE.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return n
    except Exception:
        return 0


def last(kind: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """가장 최근 항목 (kind 를 주면 그 종류 중에서)."""
    data = _load()
    for key in sorted(data, key=lambda k: int(k), reverse=True):
        entry = data[key]
        if kind is None or entry.get("kind") == kind:
            return entry
    return None


__all__ = ["LOG_FILE", "log", "reset", "last"]
