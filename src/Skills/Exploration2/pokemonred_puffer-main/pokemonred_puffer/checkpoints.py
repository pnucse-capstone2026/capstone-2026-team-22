"""체크포인트 경로 조회.

추론(play)에만 필요한 부분을 `training/checkpointing.py` 에서 떼어낸 것이다.
이 트리는 실행 전용이라 학습 쪽(모델 저장, RNG state)은 두지 않는다.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path


def latest_model_path(
    root: str | PathLike[str],
    *,
    nested: bool = False,
    pattern: str = "model_*.pt",
) -> Path | None:
    """``root`` (nested면 ``root/*/``) 아래에서 가장 최근 ``model_*.pt``."""
    base = Path(root)
    if not base.is_dir():
        return None
    models = list(base.glob(f"*/{pattern}" if nested else pattern))
    if not models:
        return None
    return max(models, key=lambda path: path.stat().st_mtime)
