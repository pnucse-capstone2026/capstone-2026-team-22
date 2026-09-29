"""SettleGate — 연출(페이드 · 스크립트 이동 · 맵 전환)이 끝나고 게임이 가라앉았는가."""

from __future__ import annotations

from typing import Any, Optional

from core.gb_memory import ADDR_BGP, ADDR_MAP_ID, scene_busy_reason

# 기다림의 안전 상한 (ready() 를 부른 횟수 = auto 가 쉬는 프레임 수). 60fps 기준 30초.
SETTLE_MAX_WAIT_FRAMES = 1800


class SettleGate:
    def __init__(self, frames: int, *, max_wait: int = SETTLE_MAX_WAIT_FRAMES) -> None:
        self.frames = max(0, int(frames))
        self.max_wait = max(1, int(max_wait))
        self.quiet = 0                   # 연속으로 조용한 프레임 수
        self.reason = ""                 # 마지막으로 조용하지 않았던 이유
        self.waited = 0                  # ready() 가 거짓을 돌려준 연속 횟수
        self.forced = False              # 방금 ready() 가 상한 때문에 참이 됐는가
        self._bgp: Optional[int] = None
        self._map: Optional[int] = None

    @property
    def enabled(self) -> bool:
        return self.frames > 0

    def update(self, pyboy: Any, gmap: Any = None) -> str:
        """한 프레임을 본다."""
        try:
            reason = scene_busy_reason(pyboy)
            bgp = int(pyboy.memory[ADDR_BGP])
            map_id = int(pyboy.memory[ADDR_MAP_ID])
        except Exception:
            return ""
        if not reason and self._bgp is not None and bgp != self._bgp:
            reason = "화면 페이드"
        if not reason and self._map is not None and map_id != self._map:
            reason = "맵 전환"
        if not reason and gmap is not None and bool(getattr(gmap, "arrival_pending", False)):
            reason = "도착 확정 대기"
        self._bgp = bgp
        self._map = map_id
        if reason:
            self.quiet = 0
            self.reason = reason
        else:
            self.quiet += 1
        return reason

    def ready(self) -> bool:
        """다음 단계로 넘어가도 되는가."""
        self.forced = False
        if not self.enabled or self.quiet >= self.frames:
            self.waited = 0
            return True
        self.waited += 1
        if self.waited >= self.max_wait:
            self.waited = 0
            self.forced = True
            return True
        return False


__all__ = ["SettleGate", "SETTLE_MAX_WAIT_FRAMES"]
