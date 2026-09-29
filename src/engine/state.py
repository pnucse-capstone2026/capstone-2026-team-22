"""입력 스레드 ↔ 메인 루프 공유 슬롯."""

from __future__ import annotations

# 메인 루프가 세션 시작 시 gmap_ref["gmap"] = gmap 으로 채운다.
gmap_ref: dict = {"gmap": None}
# 같은 자리.
pyboy_ref: dict = {"pyboy": None}
game_state: dict = {"map_id": None, "x": 0, "y": 0}
