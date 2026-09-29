"""map_types.py — 지도 셀의 의미 타입."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class BlockType(Enum):
    """셀 지형/통행 분류."""
    WALL = 1       # 통행 불가 (벽, 건물, 베기 전 나무 등)
    WATER = 2      # 물 (파도타기 시만 통행)
    GRASS = 3      # 전투용 풀 (통행 가능, 야생 인카운터 구역)
    LEDGE_U = 4    # 언덕 — 위로 점프 진입
    LEDGE_D = 5    # 언덕 — 아래로 점프 (플레이어가 북쪽에서 ↓)
    LEDGE_L = 6    # 언덕 — 왼쪽으로 점프
    LEDGE_R = 7    # 언덕 — 오른쪽으로 점프
    WARP = 8       # 워프 셀


class EntityType(Enum):
    """셀 위 상호작용 대상 (NPC·오브젝트·풀베기 나무)."""
    NPC = "npc"            # 사람 NPC (트레이너 포함)
    OBJECT = "object"      # 상호작용 오브젝트 (몬스터볼, 바위, 표지판 스프라이트 등)
    CUT_TREE = "cut_tree"  # 풀베기(Cut) 나무 (block_type=WALL + entity)


@dataclass
class Cell:
    """LocalMap 격자 한 칸 (= 이동칸 1개 = 화면 타일 2x2)."""
    cell_name: Optional[str] = None
    npc_name: Optional[str] = None
    block_type: Optional[BlockType] = None
    warp_candidate: bool = False
    entity: Optional[EntityType] = None
    frontier: bool = False
    visited: bool = False
    visited_count: int = 0

    def mark_visited(self) -> bool:
        """방문 처리하고 '첫 방문이었는지'를 반환 (Exploration RL 보상용)."""
        first = not self.visited
        self.visited = True
        self.visited_count += 1
        return first
