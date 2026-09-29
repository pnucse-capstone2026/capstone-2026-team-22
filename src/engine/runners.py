"""NavRunner / LocalNavRunner / ExploreRunner — 스킬을 프레임 단위로 굴린다."""

from __future__ import annotations

import io
from contextlib import contextmanager, redirect_stdout
from typing import Any, Dict, Optional, Tuple

from core.gb_memory import (
    ADDR_IN_BATTLE,
    ADDR_MAP_ID,
    ADDR_PLAYER_FACING,
    ADDR_PLAYER_X,
    ADDR_PLAYER_Y,
    ADDR_SCX,
    ADDR_SCY,
    ADDR_WALK_COUNTER,
    FACING_DELTA,
    is_dialogue_active,
    is_in_battle,
)
from core.closure import ENTRY_STEP, entry_kind
from core.config import battle_opt
from core.map_types import BlockType
from core.portal import GlobalMap
from Skills.Navigation import Navigation

from core import runlog
from engine import explore_gate, hm
from engine.dialogue import DialogueSkipper
from engine.frame_observer import ObservingPyBoy
from engine.input import release_button
from engine.move_learn import default_resolver, move_learn_menu_open
from engine.portal_debug import debug_current_portals, portal_text

# nav 디버그는 새 NavRunner가 만들어질 때 Navigation에 적용된다.
debug_flags: dict[str, bool] = {
    "nav_internal": False,
    # 한 걸음마다 "[nav] step N: UP → (x,y)" 를 찍을지.
    "nav_steps": False,
}

NAV_MAX_HOLD_FRAMES = 24
NAV_SETTLE_TIMEOUT = 120
NAV_MAP_WAIT_FRAMES = 90
NAV_PLAN_RETRY_FRAMES = 30
NAV_MAX_PLAN_RETRY = 3
NAV_MAX_STUCK = 3
NAV_MAX_STEPS = 600
# 목표 맵 진입 직후 X/Y가 옛 맵 워프 좌표로 남는 프레임 대비
NAV_ARRIVE_WAIT_FRAMES = 90
NAV_ARRIVE_MAX_EXTEND = 3
# 배틀 중에는 스킬만 멈추고 A 는 안 누른다.
BATTLE_AMASH_MAX_FRAMES = 1_000_000
BATTLE_AMASH_A_HOLD = 6
BATTLE_AMASH_A_GAP = 10


def player_pos_plausible(gmap: GlobalMap, map_id: int, x: int, y: int) -> bool:
    """맵 전환 직후 낡은 좌표(예: Pallet 12,11 → Lab)인지 대략 판별."""
    map_id = int(map_id)
    x, y = int(x), int(y)
    size = getattr(gmap, "map_size_cells", {}).get(map_id)
    if size is not None:
        w, h = int(size[0]), int(size[1])
        # connection 한 칸 여유. 맵 너비 밖이면 옛 맵 좌표일 가능성 큼.
        if x < -2 or y < -2 or x > w + 1 or y > h + 1:
            return False
    local = gmap.get_local_map(map_id)
    if local is not None and local.grid:
        # 관측 칸이 있는데 플레이어 칸이 전혀 없으면 아직 좌표 미갱신일 수 있음
        if (x, y) in local.grid:
            return True
        # 관측이 거의 없으면(진입 직후) size 검사만으로 통과한 경우 허용
        if len(local.grid) < 8:
            return True
        return False
    return True


_DIR_BUTTON = {
    "UP": "up",
    "DOWN": "down",
    "LEFT": "left",
    "RIGHT": "right",
    "A": "a",
}

_OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
def _same_spot(a, b, tol: int = 1) -> bool:
    """두 좌표가 사실상 같은 자리인가 (맨해튼 거리 tol 이내)."""
    if a is None or b is None:
        return False
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])) <= tol


_ACTION_DELTA = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
# 워프 칸에서 비켜설 방향. 좌우를 먼저 본다 — 문은 보통 위아래가 벽이다.
_NUDGE_DIRS = (
    ("LEFT", (-1, 0)),
    ("RIGHT", (1, 0)),
    ("UP", (0, -1)),
    ("DOWN", (0, 1)),
)
# 한 번의 Nav 에서 워프 재진입을 시도할 최대 횟수.
NAV_MAX_WARP_NUDGE = 3
# 같은 NPC 에게 막혀 제자리로 돌아온 횟수 상한.
NAV_MAX_NPC_BLOCK = 5

# --- 배틀: PP 0 인 기술을 계속 고르다 갇히는 것 방지 -----------------------
# assets/roms/pokered.sym 에서 확인한 주소.
ADDR_BATTLE_MON_MOVES = 0xD01C   # 4바이트 (기술 id)
ADDR_BATTLE_MON_PP = 0xD02D      # 4바이트 (하위 6비트가 남은 PP, 상위 2비트는 PP Up)
ADDR_CURRENT_MENU_ITEM = 0xCC26  # 열려 있는 메뉴의 커서 위치


class NavRunner:
    """Skills.Navigation 을 프레임 단위로 굴리는 실행기."""

    def __init__(self, gmap: GlobalMap, pyboy, target_map_id: int) -> None:
        self.gmap = gmap
        self.pyboy = pyboy
        self.nav = Navigation(gmap)

        if debug_flags.get("nav_internal", False):
            if hasattr(self.nav, "set_internal_edge_debug"):
                self.nav.set_internal_edge_debug(True)
            print(
                "[debug] NavRunner internal edge 로그 활성화 "
                "(set_global_destination에서 모든 맵 재계산)"
            )

        self.target_map_id = int(target_map_id)
        self.target_name = gmap.map_name(target_map_id)

        self.phase = "plan"
        self.button = None
        self.pending_action = None
        self.step_start = None
        self.hold_frames = 0
        self.settle_frames = 0
        self.wait_frames = 0
        self.plan_retry = 0
        self.stuck = 0
        self.steps = 0
        self.last_map = None
        self.finished = False
        self.success = False
        # 왜 끝났는지. play.py 가 다음 계획에 실패 사유로 넘긴다.
        self.reason = ""
        self._arrive_extends = 0
        self._paused_for_battle = False
        # 워프 칸 위에서 다시 밟기를 시도한 횟수
        self._warp_nudges = 0
        # NPC 가 길을 막고 있는가.
        self._npc_blocks = 0
        self._block_spot: Optional[Tuple[int, int, int]] = None
        self._blocking_name = ""
        # 방금 배틀이 끝났는가. 트레이너 전투 뒤의 대사는 "막는 것"이 아니다.
        self._after_battle = False
        # 한도를 넘겨 멈췄을 때 "누가 어디서 막았는지".
        self.blocked_by: Optional[Dict[str, Any]] = None
        # 이동 중 끼어든 스크립트 대화 → A 로 넘기고 끝나면 재계획
        self._dialogue = DialogueSkipper(
            pyboy, tag="nav", on_start=self._on_dialogue_start
        )

    def start(self, cur_map: int, cur_x: int, cur_y: int) -> bool:
        self.last_map = cur_map
        self._arrive_extends = 0
        self._paused_for_battle = False
        self._warp_nudges = 0
        self._npc_blocks = 0
        self._block_spot = None
        self._blocking_name = ""
        self._after_battle = False
        self.blocked_by = None

        if debug_flags.get("nav_internal", False):
            debug_current_portals(
                self.gmap,
                cur_map,
                cur_x,
                cur_y,
            )

        if cur_map == self.target_map_id:
            print(f'[nav] 이미 "{self.target_name}" 에 있습니다.')
            self._finish(True)
            return False
        if not self.nav.set_global_destination(
            cur_map,
            self.target_map_id,
            start_pos=(cur_x, cur_y),
            is_surfing=hm.can_surf(self.pyboy),
            can_cut=hm.can_cut(self.pyboy),
        ):
            self.reason = "관측된 Portal 로 이어지는 길이 없습니다"
            self._finish(False)
            return False
        print(f'[nav] 자동 이동 시작 → "{self.target_name}" (id={self.target_map_id})')
        self.phase = "plan"
        return True

    def stop(self, reason: str = "사용자 중단") -> None:
        if self.finished:
            return
        print(f"[nav] 중단: {reason}")
        self.reason = str(reason)
        self._finish(False)

    def _finish(self, success: bool) -> None:
        self._dialogue.release()
        self._release()
        self.finished = True
        self.success = success
        self.phase = "done"
        if success:
            print(f'[nav] 도착: "{self.target_name}" (총 {self.steps} 걸음)')

    def _release(self) -> None:
        release_button(self.pyboy, self.button)
        self.button = None

    def _tick_battle_pause(self) -> bool:
        """배틀 중 이동 입력을 놓고 현재 Navigation을 보존한다."""
        if not self._paused_for_battle:
            self._paused_for_battle = True
            self._release()
            self.nav.action_queue.clear()
            # 직전 대화는 트레이너가 말을 건 것이다 — 길을 막은 게 아니다.
            self._npc_blocks = 0
            self._block_spot = None
            self._blocking_name = ""
            self._after_battle = True
            print("[nav] 배틀 감지 — Navigation 일시정지")
        return False

    def pause_for_battle(self) -> None:
        self._tick_battle_pause()

    def _on_dialogue_start(self) -> None:
        """대화가 뜬 순간: 누르던 방향키를 떼고 남은 경로를 버린다."""
        self._release()
        self.nav.action_queue.clear()
        self._blocking_name = self._npc_in_front() or self._blocking_name

    def _resume_from_pause(
        self, cur_map: int, cur_x: int, cur_y: int, *, why: str
    ) -> None:
        """배틀/대화로 멈췄던 Navigation 을 현재 위치에서 다시 계획해 재개."""
        self._release()
        if why == "대화 종료" and self._note_npc_block(cur_map, cur_x, cur_y):
            return
        self.stuck = 0
        self.plan_retry = 0
        self.nav.action_queue.clear()
        print(
            f'[nav] {why} — Navigation 재개 '
            f'pos=({cur_x},{cur_y}) map="{self.gmap.map_name(cur_map)}"'
        )
        if cur_map == self.target_map_id:
            self._arrive_extends = 0
            self._begin_arrive_wait(cur_map, cur_x, cur_y)
            return
        if not self.nav.set_global_destination(
            cur_map,
            self.target_map_id,
            start_pos=(cur_x, cur_y),
            is_surfing=hm.can_surf(self.pyboy),
            can_cut=hm.can_cut(self.pyboy),
        ):
            self._finish(False)
            return
        self.phase = "mapwait"
        self.wait_frames = NAV_MAP_WAIT_FRAMES

    def _resume_after_battle(self, cur_map: int, cur_x: int, cur_y: int) -> None:
        self._paused_for_battle = False
        # 배틀 직후의 대사(트레이너 마무리 대사)는 길을 막는 NPC 가 아니다
        self._after_battle = True
        self._resume_from_pause(cur_map, cur_x, cur_y, why="배틀 종료")

    def _begin_arrive_wait(self, cur_map: int, cur_x: int, cur_y: int) -> None:
        """목표 맵 진입: 좌표·화면 안정화 후 도착 확정."""
        self.phase = "arrivewait"
        self.wait_frames = NAV_ARRIVE_WAIT_FRAMES
        print(
            f'[nav] 목표 맵 "{self.target_name}" 진입 '
            f"pos=({cur_x},{cur_y}) — 좌표 안정화 대기 "
            f"({NAV_ARRIVE_WAIT_FRAMES}f)"
        )

    def tick(self, cur_map: int, cur_x: int, cur_y: int) -> bool:
        """한 프레임 진행."""
        if self.finished:
            return False

        if int(self.pyboy.memory[ADDR_IN_BATTLE]) != 0:
            return self._tick_battle_pause()
        if self._paused_for_battle:
            self._resume_after_battle(cur_map, cur_x, cur_y)
            return False

        # 스크립트 대화: A 로 넘기는 동안 이동을 멈췄다가, 끝나면 재계획해서 재개
        if self._dialogue.busy():
            return False
        if self._dialogue.take_finished():
            self._resume_from_pause(cur_map, cur_x, cur_y, why="대화 종료")
            return False

        if cur_map != self.last_map:
            return self._on_map_change(cur_map, cur_x, cur_y)

        if self.phase in ("mapwait", "planwait", "arrivewait"):
            self.wait_frames -= 1
            if self.wait_frames > 0:
                return False
            if self.phase == "arrivewait":
                return self._finish_arrive_wait(cur_map, cur_x, cur_y)
            self.phase = "plan"
            return False
        if self.phase == "plan":
            self._plan(cur_map, cur_x, cur_y)
            return False
        if self.phase == "press":
            self._press(cur_map, cur_x, cur_y)
            return False
        if self.phase == "hold":
            self._hold()
            return False
        if self.phase == "settle":
            return self._settle(cur_map, cur_x, cur_y)
        return False

    def _finish_arrive_wait(self, cur_map: int, cur_x: int, cur_y: int) -> bool:
        if cur_map != self.target_map_id:
            # 대기 중 다른 맵으로 튕긴 경우
            self.phase = "plan"
            return False
        if not player_pos_plausible(self.gmap, cur_map, cur_x, cur_y):
            if self._arrive_extends < NAV_ARRIVE_MAX_EXTEND:
                self._arrive_extends += 1
                self.phase = "arrivewait"
                self.wait_frames = NAV_ARRIVE_WAIT_FRAMES
                print(
                    f"[nav] 좌표 아직 불안정 pos=({cur_x},{cur_y}) "
                    f"— 추가 대기 ({self._arrive_extends}/{NAV_ARRIVE_MAX_EXTEND})"
                )
                return False
            print(
                f"[nav] 경고: 안정화 후에도 좌표 이상 pos=({cur_x},{cur_y}) "
                "— 그래도 도착 처리"
            )
        print(f"[nav] 좌표 안정화 확인 pos=({cur_x},{cur_y})")
        self._finish(True)
        return True

    def _on_map_change(self, cur_map: int, cur_x: int, cur_y: int) -> bool:
        self._release()
        prev_map = self.last_map
        self.last_map = cur_map
        self.nav.action_queue.clear()
        self.stuck = 0
        self.plan_retry = 0
        # 맵이 바뀌었다 = 워프가 실제로 발동했다. 다음 워프는 다시 시도할 수 있다.
        self._warp_nudges = 0
        self._npc_blocks = 0
        self._block_spot = None
        self._blocking_name = ""

        # Portal 경로에서 이전 맵의 Portal들을 제거하고 현재 맵 구간에 맞춘다.
        self.nav.notify_map_change(cur_map)

        print(
            f"\n[nav] 맵 전환: \"{self.gmap.map_name(prev_map)}\" → "
            f"\"{self.gmap.map_name(cur_map)}\"  pos=({cur_x},{cur_y})"
        )

        if cur_map == self.target_map_id:
            # 즉시 도착 확정하지 않음 — WRAM X/Y가 한동안 옛 맵 좌표일 수 있음
            self._arrive_extends = 0
            self._begin_arrive_wait(cur_map, cur_x, cur_y)
            return True

        # 예상한 Portal 경로가 소진되었거나 다른 맵으로 들어온 경우 현재 좌표에서 전역 Portal 경로를 다시 계산한다.
        if not self.nav.portal_queue:
            print("[nav] Portal 경로 소진/예상 밖 맵 → 전역 경로 재탐색")
            if not self.nav.set_global_destination(
                cur_map,
                self.target_map_id,
                start_pos=(cur_x, cur_y),
                is_surfing=hm.can_surf(self.pyboy),
                can_cut=hm.can_cut(self.pyboy),
            ):
                self._finish(False)
                return True

        self.phase = "mapwait"
        self.wait_frames = NAV_MAP_WAIT_FRAMES
        return True

    def _plan(self, cur_map: int, cur_x: int, cur_y: int) -> None:
        if self.steps >= NAV_MAX_STEPS:
            self.stop(f"최대 스텝({NAV_MAX_STEPS}) 초과")
            return

        if self.nav.action_queue:
            self.phase = "press"
            return

        if not self.nav.portal_queue:
            if cur_map == self.target_map_id:
                if not player_pos_plausible(self.gmap, cur_map, cur_x, cur_y):
                    self._arrive_extends = 0
                    self._begin_arrive_wait(cur_map, cur_x, cur_y)
                    return
                self._finish(True)
                return

            print("[nav] 남은 Portal 경로가 없어 현재 위치에서 재탐색")
            if not self.nav.set_global_destination(
                cur_map,
                self.target_map_id,
                start_pos=(cur_x, cur_y),
                is_surfing=hm.can_surf(self.pyboy),
                can_cut=hm.can_cut(self.pyboy),
            ):
                self.stop("목표까지 연결된 Portal 경로를 찾지 못함")
                return

        ok = self.nav.plan_local_path_to_next_map(
            cur_map,
            (cur_x, cur_y),
        )

        if ok and self.nav.action_queue:
            self.plan_retry = 0
            self.phase = "press"
            return

        # A*가 성공했지만 액션이 0개라면 이미 Portal 목표 좌표에 있는 상태일 수 있다.
        if ok and not self.nav.action_queue:
            if self._nudge_off_warp(cur_map, cur_x, cur_y):
                self.phase = "press"
                return
            self.plan_retry += 1
            print(
                "[nav] 목표 Portal 위치에 있으나 이동 액션이 없습니다. "
                "맵 전환을 잠시 기다립니다."
            )
            self.phase = "planwait"
            self.wait_frames = NAV_PLAN_RETRY_FRAMES
            return

        self.plan_retry += 1
        if self.plan_retry >= NAV_MAX_PLAN_RETRY:
            if self.nav.gate_queue:
                portal_node = self.nav.gate_queue[0]
                portal_desc = portal_text(self.gmap, portal_node)
            else:
                portal_desc = "알 수 없는 Portal"

            self.stop(
                f"{portal_desc} 로 가는 LocalMap 경로를 찾지 못함 "
                "(그 구간을 아직 충분히 관측하지 않았거나 동적 장애물이 있을 수 있음)"
            )
            return

        self.phase = "planwait"
        self.wait_frames = NAV_PLAN_RETRY_FRAMES

    def _npc_in_front(self) -> str:
        """마지막으로 가려던 방향 앞칸에 있는 NPC 이름 (없으면 "")."""
        delta = _ACTION_DELTA.get(self.pending_action or "")
        if delta is None:
            return ""
        m = self.pyboy.memory
        cur_map = int(m[ADDR_MAP_ID])
        x, y = int(m[ADDR_PLAYER_X]), int(m[ADDR_PLAYER_Y])
        local = self.gmap.get_local_map(cur_map)
        if local is None:
            return ""
        cell = (getattr(local, "grid", None) or {}).get((x + delta[0], y + delta[1]))
        return str(getattr(cell, "npc_name", "") or "") if cell is not None else ""

    def _npc_near(self, cur_map: int, cur_x: int, cur_y: int) -> str:
        """사방 이웃 중 NPC 이름 하나."""
        local = self.gmap.get_local_map(int(cur_map))
        grid = (getattr(local, "grid", None) or {}) if local is not None else {}
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            cell = grid.get((int(cur_x) + dx, int(cur_y) + dy))
            name = str(getattr(cell, "npc_name", "") or "") if cell else ""
            if name:
                return name
        return ""

    def _note_npc_block(self, cur_map: int, cur_x: int, cur_y: int) -> bool:
        """대화가 **끝난 위치**가 계속 같은가."""
        if self._after_battle:
            # 트레이너 전투 뒤의 대사다 — 막힌 게 아니라 그냥 싸운 것이다
            self._after_battle = False
            self._npc_blocks = 0
            self._block_spot = None
            return False
        here = (int(cur_map), int(cur_x), int(cur_y))
        same = (
            self._block_spot is not None
            and here[0] == self._block_spot[0]
            and _same_spot(here[1:], self._block_spot[1:])
        )
        if not same:
            # 다른 자리에서 끝났다 = 아직 같은 벽에 부딪히는 중이 아니다.
            self._block_spot = here
            self._npc_blocks = 1
        else:
            self._npc_blocks += 1
        who = (
            self._blocking_name
            or self._npc_in_front()
            or self._npc_near(cur_map, cur_x, cur_y)
            or "(이름 미상)"
        )
        if self._npc_blocks >= 2:
            # 1회는 그냥 지나가는 대화일 수 있다.
            print(
                f"[nav] 대화 후 또 같은 자리({cur_x},{cur_y}) — {who} 가 길을 "
                f"막고 있습니다 ({self._npc_blocks}/{NAV_MAX_NPC_BLOCK})"
            )
        if self._npc_blocks < NAV_MAX_NPC_BLOCK:
            return False

        self.blocked_by = {
            "npc": who,
            "map_id": int(cur_map),
            "map": self.gmap.map_name(int(cur_map)),
            "pos": (int(cur_x), int(cur_y)),
            "target_map_id": int(self.target_map_id),
            "target_map": self.target_name,
            "tries": int(self._npc_blocks),
        }
        self.stop(
            f"{who} 가 {NAV_MAX_NPC_BLOCK}번 막았습니다 — "
            f'"{self.target_name}" 로 가는 길이 아직 안 열린 것 같습니다'
        )
        return True

    def _on_warp_tile(self, cur_map: int, cur_x: int, cur_y: int) -> bool:
        """지금 서 있는 칸이 워프(문·계단·게이트)인가."""
        local = self.gmap.get_local_map(int(cur_map))
        if local is None:
            return False
        cell = getattr(local, "grid", {}).get((int(cur_x), int(cur_y)))
        if cell is None:
            return False
        return bool(
            cell.block_type is BlockType.WARP or getattr(cell, "warp_candidate", False)
        )

    def _nudge_off_warp(self, cur_map: int, cur_x: int, cur_y: int) -> bool:
        """워프 칸 위에 서 있으면 옆으로 비켰다 되돌아오는 두 걸음을 넣는다."""
        if self._warp_nudges >= NAV_MAX_WARP_NUDGE:
            return False
        if not self._on_warp_tile(cur_map, cur_x, cur_y):
            return False
        local = self.gmap.get_local_map(int(cur_map))
        grid = getattr(local, "grid", None) or {}

        for action, (dx, dy) in _NUDGE_DIRS:
            side = grid.get((int(cur_x) + dx, int(cur_y) + dy))
            if side is None:
                continue
            if side.block_type is BlockType.WARP or getattr(
                side, "warp_candidate", False
            ):
                continue
            # 능력 없이 그냥 걸어갈 수 있는 칸만 (물/나무/언덕 역방향 제외).
            if entry_kind(side, (dx, dy)) != ENTRY_STEP:
                continue
            self._warp_nudges += 1
            self.nav.action_queue.extend([action, _OPPOSITE[action]])
            print(
                f"[nav] 워프 칸 위에 서 있습니다 pos=({cur_x},{cur_y}) — "
                f"{action} 로 비켰다 되돌아와 다시 밟습니다 "
                f"({self._warp_nudges}/{NAV_MAX_WARP_NUDGE})"
            )
            return True

        print(
            f"[nav] 워프 칸 위인데 비켜설 자리가 없습니다 pos=({cur_x},{cur_y}) "
            "— 주변을 더 관측해야 합니다"
        )
        return False

    def _try_hm(self, cur_map: int, cur_x: int, cur_y: int) -> bool:
        """앞을 막은 것이 나무/물이면 HM 을 쓰고 현재 위치에서 재계획한다."""
        button = _DIR_BUTTON.get(self.pending_action)
        if button is None:
            return False
        try:
            used = hm.use_for_direction(self.pyboy, button)
        except Exception as exc:
            print(f"[nav] 비전머신 사용 실패: {type(exc).__name__}: {exc}")
            runlog.log("hm", 실패=f"{type(exc).__name__}: {exc}")
            return False
        if not used:
            return False

        # 지형이 바뀌었으니 이 맵의 "다 봤음" 표시는 무효다.
        local = self.gmap.get_local_map(int(cur_map))
        if local is not None:
            local.closed = False
            local.final_done = False           # 포탈 판정도 지형이 바뀌면 무효다
            local.portal_done.clear()
        if used == "surf":
            # 물 너머가 통째로 새 미탐험이 된다.
            explore_gate.reopen_all(self.gmap, "파도타기 획득")

        # HM 사용 중 플레이어가 움직였을 수 있다(파도타기는 한 칸 전진한다).
        new_map = int(self.pyboy.memory[ADDR_MAP_ID])
        new_x = int(self.pyboy.memory[ADDR_PLAYER_X])
        new_y = int(self.pyboy.memory[ADDR_PLAYER_Y])
        self.last_map = new_map
        self.stuck = 0
        self._resume_from_pause(
            new_map,
            new_x,
            new_y,
            why="풀베기" if used == "cut" else "파도타기",
        )
        return True

    def _press(self, cur_map: int, cur_x: int, cur_y: int) -> None:
        action = self.nav.pop_action()
        if action is None:
            self.phase = "plan"
            return
        button = _DIR_BUTTON.get(action)
        if button is None:
            self.phase = "plan"
            return
        self.button = button
        self.pending_action = action
        self.step_start = (cur_map, cur_x, cur_y)
        self.hold_frames = 0
        self.pyboy.button_press(button)
        self.phase = "hold"

    def _hold(self) -> None:
        self.hold_frames += 1
        walking = int(self.pyboy.memory[ADDR_WALK_COUNTER]) != 0
        if walking or self.hold_frames >= NAV_MAX_HOLD_FRAMES:
            self._release()
            self.settle_frames = 0
            self.phase = "settle"

    def _settle(self, cur_map: int, cur_x: int, cur_y: int) -> bool:
        self.settle_frames += 1
        walking = int(self.pyboy.memory[ADDR_WALK_COUNTER]) != 0
        scx = int(self.pyboy.memory[ADDR_SCX])
        scy = int(self.pyboy.memory[ADDR_SCY])
        aligned = scx % 8 == 0 and scy % 8 == 0
        if (walking or not aligned) and self.settle_frames < NAV_SETTLE_TIMEOUT:
            return False

        moved = (cur_map, cur_x, cur_y) != self.step_start
        self.phase = "plan"
        if not moved and self.pending_action == "A":
            # 길을 막은 몬스터볼을 줍는 중이다 — A 는 원래 제자리다
            if self.nav.action_queue:
                self.phase = "press"
            return True
        if not moved:
            # 막힌 게 나무/물이면 비전머신을 쓰고 재개한다.
            if self._try_hm(cur_map, cur_x, cur_y):
                return False
            self.stuck += 1
            self.nav.action_queue.clear()
            print(f"[nav] 이동 실패({self.pending_action}) — 막힘 "
                  f"{self.stuck}/{NAV_MAX_STUCK}")
            if self.stuck >= NAV_MAX_STUCK:
                self.stop("같은 자리에서 계속 막힘 (지도와 실제가 다를 수 있음)")
            return False

        self.stuck = 0
        self.steps += 1
        # 여기서 _npc_blocks 를 지우면 안 된다 — 문지기에게 밀려난 뒤 다시 걸어 올라가는 걸음이 성공하므로 매번 초기화되어 영영 안 잡힌다.
        if debug_flags.get("nav_steps", False):
            print(
                f'\n[nav] step {self.steps}: {self.pending_action} '
                f'→ ({cur_x},{cur_y}) map="{self.gmap.map_name(cur_map)}" '
                f"남은 액션 {len(self.nav.action_queue)}개"
            )
        if self.nav.action_queue:
            self.phase = "press"
        return True


class LocalNavRunner:
    """같은 맵 안: NPC / GRASS 로 Local A* (+ NPC면 대화 종료까지 A)."""

    A_HOLD_FRAMES = 8
    A_SETTLE_FRAMES = 20
    # 대화 시작 전 A 시도 상한 / 종료 확인 idle / 전체 타임아웃
    TALK_START_MAX_A = 8
    TALK_IDLE_DONE_FRAMES = 45
    TALK_TIMEOUT_FRAMES = 60 * 60  # ~60초 @60fps 상한

    def __init__(
        self,
        gmap: GlobalMap,
        pyboy,
        *,
        mode: str,
        npc_name: Optional[str] = None,
    ) -> None:
        self.gmap = gmap
        self.pyboy = pyboy
        self.nav = Navigation(gmap)
        self.mode = mode  # "npc" | "grass"
        self.npc_name = npc_name
        self.phase = "plan"
        self.button = None
        self.pending_action = None
        self.hold_frames = 0
        self.settle_frames = 0
        self.wait_frames = 0
        self.steps = 0
        self.finished = False
        self.success = False
        self._planned = False
        self._plan_extends = 0
        self.last_map = None
        self._talk_seen = False
        self._talk_idle = 0
        self._talk_a_count = 0
        self._talk_frames = 0
        # **목표 NPC 와** 대화가 이루어졌는가. 끼어든 트레이너와의 대화는 아니다.
        self.talked_to_target = False
        self._paused_for_battle = False
        self._phase_before_battle: Optional[str] = None
        # 대화가 끝난 자리가 계속 같으면 "막혔다".
        self._npc_blocks = 0
        self._block_spot: Optional[Tuple[int, int]] = None
        # 방금 배틀이 끝났는가. 트레이너 전투 뒤의 대사는 막는 것이 아니다.
        self._after_battle = False
        # 한도를 넘겨 멈췄을 때 play.py 가 읽어 다음 계획에 넣는다.
        self.blocked_by: Optional[Dict[str, Any]] = None
        self.reason = ""
        # 이동 중 끼어든 스크립트 대화 → A 로 넘기고 끝나면 재계획.
        self._dialogue = DialogueSkipper(
            pyboy, tag="nav", on_start=self._on_dialogue_start
        )

    def start(self, cur_map: int, cur_x: int, cur_y: int) -> bool:
        self.last_map = int(cur_map)
        self._plan_extends = 0
        self._talk_seen = False
        self._talk_idle = 0
        self._talk_a_count = 0
        self._talk_frames = 0
        self._paused_for_battle = False
        self._phase_before_battle = None
        self._after_battle = False
        if self.mode == "npc" and not self.npc_name:
            print("[nav npc] 이름 필요: nav npc <이름>")
            self._finish(False)
            return False

        # 맵 Nav 직후 등: 좌표가 아직 옛 맵이면 바로 A* 하지 말고 대기
        if not player_pos_plausible(self.gmap, cur_map, cur_x, cur_y):
            print(
                f"[nav] Local 시작 좌표 불안정 pos=({cur_x},{cur_y}) "
                f"— 안정화 후 경로 계산 ({NAV_ARRIVE_WAIT_FRAMES}f)"
            )
            self.phase = "mapwait"
            self.wait_frames = NAV_ARRIVE_WAIT_FRAMES
            return True

        return self._try_plan(cur_map, cur_x, cur_y)

    def _try_plan(self, cur_map: int, cur_x: int, cur_y: int) -> bool:
        ok = False
        if self.mode == "npc":
            # A 연타는 경로 후 talk 단계에서 텍스트박스 종료까지 처리
            ok = self.nav.plan_local_path_to_npc(
                cur_map,
                (cur_x, cur_y),
                self.npc_name,
                mash_a=0,
            )
            label = f'npc "{self.npc_name}"'
        else:
            ok = self.nav.plan_local_path_to_grass(cur_map, (cur_x, cur_y))
            label = "nearest grass"
        if not ok:
            if self._plan_extends < NAV_ARRIVE_MAX_EXTEND:
                self._plan_extends += 1
                self.phase = "mapwait"
                self.wait_frames = NAV_ARRIVE_WAIT_FRAMES
                print(
                    f"[nav] Local A* 실패 — 재시도 대기 "
                    f"({self._plan_extends}/{NAV_ARRIVE_MAX_EXTEND}) "
                    f"pos=({cur_x},{cur_y})"
                )
                return True
            # 사유를 비워 두면 play 의 계획 기록에 "대화 실패: 실패" 만 남는다
            self.reason = f"{label} 앞까지 Local A* 경로 없음"
            self._finish(False)
            return False
        print(f"[nav] Local 이동 시작 → {label} from ({cur_x},{cur_y})")
        self._planned = True
        if self.nav.action_queue:
            self.phase = "press"
            return True
        if self.mode == "npc":
            print("[nav] 이미 NPC 옆 — 대화(A) 시작")
            self.phase = "talk"
            return True
        self._finish(True)
        return False

    def stop(self, reason: str = "사용자 중단") -> None:
        if self.finished:
            return
        print(f"[nav] 중단: {reason}")
        self.reason = str(reason)
        self._finish(False)

    def _finish(self, success: bool) -> None:
        self._dialogue.release()
        self._release()
        self.finished = True
        self.success = success
        self.phase = "done"
        if success:
            extra = ""
            if self.mode == "npc":
                extra = f", A×{self._talk_a_count}"
            print(f"[nav] Local 완료 ({self.mode}, steps={self.steps}{extra})")

    def _release(self) -> None:
        release_button(self.pyboy, self.button)
        self.button = None

    # ------------------------------------------------------------------
    # 누구와 말하고 있는가
    # ------------------------------------------------------------------
    def _player_pos(self) -> Tuple[int, int]:
        return (
            int(self.pyboy.memory[ADDR_PLAYER_X]),
            int(self.pyboy.memory[ADDR_PLAYER_Y]),
        )

    def _facing_npc(self, cur_map: int, cur_x: int, cur_y: int) -> Optional[str]:
        """지금 바라보는 칸에 있는 NPC 이름."""
        try:
            facing = int(self.pyboy.memory[ADDR_PLAYER_FACING])
            delta = FACING_DELTA.get(facing)
            if delta is None:
                return None
            local = self.gmap.get_local_map(int(cur_map))
            if local is None:
                return None
            cell = (getattr(local, "grid", None) or {}).get(
                (int(cur_x) + delta[0], int(cur_y) + delta[1])
            )
            return getattr(cell, "npc_name", None)
        except Exception:
            return None

    def _facing_target(self, cur_map: int, cur_x: int, cur_y: int) -> bool:
        """앞에 있는 것이 **목표 NPC** 인가."""
        if not self.npc_name:
            return True
        found = self._facing_npc(cur_map, cur_x, cur_y)
        if found is None:
            return True
        return str(found) == str(self.npc_name)

    def _tick_battle_pause(self) -> bool:
        if not self._paused_for_battle:
            self._paused_for_battle = True
            self._phase_before_battle = self.phase
            self._release()
            self.nav.action_queue.clear()
            # 트레이너가 말을 걸어 시작된 전투다 — 직전 대화를 "막힘" 으로 세지 않는다
            self._npc_blocks = 0
            self._block_spot = None
            self._after_battle = True
            print("[nav] 배틀 감지 — Local Nav 일시정지")
        return False

    def pause_for_battle(self) -> None:
        self._tick_battle_pause()

    def _on_dialogue_start(self) -> None:
        """대화가 뜬 순간: 누르던 방향키를 떼고 남은 경로를 버린다."""
        self._release()
        self.nav.action_queue.clear()

    def _resume_after_dialogue(self, cur_x: int, cur_y: int) -> None:
        """대화로 멈췄던 Local Nav 를 현재 위치에서 다시 계획해 재개."""
        self._release()
        self.nav.action_queue.clear()
        if self._note_npc_block(cur_x, cur_y):
            return
        print(f"[nav] 대화 종료 — Local Nav 재개 pos=({cur_x},{cur_y})")
        self.phase = "mapwait"
        self.wait_frames = max(20, NAV_MAP_WAIT_FRAMES // 3)

    def _note_npc_block(self, cur_x: int, cur_y: int) -> bool:
        """대화가 끝난 자리가 계속 같은가."""
        if self._after_battle:
            # 트레이너 전투 뒤의 대사는 막는 것이 아니다
            self._after_battle = False
            self._npc_blocks = 0
            self._block_spot = None
            return False
        here = (int(cur_x), int(cur_y))
        if not _same_spot(here, self._block_spot):
            self._block_spot = here
            self._npc_blocks = 1
        else:
            self._npc_blocks += 1
        if self._npc_blocks >= 2:
            print(
                f"[nav] 대화 후 또 같은 자리({cur_x},{cur_y}) — "
                f"{self.npc_name or '상대'} 가 길을 막고 있습니다 "
                f"({self._npc_blocks}/{NAV_MAX_NPC_BLOCK})"
            )
        if self._npc_blocks < NAV_MAX_NPC_BLOCK:
            return False

        who = str(self.npc_name or "(이름 미상)")
        map_id = int(self.last_map) if self.last_map is not None else -1
        name = self.gmap.map_name(map_id) if map_id >= 0 else "?"
        self.blocked_by = {
            "npc": who,
            "map_id": map_id,
            "map": name,
            "pos": here,
            "target_map": name,
            "tries": int(self._npc_blocks),
            "kind": "talk",
        }
        self.stop(
            f"{who} 와 {NAV_MAX_NPC_BLOCK}번 대화했지만 제자리입니다 "
            "— 지금 조건으로는 통과할 수 없는 것 같습니다"
        )
        return True

    def _resume_after_battle(self, cur_map: int, cur_x: int, cur_y: int) -> None:
        self._paused_for_battle = False
        self._after_battle = True          # 곧 뜨는 대사는 트레이너 마무리 대사다
        self._release()
        prev = self._phase_before_battle
        self._phase_before_battle = None
        print(
            f"[nav] 배틀 종료 — Local Nav 재개 "
            f"pos=({cur_x},{cur_y})"
        )
        self.nav.action_queue.clear()
        # 대화 중이었더라도 **앞에 있는 것이 목표 NPC 일 때만** 대화를 잇는다.
        if (
            self.mode == "npc"
            and (self._talk_seen or (prev is not None and prev.startswith("talk")))
            and self._facing_target(cur_map, cur_x, cur_y)
        ):
            self.phase = "talk"
            self._talk_idle = 0
            return
        if self.mode == "npc":
            print(f'[nav] 전투로 끊겼습니다 — "{self.npc_name}" 에게 다시 갑니다')
            self._talk_seen = False
        self.phase = "mapwait"
        self.wait_frames = max(20, NAV_MAP_WAIT_FRAMES // 3)

    def tick(self, cur_map: int, cur_x: int, cur_y: int) -> bool:
        if self.finished:
            return False

        if int(self.pyboy.memory[ADDR_IN_BATTLE]) != 0:
            return self._tick_battle_pause()
        if self._paused_for_battle:
            self._resume_after_battle(cur_map, cur_x, cur_y)
            return False

        # talk 계열은 자기 상태 기계가 A 를 관리하므로 건드리지 않는다.
        if not self.phase.startswith("talk"):
            if self._dialogue.busy():
                return False
            if self._dialogue.take_finished():
                self._resume_after_dialogue(cur_x, cur_y)
                return False

        if self.last_map is None:
            self.last_map = int(cur_map)
        elif int(cur_map) != int(self.last_map):
            self.stop(f"맵 변경 {self.last_map}→{cur_map}")
            return False

        if self.phase == "mapwait":
            self.wait_frames -= 1
            if self.wait_frames > 0:
                return False
            if not player_pos_plausible(self.gmap, cur_map, cur_x, cur_y):
                if self._plan_extends < NAV_ARRIVE_MAX_EXTEND:
                    self._plan_extends += 1
                    self.wait_frames = NAV_ARRIVE_WAIT_FRAMES
                    print(
                        f"[nav] Local 좌표 대기 연장 pos=({cur_x},{cur_y}) "
                        f"({self._plan_extends}/{NAV_ARRIVE_MAX_EXTEND})"
                    )
                    return False
            self._try_plan(cur_map, cur_x, cur_y)
            return False
        if self.phase == "press":
            self._press()
            return False
        if self.phase == "hold":
            self._hold()
            return False
        if self.phase == "settle":
            return self._settle(cur_map, cur_x, cur_y)
        if self.phase == "talk":
            self._talk_tick()
            return False
        if self.phase == "talk_hold":
            self._talk_hold()
            return False
        if self.phase == "talk_settle":
            self._talk_settle()
            return False
        if self.phase == "done":
            self._finish(True)
        return False

    def _press(self) -> None:
        action = self.nav.pop_action()
        if action is None:
            if self.mode == "npc":
                print("[nav] 이동 완료 — 대화(A) 시작 (텍스트박스 종료까지)")
                self.phase = "talk"
                self._talk_idle = 0
                return
            self._finish(True)
            return
        button = _DIR_BUTTON.get(action)
        if button is None:
            if self.mode == "npc":
                self.phase = "talk"
                return
            self._finish(True)
            return
        self.button = button
        self.pending_action = action
        self.hold_frames = 0
        self.settle_frames = 0
        self.pyboy.button_press(button)
        self.phase = "hold"

    def _hold(self) -> None:
        self.hold_frames += 1
        is_a = self.pending_action == "A"
        limit = self.A_HOLD_FRAMES if is_a else NAV_MAX_HOLD_FRAMES
        walking = int(self.pyboy.memory[ADDR_WALK_COUNTER]) != 0
        if is_a:
            if self.hold_frames >= limit:
                self._release()
                self.phase = "settle"
                self.settle_frames = 0
            return
        if walking or self.hold_frames >= limit:
            self._release()
            self.phase = "settle"
            self.settle_frames = 0

    def _settle(self, cur_map: int, cur_x: int, cur_y: int) -> bool:
        self.settle_frames += 1
        is_a = self.pending_action == "A"
        if is_a:
            if self.settle_frames < self.A_SETTLE_FRAMES:
                return False
            self.steps += 1
            self.phase = "press"
            return False

        walking = int(self.pyboy.memory[ADDR_WALK_COUNTER]) != 0
        scx = int(self.pyboy.memory[ADDR_SCX])
        scy = int(self.pyboy.memory[ADDR_SCY])
        aligned = scx % 8 == 0 and scy % 8 == 0
        if (walking or not aligned) and self.settle_frames < NAV_SETTLE_TIMEOUT:
            return False
        self.steps += 1
        self.phase = "press"
        return True

    def _talk_tick(self) -> None:
        """텍스트박스가 꺼질 때까지 A."""
        if not self._talk_seen and not self._facing_target(
            int(self.last_map or -1), *self._player_pos()
        ):
            # 앞에 있는 것이 목표가 아니다 (끼어든 트레이너 등) — 다시 간다
            who = self._facing_npc(int(self.last_map or -1), *self._player_pos())
            print(f'[nav] 앞에 있는 것은 "{who}" 입니다 — 목표에게 다시 갑니다')
            self._release()
            self.phase = "mapwait"
            self.wait_frames = max(20, NAV_MAP_WAIT_FRAMES // 3)
            return
        self._talk_frames += 1
        if self._talk_frames >= self.TALK_TIMEOUT_FRAMES:
            print("[nav] 대화 타임아웃 — 종료")
            self._finish(True)
            return

        active = is_dialogue_active(self.pyboy)
        if active:
            self._talk_seen = True
            self._talk_idle = 0
        else:
            self._talk_idle += 1
            if self._talk_seen and self._talk_idle >= self.TALK_IDLE_DONE_FRAMES:
                print("[nav] 텍스트박스 종료 확인")
                self.talked_to_target = True
                self._finish(True)
                return
            if (
                not self._talk_seen
                and self._talk_a_count >= self.TALK_START_MAX_A
                and self._talk_idle >= self.TALK_IDLE_DONE_FRAMES
            ):
                print("[nav] 대화가 시작되지 않음 — 종료")
                self._finish(True)
                return

        # 대화 중이면 A 계속 / 아직 시작 전이면 시작용 A만 제한 횟수
        should_press = active or (
            not self._talk_seen and self._talk_a_count < self.TALK_START_MAX_A
        )
        if should_press:
            self._talk_a_count += 1
            self.button = "a"
            self.pending_action = "A"
            self.hold_frames = 0
            self.settle_frames = 0
            self.pyboy.button_press("a")
            self.phase = "talk_hold"

    def _talk_hold(self) -> None:
        self.hold_frames += 1
        if self.hold_frames >= self.A_HOLD_FRAMES:
            self._release()
            self.phase = "talk_settle"
            self.settle_frames = 0

    def _talk_settle(self) -> None:
        self.settle_frames += 1
        if self.settle_frames < self.A_SETTLE_FRAMES:
            return
        self.phase = "talk"


def find_move_learn_handoff(exc: BaseException):
    """예외 사슬에서 "레벨업 기술 교체" 위임을 찾는다."""
    current: Optional[BaseException] = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if getattr(current, "reason", None) == "move_learn_replace_slot" and all(
            hasattr(current, name)
            for name in (
                "party_index",
                "new_move_id",
                "current_move_ids",
                "selected_slot",
                "cursor_pair",
            )
        ):
            return current
        current = current.__cause__ or current.__context__
    return None


class BattleRunner:
    """BattleSkill이 PyBoy tick/input을 독점하는 동안의 실행기."""

    # 한 배틀에서 기술 교체를 이만큼까지만 자동 처리한다.
    MOVE_LEARN_MAX_RESOLVES = 4

    def __init__(
        self, pyboy, skill, *, move_learn=None, on_failure: str = "amash"
    ) -> None:
        self.pyboy = pyboy
        self.skill = skill
        # engine.move_learn.MoveLearnResolver (없으면 기존대로 수동 위임)
        self.move_learn = move_learn
        # PPO 가 실패했을 때: amash = A 연타로 끝낸다 / manual·disable = 사람에게
        self.on_failure = str(on_failure or "amash").strip().lower()
        self.finished = False
        self.success = False
        self.manual_handoff = False
        self.reason = ""
        self.turns = 0
        self._move_learn_resolves = 0
        # PP 때문에 진행이 멈췄는지 세는 카운터 (battle.pp_stuck_limit)
        self._pp_stall = 0
        self._pp_last_turns = -1
        self._pp_nudges = 0
        self._pp_limit = max(1, int(battle_opt("pp_stuck_limit", 100)))
        # A 연타 폴백 상태
        self._amash = False
        self._amash_frames = 0
        self._amash_reason = ""
        self._amash_a_t = 0
        self._amash_btn: Optional[str] = None
        # Battle Skill 내부 print(턴마다 나오는 action 로그 등)를 보여 줄지.
        self._verbose_skill = bool(battle_opt("debug", False)) or bool(
            battle_opt("macro_debug", False)
        )

    @contextmanager
    def _quiet_skill(self):
        """Battle Skill 이 stdout 에 찍는 것을 삼킨다."""
        if self._verbose_skill:
            yield
            return
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                yield
        except BaseException:
            text = buf.getvalue().strip()
            if text:
                print("[battle] 스킬 출력:")
                for line in text.splitlines():
                    print(f"  {line}")
            raise

    def owns_emulator_ticks(self) -> bool:
        if self._amash:
            # A 연타는 메인 루프의 pyboy.tick() 에 얹는다.
            return False
        return bool(
            not self.finished
            and getattr(self.skill, "active", False)
        )

    def start(self) -> bool:
        self.finished = False
        self.success = False
        self.manual_handoff = False
        self.reason = ""
        self.turns = 0
        self._move_learn_resolves = 0
        self._pp_stall = 0
        self._pp_last_turns = -1
        self._pp_nudges = 0
        print("[battle] Battle Skill 발동")
        started = self._start_skill()
        if started is None:
            # 폴백이 A 연타를 잡았으면 러너가 계속 제어권을 가진다.
            return self._amash

        if not started:
            if is_in_battle(self.pyboy):
                self._fallback(
                    RuntimeError(
                        getattr(self.skill, "reason", "battle start 실패")
                    )
                )
            else:
                self._finish(
                    True,
                    getattr(
                        self.skill,
                        "reason",
                        "battle_ended_during_prepare",
                    ),
                )
            return self._amash

        return True

    def tick(self) -> bool:
        """한 PPO 결정과 BattleMacro 턴 처리를 수행한다."""
        if self.finished:
            return False
        if self._amash:
            return self._tick_amash()
        # wIsInBattle==0xFF 는 전멸 후 블랙아웃 연출이지 전투가 아니다 (ram/wram.asm:1236-1240, home/overworld.asm:356).
        if not is_in_battle(self.pyboy):
            self._finish(True, "battle_end")
            return False

        # PPO 를 부르기 전에 PP 상태를 먼저 본다.
        if self._pp_guard():
            return False
        if self._amash:
            return self._tick_amash()

        try:
            with self._quiet_skill():
                result = self.skill.step()
        except Exception as exc:
            if self._resolve_move_learn(exc):
                return self._resume_after_move_learn()
            self._fallback(exc)
            return False

        self.turns = int(getattr(self.skill, "turns", self.turns))
        if not getattr(self.skill, "active", False):
            self._finish(
                not is_in_battle(self.pyboy),
                getattr(self.skill, "reason", "battle_end"),
            )
        return result is not None

    # ------------------------------------------------------------------
    # 레벨업 기술 교체
    # ------------------------------------------------------------------
    def _start_skill(self):
        """skill.start() 를 부른다."""
        for _ in range(self.MOVE_LEARN_MAX_RESOLVES + 1):
            try:
                with self._quiet_skill():
                    return bool(self.skill.start())
            except Exception as exc:
                if self._resolve_move_learn(exc):
                    if not is_in_battle(self.pyboy):
                        return False
                    continue
                self._fallback(exc)
                return None
        self._fallback(RuntimeError("기술 교체 처리가 반복 한도를 넘었습니다"))
        return None

    def _resume_after_move_learn(self) -> bool:
        """교체를 끝낸 뒤 배틀을 다시 PPO decision menu 까지 되돌린다."""
        if not is_in_battle(self.pyboy):
            self._finish(True, "battle_end")
            return False
        started = self._start_skill()
        if started is None:
            return False
        if not started:
            self._finish(
                not is_in_battle(self.pyboy),
                getattr(self.skill, "reason", "battle_end"),
            )
            return False
        return True

    def _resolve_move_learn(self, exc: BaseException) -> bool:
        if self.move_learn is None:
            return False
        if not getattr(self.move_learn, "enabled", False):
            return False
        handoff = find_move_learn_handoff(exc)
        if handoff is None:
            return False
        if self._move_learn_resolves >= self.MOVE_LEARN_MAX_RESOLVES:
            print("[move-learn] 한 배틀에서 너무 자주 발생 - 수동 위임")
            return False

        info = {
            "party_index": handoff.party_index,
            "new_move_id": handoff.new_move_id,
            "current_move_ids": handoff.current_move_ids,
            "selected_slot": handoff.selected_slot,
            "cursor_pair": handoff.cursor_pair,
        }
        try:
            ok = bool(self.move_learn.resolve(self.pyboy, info))
        except Exception as inner:
            print(
                "[move-learn] 처리 실패 - 수동 위임: "
                f"{type(inner).__name__}: {inner}"
            )
            return False
        if ok:
            self._move_learn_resolves += 1
        return ok

    def stop(self, reason: str = "사용자 중단") -> None:
        if self.finished:
            return
        self.skill.stop(reason)
        self._finish(False, reason)

    # A 연타 폴백 PP 0 기술을 계속 고르다 갇히는 것 방지
    def _battle_moves_pp(self) -> list:
        """싸우고 있는 포켓몬의 (기술 id, 남은 PP) 4개."""
        try:
            m = self.pyboy.memory
            return [
                (int(m[ADDR_BATTLE_MON_MOVES + i]), int(m[ADDR_BATTLE_MON_PP + i]) & 0x3F)
                for i in range(4)
            ]
        except Exception:
            return []

    def _pp_guard(self) -> bool:
        """PP 때문에 갇혔으면 개입한다."""
        moves = self._battle_moves_pp()
        known = [(i, mid, pp) for i, (mid, pp) in enumerate(moves) if mid]
        if not known:
            return False                      # 아직 못 읽었다 (전투 진입 직후 등)

        live = [i for i, _mid, pp in known if pp > 0]

        # 전 기술 PP 0 — PPO 는 둘 수 있는 수가 없다. 바로 A 연타로.
        if not live:
            if not self._amash:
                self._enter_amash("전 기술 PP 0 — 게임이 Struggle 을 고릅니다")
            return False

        # 턴이 넘어가고 있으면 정상이다.
        turns = int(getattr(self.skill, "turns", 0) or 0)
        if turns != self._pp_last_turns:
            self._pp_last_turns = turns
            self._pp_stall = 0
            self._pp_nudges = 0
            return False

        self._pp_stall += 1
        if self._pp_stall < self._pp_limit:
            return False

        # 여기까지 왔다 = 같은 턴에서 pp_stuck_limit 번 제자리.
        self._pp_stall = 0
        self._pp_nudges += 1
        target = max(live, key=lambda i: moves[i][1])   # PP 가 가장 많이 남은 기술
        cursor = int(self.pyboy.memory[ADDR_CURRENT_MENU_ITEM])
        pplist = [pp for _i, _m, pp in known]
        print(
            f"[battle] {self._pp_limit}회째 같은 턴 — PP={pplist} 중 "
            f"{target + 1}번 기술(PP {moves[target][1]})을 직접 고릅니다 "
            f"(이 전투에서 {self._pp_nudges}번째)"
        )
        ok = self._pick_move_slot(target)
        runlog.log(
            "battle_pp",
            조치="기술 직접 선택",
            커서=cursor + 1 if 0 <= cursor < 4 else None,
            고른_슬롯=target + 1,
            PP=pplist,
            성공=bool(ok),
            시도=self._pp_nudges,
        )
        if ok:
            return True

        # 커서가 안 움직였다 = 기술 메뉴가 아니다. 그때만 A 연타로 넘긴다.
        print("[battle] 기술 메뉴가 아닌 것 같습니다 — A 연타로 넘깁니다")
        if not self._amash:
            self._enter_amash(
                f"PP 문제로 {self._pp_limit}회 제자리 (PP="
                f"{[pp for _i, _m, pp in known]})"
            )
        return False

    def _pick_move_slot(self, target: int) -> bool:
        """기술 메뉴 커서를 ``target`` 으로 옮기고 A 를 누른다."""
        for _ in range(8):
            cur = int(self.pyboy.memory[ADDR_CURRENT_MENU_ITEM])
            if not 0 <= cur < 4:
                return False
            if cur == target:
                self._tap("a")
                return True
            self._tap("down" if cur < target else "up")
            if int(self.pyboy.memory[ADDR_CURRENT_MENU_ITEM]) == cur:
                return False        # 커서가 안 움직였다 = 기술 메뉴가 아니다
        return False

    def _tap(self, button: str, frames: int = 8) -> None:
        """버튼 한 번."""
        try:
            self.pyboy.button_press(button)
            for _ in range(max(1, int(frames))):
                self.pyboy.tick()
            self.pyboy.button_release(button)
            for _ in range(max(1, int(frames))):
                self.pyboy.tick()
        except Exception:
            pass

    def _fallback(self, exc: BaseException) -> None:
        reason = f"{type(exc).__name__}: {exc}"
        try:
            self.skill.stop(reason)
        except Exception:
            pass
        # 기술 교체 화면에서 온 실패는 A 연타로 풀 수 없다 — 여기서 A 는 "슬롯 0 삭제 확정"이라 누르면 안 되고, 안 누르면 화면에서 못 나간다.
        if find_move_learn_handoff(exc) is not None:
            self._to_manual(reason)
            return
        if self.on_failure == "amash" and is_in_battle(self.pyboy):
            self._enter_amash(reason)
            return
        self._to_manual(reason)

    def _to_manual(self, reason: str) -> None:
        self.manual_handoff = True
        self.reason = str(reason)
        self.finished = True
        self.success = False
        print(f"[battle] 자동 처리 실패 - 수동 조작으로 전환: {self.reason}")

    def _enter_amash(self, reason: str) -> None:
        """PPO 가 둘 수 없는 상황을 A 연타로 빠져나간다."""
        self._amash = True
        self._amash_reason = reason
        self._amash_frames = 0
        self._amash_a_t = 0
        print(f"[battle] PPO 실패 - A 연타로 마무리합니다: {reason}")
        runlog.log("battle_fallback", 방식="A 연타", 사유=reason)

    def _amash_release(self) -> None:
        if self._amash_btn is not None:
            try:
                self.pyboy.button_release(self._amash_btn)
            except Exception:
                pass
            self._amash_btn = None

    def _tick_amash(self) -> bool:
        if not is_in_battle(self.pyboy):
            self._amash_release()
            self._amash = False
            self._finish(True, f"a_mash_end ({self._amash_reason})")
            return False

        self._amash_frames += 1
        if self._amash_frames > BATTLE_AMASH_MAX_FRAMES:
            self._amash_release()
            self._amash = False
            self._to_manual(
                f"A 연타 {BATTLE_AMASH_MAX_FRAMES}프레임 초과 "
                f"({self._amash_reason})"
            )
            return False

        # 레벨업 기술 교체 메뉴에서 A 한 번 = "슬롯 0 삭제 확정" 이다 (learn_move.asm:141-142 가 wCurrentMenuItem 을 0 으로 초기화한다).
        info = move_learn_menu_open(self.pyboy)
        if info is not None:
            self._amash_release()
            self._amash_a_t = 0
            resolver = self.move_learn or default_resolver()
            if resolver is not None and getattr(resolver, "enabled", False):
                try:
                    resolver.resolve(self.pyboy, info)
                except Exception as exc:
                    print(
                        "[battle] 기술 교체 처리 실패: "
                        f"{type(exc).__name__}: {exc}"
                    )
            else:
                # 지울 기술을 고를 방법이 없다.
                self._amash = False
                self._to_manual(
                    "레벨업 기술 교체 화면 - 지울 기술을 직접 고르세요 "
                    f"({self._amash_reason})"
                )
            return False

        self._amash_a_t += 1
        if self._amash_a_t == 1:
            try:
                self.pyboy.button_press("a")
                self._amash_btn = "a"
            except Exception:
                pass
        elif self._amash_a_t == BATTLE_AMASH_A_HOLD:
            self._amash_release()
        elif self._amash_a_t >= BATTLE_AMASH_A_HOLD + BATTLE_AMASH_A_GAP:
            self._amash_a_t = 0
        return False

    def _finish(self, success: bool, reason: str) -> None:
        self._amash_release()
        self._amash = False
        self.finished = True
        self.success = bool(success)
        self.reason = str(reason)
        self.turns = int(getattr(self.skill, "turns", self.turns))
        print(
            f"[battle] 종료 success={self.success} "
            f"turns={self.turns} reason={self.reason}"
        )


class ExploreRunner:
    """ExplorationSkill 을 프레임 단위로 굴리는 실행기."""

    # 이 값 이상이면 "상한 없음"으로 본다 (scripts/play.py 가 넘긴다).
    NO_LIMIT_STEPS = 1_000_000_000

    def __init__(
        self,
        gmap: GlobalMap,
        pyboy,
        skill,
        *,
        observed=None,
    ) -> None:
        self.gmap = gmap
        self.pyboy = pyboy
        self.skill = skill
        # RL 이 돌리는 프레임마다 GlobalMap 을 관측하는 프록시.
        self._observed = (
            observed if observed is not None else ObservingPyBoy(pyboy, gmap)
        )
        self.phase = "plan"
        self.button: Optional[str] = None
        self.hold_frames = 0
        self.settle_frames = 0
        self.finished = False
        self.success = False
        # 스킬이 스스로 목표를 이뤘는가 (SeekWarp: 워프 통과).
        self.achieved = False
        self._paused_for_battle = False
        # 이번 대화를 스킬에 이미 알렸는가 (대화당 한 번만 알린다)
        self._dialogue_told = False
        # 탐험을 시작한 맵. 이 맵이 최종 종료 조건에 닿으면 에피소드를 끊는다.
        self.start_map: Optional[int] = None
        self._close_counter = 0
        # 탐험 중 끼어든 스크립트 대화 → A 로 넘기고 끝나면 그대로 재개 탐험 중에는 대화가 수십 번 걸린다.
        self._dialogue = DialogueSkipper(
            pyboy, tag="explore", on_start=self._release, verbose=False
        )

    def owns_emulator_ticks(self) -> bool:
        """RL pulse 는 run_action_on_emulator 안에서 tick → 메인 루프 tick 스킵."""
        if self.finished or self._paused_for_battle:
            return False
        if int(self.pyboy.memory[ADDR_IN_BATTLE]) != 0:
            return False
        if self._dialogue.pending():
            return False
        return bool(getattr(self.skill, "pulse_action", False)) and callable(
            getattr(self.skill, "run_action_on_emulator", None)
        )

    def start(self, map_id: int, *, max_steps: Optional[int] = None) -> bool:
        """탐험 시작."""
        self.finished = False
        self.success = False
        self.start_map = int(map_id)
        self._close_counter = 0
        self._paused_for_battle = False
        self._dialogue_told = False
        self._dialogue.reset()
        self.phase = "plan"
        # 관측 프록시는 __init__ 에서 만든 것을 재사용한다 (같은 객체여야 bind_pyboy 가 재등록을 건너뛴다).
        if not self.skill.bind_pyboy(self._observed):
            self.finished = True
            return False
        started = (
            self.skill.start(map_id, max_steps=int(max_steps))
            if max_steps is not None
            else self.skill.start(map_id)
        )
        if not started:
            self.finished = True
            return False
        return True

    def stop(self, reason: str = "사용자 중단") -> None:
        if self.finished:
            return
        self._dialogue.release()
        self._release()
        self.skill.stop(reason)
        self.finished = True
        self.success = False

    def _finish(self, success: bool) -> None:
        self._dialogue.release()
        self._release()
        if self.skill.active:
            self.skill.stop("완료" if success else "실패")
        self.finished = True
        self.success = success
        self.achieved = bool(getattr(self.skill, "achieved", success))

    def _release(self) -> None:
        release_button(self.pyboy, self.button)
        self.button = None

    def tick(self, cur_map: int, cur_x: int, cur_y: int) -> bool:
        """True면 한 액션 사이클 완료."""
        if self.finished:
            return False

        if int(self.pyboy.memory[ADDR_IN_BATTLE]) != 0:
            if not self._paused_for_battle:
                self._paused_for_battle = True
                self._release()
                self._notify_skill("battle")
                print("[explore] 배틀 감지 — Exploration 일시정지")
            return False

        if self._paused_for_battle:
            self._paused_for_battle = False
            self._release()
            self.phase = "plan"
            print("[explore] 배틀 종료 — Exploration 재개")
            return False

        # 스크립트 대화: A 로 넘기는 동안 RL 액션을 멈췄다가 끝나면 그대로 재개.
        if self._dialogue.busy():
            # **대화가 시작될 때 한 번만** 알린다. 매 프레임 알리면 스킬 쪽에서
            # 같은 대화를 수백 번 센다 (로그도 그만큼 찍힌다).
            if not self._dialogue_told:
                self._dialogue_told = True
                self._notify_skill("dialogue")
            return False
        if self._dialogue.take_finished():
            self._dialogue_told = False
            self._release()
            self.phase = "plan"
            return False

        # RL pulse 전용 (휴리스틱 hold 경로 없음)
        if not self.owns_emulator_ticks():
            print("[explore] RL pulse 불가 — Exploration 중단")
            self._finish(False)
            return False

        action = self.skill.propose_action(
            self.gmap, cur_map, (cur_x, cur_y)
        )
        if action is None:
            self._finish(True)
            return False
        before = (int(cur_map), int(cur_x), int(cur_y))
        self.skill.run_action_on_emulator(action)
        self._try_hm_after(before, action)
        self._check_closed(cur_map, cur_x, cur_y)
        return True

    def _try_hm_after(self, before: Tuple[int, int, int], action: str) -> None:
        """제자리였고 앞이 나무·물이면 비전머신을 쓴다 (Navigation 과 같은 처리)."""
        button = _DIR_BUTTON.get(str(action))
        if button is None:
            return
        m = self.pyboy.memory
        now = (int(m[ADDR_MAP_ID]), int(m[ADDR_PLAYER_X]), int(m[ADDR_PLAYER_Y]))
        if now != before:
            return                      # 움직였다 = 막힌 게 아니다
        try:
            used = hm.use_for_direction(self.pyboy, button)
        except Exception as exc:
            print(f"[explore] 비전머신 사용 실패: {type(exc).__name__}: {exc}")
            return
        if not used:
            return
        print(f"[explore] {'풀베기' if used == 'cut' else '파도타기'} 사용 — 길이 열렸습니다")
        runlog.log("hm", 사용=used, 어디="explore", 맵=self.gmap.map_name(before[0]))
        local = self.gmap.get_local_map(before[0])
        if local is not None:
            local.closed = False        # 지형이 바뀌었다 — 다 봤다는 표시를 지운다

    def _notify_skill(self, kind: str) -> None:
        """배틀·대화로 이동이 끊겼다고 스킬에 알린다 (구현한 스킬만)."""
        notify = getattr(self.skill, "notify_interrupt", None)
        if callable(notify):
            try:
                notify(kind)
            except Exception:
                pass

    def _check_closed(self, cur_map: int, cur_x: int, cur_y: int) -> None:
        """시작한 맵이 최종 종료 조건에 닿았으면 에피소드를 끊는다."""
        if self.finished or self.start_map is None:
            return
        # SeekWarp 처럼 "통과"가 목적인 스킬은 종료 조건이 성립해도 끝낼 이유가 없다.
        if getattr(self.skill, "ignore_closure", False):
            return
        if not explore_gate.stop_on_closed():
            return
        every = explore_gate.check_every()
        self._close_counter += 1
        if self._close_counter < every:
            return
        self._close_counter = 0
        # 시작한 맵을 떠나 있으면 그 맵 안의 밟은 칸을 시드로 판정한다.
        pos = (int(cur_x), int(cur_y)) if int(cur_map) == self.start_map else None
        try:
            # 이 맵만 본다(recurse=False).
            done, why = explore_gate.map_done(
                self.gmap, self.pyboy, self.start_map, pos, recurse=False
            )
        except Exception as exc:
            print(f"[explore] 닫힘 판정 실패: {type(exc).__name__}: {exc}")
            return
        if not done:
            return
        name = self.gmap.map_name(self.start_map)
        steps = int(getattr(self.skill, "steps", 0) or 0)
        limit = int(getattr(self.skill, "max_steps", 0) or 0)
        # 콘솔 explore 는 상한이 사실상 없다 — 큰 숫자를 그대로 찍으면 헷갈린다.
        budget = (
            f"{steps} / {limit} (max_steps 를 다 쓰지 않고 중단)"
            if limit < self.NO_LIMIT_STEPS
            else f"{steps} (상한 없음 — stop 전에 최종 종료 조건으로 끝남)"
        )
        print(
            f'[explore] "{name}" 최종 종료 조건 충족 — 탐험을 끝냅니다\n'
            f"          스텝  : {budget}\n"
            f"          이유  : {why}"
        )
        runlog.log(
            "explore_end",
            맵=name,
            map_id=self.start_map,
            스텝=steps,
            max_steps=limit if limit < self.NO_LIMIT_STEPS else None,
            사유=why,
            조기종료=True,
        )
        self._finish(True)

    def pause_for_battle(self) -> None:
        if self.finished or self._paused_for_battle:
            return
        self._paused_for_battle = True
        self._release()
        print("[explore] 배틀 감지 — Exploration 일시정지")

