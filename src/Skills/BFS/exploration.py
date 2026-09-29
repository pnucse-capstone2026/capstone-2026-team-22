"""BFS — 규칙 기반 탐험."""

from __future__ import annotations

import contextlib
import io
from collections import deque
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from core import mapstore, runlog
from core.closure import (
    GATE_HM_NAMES,
    ClosureResult,
    closure_for,
    gate_label,
    is_exit_free_map,
    pokecenter_done,
)
from core.gb_memory import ADDR_IN_BATTLE, ADDR_MAP_ID, ADDR_PLAYER_X, ADDR_PLAYER_Y, ADDR_TILESET
from core.map_types import BlockType, EntityType
from core.rl_action import ACTION_NAME_TO_INDEX, DEFAULT_ACTION_FREQ, execute_rl_action
from core.closure import warp_dest
from core.warp_targets import seek_targets, unlinked_exits

Coord = Tuple[int, int]
FailKey = Tuple[int, str, Coord]

_DIRS: Tuple[Tuple[str, Coord], ...] = (
    ("DOWN", (0, 1)), ("UP", (0, -1)), ("LEFT", (-1, 0)), ("RIGHT", (1, 0)),
)
_MOVES = {"UP", "DOWN", "LEFT", "RIGHT"}
_DELTA = dict(_DIRS)
_LEDGES = {BlockType.LEDGE_U, BlockType.LEDGE_D, BlockType.LEDGE_L, BlockType.LEDGE_R}
WAIT = "WAIT"                 # 맵 전환 직후 LocalMap 이 아직 없을 때 프레임만 돌린다
_MAX_WAIT = 12
_INTERRUPT_LIMIT = 4          # 가는 도중 끊긴 횟수가 이만큼이면 그 목표를 실패로 한 번 센다
_SETTLE_ACTIONS = 4
_ARRIVE_WAITS = 2             # 맵이 바뀐 직후 먼저 기다릴 횟수 (화면 전환 중 입력이 먹힌다)
_ARRIVE_GRACE = 2             # 맵이 바뀐 뒤 이만큼의 이동은 어긋나도 막힌 칸으로 세지 않는다
_TRY_PER_PLAN = 24
_TRAVEL_SCAN = 40             # 할 일이 남은 맵을 찾을 때 볼 맵 수 상한
_TRAVEL_REVISIT = 3           # 같은 목적지로 가면서 같은 맵을 이만큼 다시 밟으면 포기
_STUCK_WINDOW = 10            # 최근 이만큼의 걸음(두 칸 왕복 5번)을 보고 제자리인지 판단한다
_STUCK_CELLS = 2              # 그 사이 밟은 칸이 이만큼 이하면 제자리 왕복이다
_STUCK_ROUNDS = 3             # 제자리 왕복이 이만큼 반복되면 이번 탐험을 끝낸다
# 마을·도시 맵 id.
_CITY_MAP_IDS = frozenset(range(11))


# ---------------------------------------------------------------------------
# 순수 함수 (테스트 대상)
# ---------------------------------------------------------------------------
def bfs_distances(reachable: Iterable[Coord], start: Coord) -> Dict[Coord, int]:
    """도달 영역 안에서 start 로부터의 걸음 수 (상하좌우)."""
    cells = set(reachable)
    start = (int(start[0]), int(start[1]))
    if start not in cells:
        return {}
    dist = {start: 0}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for _, (dx, dy) in _DIRS:
            nxt = (cur[0] + dx, cur[1] + dy)
            if nxt in cells and nxt not in dist:
                dist[nxt] = dist[cur] + 1
                queue.append(nxt)
    return dist


def _gate_note(res: ClosureResult) -> str:
    """경계를 막은 것이 물/나무면 " (나무에 막혀 있습니다 — 풀베기)" 같은 꼬리말."""
    label = gate_label(getattr(res, "gated_by", ()) or ())
    if not label:
        return ""
    hms = " · ".join(
        GATE_HM_NAMES[g] for g in ("water", "cut_tree", "snorlax", "silph_scope") if g in (res.gated_by or ())
    )
    return f" ({label}에 막혀 있습니다 — {hms}가 있어야 넘어갑니다)"


def frontier_stands(
    res: ClosureResult,
    pos: Coord,
    *,
    skip: Callable[[Coord], bool] = lambda _c: False,
    blocked: Optional[Set[Coord]] = None,
    extra: Iterable[Coord] = (),
) -> List[Tuple[Coord, List[Coord]]]:
    """미관측 경계를 보러 설 칸들."""
    dist = bfs_distances(res.reachable, pos)
    if blocked:
        open_dist = bfs_distances(set(res.reachable) - set(blocked), pos)
    else:
        open_dist = dist
    stands: Dict[Coord, List[Coord]] = {}
    for f in set(res.frontier) | set(extra):
        if skip(f):
            continue
        for _, (dx, dy) in _DIRS:
            s = (f[0] - dx, f[1] - dy)
            if s not in res.reachable:
                continue
            if blocked and s in dist and s not in open_dist:
                continue
            stands.setdefault(s, []).append(f)

    def key(s: Coord):
        far = dist.get(s)
        manhattan = abs(s[0] - pos[0]) + abs(s[1] - pos[1])
        return (0, far, s) if far is not None else (1, manhattan, s)

    return [(s, sorted(stands[s])) for s in sorted(stands, key=key)]


def unseen_connections(gmap: Any, map_id: int, res: ClosureResult) -> List[Coord]:
    """warps.json 이 아는 **맵 밖 출구** 중 아직 화면에 안 들어온 칸 (도달 영역에 붙은 것만)."""
    static = (getattr(gmap, "warp_at", None) or {}).get(int(map_id)) or {}
    local = gmap.get_local_map(int(map_id))
    grid = getattr(local, "grid", None) or {}
    size = (getattr(gmap, "map_size_cells", None) or {}).get(int(map_id))
    graph = getattr(gmap, "graph", None)
    out: List[Coord] = []
    for coord, point in static.items():
        if coord in grid:
            continue
        if size and 0 <= coord[0] < size[0] and 0 <= coord[1] < size[1]:
            continue                                   # 맵 안 문은 화면으로 보인다
        if str(point.get("kind", "")) != "connection":
            dest = warp_dest(gmap, map_id, point)
            if dest is None:
                continue
            if graph is not None and (graph.has_edge(int(map_id), dest) or graph.has_edge(dest, int(map_id))):
                continue
        if any((coord[0] - dx, coord[1] - dy) in res.reachable for _, (dx, dy) in _DIRS):
            out.append(tuple(coord))
    return sorted(out)


def walk_cells(grid: Dict[Coord, Any], pos: Coord, actions: Iterable[str]) -> List[Coord]:
    """액션을 따라가면 차례로 서게 될 칸."""
    out: List[Coord] = []
    cur = (int(pos[0]), int(pos[1]))
    for action in actions:
        d = _DELTA.get(action)
        if d is None:
            continue
        cur = step_target(grid, cur, action)
        out.append(cur)
    return out


def step_target(grid: Dict[Coord, Any], pos: Coord, action: str) -> Coord:
    dx, dy = _DELTA[action]
    nxt = (pos[0] + dx, pos[1] + dy)
    cell = grid.get(nxt)
    if cell is not None and cell.block_type in _LEDGES:
        nxt = (nxt[0] + dx, nxt[1] + dy)
    return nxt


def seed_cell(local: Any, size: Optional[Coord]) -> Optional[Coord]:
    """현재 위치가 아닌 맵을 판정할 때 BFS 를 시작할 칸 — 밟아 본 칸 하나."""
    grid = getattr(local, "grid", None) or {}
    fallback = None
    for c, cell in sorted(grid.items()):
        if size and not (0 <= c[0] < size[0] and 0 <= c[1] < size[1]):
            continue
        if getattr(cell, "visited", False):
            return c
        if fallback is None and cell.block_type is None and not getattr(cell, "warp_candidate", False):
            fallback = c
    return fallback


# ---------------------------------------------------------------------------
# 스킬
# ---------------------------------------------------------------------------
class BFSExplorationSkill:
    """ExploreRunner 계약 구현."""

    pulse_action = True
    # 폐영역이 성립해도 러너가 끊지 않는다 — 출구 통과 · 다른 맵 이동이 남아 있다
    ignore_closure = True

    def __init__(
        self,
        *,
        device: str = "cpu",
        greedy: bool = False,
        cheat_profile: str = "off",
        model_path: Any = None,
        max_steps: Optional[int] = None,
        action_freq: int = DEFAULT_ACTION_FREQ,
        capabilities: Optional[Callable[[], Tuple[bool, bool]]] = None,
        cross_warps: bool = True,
        travel: bool = True,
        fail_limit: int = 2,
        travel_max_hops: int = 2,
        home_map_only: bool = False,
        free_roam_tilesets: Iterable[int] = (),
        tileset: Optional[Callable[[], Optional[int]]] = None,
        verbose: bool = True,
    ) -> None:
        # device/greedy/model_path 는 factory 가 공통으로 넘긴다 — 규칙 기반이라 안 쓴다
        self.max_steps = int(max_steps or 900)
        self.action_freq = int(action_freq)
        self._capabilities = capabilities
        self.cross_warps = bool(cross_warps)
        self.travel = bool(travel)
        # True 면 **스킬이 시작된 맵**만 넓힌다 (exploration.bfs_home_map_only).
        self.home_map_only = bool(home_map_only)
        self._home: Optional[int] = None
        # home_map_only 의 **예외 타일셋** (exploration.bfs_free_roam_tilesets, 17 = 동굴).
        self.free_roam_tilesets = frozenset(int(t) for t in (free_roam_tilesets or ()))
        # 현재 타일셋 조회 (play.py 가 넣어 준다). 없으면 붙어 있는 pyboy 의 RAM 을 읽는다.
        self._tileset = tileset
        # 이번 에피소드가 예외(묶지 않고 넓히기)로 돌고 있는가. start() 가 정한다.
        self._free_roam = False
        self.fail_limit = max(1, int(fail_limit))
        # 할 일이 남은 다른 맵으로 옮겨 갈 때 **이만큼 떨어진 맵까지만** 본다 (GlobalMap 홉).
        self.travel_max_hops = max(1, int(travel_max_hops))
        self._fail_why = ""          # 다음 _fail 기록에 붙일 이유 (로그용)
        self._battle_cut = False     # 지금 목표가 전투로 끊겼는가 (실패로 세지 않는다)
        self.verbose = bool(verbose)

        self._pyboy: Any = None
        self._nav: Any = None
        self.active = False
        self.steps = 0
        self.achieved = False
        self.reason = ""

        # 세션 동안 유지 (start 에서 지우지 않는다)
        self._fails: Dict[FailKey, int] = {}
        self._fails_caps: Optional[Tuple[bool, bool]] = None
        self._block_hits: Dict[Tuple[int, Coord], int] = {}
        # 목표로 가는 도중 끊긴 횟수 (배틀 뒤 밀림 · NPC 와 부딪힘). 목표 실패와 따로 센다
        self._interrupts: Dict[FailKey, int] = {}
        # 이동 중 (목적지, 지나간 맵) 방문 횟수. 같은 맵을 되밟으면 왕복하는 중이다.
        self._travel_visits: Dict[Tuple[int, int], int] = {}
        # 최근 밟은 칸 (제자리 왕복 감지용)
        self._recent: deque = deque(maxlen=_STUCK_WINDOW)
        self._recent_goals: deque = deque(maxlen=_STUCK_WINDOW)
        self._stuck_keys: Set[FailKey] = set()
        self._stuck_rounds = 0
        # 직전 경로 계획이 "막힌 칸을 지나야 해서" 버려졌는가 (A* 실패와 구분한다)
        self._blocked_reject = False
        # 직전 액션이 스크립트에 걸렸는가 (트레이너가 "!" 를 띄우고 걸어옴 · 배틀 · 대화).
        self._scripted_seen = False
        # 직전 대화가 시작된 (맵, 서 있던 자리, 가려던 칸). 같은 것이 또 나오면 막힌 것이다.
        self._dialogue_spot: Optional[Tuple[int, Coord, Coord]] = None
        # 지금 대화가 열려 있는가.
        self._dialogue_open = False
        # 이번 목표로 가는 동안 대화가 떴는가 (문지기에게 막히는 경우)
        self._dialogue_on_the_way = False
        # 제자리 실패 뒤 같은 입력을 한 번 더 넣어 봤는가 (방향 전환 한 번)
        self._retried = False

        # 에피소드 상태
        self._cur_map: Optional[int] = None
        self._maps_before: Optional[Set[int]] = None
        self.new_maps: List[int] = []
        self._queue: List[str] = []
        self._pending: Optional[Tuple[str, int, Any]] = None
        # 이번 목표로 가려고 **서려던 칸**. 거기 섰는지로 "가서 해 봤다" 를 가른다.
        self._pending_goal: Optional[Coord] = None
        self._last_pos: Optional[Coord] = None
        self._last_action: Optional[str] = None
        self._expect: Optional[Coord] = None
        self._interrupted = False
        self._waits = 0
        self._since_change = _SETTLE_ACTIONS
        self._arrive_waits = 0
        self._moves_since_change = _ARRIVE_GRACE
        self.counts: Dict[str, int] = {}

    # -- 계약 ---------------------------------------------------------------
    def bind_pyboy(self, pyboy: Any) -> bool:
        self._pyboy = pyboy
        return True

    def retry_unfinished(self, gmap: Any) -> None:
        """아직 안 이어진 출구가 남은 맵은 접어 뒀던 경계를 다시 시도한다."""
        graph = getattr(gmap, "graph", None)
        if graph is None:
            return
        for mid in [int(n) for n in graph.nodes]:
            if not self.unreached_exits(gmap, mid):
                continue
            stale = [k for k in self._fails
                     if k[0] == mid and k[1] == "f" and k not in self._stuck_keys]
            for key in stale:
                del self._fails[key]
                self._interrupts.pop(key, None)

    def start(self, map_id: int, *, max_steps: Optional[int] = None) -> bool:
        if self._pyboy is None:
            return False
        if max_steps is not None:
            self.max_steps = int(max_steps)
        self.active = True
        self.steps = 0
        self.achieved = False
        self.reason = ""
        self._cur_map = int(map_id)
        self._home = int(map_id)
        self._free_roam = self._free_roam_here()
        if self._free_roam:
            self._say("동굴에서 시작 — 시작한 맵에 묶지 않고 넓힙니다 (마을에 닿으면 끝)")
        self._maps_before = None
        self.new_maps = []
        self._queue = []
        self._pending = None
        self._last_pos = None
        self._last_action = None
        self._expect = None
        self._interrupted = False
        self._waits = 0
        self._since_change = _SETTLE_ACTIONS
        self._arrive_waits = 0
        self._moves_since_change = _ARRIVE_GRACE
        self._travel_visits = {}
        self._recent = deque(maxlen=_STUCK_WINDOW)
        self._recent_goals = deque(maxlen=_STUCK_WINDOW)
        self._stuck_rounds = 0
        self.counts = {"경계": 0, "출구": 0, "이동": 0}
        return True

    def stop(self, reason: str = "중단") -> None:
        if not self.active:
            return
        self.active = False
        summary = (
            f"경계 {self.counts.get('경계', 0)} · 출구 {self.counts.get('출구', 0)} · "
            f"이동 {self.counts.get('이동', 0)} · 새 맵 {len(self.new_maps)}"
        )
        self.reason = f"{reason} ({summary})"
        self._say(f"종료 — {self.reason}")

    def run_action_on_emulator(self, action_name: str) -> None:
        if action_name == WAIT:
            for _ in range(max(1, self.action_freq)):
                self._pyboy.tick(1, render=True)
            return
        if action_name == "A":
            # 길을 막은 몬스터볼 줍기. 대화창("…을 주웠다!")은 러너가 넘긴다.
            self._pyboy.button("a", 8)
            for _ in range(max(1, self.action_freq)):
                self._pyboy.tick(1, render=True)
            return
        execute_rl_action(self._pyboy, action_name, action_freq=self.action_freq, realtime=True)
        if self._scripted():
            self._scripted_seen = True

    def notify_interrupt(self, kind: str = "") -> None:
        """러너가 배틀·대화로 이동을 끊었다고 알려 준다 (engine.ExploreRunner)."""
        self._scripted_seen = True
        if str(kind) != "dialogue":
            # 전투다 — 이어지던 대화 기록을 지운다 (트레이너는 막은 것이 아니다)
            self._dialogue_spot = None
            self._dialogue_open = False
            # **전투로 끊긴 것은 "못 가서" 가 아니다.** 동굴은 몇 걸음마다 야생 전투가
            # 나서, 이걸 끊김으로 세면 멀리 있는 경계가 싸우다 지쳐 접힌다
            # (돌산터널 남쪽 출구 앞 경계가 그렇게 빠졌다).
            self._battle_cut = True
            return
        if self._dialogue_open:
            return                            # 같은 대화의 두 번째 프레임부터는 무시
        self._dialogue_open = True
        self._dialogue_on_the_way = True
        here = self._player_spot()
        if here is None:
            return
        map_id, pos = here
        target = self._expect or (
            step_target({}, pos, self._last_action)
            if self._last_action in _MOVES else None
        )
        if target is None:
            return
        spot = (map_id, pos, tuple(target))
        key = (map_id, tuple(target))
        if self._dialogue_spot != spot:
            # 처음 본 자리 — 한 번으로는 막혔다고 할 수 없다 (그냥 말을 건 대화일 수 있다)
            self._dialogue_spot = spot
            self._block_hits[key] = max(1, self._block_hits.get(key, 0))
            return
        self._block_hits[key] = self._block_hits.get(key, 0) + 1
        if self._block_hits[key] >= self.fail_limit:
            self._say(f"{target} 로 가려 할 때마다 대화가 떠서 못 지나갑니다 — 막힌 칸으로 둡니다")

    def _player_spot(self) -> Optional[Tuple[int, Coord]]:
        """지금 (맵, 플레이어 좌표)."""
        try:
            m = self._pyboy.memory
            return int(m[ADDR_MAP_ID]) & 0xFF, (int(m[ADDR_PLAYER_X]), int(m[ADDR_PLAYER_Y]))
        except Exception:
            return None

    def _scripted(self) -> bool:
        """지금 게임이 스크립트에 붙들려 있는가 (배틀 · 대화 · 트레이너가 걸어오는 중)."""
        try:
            if int(self._pyboy.memory[ADDR_IN_BATTLE]) != 0:
                return True
        except Exception:
            return False
        try:
            addr = self._pyboy.symbol_lookup("wJoyIgnore")[1]
            return int(self._pyboy.memory[addr]) != 0
        except Exception:
            return False

    def memory_snapshot(self, gmap: Any) -> Dict[str, Dict[str, Any]]:
        """지금 기억하고 있는 "못 가는 곳" 을 맵 이름별로 정리한다."""
        out: Dict[str, Dict[str, Any]] = {}

        def slot(mid: int) -> Dict[str, Any]:
            name = gmap.map_name(int(mid)) if hasattr(gmap, "map_name") else str(mid)
            info = out.setdefault(str(name), {
                "map_id": int(mid),
                "막힌_칸": [],
                "포기한_경계": [],
                "포기한_출구": [],
                "포기한_이동": [],
            })
            return info

        def spot(coord: Coord) -> str:
            return f"{int(coord[0])},{int(coord[1])}"      # 한 줄로 읽히게

        for (mid, coord), hits in sorted(self._block_hits.items()):
            if hits >= self.fail_limit:
                slot(mid)["막힌_칸"].append(spot(coord))
        for (mid, kind, coord), hits in sorted(self._fails.items()):
            if hits < self.fail_limit:
                continue
            info = slot(mid)
            if kind == "f":
                info["포기한_경계"].append(spot(coord))
            elif kind == "w":
                info["포기한_출구"].append(spot(coord))
            else:
                info["포기한_이동"].append(
                    gmap.map_name(int(mid)) if hasattr(gmap, "map_name") else str(mid)
                )
        return out

    def dump_memory(self, gmap: Any) -> None:
        """위 스냅샷을 runtime/current_blocked.json 에 덮어쓴다."""
        try:
            mapstore.save_blocked(self.memory_snapshot(gmap))
        except Exception:
            pass                       # 기록 실패가 실행을 막지는 않는다

    def unreached_exits(self, gmap: Any, map_id: int) -> List[Coord]:
        """아직 안 이어진 출구 중 **포기하지 않은** 것."""
        if is_exit_free_map(gmap.map_name(int(map_id)) if hasattr(gmap, "map_name") else ""):
            return []             # 포켓몬센터는 출구를 안 따진다
        try:
            got = unlinked_exits(gmap, int(map_id))
        except Exception:
            return []
        return [c for c in got if not self._failed((int(map_id), "w", c))]

    def forget_blocks(self) -> None:
        """막힌 칸 · 실패한 목표 기억을 지운다."""
        if self._block_hits or self._fails:
            self._say("새 이벤트 — 막혔던 길·목표를 다시 시도합니다")
        self._block_hits.clear()
        self._fails.clear()
        self._stuck_keys.clear()
        self._interrupts.clear()
        if self._nav is not None and hasattr(self._nav, "forget_object_tries"):
            self._nav.forget_object_tries()    # 비키지 않던 오브젝트도 다시 시도

    def refresh_map_status(self, gmap: Any, cur_map: Optional[int] = None, pos: Optional[Coord] = None) -> None:
        """LLM 에 보여 줄 맵 상태를 **이 스킬의 판단**에 맞춘다 (계획 직전에 play.py 가 부른다)."""
        graph = getattr(gmap, "graph", None)
        if graph is None:
            return
        caps = self._caps()
        self._sync_fail_memory(caps)
        sizes = getattr(gmap, "map_size_cells", {}) or {}
        for mid in [int(n) for n in graph.nodes]:
            local = gmap.get_local_map(mid)
            if local is None:
                continue
            size = sizes.get(mid)
            seed = None
            if cur_map is not None and mid == int(cur_map) and pos is not None:
                p = (int(pos[0]), int(pos[1]))
                if not size or (0 <= p[0] < size[0] and 0 <= p[1] < size[1]):
                    seed = p
            if seed is None:
                seed = seed_cell(local, size)
            if seed is None:
                local.explore_exhausted = False
                continue
            res = None
            try:
                work, res = self._local_work_res(gmap, mid, seed, caps)
                # 갈 수 있는 곳을 다 봤고 **경계가 물·나무·잠만보·유령 같은 조건으로 막혀 있으면**, 남은 출구는 그 너머다 — 할 일로 세면 그 맵이 영영 "미완료" 다.
                gated = res is not None and bool(res.gated_by)
                if not work and not gated:
                    work = bool(self.unreached_exits(gmap, mid))
            except Exception:
                work = True
            local.explore_exhausted = not work
            # **무엇이 막았는지** 남긴다 — 물이면 파도타기, 나무면 풀베기라야
            # 열린다. 프롬프트에 "탐험 완료 (나무 너머는 아직)" 로 나간다.
            local.explore_gated_by = (
                set(res.gated_by) if (res is not None and not work) else set()
            )
            # 포기한 목표·막힌 칸이 남아 있는 맵만 "막힌 곳 너머는 아직" 이라고 한다
            local.explore_gave_up = bool(
                not work and (self._blocked(mid) or any(k[0] == mid for k in self._fails))
            )
        self.dump_memory(gmap)

    def has_work(self, gmap: Any, map_id: int, pos: Coord) -> bool:
        """이 스킬을 돌려서 얻을 것이 있는가 (play.py 가 탐험을 건너뛸지 정할 때)."""
        caps = self._caps()
        self._sync_fail_memory(caps)
        if self._local_work(gmap, int(map_id), tuple(pos), caps):
            return True
        if self.home_map_only and not self._free_roam_here():
            return False                       # 다른 맵의 일은 이 모드의 일이 아니다 (동굴은 예외)
        return self.travel and bool(self._work_maps(gmap, int(map_id), caps, limit=1))

    def propose_action(self, gmap: Any, map_id: int, current_pos: Coord) -> Optional[str]:
        if not self.active:
            return None
        map_id = int(map_id)
        pos = (int(current_pos[0]), int(current_pos[1]))
        if self._maps_before is None:
            graph = getattr(gmap, "graph", None)
            self._maps_before = {int(n) for n in graph} if graph is not None else set()
            # 아직 안 이어진 출구가 남은 맵은 접어 뒀던 경계를 다시 시도한다
            self.retry_unfinished(gmap)

        if map_id != self._cur_map:
            self._on_map_change(gmap, map_id)
        if self.steps >= self.max_steps:
            self.stop("스텝 소진")
            return None
        if self._arrive_waits > 0:
            self._arrive_waits -= 1
            return WAIT

        self._check_arrival(gmap, map_id, pos)
        if self._note_stuck(gmap, map_id, pos):
            self.stop("제자리를 벗어나지 못함")
            return None
        self._drop_seen_frontier(gmap, map_id)

        if not self._queue:
            planned = self._plan(gmap, map_id, pos)
            if planned == WAIT:
                self._waits += 1
                if self._waits > _MAX_WAIT:
                    self.stop("맵 정보가 준비되지 않음")
                    return None
                return WAIT
            if not planned and self._since_change < _SETTLE_ACTIONS:
                # 워프 직후엔 RAM 맵 번호만 바뀌고 좌표·LocalMap·GlobalMap 간선이 몇 프레임 늦게 들어온다.
                self._since_change += 1
                return WAIT
            if not planned:
                if self.reason_is_done():
                    self.achieved = True
                self.dump_memory(gmap)         # 이번 탐험에서 새로 안 것까지 기록
                self.stop(self._why or "할 일 없음")
                return None
        self._waits = 0
        action = self._queue.pop(0)
        self.steps += 1
        self._dialogue_open = False           # 다시 움직인다 = 그 대화는 끝났다
        self._last_pos = pos
        self._last_action = action
        grid = getattr(gmap.get_local_map(map_id), "grid", None) or {}
        # 바로 뒤가 "A" 인 방향 입력은 **이동이 아니라 그쪽을 보는 것**이다 (길 막은 볼 줍기).
        facing_only = action in _MOVES and bool(self._queue) and self._queue[0] == "A"
        self._expect = (
            step_target(grid, pos, action) if action in _MOVES and not facing_only else None
        )
        self._moves_since_change += 1
        return action

    # -- 내부 ---------------------------------------------------------------
    _why = ""

    def reason_is_done(self) -> bool:
        return self._why.startswith("갈 수 있는 곳을 다 봄")

    def _say(self, text: str) -> None:
        if self.verbose:
            print(f"[bfs] {text}")

    def _current_tileset(self) -> Optional[int]:
        """지금 서 있는 맵의 타일셋 (wCurMapTileset)."""
        if self._tileset is not None:
            try:
                got = self._tileset()
                return None if got is None else int(got)
            except Exception:
                return None
        if self._pyboy is None:
            return None
        try:
            return int(self._pyboy.memory[ADDR_TILESET])
        except Exception:
            return None

    def _free_roam_here(self) -> bool:
        """지금 서 있는 맵이 ``home_map_only`` 의 예외(동굴)인가."""
        if not (self.home_map_only and self.free_roam_tilesets):
            return False
        tileset = self._current_tileset()
        return tileset is not None and tileset in self.free_roam_tilesets

    @staticmethod
    def _leads_outside(gmap: Any, map_id: int, coord: Coord) -> bool:
        """그 출구가 **건물·동굴 밖**으로 나가는가 (warps.json 의 ``LAST_MAP`` · 도로 connection)."""
        static = (getattr(gmap, "warp_at", None) or {}).get(int(map_id)) or {}
        x, y = int(coord[0]), int(coord[1])
        for c in ((x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            point = static.get(c)
            if point is not None:
                return (str(point.get("dest_const", "")) == "LAST_MAP"
                        or str(point.get("kind", "")) == "connection")
        return False

    @property
    def _restricted(self) -> bool:
        """이번 에피소드를 시작한 맵에 묶는가 (home_map_only 이고 동굴 예외가 아닐 때)."""
        return self.home_map_only and not self._free_roam

    def _caps(self) -> Tuple[bool, bool]:
        if self._capabilities is None:
            return False, False
        try:
            c, s = self._capabilities()
            return bool(c), bool(s)
        except Exception:
            return False, False

    def _sync_fail_memory(self, caps: Tuple[bool, bool]) -> None:
        if self._fails_caps != caps:
            if self._fails_caps is not None and self._fails:
                self._say("비전머신 능력이 바뀌어 막혔던 목표를 다시 시도합니다")
            self._fails.clear()
            self._fails_caps = caps

    def _failed(self, key: FailKey) -> bool:
        return self._fails.get(key, 0) >= self.fail_limit

    def _fail(self, key: FailKey, *, hard: bool = False) -> None:
        before = self._fails.get(key, 0)
        self._fails[key] = self.fail_limit if hard else before + 1
        if before < self.fail_limit <= self._fails[key]:
            # **접은 순간을 남긴다.** 접힌 목표는 다시 안 고르므로 "왜 저기를 안 가지"
            # 의 답이 여기뿐이다 (돌산터널 남쪽 출구 앞 경계가 말없이 빠졌던 일).
            map_id, kind, target = key
            runlog.log(
                "bfs_fail",
                map_id=int(map_id),
                종류={"f": "경계", "w": "출구", "m": "이동"}.get(kind, kind),
                목표=list(target),
                즉시=bool(hard),
                이유=self._fail_why or "",
            )
        self._fail_why = ""

    def _navigation(self, gmap: Any, caps: Tuple[bool, bool]):
        if self._nav is None or self._nav.global_map is not gmap:
            from Skills.Navigation.navigation import Navigation
            self._nav = Navigation(gmap)
        self._nav.can_cut, self._nav.is_surfing = caps
        return self._nav

    def _quiet(self):
        # Navigation 의 A* 는 계획마다 여러 줄을 찍는다. 탐험은 수백 번 계획하므로 막는다.
        return contextlib.redirect_stdout(io.StringIO())

    def _on_map_change(self, gmap: Any, map_id: int) -> None:
        kind = self._pending[0] if self._pending else None
        if map_id not in (self._maps_before or set()) and map_id not in self.new_maps:
            self.new_maps.append(map_id)
            self.achieved = True
            self._say(f"새 맵 등록: {gmap.map_name(map_id)}")
        elif kind == "출구":
            self._say(f"출구 통과 → {gmap.map_name(map_id)}")
        self._cur_map = map_id
        self._queue = []
        self._pending = None
        self._expect = None
        self._last_pos = None
        self._since_change = 0
        self._arrive_waits = _ARRIVE_WAITS
        self._moves_since_change = 0

    @staticmethod
    def _person_on(gmap: Any, map_id: int, coord: Coord) -> bool:
        """그 칸에 사람·오브젝트가 서 있는가 (관측 기준)."""
        grid = getattr(gmap.get_local_map(int(map_id)), "grid", None) or {}
        cell = grid.get(tuple(coord))
        return cell is not None and getattr(cell, "entity", None) in (
            EntityType.NPC, EntityType.OBJECT
        )

    def _blocked(self, map_id: int) -> Set[Coord]:
        return {c for (m, c), n in self._block_hits.items() if m == map_id and n >= self.fail_limit}

    def _check_arrival(self, gmap: Any, map_id: int, pos: Coord) -> None:
        """직전 액션으로 들어가려던 칸에 실제로 섰는가."""
        expect, self._expect = self._expect, None
        self._scripted_seen = False
        if expect is None or pos == expect:
            self._retried = False
            return
        self._interrupted = True
        if not self._retried and pos == self._last_pos and self._last_action in _MOVES:
            # 제자리다 = 방향만 돌았을 수 있다 (Gen1). 같은 입력을 한 번 더 넣어 본다.
            self._retried = True
            self._queue.insert(0, self._last_action)
            return
        self._retried = False
        self._queue = []                 # 계획이 어긋났다 — 다시 짠다

    def _note_stuck(self, gmap: Any, map_id: int, pos: Coord) -> bool:
        """최근 걸음이 두 칸 사이 왕복이면 지금 목표를 접는다."""
        self._recent.append((int(map_id), tuple(pos)))
        if len(self._recent) < _STUCK_WINDOW:
            return False
        if len({p for p in self._recent}) > _STUCK_CELLS:
            return False

        spots = set(self._recent)
        self._recent.clear()
        self._stuck_rounds += 1
        # 지금 목표만 접으면 안 된다 — 두 맵 경계를 오갈 때는 맵이 바뀔 때마다 목표가 지워져, 접을 목표가 없거나 반대편 목표가 남아 같은 왕복을 되풀이한다.
        goals = list(self._recent_goals)
        if self._pending is not None:
            goals.append(self._pending)
        self._recent_goals.clear()
        keys: Set[FailKey] = set()
        for kind, mid, data in goals:
            if kind == "경계":
                keys.update((mid, "f", f) for f in data or ())
            elif kind == "출구":
                keys.add((mid, "w", data))
            else:
                keys.add((int(data), "m", (0, 0)))
        warp_at = getattr(gmap, "warp_at", None) or {}
        for mid, (x, y) in spots:
            for coord in warp_at.get(int(mid), {}):
                if abs(coord[0] - x) + abs(coord[1] - y) <= 1:
                    keys.add((int(mid), "w", (int(coord[0]), int(coord[1]))))
        for key in keys:
            self._fail_why = "제자리 왕복"
            self._fail(key, hard=True)
        self._stuck_keys |= keys
        self._pending = None
        self._pending_goal = None
        self._queue = []
        self._say(
            f"{gmap.map_name(map_id)}: {pos} 근처에서 제자리 왕복 — "
            f"이 자리 목표 {len(keys)}개는 더 안 봅니다 ({self._stuck_rounds}/{_STUCK_ROUNDS})"
        )
        return self._stuck_rounds >= _STUCK_ROUNDS

    def _drop_seen_frontier(self, gmap: Any, map_id: int) -> None:
        """보러 가던 경계가 가는 도중 화면에 들어왔으면 끝까지 걷지 않고 다시 계획한다."""
        if not self._queue or self._pending is None or self._pending[0] != "경계":
            return
        if self._pending[1] != map_id:
            return
        grid = getattr(gmap.get_local_map(map_id), "grid", None) or {}
        if all(grid.get(f) is not None for f in self._pending[2]):
            self._queue = []
            self._pending = None

    def _close_pending(self, gmap: Any, map_id: int, res: Optional[ClosureResult], pos: Coord) -> None:
        """큐를 다 썼는데 목표가 안 이뤄졌으면 실패로 센다."""
        if self._pending is None:
            return
        kind, mid, data = self._pending
        goal, self._pending_goal = self._pending_goal, None
        interrupted, self._interrupted = self._interrupted, False
        battle_cut, self._battle_cut = self._battle_cut, False
        self._pending = None
        if interrupted and battle_cut:
            return                    # 전투로 끊겼다 — 실패로도 끊김으로도 세지 않는다
        if goal is not None and tuple(pos) == tuple(goal):
            interrupted = False       # 가려던 자리에 섰다 = 도중에 끊긴 게 아니다
        if kind == "출구" and self._dialogue_on_the_way:
            # 가는 길에 대화가 떴다 = 문지기가 막았다. 배틀과 달리 이건 진짜 실패다
            interrupted = False
        if mid != map_id:
            return
        if kind == "경계":
            keys = [(mid, "f", f) for f in data]
        elif kind == "출구":
            keys = [(mid, "w", data)]
        else:
            keys = [(int(data), "m", (0, 0))]
        if interrupted:
            for key in keys:
                self._interrupts[key] = self._interrupts.get(key, 0) + 1
                if self._interrupts[key] % _INTERRUPT_LIMIT == 0:
                    self._fail_why = f"가는 도중 {_INTERRUPT_LIMIT}번 끊김"
                    self._fail(key)
            return
        if kind == "경계":
            if res is None:
                return
            grid = getattr(gmap.get_local_map(map_id), "grid", None) or {}
            for key in keys:
                f = key[2]
                if f in res.frontier or f not in grid:
                    self._fail_why = "가서 섰는데도 안 보임"
                    self._fail(key)                # 가서 섰는데도 안 보인다
        else:
            for key in keys:
                self._fail_why = "끝까지 갔는데 맵이 안 바뀜"
                self._fail(key)                    # 끝까지 밟았는데 맵이 안 바뀜
            if kind == "출구":
                self._fail_sibling_exits(gmap, mid, data)

    def _fail_sibling_exits(self, gmap: Any, map_id: int, coord: Coord) -> None:
        """같은 곳으로 이어지는 옆 출구 칸도 같이 접는다."""
        static = (getattr(gmap, "warp_at", None) or {}).get(int(map_id)) or {}
        point = static.get(tuple(coord))
        if not point:
            return
        dest = point.get("dest_id")
        const = str(point.get("dest_const", ""))
        if dest is None and not const:
            return
        def same(info: dict) -> bool:
            return (dest is not None and info.get("dest_id") == dest) or bool(
                const and str(info.get("dest_const", "")) == const
            )

        # **붙어 있는 칸만** 같은 출구다. 목적지만 같다고 묶으면 돌산터널의 북쪽 입구와
        # 남쪽 출구(둘 다 LAST_MAP = 10번도로)가 한 출구로 묶여, 북쪽에서 한 번 실패하면
        # 남쪽 출구까지 같이 접혔다.
        group, todo = {tuple(coord)}, [tuple(coord)]
        while todo:
            cx, cy = todo.pop()
            for nb in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                info = static.get(nb)
                if nb not in group and info and same(info):
                    group.add(nb)
                    todo.append(nb)
        for other in group - {tuple(coord)}:
            self._fail_why = "옆 칸 출구가 막힘"
            self._fail((int(map_id), "w", (int(other[0]), int(other[1]))))

    def _closure(self, gmap: Any, map_id: int, pos: Coord, caps: Tuple[bool, bool]) -> ClosureResult:
        can_cut, can_surf = caps
        return closure_for(gmap, map_id, pos, is_surfing=can_surf, can_cut=can_cut, mark=False)

    def _warp_targets(self, gmap: Any, map_id: int, pos: Coord, caps: Tuple[bool, bool]) -> List[Coord]:
        if not self.cross_warps:
            return []
        can_cut, can_surf = caps
        try:
            got = seek_targets(gmap, map_id, pos, can_cut=can_cut, can_surf=can_surf)
        except Exception:
            return []
        return [t for t in got if not self._failed((map_id, "w", t))]

    def _local_work(self, gmap: Any, map_id: int, pos: Coord, caps) -> bool:
        return self._local_work_res(gmap, map_id, pos, caps)[0]

    def _local_work_res(self, gmap: Any, map_id: int, pos: Coord, caps):
        """(할 일이 남았나, 그때의 closure 결과)."""
        local = gmap.get_local_map(map_id)
        if local is None:
            return False, None
        name = gmap.map_name(map_id) if hasattr(gmap, "map_name") else ""
        if pokecenter_done(local, name):
            return False, None    # 포켓몬센터: 간호사를 봤으면 끝 — 구석까지 보러 오지 않는다
        res = self._closure(gmap, map_id, pos, caps)
        if frontier_stands(res, pos, skip=lambda f: self._failed((map_id, "f", f)), blocked=self._blocked(map_id),
                           extra=unseen_connections(gmap, map_id, res)):
            return True, res
        if is_exit_free_map(gmap.map_name(map_id) if hasattr(gmap, "map_name") else ""):
            return False, res     # 포켓몬센터: 다 봤으면 끝 (문은 들어온 입구뿐이다)
        return bool(self._warp_targets(gmap, map_id, pos, caps)), res

    def _work_maps(self, gmap: Any, map_id: int, caps, *, limit: int = 3) -> List[int]:
        """할 일이 남은 다른 맵."""
        graph = getattr(gmap, "graph", None)
        if graph is None or map_id not in graph:
            return []
        order: List[int] = []
        seen = {map_id}
        queue = deque([(map_id, 0)])
        while queue and len(order) < _TRAVEL_SCAN:
            cur, hops = queue.popleft()
            if hops >= self.travel_max_hops:
                continue                       # 더 먼 맵은 플래너가 정할 일이다
            for nxt in sorted(int(n) for n in graph.successors(cur)):
                if nxt not in seen:
                    seen.add(nxt)
                    order.append(nxt)
                    queue.append((nxt, hops + 1))
        found: List[int] = []
        sizes = getattr(gmap, "map_size_cells", {}) or {}
        for mid in order:
            if self._failed((mid, "m", (0, 0))):
                continue
            local = gmap.get_local_map(mid)
            seed = seed_cell(local, sizes.get(mid)) if local is not None else None
            if seed is None:
                continue
            if self._local_work(gmap, mid, seed, caps):
                found.append(mid)
                if len(found) >= limit:
                    break
        return found

    def _plan(self, gmap: Any, map_id: int, pos: Coord):
        """self._queue 를 채운다."""
        caps = self._caps()
        self._sync_fail_memory(caps)
        self._why = ""
        if gmap.get_local_map(map_id) is None:
            return WAIT
        res = self._closure(gmap, map_id, pos, caps)
        if res.reason in ("seed_off_map", "no_local_map", "no_grid"):
            return WAIT
        self._close_pending(gmap, map_id, res, pos)
        nav = self._navigation(gmap, caps)
        name = gmap.map_name(map_id)

        # 0-1.
        if self._free_roam and map_id != self._home and int(map_id) in _CITY_MAP_IDS:
            self.achieved = True
            self._why = f"동굴을 나와 마을({name})에 도착"
            return False

        # 0. 시작한 맵만 넓히는 모드 — 출구를 넘어와 있으면 **아무것도 넓히지 않고** 돌아간다.
        if self._restricted and self._home is not None and map_id != self._home:
            actions = self._travel_actions(nav, map_id, pos, self._home, caps, self._blocked(map_id))
            if actions:
                self._queue = actions
                self._pending = ("이동", map_id, self._home)
                self._pending_goal = None
                self.counts["복귀"] = self.counts.get("복귀", 0) + 1
                self._say(f"{name} 노드 등록 — 시작한 {gmap.map_name(self._home)} 로 돌아갑니다")
                return True
            # 돌아갈 길이 없다 (워프 패널·일방통행). 여기서 끝내고 플래너에게 넘긴다.
            self._why = f"시작한 맵({gmap.map_name(self._home)})으로 돌아갈 길이 없음"
            return False

        # 1. 경계
        blocked = self._blocked(map_id)
        extra = unseen_connections(gmap, map_id, res)
        stands = frontier_stands(res, pos, skip=lambda f: self._failed((map_id, "f", f)), blocked=blocked, extra=extra)
        tried = 0
        no_path: Dict[Coord, int] = {}       # 경계 칸 -> A* 가 실패한 설 칸 수
        seen_for: Dict[Coord, int] = {}      # 경계 칸 -> 이번에 본 설 칸 수
        for stand, cells in stands:
            for f in cells:
                seen_for[f] = seen_for.get(f, 0) + 1
            if tried >= _TRY_PER_PLAN:
                continue
            if stand == pos:
                # 바로 옆인데도 안 보인다 = 이 자리에서는 볼 수 없는 칸이다
                for f in cells:
                    self._fail_why = "바로 옆에 섰는데 안 보임"
                    self._fail((map_id, "f", f), hard=True)
                continue
            tried += 1
            actions = self._path(nav, map_id, pos, stand, "bfs-frontier", blocked)
            if actions:
                self._queue = actions
                self._pending = ("경계", map_id, cells)
                self._recent_goals.append(self._pending)
                self._pending_goal = stand
                self.counts["경계"] = self.counts.get("경계", 0) + 1
                self._say(f"{name}: 미관측 경계 {len(res.frontier) + len(extra)}칸 — {stand} 까지 {len(actions)}걸음")
                return True
            for f in cells:
                no_path[f] = no_path.get(f, 0) + 1
        # 어떤 경계 칸은 **설 자리가 여럿**이다.
        all_stands_tried = bool(stands) and tried >= len(stands)
        for f, missed in no_path.items():
            self._fail_why = (
                "설 자리 전부 A* 실패" + (" (막힌 칸을 지나야 함)" if self._blocked_reject else "")
            )
            if missed >= seen_for.get(f, 0):
                self._fail((map_id, "f", f), hard=self._blocked_reject or all_stands_tried)
        if stands and tried:
            self._say(
                f"{name}: 경계 {len(res.frontier) + len(extra)}칸 — 설 자리 {len(stands)}곳 중 "
                f"{tried}곳까지 길을 찾아봤지만 닿지 않습니다{_gate_note(res)}"
            )

        # 2. 출구
        size = (getattr(gmap, "map_size_cells", None) or {}).get(map_id)
        grid = getattr(gmap.get_local_map(map_id), "grid", None) or {}
        targets = self._warp_targets(gmap, map_id, pos, caps)
        if self._free_roam and self._free_roam_here():
            # 동굴 안에서는 **사다리(다른 층)를 먼저** 탄다.
            targets = sorted(targets, key=lambda t: self._leads_outside(gmap, map_id, t))
        for target in targets[:_TRY_PER_PLAN]:
            goal, tail = self._warp_approach(target, size, grid)
            if goal is None:
                self._fail_why = "출구로 다가설 칸이 없음"
                self._fail((map_id, "w", target), hard=True)
                continue
            actions = [] if goal == pos else self._path(nav, map_id, pos, goal, "bfs-warp", blocked)
            if goal != pos and not actions:
                self._fail_why = "출구까지 A* 실패"
                self._fail((map_id, "w", target), hard=self._blocked_reject)
                continue
            if not actions and not tail:
                self._fail((map_id, "w", target), hard=True)   # 문 위에 서 있는데 안 넘어갔다
                continue
            self._queue = actions + tail
            self._pending = ("출구", map_id, target)
            self._recent_goals.append(self._pending)
            self._pending_goal = goal
            self._dialogue_on_the_way = False
            self.counts["출구"] = self.counts.get("출구", 0) + 1
            self._say(f"{name}: 안 지나가 본 출구 {target} 로 ({len(self._queue)}걸음)")
            return True

        # 3. 이동 (시작한 맵만 넓히는 모드에서는 하지 않는다 — 동굴 예외는 한다)
        if self.travel and not self._restricted:
            for dest in self._work_maps(gmap, map_id, caps):
                key = (int(dest), map_id)
                self._travel_visits[key] = self._travel_visits.get(key, 0) + 1
                if self._travel_visits[key] > _TRAVEL_REVISIT:
                    # 같은 맵을 또 밟고 있다 = 두 맵 사이를 왕복하는 중이다.
                    self._fail_why = "두 맵 사이 왕복"
                    self._fail((int(dest), "m", (0, 0)), hard=True)
                    self._say(f"{gmap.map_name(dest)} 쪽으로 가려다 같은 자리를 맴돌아 접습니다")
                    continue
                actions = self._travel_actions(nav, map_id, pos, dest, caps, blocked)
                if actions:
                    self._queue = actions
                    self._pending = ("이동", map_id, dest)
                    self._recent_goals.append(self._pending)
                    self._pending_goal = None
                    self.counts["이동"] = self.counts.get("이동", 0) + 1
                    self._say(f"{name} 은 다 봤음 — 할 일이 남은 {gmap.map_name(dest)} 쪽으로 이동")
                    return True
                self._fail((dest, "m", (0, 0)), hard=self._blocked_reject)

        self._why = "갈 수 있는 곳을 다 봄" if not stands else "남은 곳까지 경로 없음"
        return False

    def _path(self, nav: Any, map_id: int, pos: Coord, goal: Coord, label: str,
              blocked: Optional[Set[Coord]] = None) -> List[str]:
        self._blocked_reject = False
        with self._quiet():
            ok = nav.plan_local_path_to_coords(map_id, pos, [goal], label=label)
        return self._take_actions(nav, map_id, pos, blocked) if ok else []

    def _take_actions(self, nav: Any, map_id: int, pos: Coord, blocked: Optional[Set[Coord]]) -> List[str]:
        """Navigation 이 만든 큐를 꺼낸다."""
        # "A" 는 길을 막은 몬스터볼을 줍는 입력이다 (Navigation 이 길이 없을 때만 넣는다)
        actions = [a for a in list(nav.action_queue) if a in ACTION_NAME_TO_INDEX or a == "A"]
        nav.action_queue.clear()
        self._blocked_reject = False
        if blocked:
            grid = getattr(nav.global_map.get_local_map(map_id), "grid", None) or {}
            if any(c in blocked for c in walk_cells(grid, pos, actions)):
                self._blocked_reject = True    # 길은 있는데 못 지나가는 칸을 지나야 한다
                return []
        return actions

    @staticmethod
    def _warp_approach(target: Coord, size: Optional[Coord], grid: Dict[Coord, Any]):
        """(걸어갈 칸, 도착 후 누를 액션)."""
        def in_map(c: Coord) -> bool:
            if not size:
                return True                   # 크기를 모르면 맵 안으로 본다
            return 0 <= c[0] < size[0] and 0 <= c[1] < size[1]

        if in_map(target):
            return target, []
        for action, (dx, dy) in _DIRS:
            inner = (target[0] - dx, target[1] - dy)
            cell = grid.get(inner)
            if in_map(inner) and cell is not None and cell.block_type is not BlockType.WALL:
                return inner, [action]
        return None, []

    def _travel_actions(self, nav: Any, map_id: int, pos: Coord, dest: int, caps,
                        blocked: Optional[Set[Coord]] = None) -> List[str]:
        can_cut, can_surf = caps
        self._blocked_reject = False
        with self._quiet():
            ok = nav.set_global_destination(
                map_id, dest, start_pos=pos, is_surfing=can_surf, can_cut=can_cut
            ) and nav.plan_local_path_to_next_map(map_id, pos)
        return self._take_actions(nav, map_id, pos, blocked) if ok else []


ExplorationSkill = BFSExplorationSkill   # exploration_factory 가 쓰는 공통 이름
