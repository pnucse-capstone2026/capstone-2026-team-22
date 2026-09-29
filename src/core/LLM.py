"""LLM 플래너 — GlobalMap + current_events → 다음 목표 맵 이름."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple

from core.config import ENV_PATH, exploration_opt, planner_opt
from core import mapstore, runlog
from core.closure import gate_note, pokecenter_done, unresolved_warps
from core.game_data import data_path, load_map_names

ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = ROOT / "prompts"
RUNTIME_DIR = ROOT / "runtime"
CURRENT_EVENTS_FILE = RUNTIME_DIR / "current_events.json"
LLM_REPLY_FILE = RUNTIME_DIR / "llm_reply.txt"
PROMPT_PREVIEW_FILE = RUNTIME_DIR / "last_planner_prompt.md"
# 이벤트는 **전부** 넣는다 (한 줄로 압축).
_EVENT_RECENT_TAIL = 5
_DEFAULT_GEMINI_MODEL = "gemini-flash-latest"


def _load_dotenv() -> None:
    """settings/.env 로드 (없어도 무시)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ENV_PATH, override=False)


def _google_api_key() -> Optional[str]:
    _load_dotenv()
    key = (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip()
    return key or None


def _gemini_model_name() -> str:
    _load_dotenv()
    return (os.environ.get("GEMINI_MODEL") or _DEFAULT_GEMINI_MODEL).strip()


# ---------------------------------------------------------------------------
# Goal / Context
# ---------------------------------------------------------------------------
@dataclass
class Goal:
    """LLM 이 내는 다음 목표."""

    target_map_name: str
    need_explore: int = 0            # 1 = 목표 맵이 관측 전이라 경유지에서 탐험해야 한다
    skill: str = ""
    target_npc: str = ""             # 목표 맵에서 관측된 NPC 하나 (없으면 빈 문자열)
    todo: str = ""                   # 다음 해야할 일 (이벤트 / 회복 / 탐험 …)
    reason: str = ""
    # 코드 안전망이 걸러 낸 것 (기록용)
    dropped_npc: str = ""
    note: str = ""
    # 코드 검사가 목표를 바꿨을 때, **LLM 이 원래 고른 것** (기록용)
    original_skill: str = ""
    original_target_map: str = ""
    # 코드가 채움 (LLM 출력이 아님)
    target_map_id: Optional[int] = None
    nav_map_id: Optional[int] = None  # 실제로 Nav 에 넣을 id
    nav_map_name: Optional[str] = None

    @property
    def next_event(self) -> str:
        """예전 이름 호환 — `다음 이벤트` 는 이제 `다음 해야할 일` 이다."""
        return self.todo


@dataclass
class PlannerContext:
    """플래너 입력 (프롬프트에 넣을 동적 부분)."""

    current_map_id: int
    current_map_name: str
    last_event: Optional[str]
    recent_events: List[str] = field(default_factory=list)
    known_map_names: List[str] = field(default_factory=list)
    known_summary: str = ""  # LLM용: 이름만 (id 없음)
    catalog_map_names: List[str] = field(default_factory=list)
    current_npc_names: List[str] = field(default_factory=list)
    # 맵 이름 → 그 맵에서 **한 번이라도 관측한** NPC 이름.
    known_npcs: Dict[str, List[str]] = field(default_factory=dict)
    # 맵 이름 → 탐험 종료조건 상태 ("탐험 완료" / "미완료 · 안 들어가 본 문 2곳").
    map_status: Dict[str, str] = field(default_factory=dict)
    # **현재 맵에서 걸어서 바로 갈 수 있는 맵 이름**. GlobalMap 의 간선 그대로다.
    # 이게 없으면 "지금 서 있는 도시 안에 체육관이 있다"를 LLM 이 알 길이 없어,
    # 이름을 한글 지명으로 짐작하다가 엉뚱한 도시를 목표로 잡는다.
    connected_maps: List[str] = field(default_factory=list)
    # 최근 계획과 그 결과 (오래된 것 → 최근). play.py 가 채운다.
    plan_history: List[str] = field(default_factory=list)
    # 지난 이동에서 길이 막혔던 사실.
    obstacles: List[str] = field(default_factory=list)
    # 맵 이름 → NPC 가 막고 있는 칸 ("x,y" 또는 "x,y (NPC이름)").
    blocked_cells: Dict[str, List[str]] = field(default_factory=dict)
    # **직전 사이클**의 이동 실패 한 건. 다음 계획에만 넣고 바로 버린다
    # (오래 들고 있으면 이미 뚫린 길을 계속 피하게 된다).
    nav_fail: str = ""
    # 파티 체력/PP — 이미 사람이 읽는 문자열로 만들어져 들어온다.
    party_status: List[str] = field(default_factory=list)
    # 가진 배지 이름.
    badges: List[str] = field(default_factory=list)
    # 아직 없는 배지.
    missing_badges: List[str] = field(default_factory=list)
    # 비전머신을 지금 쓸 수 있는지와 못 쓰는 이유 (engine/hm.field_move_status).
    hm_status: List[str] = field(default_factory=list)
    # 지금은 말을 걸어도 되는 점원 (``TALK_OK_WITH_ITEM`` 조건을 만족한 이름).
    talk_ok_npcs: List[str] = field(default_factory=list)
    # name → id (코드 전용, 프롬프트에 안 넣음)
    catalog_ids: Dict[str, int] = field(default_factory=dict)
    known_ids: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_global_map(
        cls,
        gmap,
        *,
        cur_map_id: int,
        last_event: Optional[str] = None,
        recent_events: Optional[List[str]] = None,
        party_status: Optional[List[str]] = None,
        badges: Optional[List[str]] = None,
        missing_badges: Optional[List[str]] = None,
        hm_status: Optional[List[str]] = None,
        talk_ok_npcs: Optional[List[str]] = None,
        obstacles: Optional[List[str]] = None,
        blocked_cells: Optional[Dict[str, List[str]]] = None,
        nav_fail: str = "",
    ) -> "PlannerContext":
        cur_map_id = int(cur_map_id)
        # 디버깅용 덤프(runtime/current_map.json · current_npc.json).
        # **프롬프트에는 안 쓴다** — LLM 이 보는 것은 이번 세션 GlobalMap 뿐이다.
        # 지난 실행 기록을 먹이면 Portal·격자가 백지라 못 가는 곳을 목표로 잡는다.
        live_maps, live_npcs = mapstore.dump(gmap)

        known_ids: Dict[str, int] = {}
        for mid in getattr(gmap, "graph", {}).nodes:
            name = gmap.map_name(int(mid))
            known_ids[name] = int(mid)

        catalog_ids = _load_catalog_ids()
        if recent_events is None:
            recent_events = list_event_names()
        if last_event is None:
            last_event = get_last_event()

        cur_npcs: List[str] = []
        local = gmap.get_local_map(cur_map_id)
        if local is not None and hasattr(local, "list_npc_names"):
            cur_npcs = local.list_npc_names()

        known_npcs: Dict[str, List[str]] = {
            name: list(live_npcs.get(name) or []) for name in known_ids
        }
        map_status: Dict[str, str] = {
            name: map_status_line(gmap.get_local_map(mid))
            for name, mid in known_ids.items()
        }
        # **지금 걸어서 갈 수 있는 맵인가.** 관측만 됐고 길은 아직 못 이은 맵이
        # 있다 (창 너머로 본 도로 등). 표시해 두지 않으면 LLM 이 그런 맵을
        # 계속 목표로 내고 Navigation 이 매번 "길이 없습니다" 로 실패한다.
        unreachable = _unreachable_maps(gmap, cur_map_id, known_ids)
        for name in unreachable:
            map_status[name] = f"{map_status.get(name, '미완료')} · 지금 가는 길 없음"

        # 현재 맵과 이어진 맵 (문·계단·도로). 지도 그래프가 이미 들고 있다.
        connected: List[str] = []
        try:
            graph = getattr(gmap, "graph", None)
            if graph is not None and cur_map_id in graph:
                connected = sorted(
                    gmap.map_name(int(n)) for n in graph.successors(cur_map_id)
                )
        except Exception:
            connected = []

        return cls(
            current_map_id=cur_map_id,
            current_map_name=gmap.map_name(cur_map_id),
            last_event=last_event,
            recent_events=list(recent_events),
            known_map_names=sorted(known_ids.keys()),
            known_summary=_llm_global_summary(live_maps),
            catalog_map_names=sorted(catalog_ids.keys()),
            current_npc_names=cur_npcs,
            known_npcs=known_npcs,
            map_status=map_status,
            connected_maps=connected,
            obstacles=list(obstacles or []),
            blocked_cells={k: list(v) for k, v in (blocked_cells or {}).items() if v},
            nav_fail=str(nav_fail or ""),
            party_status=list(party_status or []),
            badges=list(badges or []),
            missing_badges=list(missing_badges or []),
            hm_status=list(hm_status or []),
            talk_ok_npcs=list(talk_ok_npcs or []),
            catalog_ids=catalog_ids,
            known_ids=known_ids,
        )


def _unreachable_maps(gmap: Any, cur_map_id: int, known_ids: Dict[str, int]) -> Set[str]:
    """현재 맵에서 **걸어서 못 가는** 관측 맵 이름."""
    graph = getattr(gmap, "graph", None)
    if graph is None or int(cur_map_id) not in graph:
        return set()
    try:
        import networkx as nx
        # **방향을 보지 않는다.** 간선이 한쪽 방향만 기록된 구간이 있다 (들어온
        # 문의 반대 방향 등). 방향 그래프로 따지면 걸어온 길인데도 "못 간다" 가
        # 되어, 코드 검사가 맞는 목표를 현재 맵으로 바꿔 버린다.
        seen = nx.node_connected_component(graph.to_undirected(as_view=True), int(cur_map_id))
    except Exception:
        return set()
    return {name for name, mid in known_ids.items() if int(mid) not in seen}


def map_status_line(local: Any) -> str:
    """LLM 에 보여 줄 한 맵의 탐험 상태."""
    if local is None:
        return "미완료"
    if pokecenter_done(local):
        return "탐험 완료"            # 간호사를 봤으면 끝 (구석 PC 쪽은 안 본다)
    if getattr(local, "closed", False):
        # **무엇에 막혔는지 그대로 쓴다** — "물 너머는 아직" 과 "나무 너머는
        # 아직" 은 뚫는 방법이 다르다(파도타기 vs 풀베기). 뭉뚱그리면 플래너가
        # 엉뚱한 비전머신을 찾아 나선다.
        note = gate_note(getattr(local, "closed_gated_by", ()))
        line = "탐험 완료" + (f" ({note})" if note else "")
        # 이 맵은 다 봤지만 **관측한 워프 너머 구역**이 남았으면 그것도 말한다 — 플래너가 그 문(사다리·계단)을 목표로 잡을 수 있게.
        pending = [
            c for c, status in (getattr(local, "portal_done", None) or {}).items()
            if not (status[0] if isinstance(status, tuple) else status)
        ]
        if pending and not getattr(local, "final_done", False):
            line += f" · 포탈 {len(pending)}곳 미완료"
        return line
    # 탐험 스킬(skill: bfs)이 "남은 곳이 전부 막혔다"고 판단한 맵.
    if getattr(local, "explore_exhausted", False):
        # 물·나무가 막고 있으면 그것부터 말한다 — 뚫는 방법이 분명하기 때문이다 (파도타기·풀베기).
        note = gate_note(getattr(local, "explore_gated_by", ()))
        if note:
            return f"탐험 완료 ({note})"
        gave_up = getattr(local, "explore_gave_up", False)
        return "탐험 완료" + (" (막힌 곳 너머는 아직)" if gave_up else "")
    try:
        doors = len(unresolved_warps(local))
    except Exception:
        doors = 0
    return "미완료" + (f" · 안 들어가 본 문 {doors}곳" if doors else "")


def _llm_global_summary(maps: Dict[str, Dict[str, Any]]) -> str:
    """LLM용 맵 요약 — 이름만 (map_id 제외)."""
    if not maps:
        return "(아직 아는 맵 없음)"
    lines: List[str] = []
    for name in sorted(maps):
        info = maps[name] or {}
        exits = info.get("exits") or []
        pois = info.get("pois") or []
        lines.append(
            f"- {name}: 관측 {int(info.get('cells') or 0)}칸, "
            f"출구→[{', '.join(exits) if exits else '없음'}], "
            f"POI=[{', '.join(pois) if pois else '없음'}]"
        )
    return "\n".join(lines)


def _load_catalog_ids() -> Dict[str, int]:
    path = data_path("map_data.json")
    if not path:
        return {}
    return {name: mid for mid, name in load_map_names(path).items()}


def split_explore() -> bool:
    """탐험을 Coverage / SeekWarp 둘로 나눠 쓰는가 (``exploration.auto_split``)."""
    return bool(exploration_opt("auto_split", False))


def planning_prompt_dir() -> str:
    return "planning_split" if split_explore() else "planning"


def _read_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    if not path.is_file():
        return f"(missing prompts/{name})"
    return path.read_text(encoding="utf-8").strip()


def npc_by_map_block(ctx: PlannerContext) -> str:
    """맵별 관측 NPC 목록."""
    names = ctx.known_map_names or sorted(ctx.known_npcs.keys())
    if not names:
        return "- (아직 관측한 맵 없음)"
    lines: List[str] = []
    for name in names:
        npcs = ctx.known_npcs.get(name) or []
        here = " (지금 여기)" if name == ctx.current_map_name else ""
        body = ", ".join(npcs) if npcs else "(없음)"
        lines.append(f"- {name}{here}: {body}")
    return "\n".join(lines)


def events_block(ctx: PlannerContext) -> str:
    """지금까지 일어난 일 **전부**를 한 줄로 + 마지막 몇 개를 따로."""
    names = [str(n) for n in ctx.recent_events if str(n).strip()]
    if not names:
        return "(아직 없음)"
    line = f"({len(names)}개, 오래된 순) " + ", ".join(names)
    tail = names[-_EVENT_RECENT_TAIL:]
    if len(names) > len(tail):
        line += "\n마지막 " + str(len(tail)) + "개: " + " → ".join(tail)
    return line


def map_block(ctx: PlannerContext) -> str:
    """관측한 맵을 **상태별로 접어서** 보여 준다."""
    names = ctx.known_map_names or sorted(ctx.map_status)
    if not names:
        return "- (아직 관측한 맵 없음)"

    buckets: Dict[str, List[str]] = {}
    for name in names:
        status = str(ctx.map_status.get(name, "미완료"))
        # 구분자는 " · " 다. 라벨 안의 가운뎃점("물·나무 너머는 아직")은 자르면 안 된다.
        parts = [p.strip() for p in status.split(" · ") if p.strip()]
        base = parts[0] if parts else "미완료"
        extra = [p for p in parts[1:]]
        if "지금 가는 길 없음" in status:
            base, extra = "지금 가는 길 없음", [p for p in extra if "지금 가는 길" not in p]
        label = name
        notes = (["지금 여기"] if name == ctx.current_map_name else []) + extra
        if notes:
            label += " (" + ", ".join(notes) + ")"
        buckets.setdefault(base, []).append(label)

    order = ["미완료", "지금 가는 길 없음"]
    order += sorted(k for k in buckets if k not in order and k != "탐험 완료")
    order += ["탐험 완료"]
    lines: List[str] = []
    for key in order:
        got = buckets.get(key)
        if not got:
            continue
        tail = f" ({len(got)}곳)" if key == "탐험 완료" and len(got) > 3 else ""
        lines.append(f"- {key}: {', '.join(got)}{tail}")
    return "\n".join(lines)


def blocked_cells_block(ctx: PlannerContext) -> str:
    """NPC 가 막고 있는 칸 (맵별 좌표) + 길이 막혔던 이동."""
    if not ctx.blocked_cells and not ctx.obstacles:
        return ""
    rows = []
    for name, cells in sorted(ctx.blocked_cells.items()):
        shown = []
        for cell in cells:
            spot, _, npc = str(cell).partition(" ")      # "2,9" 또는 "2,9 (NPC이름)"
            shown.append(f"({spot})" + (f" {npc}" if npc else ""))
        rows.append(f"- {name}: " + ", ".join(shown))
    rows += [f"- {n}" for n in ctx.obstacles]
    return (
        "## 막힌 칸 (맵별 좌표 — NPC 가 길을 막고 있는 곳)\n"
        + "\n".join(rows)
        + "\n이 칸들은 **NPC 가 막고 있는 것이고, 앞선 이벤트가 아직 진행되지 않아서** 막혀 있습니다.\n"
        "조건이 그대로면 다시 가도 결과가 같습니다. **그 길을 다시 목표로 삼지 말고**,\n"
        "`## 지금까지 일어난 일` 과 대조해 빠진 선행 이벤트를 찾아 그것을 목표로 삼으세요.\n\n"
    )


def build_user_prompt(ctx: PlannerContext) -> str:
    """동적 user 메시지 — LLM 에는 이름만 (좌표·map_id 없음)."""
    badge_block = ", ".join(ctx.badges) if ctx.badges else "(아직 없음)"
    # **건너뛴 체육관을 스스로 찾게 한다.** 길이 막혔을 때 "배지가 모자란 것은
    # 아닌가" 를 먼저 따지지 않으면, 없는 길을 계속 두드린다.
    missing_block = (
        f"\n- 아직 없는 배지: {', '.join(ctx.missing_badges)}"
        if ctx.missing_badges
        else ""
    )
    hm_block = (
        "\n- 비전머신 — " + " / ".join(ctx.hm_status) if ctx.hm_status else ""
    )
    if ctx.talk_ok_npcs:
        hm_block += (
            "\n- 말을 걸어도 되는 점원: " + ", ".join(ctx.talk_ok_npcs)
            + " (가방에 건넬 물건이 있다 — 구매 메뉴가 아니라 이벤트다)"
        )

    # 계획 기록과 직전 실패는 같은 사실이다 — 한 블록으로 합친다.
    history_block = ""
    if ctx.plan_history:
        rows = "\n".join(str(line) for line in ctx.plan_history[-5:])
        tip = (
            "**직전이 실패입니다.** 유형과 `사유:` 줄(있으면 — 어디서 왜 막혔는지)을 보고 빠진 선행 이벤트가 있는지 따진 뒤,\n"
            "같은 목표를 같은 방법으로 또 내지 마세요. 판단은 `이유:` 첫 문장에 씁니다.\n\n"
            if ctx.nav_fail
            else "**같은 목표를 같은 방법으로 또 내지 마세요.**\n\n"
        )
        history_block = (
            "## 최근 계획과 결과 (오래된 것 → 최근)\n" f"{rows}\n" + tip
        )
    fail_block = (
        f"## 직전 계획은 실행되지 못했습니다\n{ctx.nav_fail}\n\n"
        if ctx.nav_fail and not ctx.plan_history
        else ""
    )
    blocked_block = blocked_cells_block(ctx)
    party_block = "\n".join(f"- {n}" for n in ctx.party_status) or "- (파티 없음)"

    linked = ", ".join(ctx.connected_maps) if ctx.connected_maps else "(아직 없음)"

    return f"""{history_block}{fail_block}{blocked_block}## 현재 상태
- 현재 맵: {ctx.current_map_name}
- 여기서 바로 갈 수 있는 맵: {linked}
  (지금 서 있는 곳과 이어진 맵. 가려는 곳이 이 줄에 있으면 그대로 목표 맵으로 쓴다)
- 배지: {len(ctx.badges)}/8 — {badge_block} (이미 딴 체육관으로 돌아갈 이유는 없다){missing_block}{hm_block}

## 관측한 맵 (`목표 맵:` 은 **반드시 이 중에서** · 탐험 상태별)
{map_block(ctx)}

## 관측한 NPC (맵별 — `목표 NPC:` 는 **목표 맵 줄**의 사람 중에서)
{npc_by_map_block(ctx)}

## 지금까지 일어난 일
{events_block(ctx)}

## 파티 상태 (체력 / 기술별 남은 PP — 회복이 먼저인지 판단)
{party_block}

## 할 일
위 시스템 지침대로 **다섯 줄만, 이 순서로** 출력하세요. 다른 문장 금지.
먼저 할 일과 이유를 정하고, 목표 맵·목표 NPC 는 **그 두 줄에 적은 장소·사람**으로 씁니다.

다음 해야할 일:
이유:
목표 맵:
목표 NPC:
사용할 스킬:
"""


# 노랑시티 게이트 네 곳(ROUTE_5/6/7/8_GATE)의 경비원.
SAFFRON_GATE_GUARDS = frozenset({"ROUTE5GATE_GUARD", "ROUTE6GATE_GUARD", "ROUTE7GATE_GUARD", "ROUTE8GATE_GUARD"})
GAVE_SAFFRON_DRINK_EVENT = "Gave Saffron Guards Drink"


def _npc_reject_reason(
    npc: str, target_map: str, ctx: PlannerContext
) -> str:
    """이 NPC 를 대화 대상으로 삼을 수 없는 이유."""
    if not npc:
        return ""
    if is_no_talk_npc(npc, ctx.talk_ok_npcs):
        return (
            f"'{npc}' 는 상점 점원입니다 — 말을 걸면 구매 메뉴가 열려 "
            "빠져나올 수 없으므로 대화 대상이 될 수 없습니다"
        )
    if is_placeholder_npc(npc):
        return (
            f"'{npc}' 는 정체를 못 알아낸 스프라이트의 임시 라벨이라 "
            "특정 인물로 지목할 수 없습니다"
        )
    if (
        npc in SAFFRON_GATE_GUARDS
        and GAVE_SAFFRON_DRINK_EVENT in (ctx.recent_events or [])
        and bool(planner_opt("open_gates_after_drink", True))
    ):
        return (
            f"'{npc}' 에게는 이미 음료를 건넸습니다({GAVE_SAFFRON_DRINK_EVENT}) — 노랑시티 게이트 "
            "네 곳이 모두 열렸으니 경비원에게 다시 말 걸 필요 없이 지나가면 됩니다"
        )
    if npc not in (ctx.known_npcs.get(target_map) or []):
        return f"'{npc}' 는 '{target_map}' 에서 관측된 NPC 가 아닙니다"
    return ""


SKILL_NAMES = ("Navigation", "Exploration")
SPLIT_SKILL_NAMES = ("Navigation", "Coverage", "SeekWarp")


def explore_skill_names(split: Optional[bool] = None) -> Tuple[str, ...]:
    """지금 설정에서 LLM 이 고를 수 있는 탐험 스킬 이름."""
    use_split = split_explore() if split is None else bool(split)
    return SPLIT_SKILL_NAMES[1:] if use_split else SKILL_NAMES[1:]


def normalize_skill(raw: Any, split: Optional[bool] = None) -> str:
    """`사용할 스킬:` 줄을 지금 설정의 스킬 이름 중 하나로."""
    use_split = split_explore() if split is None else bool(split)
    key = str(raw or "").strip().lower().replace(" ", "").replace("_", "")
    cover = key.startswith(("cover", "탐험", "explor", "커버", "익스플"))
    seek = key.startswith(("seek", "warp", "워프", "시크"))
    if not use_split:
        return "Exploration" if (cover or seek) else "Navigation"
    if cover:
        return "Coverage"
    if seek:
        return "SeekWarp"
    return "Navigation"


def is_unreachable_map(ctx: PlannerContext, map_name: str) -> bool:
    """지금 걸어서 갈 수 없다고 **프롬프트에 적어 준** 맵인가."""
    status = str((ctx.map_status or {}).get(str(map_name or ""), ""))
    return "지금 가는 길 없음" in status


def is_explored_map(ctx: PlannerContext, map_name: str) -> bool:
    """그 맵의 탐험이 끝났다고 **판정되어 있는가**."""
    status = str((ctx.map_status or {}).get(str(map_name or ""), ""))
    return status.startswith("탐험 완료")


def apply_code_checks(goal: Goal, ctx: PlannerContext) -> Goal:
    """LLM 출력에 **코드가 확실히 아는 것**만 건다."""
    goal.skill = normalize_skill(goal.skill)
    reason = _npc_reject_reason(goal.target_npc, goal.target_map_name, ctx)
    if reason:
        goal.dropped_npc = goal.target_npc
        goal.target_npc = ""
        goal.note = reason

    if goal.target_map_name and is_unreachable_map(ctx, goal.target_map_name):
        # 길이 없는 맵으로는 Navigation 이 매번 실패한다.
        note = (
            f"'{goal.target_map_name}' 로 가는 길이 아직 없습니다 "
            f"(관측만 된 맵) — 지금 있는 '{ctx.current_map_name}' 탐험으로 내립니다"
        )
        goal.original_skill = goal.skill
        goal.original_target_map = goal.target_map_name
        goal.target_map_name = ctx.current_map_name
        goal.target_npc = ""
        goal.skill = explore_skill_names()[0]
        goal.note = f"{goal.note} / {note}" if goal.note else note
        return goal

    if goal.skill != "Navigation" and is_explored_map(
        ctx, goal.target_map_name
    ):
        note = (
            f"'{goal.target_map_name}' 는 이미 탐험이 끝난 맵이라 "
            f"{goal.skill} 로는 더 볼 것이 없습니다 — Navigation 으로 내립니다"
        )
        goal.skill = "Navigation"
        goal.note = f"{goal.note} / {note}" if goal.note else note
    return goal


def split_prompt_messages(messages: List[Dict[str, str]]) -> Tuple[str, str]:
    """[{role, content}] → (system, user) 두 덩어리."""
    system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
    user = "\n\n".join(m["content"] for m in messages if m.get("role") == "user")
    return system, user


def build_prompt_messages(ctx: PlannerContext) -> List[Dict[str, str]]:
    """OpenAI/Claude 스타일 messages 리스트."""
    folder = planning_prompt_dir()
    system = "\n\n".join(
        [
            _read_prompt(f"{folder}/system.md"),
            _read_prompt(f"{folder}/rules.md"),
            _read_prompt(f"{folder}/fewshot.md"),
        ]
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": build_user_prompt(ctx)},
    ]


def dump_prompt_preview(
    ctx: PlannerContext, path: Optional[Path] = None
) -> Path:
    """디버그용: 조립된 프롬프트를 파일로 저장 (실제로 보내는 것과 같아야 한다)."""
    out = path or PROMPT_PREVIEW_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    for msg in build_prompt_messages(ctx):
        parts.append(f"# role: {msg['role']}\n\n{msg['content']}\n")
    out.write_text("\n---\n\n".join(parts), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Resolve / validate Goal against maps
# ---------------------------------------------------------------------------
def _resolve_name(name: str, table: Dict[str, int]) -> Optional[int]:
    if not name:
        return None
    if name in table:
        return table[name]
    low = name.lower()
    hits = [(n, i) for n, i in table.items() if n.lower() == low]
    if len(hits) == 1:
        return hits[0][1]
    hits = [(n, i) for n, i in table.items() if low in n.lower()]
    if len(hits) == 1:
        return hits[0][1]
    return None


def enrich_goal(goal: Goal, ctx: PlannerContext) -> Goal:
    """need_explore / nav 대상을 코드로 확정·보정."""
    tid = _resolve_name(goal.target_map_name, ctx.catalog_ids)
    if tid is None:
        tid = _resolve_name(goal.target_map_name, ctx.known_ids)
    goal.target_map_id = tid

    known = tid is not None and goal.target_map_name in ctx.known_ids
    # 이름이 known_ids 키와 다를 수 있어 id로도 확인
    if tid is not None and tid in ctx.known_ids.values():
        known = True

    if known:
        # 탐험 스킬(Exploration / Coverage)을 골랐다는 것은 "여기서 더 봐야 한다"는 뜻이다
        goal.need_explore = 1 if goal.skill in ("Exploration", "Coverage") else 0
        goal.nav_map_id = tid
        goal.nav_map_name = goal.target_map_name
    else:
        # 관측 목록 밖 이름 — 프롬프트가 금지한 경우다.
        goal.need_explore = 1
        goal.nav_map_id = ctx.current_map_id
        goal.nav_map_name = ctx.current_map_name

    return goal


# 말을 걸면 안 되는 NPC.
NO_TALK_NPC_MARKS = ("CLERK",)

# **조건이 맞으면 말을 걸어도 되는 점원.** 이름 → 가방에 있어야 할 아이템 id.
# 자전거 가게 점원은 상품권이 있으면 구매 메뉴 없이 자전거를 준다 — 그게 이벤트다.
# 상품권이 없으면 "사겠습니까?" 메뉴가 열리므로 그때는 여전히 금지다.
TALK_OK_WITH_ITEM = {
    "BIKESHOP_CLERK": 0x2D,
}


# `NPC_4`, `HiddenNPC_3_5` 처럼 이름을 못 알아낸 스프라이트에 붙는 임시 라벨.
_PLACEHOLDER_NPC = re.compile(r"^(NPC|HiddenNPC)_\d", re.I)


def is_placeholder_npc(name: str) -> bool:
    """정체를 못 알아낸 스프라이트인가."""
    return bool(_PLACEHOLDER_NPC.match(str(name or "").strip()))


def is_no_talk_npc(name: str, allow: Any = ()) -> bool:
    """상점 점원처럼 말을 걸면 안 되는 대상인가."""
    up = str(name or "").strip().upper()
    if up in {str(a).strip().upper() for a in (allow or ())}:
        return False
    return any(mark in up for mark in NO_TALK_NPC_MARKS)


def nav_npc_for_goal(goal: Goal, allow: Any = ()) -> str:
    """plan apply 에서 Local Nav 대상으로 쓸 NPC 이름 (없으면 빈 문자열)."""
    name = goal.target_npc or ""
    return "" if is_no_talk_npc(name, allow) else name



def parse_goal_text(text: str) -> Goal:
    """모델 응답 → Goal."""
    raw = text.strip()
    if not raw:
        raise ValueError("빈 목표 응답")

    # JSON 으로 와도 받는다
    if "{" in raw:
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        blob = fence.group(1) if fence else None
        if blob is None:
            brace = re.search(r"\{.*\}", raw, re.DOTALL)
            blob = brace.group(0) if brace else None
        if blob:
            data = json.loads(blob)
            g = lambda *keys: next((str(data[k]).strip() for k in keys if data.get(k)), "")
            return Goal(
                target_map_name=g("target_map_name", "목표 맵", "목표"),
                skill=g("skill", "사용할 스킬"),
                target_npc=_none_str(g("target_npc", "목표 NPC", "NPC")),
                todo=g("todo", "다음 해야할 일", "next_event", "다음 이벤트"),
                reason=g("reason", "이유"),
            )

    fence = re.search(r"```(?:string|text)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()

    target = skill = npc = todo = reason = ""
    for line in raw.splitlines():
        line = line.strip().strip("\"'` ")
        if not line:
            continue
        m = re.match(r"^(?:사용할\s*스킬|스킬|skill)\s*[:：]\s*(.+)$", line, re.I)
        if m:
            skill = m.group(1).strip(); continue
        m = re.match(r"^(?:목표\s*NPC|대상\s*NPC|target_npc|NPC)\s*[:：]\s*(.+)$", line, re.I)
        if m:
            npc = m.group(1).strip(); continue
        m = re.match(r"^(?:목표\s*맵|목표\s*위치|목표|target(?:_map(?:_name)?)?)\s*[:：]\s*(.+)$", line, re.I)
        if m:
            target = m.group(1).strip(); continue
        m = re.match(
            r"^(?:다음\s*해야\s*할\s*일|다음\s*할\s*일|해야\s*할\s*일|다음\s*이벤트|이벤트|todo|next_event)\s*[:：]\s*(.+)$",
            line, re.I,
        )
        if m:
            todo = m.group(1).strip(); continue
        m = re.match(r"^(?:이유|reason)\s*[:：]\s*(.+)$", line, re.I)
        if m:
            reason = m.group(1).strip(); continue

    if not target:
        # 라벨의 첫 글자가 깨져 나오는 경우 (예: "目표 맵:" — 한자 目).
        for line in raw.splitlines():
            line = line.strip().strip("\"'` ")
            m = re.match(r"^\S{0,3}표\s*(?:맵)?\s*[:：]\s*(.+)$", line)
            if m and not line.startswith(("이유", "다음", "회복", "NPC", "사용")):
                target = m.group(1).strip()
                break

    if not target:
        body = "\n".join(f"    | {ln}" for ln in text.strip().splitlines())
        raise ValueError(
            f"'목표 맵:' 줄을 찾지 못했습니다 (prompts/{planning_prompt_dir()}/system.md 의 5줄 형식). "
            f"받은 응답:\n{body}"
        )

    return Goal(
        target_map_name=_none_str(target) or target,
        skill=skill,
        target_npc=_none_str(npc),
        todo=todo,
        reason=reason,
    )


def _none_str(s: str) -> str:
    return "" if str(s or "").strip() in ("없음", "none", "None", "-", "(없음)", "") else str(s).strip()


# ---------------------------------------------------------------------------
# Planner protocol
# ---------------------------------------------------------------------------
class Planner(Protocol):
    def propose_goal(self, ctx: PlannerContext) -> Goal: ...

    def ask_text(self, system: str, user: str) -> str:
        """프롬프트 한 쌍을 보내고 응답 텍스트를 그대로 받는다."""
        ...

    def describe(self) -> str:
        """플래너 종류와 상태를 한 줄로."""
        ...


class ReplyFilePlanner:
    """Mock 없음."""

    def __init__(
        self,
        reply_file: Path = LLM_REPLY_FILE,
        *,
        reason: str = "",
    ) -> None:
        self.reply_file = Path(reply_file)
        # 왜 API 대신 이 모드인지 (배너에 표시)
        self.reason = reason

    def describe(self) -> str:
        why = f" — {self.reason}" if self.reason else ""
        return f"ReplyFilePlanner (수동: {self.reply_file.name}){why}"

    def propose_goal(self, ctx: PlannerContext) -> Goal:
        # 사용자가 답한 프롬프트는 "이번에 덮어쓰기 전"의 것이다.
        answered_prompt_mtime = (
            PROMPT_PREVIEW_FILE.stat().st_mtime
            if PROMPT_PREVIEW_FILE.is_file()
            else None
        )
        dump_prompt_preview(ctx)

        if not self.reply_file.is_file():
            raise FileNotFoundError(
                f"{self.reply_file.as_posix()} 없음. "
                "runtime/last_planner_prompt.md 를 LLM에 넣고 "
                "맵 이름 한 줄을 llm_reply.txt 에 저장한 뒤 다시 plan 하세요."
            )

        # 답이 그 프롬프트보다 오래됐다면 더 이전 plan 의 답이다.
        if (
            answered_prompt_mtime is not None
            and self.reply_file.stat().st_mtime < answered_prompt_mtime
        ):
            stale = self.reply_file.with_suffix(".stale.txt")
            self.reply_file.replace(stale)
            raise FileNotFoundError(
                f"{self.reply_file.as_posix()} 가 이전 plan 의 답입니다 "
                f"(→ {stale.name} 로 치움). "
                "새 last_planner_prompt.md 로 다시 받아 저장한 뒤 plan 하세요."
            )

        text = self.reply_file.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(
                f"{self.reply_file.as_posix()} 가 비어 있음. 맵 이름 한 줄을 넣으세요."
            )
        goal = parse_goal_text(text)
        goal.reason = goal.reason or "from runtime/llm_reply.txt"

        # 소비 완료 — 다음 plan 이 같은 답을 재사용하지 않도록 치운다.
        self.reply_file.replace(self.reply_file.with_suffix(".used.txt"))
        return goal

    def ask_text(self, system: str, user: str) -> str:
        """수동 모드의 2단계."""
        answered = (
            PROMPT_PREVIEW_FILE.stat().st_mtime
            if PROMPT_PREVIEW_FILE.is_file() else None
        )
        try:
            PROMPT_PREVIEW_FILE.parent.mkdir(parents=True, exist_ok=True)
            PROMPT_PREVIEW_FILE.write_text(
                (system + "\n\n" + user).strip() + "\n", encoding="utf-8"
            )
        except OSError:
            pass
        if not self.reply_file.is_file():
            raise FileNotFoundError(
                f"{self.reply_file.as_posix()} 없음. "
                "runtime/last_planner_prompt.md (검증 단계) 를 LLM 에 넣고 "
                "답을 llm_reply.txt 에 저장한 뒤 다시 plan 하세요."
            )
        if answered is not None and self.reply_file.stat().st_mtime < answered:
            stale = self.reply_file.with_suffix(".stale.txt")
            self.reply_file.replace(stale)
            raise FileNotFoundError(
                f"{self.reply_file.as_posix()} 가 이전 단계의 답입니다 "
                f"(→ {stale.name}). 검증 단계 프롬프트로 다시 받아 주세요."
            )
        text = self.reply_file.read_text(encoding="utf-8").strip()
        self.reply_file.replace(self.reply_file.with_suffix(".used.txt"))
        return text


class ApiPlanner:
    """Google Gemini API로 맵 이름 한 줄을 받아 Goal로 만든다."""

    def __init__(
        self,
        model: Optional[str] = None,
        *,
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model or _gemini_model_name()
        self.api_key = (api_key or _google_api_key() or "").strip() or None

    def describe(self) -> str:
        return f"ApiPlanner (Gemini {self.model})"

    def ask_text(self, system: str, user: str) -> str:
        if not self.api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY(또는 GEMINI_API_KEY) 없음. "
                "settings/.env 에 키를 넣으세요. (settings/.env.example 참고)"
            )
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ImportError(
                "google-genai 미설치. pip install google-genai python-dotenv"
            ) from exc

        client = genai.Client(api_key=self.api_key)
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system or None,
                    temperature=0.2,
                ),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Gemini 호출 실패 (model={self.model}): {exc}"
            ) from exc
        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise RuntimeError(f"Gemini 빈 응답 (model={self.model})")

        # 디버그/재현용으로도 저장
        try:
            LLM_REPLY_FILE.parent.mkdir(parents=True, exist_ok=True)
            LLM_REPLY_FILE.write_text(text + "\n", encoding="utf-8")
        except OSError:
            pass
        return text

    def propose_goal(self, ctx: PlannerContext) -> Goal:
        system, user = split_prompt_messages(build_prompt_messages(ctx))
        dump_prompt_preview(ctx)
        text = self.ask_text(system, user)
        goal = parse_goal_text(text)
        if not goal.reason:
            goal.reason = f"(모델 이유 없음 · {self.model})"
        return goal


# claude -p 는 Claude Code 이므로 그냥 두면 파일을 읽거나 셸을 돌리려 할 수 있다.
_CLAUDE_DENY_TOOLS = "Bash,Edit,Write,Read,Glob,Grep,WebFetch,WebSearch,NotebookEdit,Task"
# claude CLI 가 받는 --effort 값.
_CLAUDE_EFFORTS = ("low", "medium", "high", "xhigh", "max")

_IS_WINDOWS = os.name == "nt"
# npm 전역 설치가 shim(claude.cmd) 옆에 두는 진짜 실행 파일.
_NPM_CLAUDE_EXE = Path("node_modules") / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"


def _resolve_claude_cli(cli: str) -> str:
    """CLI 이름을 subprocess 가 셸 없이 바로 띄울 수 있는 경로로 바꾼다."""
    found = shutil.which(cli)
    if not found:
        return cli
    if _IS_WINDOWS and Path(found).suffix.lower() != ".exe":
        exe = Path(found).parent / _NPM_CLAUDE_EXE
        if exe.is_file():
            return str(exe)
    return found


class ClaudeCliPlanner:
    """Claude Code CLI(`claude -p`)로 맵 이름 한 줄을 받아 Goal로 만든다."""

    def __init__(
        self,
        *,
        cli: Optional[str] = None,
        model: Optional[str] = None,
        effort: Optional[str] = None,
        timeout_sec: Optional[int] = None,
    ) -> None:
        self.cli = (cli or str(planner_opt("claude_cli", "claude") or "claude")).strip()
        self.exe = _resolve_claude_cli(self.cli)
        raw_model = (
            model if model is not None else planner_opt("claude_model", "")
        )
        self.model = str(raw_model or "").strip() or None
        raw_effort = (
            effort if effort is not None else planner_opt("claude_effort", "")
        )
        self.effort = str(raw_effort or "").strip().lower() or None
        if self.effort is not None and self.effort not in _CLAUDE_EFFORTS:
            print(
                f"[planner] claude_effort={self.effort!r} 는 알 수 없는 값 — "
                f"CLI 기본값을 씁니다. 가능한 값: {', '.join(_CLAUDE_EFFORTS)}"
            )
            self.effort = None
        self.timeout_sec = int(
            timeout_sec if timeout_sec is not None else planner_opt("timeout_sec", 120)
        )

    def describe(self) -> str:
        model = f" · {self.model}" if self.model else ""
        effort = f" · effort={self.effort}" if self.effort else ""
        return f"ClaudeCliPlanner ({self.cli} -p{model}{effort})"

    def ask_text(self, system: str, user: str) -> str:
        cmd = [self.exe, "-p", "--output-format", "text"]
        if system:
            cmd += ["--system-prompt", system]
        if self.model:
            cmd += ["--model", self.model]
        if self.effort:
            cmd += ["--effort", self.effort]
        cmd += ["--disallowed-tools", _CLAUDE_DENY_TOOLS]

        try:
            proc = subprocess.run(
                cmd,
                input=user,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self.timeout_sec,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"{self.exe} 실행 실패 — Claude Code CLI 를 띄울 수 없습니다. "
                "settings/config.yaml 의 planner.claude_cli 를 확인하세요."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"{self.cli} 응답이 {self.timeout_sec}초를 넘겼습니다 "
                "(settings/config.yaml 의 planner.timeout_sec)"
            ) from exc

        if proc.returncode != 0:
            lines = (proc.stderr or "").strip().splitlines()
            tail = lines[-1] if lines else f"rc={proc.returncode}"
            raise RuntimeError(f"{self.cli} 호출 실패: {tail}")

        text = (proc.stdout or "").strip()
        if not text:
            raise RuntimeError(f"{self.cli} 빈 응답")

        # 디버그/재현용으로도 저장 (ApiPlanner 와 동일)
        try:
            LLM_REPLY_FILE.parent.mkdir(parents=True, exist_ok=True)
            LLM_REPLY_FILE.write_text(text + "\n", encoding="utf-8")
        except OSError:
            pass
        return text

    def propose_goal(self, ctx: PlannerContext) -> Goal:
        system, user = split_prompt_messages(build_prompt_messages(ctx))
        dump_prompt_preview(ctx)
        text = self.ask_text(system, user)
        goal = parse_goal_text(text)
        if not goal.reason:
            goal.reason = f"(모델 이유 없음 · {self.cli})"
        return goal


def _claude_planner_or_reason() -> Tuple[Optional[Planner], str]:
    """Claude Code CLI 를 쓸 수 있으면 (planner, ""), 아니면 (None, 사유)."""
    cli = str(planner_opt("claude_cli", "claude") or "claude").strip()
    if shutil.which(cli) is None:
        return None, f"{cli} CLI 없음 (PATH 확인)"
    return ClaudeCliPlanner(), ""


def _gemini_planner_or_reason() -> Tuple[Optional[Planner], str]:
    """Gemini API 를 쓸 수 있으면 (planner, ""), 아니면 (None, 사유)."""
    if not _google_api_key():
        return None, f"GOOGLE_API_KEY 없음 ({ENV_PATH} 확인)"
    from importlib.util import find_spec

    if find_spec("google.genai") is None:
        return None, "google-genai 미설치 (pip install google-genai)"
    return ApiPlanner(), ""


def default_planner() -> Planner:
    """settings/config.yaml 의 planner.backend 에 따라 플래너를 만든다."""
    backend = str(planner_opt("backend", "auto") or "auto").strip().lower()

    if backend == "manual":
        return ReplyFilePlanner(reason="config.yaml 의 planner.backend: manual")

    if backend == "claude":
        planner, why = _claude_planner_or_reason()
        return planner or ReplyFilePlanner(reason=why)

    if backend == "gemini":
        planner, why = _gemini_planner_or_reason()
        return planner or ReplyFilePlanner(reason=why)

    if backend != "auto":
        return ReplyFilePlanner(
            reason=f"planner.backend 값이 이상함: {backend!r} "
            "(claude | gemini | manual | auto)"
        )

    # auto — claude → gemini 순으로 쓸 수 있는 것
    reasons = []
    for probe in (_claude_planner_or_reason, _gemini_planner_or_reason):
        planner, why = probe()
        if planner is not None:
            return planner
        reasons.append(why)
    return ReplyFilePlanner(reason=" / ".join(reasons))


# ---------------------------------------------------------------------------
# runtime/current_events.json — 이벤트 이름만 기록
# ---------------------------------------------------------------------------
def _empty_events_payload() -> Dict[str, Any]:
    return {"last_event": None, "events": []}


def reset_current_events() -> None:
    """play.py 세션 시작 시 current_events.json 을 비운다."""
    save_events(_empty_events_payload())


def load_events() -> Dict[str, Any]:
    """current_events.json 로드."""
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    if not CURRENT_EVENTS_FILE.is_file():
        payload = _empty_events_payload()
        save_events(payload)
        return payload
    try:
        data = json.loads(CURRENT_EVENTS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = _empty_events_payload()
        save_events(payload)
        return payload
    events = data.get("events") or []
    # 문자열 리스트로 정규화 (예전 {name, map_id} 형태면 name만)
    names: List[str] = []
    for item in events:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict) and item.get("name"):
            names.append(str(item["name"]))
    last = data.get("last_event") or (names[-1] if names else None)
    return {"last_event": last, "events": names}


def save_events(payload: Dict[str, Any]) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    CURRENT_EVENTS_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


# 안 넣는다.
IGNORED_EVENT_NAMES = frozenset(
    {"Bought Museum Ticket"} | {f"Beat Power Plant Voltorb {i}" for i in range(8)}
)

# **세션에서 처음 한 번만** 받는 이벤트. 기록은 하되, 비트가 다시 켜질 때마다
# "새 이벤트" 로 재계획을 거는 일은 없게 한다. 포켓몬 타워 5층 정화 구역은 그 칸을
# 밟을 때마다 비트가 켜져서, 한 번 드나드는 사이에 8번 재계획이 걸렸다.
ONCE_EVENT_NAMES = frozenset({"In Purified Zone"})


def list_event_names() -> List[str]:
    names = list(load_events().get("events") or [])
    return [n for n in names if n not in IGNORED_EVENT_NAMES]


def get_last_event() -> Optional[str]:
    last = load_events().get("last_event")
    if not last or str(last) in IGNORED_EVENT_NAMES:
        return None
    return str(last)


def record_event(name: str) -> None:
    """게임에서 이벤트 발생 시 호출 — 이름만 append."""
    name = str(name).strip()
    if not name or name in IGNORED_EVENT_NAMES:
        return
    data = load_events()
    events = list(data.get("events") or [])
    if name in events:
        return
    # 예전에 쌓인 중복도 이 참에 정리한다 (첫 등장 순서 유지).
    seen: set = set()
    cleaned = [n for n in events if not (n in seen or seen.add(n))]
    cleaned.append(name)
    save_events({"last_event": name, "events": cleaned})


# ---------------------------------------------------------------------------
# runtime/logging.json — 계획/검증 이력 (core/runlog.py 가 파일을 관리)
# ---------------------------------------------------------------------------
def reset_plan_log() -> None:
    """play.py 세션 시작 시 로그를 비운다."""
    runlog.reset("play.py 세션 시작")


def record_plan(
    goal: Goal,
    *,
    current_map: str = "",
    last_event: Optional[str] = None,
    cycle: Optional[int] = None,
    party: Optional[List[str]] = None,
) -> int:
    """확정된 계획 한 건을 기록."""
    return runlog.log(
        "plan",
        cycle=int(cycle) if cycle is not None else None,
        현재_맵=current_map or None,
        최근_이벤트=last_event,
        사용할_스킬=goal.skill or "",
        목표_맵=goal.target_map_name or "",
        목표_NPC=goal.target_npc or "없음",
        버린_NPC=goal.dropped_npc or None,
        비고=goal.note or None,
        다음_해야할_일=goal.todo or "",
        이유=goal.reason or "",
        파티=list(party) if party else None,
        nav_map_id=int(goal.nav_map_id) if goal.nav_map_id is not None else None,
        nav_map=goal.nav_map_name or goal.target_map_name or None,
        need_explore=int(goal.need_explore),
    )
