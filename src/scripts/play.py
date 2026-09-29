#play.py — LLM + Skills(Navigation/Exploration) + Map 통합 실행기.


from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# legacy gym 을 여기서 미리 잡아 둔다.
try:  # noqa: E402
    import gym  # noqa: F401
except Exception:  # gym 이 없는 환경이면 그냥 넘어간다
    pass

from core.config import (  # noqa: E402
    ENV_PATH,
    auto_opt,
    battle_opt,
    event_opt,
    exploration_opt,
    load_config,
    planner_opt,
    recording_opt,
)
from core.gb_memory import (  # noqa: E402
    ADDR_MAP_ID,
    BATTLE_NONE,
    PLAYABLE_BATTLE_KINDS,
    battle_kind,
    battle_phase,
    is_in_battle,
)
from core.LLM import (  # noqa: E402
    PlannerContext,
    default_planner,
    dump_prompt_preview,
    get_last_event,
    ReplyFilePlanner,
    list_event_names,
    enrich_goal,
    nav_npc_for_goal,
    record_event,
    is_no_talk_npc,
    TALK_OK_WITH_ITEM,
    record_plan,
    reset_plan_log,
    apply_code_checks,
    planning_prompt_dir,
)
from core import mapstore, runlog  # noqa: E402
from core.closure import set_story_gate_checker  # noqa: E402
from core.map_types import BlockType  # noqa: E402
from core.event_tracker import EventWatcher, is_ignored_event_name  # noqa: E402
from core.portal import GlobalMap  # noqa: E402
from Skills.exploration_factory import (  # noqa: E402
    RULE_BASED,
    exploration_skill_choice,
    exploration_skill_label,
    make_exploration_skill,
)
from core.warp_targets import seek_targets  # noqa: E402
from Skills.battle_factory import flee_battle, make_battle_skill  # noqa: E402
from engine.move_learn import (  # noqa: E402
    MoveLearnResolver,
    move_learn_menu_open,
    set_default_resolver,
)
from Skills.Navigation import (  # noqa: E402
    NavMission,
    Navigation,
    debug_flags as nav_debug_flags,
)

from engine.cheats import (  # noqa: E402
    apply_auto_win_battle,
    apply_noclip_cheat,
    apply_skip_mt_moon_fossils,
    apply_skip_bill_cell_separator,
    apply_skip_vermilion_gym_puzzle,
    MAP_ID_GAME_CORNER,
    apply_buy_saffron_drink,
    apply_skip_game_corner_poster,
    apply_wake_snorlax,
    cheat_flags,
    disable_wild_encounters,
    handle_noclip_command,
    is_unbeatable_ghost,
)
from engine.dialogue import DialogueSkipper  # noqa: E402
from engine.settle import SettleGate  # noqa: E402
from engine.recorder import DemoRecorder  # noqa: E402
from engine.frame_observer import ObservingPyBoy  # noqa: E402
from engine.debug_view import handle_view_command  # noqa: E402
from engine.play_photo import save_next_play_photo  # noqa: E402
from engine.party_cheat import apply_party_mon_from_config  # noqa: E402
from engine.portal_debug import (  # noqa: E402
    print_current_node,
    print_target_list,
    resolve_target,
)
from engine.runners import (  # noqa: E402
    BattleRunner,
    ExploreRunner,
    LocalNavRunner,
    NavRunner,
    debug_flags as runner_debug_flags,
)
from engine.session import (  # noqa: E402
    ViewGate,
    handle_local_action,
    open_emulator,
    print_local_map_block,
    read_player,
    render_views,
    save_play_state,
)
from engine import explore_gate, hm  # noqa: E402
from engine.party_status import (  # noqa: E402
    BADGE_NAMES,
    badges_from_ref,
    party_lines_from_ref,
)
from engine.state import gmap_ref, pyboy_ref  # noqa: E402

_CFG = load_config()
EVENT_POLL_FRAMES = int(event_opt("poll_every_frames"))

_cmd: dict = {
    "action": None,
    "rec": None,
    "nav_target": None,
    "nav_local": None,
    "plan_apply": False,
}
_game: dict = {"map_id": None, "x": 0, "y": 0}
# 재생성 가능하도록 슬롯에 담는다 (plan reload).
_planner_ref: dict = {"planner": default_planner()}
_last_goal = {"goal": None}
_plan_chain: list = []  # map | npc | grass steps

# 한 걸음마다 LocalMap 전체를 찍을지. 평소 터미널을 깨끗하게 두려고 기본 off.
_debug_flags = {"localmap": False}

# auto 모드 — LLM plan → Navigation → Exploration 을 사람 입력 없이 반복
_auto: dict = {
    "on": False,
    # 상태 기계 단계 plan          계획을 세우고 nav 를 건다 nav_wait      nav(맵 → 있으면 NPC)가 끝나기를 기다린다 explore…
    "phase": "plan",
    "goal": None,
    "cycle": 0,
    "nav_ok": False,        # 마지막 Nav 성공 여부
    "npc_talked": False,    # 이번 사이클에 NPC Local Nav(대화)까지 마쳤는가
    "plan_fails": 0,        # 연속 플래너 실패 횟수
    "max_cycles": 50,
    "steps": 0,
    "max_steps": 0,         # 0 = 스텝 제한 없음
}

# play.py 실행 하나 동안의 누적 액션 수.
_SESSION = {"steps": 0}


# 응답 형식이 한 번 어긋나는 것만으로 무인 모드가 죽지 않도록 이만큼은 재시도한다.
AUTO_PLAN_FAIL_HINT_EVERY = 5
# 배틀인데 러너도 사람도 진행시키지 않을 때, 몇 프레임 뒤부터 A 로 밀지.
BATTLE_IDLE_FRAMES = 180
BATTLE_IDLE_A_EVERY = 16

def _planner_line() -> str:
    """시작 배너/plan reload 용 한 줄 설명."""
    planner = _planner_ref["planner"]
    describe = getattr(planner, "describe", None)
    return describe() if callable(describe) else type(planner).__name__


def _reload_planner() -> None:
    """settings/.env + config.yaml 을 다시 읽어 플래너를 새로 만든다."""
    try:
        from dotenv import load_dotenv

        # override=True 여야 방금 고친 키가 기존 환경변수를 덮어쓴다
        load_dotenv(ENV_PATH, override=True)
    except ImportError:
        print("[planner] python-dotenv 미설치 — 환경변수만 확인합니다")
    load_config(reload=True)
    _planner_ref["planner"] = default_planner()
    print(f"[planner] {_planner_line()}")


AVAILABLE_SKILLS = (
    (
        "Auto",
        "LLM Planning → Navigation → Exploration 자동 반복 (auto_split 이면 SeekWarp/Coverage)",
        "auto  (중단: stop)",
    ),
    (
        "Navigation",
        "맵 / NPC / GRASS / Pokemon Center 힐",
        "nav | nav npc | nav grass | nav heal",
    ),
    (
        "Exploration",
        "현재 맵 탐험 (기본: config 의 exploration.skill)",
        "explore | explore coverage | explore seekwarp",
    ),
    (
        "Battle",
        "배틀 감지 시 PPO로 자동 전투",
        "settings/config.yaml battle.enabled",
    ),
)


HELP = """
명령
  skills              사용 가능한 Skills 목록
  nav [목표]          Navigation 시작 (맵 이름/id)
  nav npc <이름>      현재 맵 NPC로 Local Nav 후 A 연타
  nav grass           Known 전체 최근접 GRASS (필요 시 맵 Nav→Local)
  nav heal            최근접 Pokemon Center → Nurse 대화(힐)
  explore             탐험 시작 — 스텝 상한 없음, stop 으로 중단
                      (맵이 최종 종료 조건에 닿으면 알아서 끝난다 — exploration.close_*)
  explore coverage    Coverage 스킬로 (격자 넓히기) / explore seekwarp: 안 가본 워프 통과
  auto                plan → nav → explore 자동 반복 (중단: stop)
  debug on|off        nav/explore 한 걸음마다 LocalMap 출력 (기본 off)
  plan                LLM 계획 (Planning 1회 + 코드 안전망). 실행은 안 함
  plan apply          위 계획을 실행 — 맵 Nav → (있으면) NPC Nav → 도착 후 탐험
  plan reload         settings/.env + config.yaml 다시 읽어 플래너 재생성
  npc / npcs          현재 LocalMap NPC 이름 목록
  event               감지된 이벤트 목록 보기
  event [이름]        수동으로 이벤트 이름 추가 (디버그)
  (기록)              계획·검증·회복·종료 조건 판정은 runtime/logging.json 에 남는다
  speed [배속]        게임 속도 보기/바꾸기 (0 = 제한 없음, 예: speed 1 / speed 0)
  rec on|off          시연 영상 녹화 시작/끝 (recordings/ 에 mp4 · auto 와 따로) · rec: 상태
  stop                활성 스킬 중단
  photo               누적 방문/관측/벽을 runtime/playphoto/<번호>.png로 저장
  view … / noclip …   test_map 과 동일
  global / local …    지도 디버그 (local 은 탐험 종료 조건 판정도 같이 출력)
  help
"""


# 지금 적용된 에뮬레이션 배속. settings/config.yaml 의 emulation_speed 로 시작한다.
_SPEED = {"value": int(_CFG.get("emulation_speed", 3))}
_SPEED_MAX = 30


def handle_speed_command(line: str) -> None:
    """speed | speed <배속>  — 게임 진행 속도를 바꾼다 (0 = 제한 없음)."""
    pyboy = pyboy_ref.get("pyboy")
    parts = line.strip().lower().split()
    if len(parts) == 1:
        now = _SPEED["value"]
        label = "제한 없음" if now == 0 else f"{now}배"
        print(f"[speed] 현재 {label}  (사용: speed <0~{_SPEED_MAX}>, 0 = 제한 없음)")
        return
    try:
        value = int(parts[1])
    except ValueError:
        print(f"[speed] 숫자를 적으세요 — speed <0~{_SPEED_MAX}> (0 = 제한 없음)")
        return
    if not 0 <= value <= _SPEED_MAX:
        print(f"[speed] 0 ~ {_SPEED_MAX} 사이여야 합니다 (0 = 제한 없음)")
        return
    if pyboy is None:
        print("[speed] 에뮬레이터가 아직 준비되지 않았습니다")
        return
    try:
        pyboy.set_emulation_speed(value)
    except Exception as exc:
        print(f"[speed] 변경 실패: {type(exc).__name__}: {exc}")
        return
    _SPEED["value"] = value
    label = "제한 없음" if value == 0 else f"{value}배"
    print(f"[speed] -> {label}  (config 의 emulation_speed 는 그대로)")


# ===========================================================================
# 입력 스레드
# ===========================================================================
def _print_skills() -> None:
    print("\n[skills] 사용 가능한 Skills:")
    for name, desc, usage in AVAILABLE_SKILLS:
        print(f"  · {name:12}  {desc}")
        print(f"                 사용: {usage}")
    print()


def _input_thread() -> None:
    print(
        "\n[명령] skills | auto | nav | nav npc | nav grass | nav heal | npc | "
        "explore | plan | plan apply | event | stop | debug on|off | "
        "photo | speed | rec on|off | view … | noclip | global | local | help"
    )
    while True:
        try:
            line = input().strip()
        except EOFError:
            break
        if not line:
            continue
        key = line.lower()
        if key in ("help", "?", "h"):
            print(HELP)
        elif key == "skills":
            _print_skills()
        elif key.startswith("noclip"):
            handle_noclip_command(line)
        elif key.startswith("view"):
            handle_view_command(line)
        elif key == "speed" or key.startswith("speed "):
            handle_speed_command(line)
        elif key == "rec" or key.startswith("rec "):
            parts = key.split()
            sub = parts[1] if len(parts) >= 2 else "status"
            if sub in ("on", "off", "status"):
                _cmd["rec"] = sub
            else:
                print("[rec] 사용: rec on | rec off | rec")
        elif key == "stop" or key == "nav stop":
            _auto_stop("사용자 중단")
            _cmd["action"] = "stop"
            _plan_chain.clear()
        elif key == "auto" or key.startswith("auto "):
            _cmd["action"] = "auto_start"
        elif key == "debug" or key.startswith("debug "):
            _handle_debug_line(line)
        elif key == "explore" or key.startswith("explore "):
            # 콘솔에서 직접 친 것은 상한 없이 돌리고 stop 으로 끊는다.
            parts = line.split()
            kind = parts[1].lower() if len(parts) >= 2 else None
            if kind not in (None, "coverage", "seekwarp"):
                print("[explore] 쓸 수 있는 값: explore | explore coverage | explore seekwarp")
            else:
                _cmd["explore_manual"] = True
                _cmd["explore_kind"] = kind
                _cmd["action"] = "explore_start"
        elif key == "plan" or key.startswith("plan "):
            _handle_plan_line(line)
        elif key == "event" or key.startswith("event "):
            _handle_event_line(line)
        elif key in ("npc", "npcs"):
            _print_current_npcs()
        elif key == "nav" or key.startswith("nav "):
            _handle_nav_line(line)
        elif key == "global":
            _cmd["action"] = "global"
        elif key in ("photo", "playphoto", "map photo"):
            _cmd["action"] = "photo"
        elif key == "local":
            _cmd["action"] = "local"
        elif key == "local split":
            _cmd["action"] = "local_split"
        elif key == "local file":
            _cmd["action"] = "local_file"
        elif key == "local coords":
            _cmd["action"] = "local_coords"
        else:
            print(f"[play] 알 수 없는 명령: {line!r}  (help)")


def _print_current_npcs() -> None:
    gmap = gmap_ref.get("gmap")
    cur_map = _game.get("map_id")
    if gmap is None or cur_map is None:
        print("[npc] GlobalMap/현재 맵 없음")
        return
    local = gmap.get_local_map(int(cur_map))
    if local is None:
        print("[npc] LocalMap 없음")
        return
    names = local.list_npc_names() if hasattr(local, "list_npc_names") else []
    print(f"[npc] map={gmap.map_name(int(cur_map))}  count={len(names)}")
    for name in names:
        coords = local.npc_coords(name)
        print(f"  - {name}: {coords}")


def _handle_event_line(line: str) -> None:
    parts = line.split(maxsplit=1)
    arg = parts[1].strip() if len(parts) > 1 else ""
    if not arg:
        names = list_event_names()
        print(f"[event] last={get_last_event()!r}  count={len(names)}")
        if names:
            print("  recent:", ", ".join(names[-8:]))
        print("  file: runtime/current_events.json")
        return
    if is_ignored_event_name(arg):
        print(
            f"[event] skip {arg!r} — 관장·라이벌이 아닌 전투/자리표시자는 "
            "current_events.json 에 넣지 않습니다"
        )
        return
    record_event(arg)
    print(f"[event] recorded → {arg!r}  (last={get_last_event()!r})")


def _handle_plan_line(line: str) -> None:
    parts = line.split()
    if len(parts) >= 2 and parts[1].lower() == "reload":
        _reload_planner()
        return
    apply = len(parts) >= 2 and parts[1].lower() == "apply"
    if apply:
        _cmd["plan_apply"] = True
    _cmd["action"] = "plan"



# NPC 가 길을 막아 Navigation 이 접힌 기록.
_OBSTACLES: list[str] = []
_OBSTACLE_MAX = 5
# 맵 이름 -> 길을 막은 NPC 기록.
_BLOCKED_NPCS: dict = {}


# 최근 계획과 그 결과. 프롬프트에 넣고 runtime/past_plan.json 에도 남긴다.
_PLAN_HISTORY: list = []
_PLAN_HISTORY_MAX = 5
# 이번 사이클에 일어난 일들 (계획 → 결과로 묶어 위 기록에 넣는다)
_CYCLE_NOTES: list = []


def _cycle_note(text: str) -> None:
    """이번 사이클에 일어난 일 한 줄."""
    line = str(text or "").strip()
    if line and line not in _CYCLE_NOTES:
        _CYCLE_NOTES.append(line)


def _short(text: str, limit: int = 160) -> str:
    """프롬프트에 넣을 한 줄로 줄인다 (이유가 길면 뒤를 자른다)."""
    line = " ".join(str(text or "").split())
    return line if len(line) <= limit else line[: limit - 1] + "…"


# 결과 문장 → 한 단어 유형.
_RESULT_TAGS = (
    ("막힘", ("막았습니다", "막고 있", "막힌")),
    ("길 없음", ("이어지는 길이 없", "가는 길이 없", "경로 없음", "길을 못", "도달 가능한 Portal")),
    ("대화 실패", ("대화 실패", "대화했지만 달라진 게 없")),
    ("탐험 무소득", ("더 볼 것이 없", "관측이 안 늘", "볼 것이 없다")),
    ("이벤트 발생", ("이벤트 발생",)),
    ("성공", ("이동 성공", "도착")),
)


def _result_tag(notes) -> str:
    """이번 사이클 결과를 한 단어로."""
    text = " ".join(str(n) for n in (notes or []))
    for tag, keys in _RESULT_TAGS:
        if any(k in text for k in keys):
            return tag
    return "진행 없음"


def _plan_history_lines() -> list:
    """프롬프트용 블록."""
    out = []
    rows = _PLAN_HISTORY[-_PLAN_HISTORY_MAX:]
    for i, row in enumerate(rows):
        goal = str(row.get("목표") or "")
        tag = str(row.get("결과유형") or "") or _result_tag(row.get("결과"))
        here = "  ← 직전" if i == len(rows) - 1 else ""
        block = [f"- [{row.get('사이클')}] {goal} — {tag}{here}"]
        if row.get("LLM이_고른_것"):
            # LLM 이 고른 것을 코드가 바꿨다 — 그대로 보여 줘야 LLM 이 "내가 같은 걸 또 골랐다" 고 착각하지 않는다
            block.append(
                f"  (당신이 고른 것은 {row['LLM이_고른_것']} 였고 코드가 바꿨습니다)"
            )
        # 직전 한 건만 하려던 일까지 보여 준다. 오래된 것은 이유 한 줄이면 충분하다.
        last = i == len(rows) - 1
        detail = " / ".join(
            part
            for part in (
                f"하려던 일: {_short(row.get('다음_해야할_일') or '', 60)}"
                if last and row.get("다음_해야할_일") else "",
                f"이유: {_short(row.get('이유') or '', 80 if last else 50)}"
                if row.get("이유") else "",
            )
            if part
        )
        if detail:
            block.append(f"  {detail}")
        why = _history_fail_reason(row, tag)
        if why:
            block.append(f"  사유: {why}")
        out.append("\n".join(block))
    return out


# 실패 사유 줄에서 뺄 결과 문장 — 무언가 된 기록은 사유가 아니다.
_NOT_A_REASON = ("이동 성공", "대화함")


def _history_fail_reason(row: dict, tag: str) -> str:
    """실패한 계획의 사유 한 줄 (`planner.history_fail_reason`)."""
    if not bool(planner_opt("history_fail_reason", True)):
        return ""
    if tag in _PROGRESS_TAGS:
        return ""
    parts = [
        str(n) for n in (row.get("결과") or [])
        if str(n).strip() and not str(n).startswith(_NOT_A_REASON)
    ]
    note = str(row.get("비고") or "").strip()
    if note:
        parts.append(f"코드가 바꾼 이유: {note}")
    return _short(" / ".join(parts), 220) if parts else ""


# 아무것도 못 한 사이클이 이만큼 이어지면 막힘 기억을 한 번 지우고 다시 본다.
_DEAD_CYCLE_LIMIT = 3
_DEAD_CYCLES = {"n": 0}
_PROGRESS_TAGS = ("성공", "이벤트 발생")


def _flush_cycle(goal) -> None:
    """직전 사이클의 계획과 결과를 기록에 넣는다 (다음 계획 직전에 부른다)."""
    if goal is None and not _CYCLE_NOTES:
        return
    skill = str(getattr(goal, "skill", "") or "")
    target = str(getattr(goal, "target_map_name", "") or "")
    npc = str(getattr(goal, "target_npc", "") or "")
    label = f"{skill} → {target}" if target else skill or "(계획 없음)"
    if npc:
        label += f" ({npc})"
    original = ""
    o_target = str(getattr(goal, "original_target_map", "") or "")
    if o_target:
        o_skill = str(getattr(goal, "original_skill", "") or skill)
        original = f"{o_skill} → {o_target}"
    tag = _result_tag(_CYCLE_NOTES)
    _DEAD_CYCLES["n"] = 0 if tag in _PROGRESS_TAGS else _DEAD_CYCLES["n"] + 1
    _PLAN_HISTORY.append({
        "사이클": int(_auto.get("cycle") or 0),
        "목표": label,
        "LLM이_고른_것": original,
        "비고": str(getattr(goal, "note", "") or ""),
        "다음_해야할_일": str(getattr(goal, "todo", "") or ""),
        "이유": str(getattr(goal, "reason", "") or ""),
        "결과유형": tag,
        "결과": list(_CYCLE_NOTES),
        "시각": time.strftime("%H:%M:%S"),
    })
    del _PLAN_HISTORY[:-_PLAN_HISTORY_MAX]
    _CYCLE_NOTES.clear()
    try:
        mapstore.save_past_plans(_PLAN_HISTORY)
    except Exception:
        pass


def _unblock_after_event(gmap, pyboy=None) -> None:
    """스토리 이벤트가 났다 — **막혀 있던 것들을 전부 다시 본다.**"""
    forget = getattr(_explore_state.get("default"), "forget_blocks", None)
    if callable(forget):
        forget()
    try:
        explore_gate.reopen_all(gmap, "새 이벤트")
    except Exception:
        pass
    if _OBSTACLES or _BLOCKED_NPCS:
        _OBSTACLES.clear()
        _BLOCKED_NPCS.clear()
        _dump_blocked_npcs()
        print("[nav] 새 이벤트 — 막힘 기록을 지웁니다")
    _NAV_FAIL.clear()


def _dump_blocked_npcs() -> None:
    try:
        mapstore.save_blocked_npcs(_BLOCKED_NPCS)
    except Exception:
        pass                      # 기록 실패가 실행을 막지는 않는다

# NPC 대화를 시작할 때의 (맵, 이름, 그때까지의 이벤트 수).
_TALK_WATCH: dict = {}

# **직전 사이클**의 이동 실패 한 건. 다음 Planning 프롬프트에 한 번만 넣고 버린다.
# (계속 들고 있으면 나중에 길이 뚫려도 그 목표를 영영 피하게 된다)
_NAV_FAIL: dict = {}


def _note_nav_fail(target_map: str, cur_map: str, reason: str) -> None:
    _cycle_note(str(reason or ""))
    _NAV_FAIL.clear()
    _NAV_FAIL.update(target=str(target_map), map=str(cur_map), reason=str(reason))
    print(f"[nav] 실패 기록 — 다음 계획에만 반영합니다: {reason}")
    runlog.log(
        "nav_fail",
        cycle=int(_auto["cycle"]) if _auto.get("on") else None,
        현재_맵=cur_map,
        목표_맵=target_map,
        사유=reason,
    )


def _take_nav_fail() -> str:
    """프롬프트에 넣을 한 문단."""
    info = dict(_NAV_FAIL)
    _NAV_FAIL.clear()
    if not info:
        return ""
    return (
        f'- 직전에 "{info["map"]}" 에서 "{info["target"]}" 로 이동하려 했지만 '
        f"실패했습니다 ({info['reason']})."
    )


def _map_name(map_id) -> str:
    gmap = gmap_ref.get("gmap")
    try:
        return gmap.map_name(int(map_id)) if gmap is not None else str(map_id)
    except Exception:
        return str(map_id)


def _talk_begin(cur_map: int, npc_name: str) -> None:
    _TALK_WATCH.clear()
    if npc_name:
        _TALK_WATCH.update(
            map_id=int(cur_map),
            map=_map_name(cur_map),
            npc=str(npc_name),
            events=len(list_event_names()),
        )


def npc_unreachable_detail(local, npc_name: str) -> str:
    """그 사람 앞까지 길이 없을 때, **왜** 없는지 한 구절 (프롬프트용)."""
    if local is None or not npc_name:
        return ""
    try:
        cells = list(local.npc_coords(npc_name) or [])
    except Exception:
        cells = []
    if not cells:
        return "그 사람의 자리가 지금 맵 지도에 없습니다"
    stands = []
    for cell in cells:
        stands += Navigation._npc_interaction_stand_tiles(local, tuple(cell))
    if not stands:
        return "그 사람 앞에 설 수 있는 칸이 아직 관측되지 않았습니다 (그 주변이 미관측)"
    return (
        "관측됐지만 지금 서 있는 곳과 이어진 길이 없음 (벽·잠긴 문 너머이거나 다른 층 계단·워프로 "
        "들어가는 자리) — 같은 목표를 다시 내도 같은 결과"
    )


def _npc_unreachable_detail(map_id, npc_name: str) -> str:
    if not bool(planner_opt("history_fail_reason", True)):
        return ""                   # 예전 문장 그대로 ("앞까지 가는 길이 없습니다")
    gmap = gmap_ref.get("gmap")
    if gmap is None or map_id is None:
        return ""
    try:
        return npc_unreachable_detail(gmap.get_local_map(int(map_id)), npc_name)
    except Exception:
        return ""


def _talk_end(reached: bool = True) -> str:
    """대화가 끝났다."""
    watch = dict(_TALK_WATCH)
    _TALK_WATCH.clear()
    if not watch:
        return ""
    if not reached:
        line = f"{watch['npc']} 앞까지 가는 길이 없습니다"
        detail = _npc_unreachable_detail(watch.get("map_id"), str(watch.get("npc") or ""))
        if detail:
            line += f" — {detail}"
        print(f"[nav] {line}")
        return line
    if len(list_event_names()) != int(watch.get("events") or 0):
        return ""                   # 이벤트가 늘었다 = 대화가 무언가를 했다
    line = f"{watch['npc']} 와 대화했지만 달라진 게 없습니다"
    print(f"[nav] {line}")
    return line


def _blocked_cells(gmap) -> dict:
    """맵 이름 -> NPC 가 막고 있는 칸."""
    out: dict = {}
    snap = getattr(_explore_state.get("default"), "memory_snapshot", None)
    if callable(snap):
        try:
            for name, info in (snap(gmap) or {}).items():
                for cell in info.get("막힌_칸") or []:
                    out.setdefault(str(name), []).append(str(cell))
        except Exception as exc:
            print(f"[plan] 막힌 칸 읽기 실패: {type(exc).__name__}: {exc}")
    for name, rows in _BLOCKED_NPCS.items():
        for row in rows:
            spot = str(row.get("막은_자리") or "")
            if not spot:
                continue
            cells = out.setdefault(str(name), [])
            npc = str(row.get("NPC") or "")
            named = f"{spot} ({npc})" if npc and "미상" not in npc else spot
            if spot in cells:
                cells[cells.index(spot)] = named      # 같은 칸이면 이름 있는 쪽으로
            elif named not in cells:
                cells.append(named)
    return out


def _note_obstacle(info: dict) -> None:
    """NavRunner.blocked_by 를 사람이 읽는 한 줄로 남긴다."""
    line = (
        f"\"{info.get('map')}\" 에서 {info.get('npc')} 가 길을 막아 "
        f"\"{info.get('target_map')}\" 로 못 갔습니다 "
        f"({info.get('tries')}번 시도)"
    )
    if line in _OBSTACLES:
        return
    _OBSTACLES.append(line)
    del _OBSTACLES[:-_OBSTACLE_MAX]
    rows = _BLOCKED_NPCS.setdefault(str(info.get("map") or "?"), [])
    pos = tuple(info.get("pos") or ())
    rows.append({
        "NPC": str(info.get("npc") or ""),
        "막은_자리": f"{pos[0]},{pos[1]}" if len(pos) == 2 else "",
        "못_간_맵": str(info.get("target_map") or ""),
        "시도": int(info.get("tries") or 0),
        "시각": time.strftime("%H:%M:%S"),
    })
    _dump_blocked_npcs()
    # 이번 이동이 실패한 이유이기도 하다 — 다음 계획 한 번에는 실패 사유로도 보인다.
    _NAV_FAIL.clear()
    _NAV_FAIL.update(
        target=str(info.get("target_map") or ""), map=str(info.get("map") or ""), reason=line
    )
    print(f"[nav] 막힘 기록 — 다음 계획에 반영합니다: {line}")
    _cycle_note(line)
    runlog.log(
        "blocked",
        cycle=int(_auto["cycle"]) if _auto.get("on") else None,
        NPC=info.get("npc"),
        맵=info.get("map"),
        좌표=list(info.get("pos") or ()),
        목표_맵=info.get("target_map"),
        시도=info.get("tries"),
    )


def _print_party(indent: str = "  ") -> None:
    """현재 파티의 체력/PP 를 한 줄씩."""
    lines = party_lines_from_ref(pyboy_ref)
    if not lines:
        print(f"{indent}(파티 없음)")
        return
    for line in lines:
        print(f"{indent}{line}")


def _propose_goal(ctx):
    """1단계(계획) 한 번."""
    try:
        return _planner_ref["planner"].propose_goal(ctx)
    except (FileNotFoundError, ValueError) as exc:
        # FileNotFoundError 는 수동 모드에서 답 파일을 기다리는 정상 상태지만, ValueError(형식 파싱 실패)는 어느 백엔드에서나 날 수 있다.
        manual = isinstance(_planner_ref["planner"], ReplyFilePlanner)
        print(f"  [{'대기' if manual else 'planner'}] {exc}")
        if manual:
            print("  1) last_planner_prompt.md 를 LLM에 넣기")
            print("  2) 다섯 줄 응답 → runtime/llm_reply.txt")
            print("  3) plan 또는 plan apply 다시\n")
        else:
            print(f"  (응답 형식은 prompts/{planning_prompt_dir()}/system.md 의 다섯 줄)\n")
        return None
    except Exception as exc:
        # 백엔드 무관(claude CLI / Gemini API) 호출 실패
        print(f"  [planner] {exc}\n")
        return None


def _print_planning(ctx, goal) -> None:
    """LLM 이 실제로 뭘 뽑았는지 그대로 보여준다 (코드 안전망 적용 후)."""
    print("[계획] LLM 이 뽑은 것")
    print(f"  사용할 스킬     : {goal.skill or '(없음)'}")
    print(f"  목표 맵         : {goal.target_map_name or '(없음)'}")
    print(f"  목표 NPC        : {goal.target_npc or '없음'}")
    print(f"  다음 해야할 일  : {goal.todo or '(없음)'}")
    print(f"  이유            : {goal.reason or '(없음)'}")
    runlog.log(
        "planning",
        cycle=int(_auto["cycle"]) if _auto.get("on") else None,
        현재_맵=ctx.current_map_name,
        사용할_스킬=goal.skill or "",
        목표_맵=goal.target_map_name or "(없음)",
        목표_NPC=goal.target_npc or "없음",
        다음_해야할_일=goal.todo or "",
        이유=goal.reason or "",
        관측_맵=list(ctx.known_map_names),
        맵_상태=dict(ctx.map_status),
        관측_NPC=dict(ctx.known_npcs),
        파티=list(ctx.party_status),
        배지=list(ctx.badges),
        이벤트수=len(ctx.recent_events),
        스텝=int(_SESSION["steps"]),
    )


def _plan(ctx):
    """계획 한 번 = LLM 호출 1회 + 코드 안전망."""
    goal = _propose_goal(ctx)
    if goal is None:
        return None
    apply_code_checks(goal, ctx)
    _print_planning(ctx, goal)
    if goal.dropped_npc:
        print(f'  버림            : "{goal.dropped_npc}" — {goal.note}')
    return goal


# 스토리 경계(6층 유령 등)가 열렸는지 판정 코드가 알 수 있게 — core 는 게임 메모리를 안 읽는다.
set_story_gate_checker(
    lambda gate: explore_gate.story_gate_open(pyboy_ref.get("pyboy"), gate)
)


def _mark_new_stairs(gmap, map_id: int, dest_const: str) -> None:
    """스위치로 새로 열린 계단을 **안 가 본 워프(w)** 로 바로 지도에 올린다."""
    local = gmap.get_local_map(int(map_id))
    if local is None:
        return
    for coord, point in (getattr(gmap, "warp_at", {}) or {}).get(int(map_id), {}).items():
        if str(point.get("dest_const", "")) != dest_const:
            continue
        cell = local.ensure_cell(int(coord[0]), int(coord[1]))
        cell.warp_candidate = True
        if cell.block_type is BlockType.WALL:
            cell.block_type = None        # 막혀 있던 벽이 계단이 됐다
        print(f"[map] {gmap.map_name(int(map_id))} {tuple(coord)} 에 새 계단 — 안 가 본 워프로 표시")
    local.closed = False
    local.explore_exhausted = False


def _missing_badges(ref=None) -> list:
    """아직 없는 배지 이름."""
    have = set(badges_from_ref(ref))
    return [name for name in BADGE_NAMES if name not in have]


def _talk_ok_npcs(ref=None) -> list:
    """지금은 말을 걸어도 되는 점원 — 건넬 물건이 가방에 있는 경우 (``TALK_OK_WITH_ITEM``)."""
    if ref is None:
        ref = pyboy_ref
    pyboy = ref.get("pyboy")
    if pyboy is None or not cheat_flags.get("bike_clerk_with_voucher", True):
        return []
    from core.gb_memory import has_bag_item
    return [name for name, item in TALK_OK_WITH_ITEM.items() if has_bag_item(pyboy, item)]


def _hm_status(ref=None) -> list:
    """비전머신을 쓸 수 있는지 · 못 쓰면 무엇이 없어서인지."""
    if ref is None:
        ref = pyboy_ref
    pyboy = ref.get("pyboy")
    if pyboy is None:
        return []
    try:
        return hm.field_move_status(pyboy)
    except Exception:
        return []


def _run_plan(gmap: GlobalMap, cur_map: int, *, apply: bool, pos=None):
    """plan / plan apply 본체."""
    # 맵 상태(탐험 완료/미완료)를 탐험 스킬의 판단에 맞춘다 (skill: bfs).
    _flush_cycle(_auto.get("goal"))
    if _DEAD_CYCLES["n"] >= _DEAD_CYCLE_LIMIT:
        # 몇 사이클째 아무 일도 못 했다.
        print(
            f"[auto] {_DEAD_CYCLES['n']} 사이클째 아무 진행이 없습니다 — "
            "막혔던 기억을 지우고 다시 봅니다"
        )
        _unblock_after_event(gmap)
        _DEAD_CYCLES["n"] = 0
    refresh = getattr(_explore_state.get("default"), "refresh_map_status", None)
    if callable(refresh):
        try:
            refresh(gmap, int(cur_map), pos)
        except Exception as exc:
            print(f"[plan] 맵 상태 갱신 실패: {type(exc).__name__}: {exc}")
    # 이벤트는 runtime/current_events.json → PlannerContext 파티 체력/PP 는 LLM 이 회복(포켓몬센터 목표)을 먼저 낼지 판단하는 근거다.
    ctx = PlannerContext.from_global_map(
        gmap,
        cur_map_id=int(cur_map),
        party_status=party_lines_from_ref(pyboy_ref),
        badges=badges_from_ref(pyboy_ref),
        missing_badges=_missing_badges(pyboy_ref),
        hm_status=_hm_status(pyboy_ref),
        talk_ok_npcs=_talk_ok_npcs(pyboy_ref),
        obstacles=list(_OBSTACLES),
        blocked_cells=_blocked_cells(gmap) if bool(planner_opt("blocked_cells", True)) else {},
        nav_fail=_take_nav_fail(),
    )
    ctx.plan_history = _plan_history_lines()
    prompt_path = dump_prompt_preview(ctx)
    print(f"\n[plan] prompt → {prompt_path}")
    print(f"  last_event : {ctx.last_event or '(없음)'}")
    print(f"  events     : {len(ctx.recent_events)}개 (current_events.json)")
    goal = _plan(ctx)
    if goal is None:
        return None
    goal = enrich_goal(goal, ctx)

    _last_goal["goal"] = goal
    # 계획 내용은 위 _print_planning 이 이미 찍었다 (여기서 또 찍지 않는다)
    n = record_plan(
        goal,
        current_map=ctx.current_map_name,
        last_event=ctx.last_event,
        cycle=int(_auto["cycle"]) if _auto.get("on") else None,
        party=list(ctx.party_status),
    )
    print(f"[plan] log → runtime/logging.json (#{n})")
    rec = _REC.get("rec")
    if rec is not None and rec.active:
        rec.hold()                    # 새 목표가 패널에 뜬 화면을 plan_hold_sec 초

    if not apply:
        print("  (이동하려면: plan apply)\n")
        return goal


    if goal.nav_map_id is None:
        print("[plan apply] nav 대상 id 없음 — catalog/known 이름 확인\n")
        return goal
    if goal.nav_map_id not in {int(m) for m in gmap.graph.nodes}:
        print(
            f"[plan apply] '{goal.nav_map_name}' 아직 GlobalMap에 없음. "
            "explore 로 관측을 넓히세요.\n"
        )
        _note_nav_fail(
            str(goal.nav_map_name or goal.target_map_name or "?"),
            ctx.current_map_name,
            "그 맵이 아직 관측 그래프에 없습니다",
        )
        return goal
    _plan_chain.clear()
    # 1) 맵 이동 (이미 그 맵이면 스킵은 runner가 처리)
    _plan_chain.append({"kind": "map", "id": int(goal.nav_map_id)})
    # 2) NPC Local Nav — **현재 맵에서 관측된** NPC 만 대상이 된다.
    nav_npc = nav_npc_for_goal(goal, _talk_ok_npcs())
    if nav_npc:
        _plan_chain.append({"kind": "npc", "name": nav_npc})
    _start_next_plan_step()
    return goal


def _npc_observed(name: str) -> bool:
    """현재 맵 LocalMap 에 그 NPC 좌표가 관측돼 있는지."""
    gmap = gmap_ref.get("gmap")
    cur_map = _game.get("map_id")
    if not name or gmap is None or cur_map is None:
        return False
    local = gmap.get_local_map(int(cur_map))
    if local is None:
        return False
    try:
        return bool(local.npc_coords(name))
    except Exception:
        return False


def _start_next_plan_step() -> None:
    """plan_chain 앞쪽 단계를 _cmd 로 넣는다."""
    if not _plan_chain:
        return
    step = _plan_chain.pop(0)
    if step["kind"] == "map":
        _cmd["nav_target"] = int(step["id"])
        _cmd["nav_local"] = None
        _cmd["action"] = "nav_start"
        print(f"[plan apply] nav map id={step['id']}")
    elif step["kind"] == "npc":
        # 상점 점원에게 말을 걸면 대화가 아니라 **구매 메뉴**가 뜬다.
        if is_no_talk_npc(step["name"], _talk_ok_npcs()):
            print(
                f"[plan apply] {step['name']!r} 는 상점 점원 — 대화하지 않습니다"
            )
            return
        # 아직 관측 전이면 Local A* 가 좌표를 못 찾아 3번 재시도하다 실패한다.
        if not _npc_observed(step["name"]):
            print(
                f"[plan apply] NPC {step['name']!r} 아직 관측 전 — "
                "NPC 단계 생략 (explore 로 관측 필요)"
            )
            return
        _cmd["nav_target"] = None
        _cmd["nav_local"] = {"mode": "npc", "npc_name": step["name"]}
        _cmd["action"] = "nav_local_start"
        print(f"[plan apply] nav npc {step['name']!r}")
    elif step["kind"] == "grass":
        _cmd["nav_target"] = None
        _cmd["nav_local"] = {"mode": "grass", "npc_name": None}
        _cmd["action"] = "nav_local_start"
        print("[plan apply] nav grass (Local)")
    elif step["kind"] == "heal":
        _start_heal_npc_step(step.get("npc_name"))


def _continue_plan_chain_after_nav() -> None:
    """맵/Local Nav 성공 후 남은 plan_chain 단계 시작."""
    if _plan_chain:
        _start_next_plan_step()


# ===========================================================================
# debug — 한 걸음마다 LocalMap 찍기 (평소 off)
# ===========================================================================
def _set_debug(on: bool) -> None:
    """경로 계산 과정 출력을 한 번에 켜고 끈다."""
    _debug_flags["localmap"] = on
    nav_debug_flags["matrix"] = on          # Skills/Navigation — 행렬·경로 시각화
    runner_debug_flags["nav_steps"] = on    # engine/runners — [nav] step N


def _handle_debug_line(line: str) -> None:
    parts = line.split(maxsplit=1)
    opt = parts[1].strip().lower() if len(parts) > 1 else "status"
    if opt in ("on", "off"):
        _set_debug(opt == "on")
    elif opt != "status":
        print("[debug] 사용법: debug on | debug off | debug status")
        return
    state = "on" if _debug_flags["localmap"] else "off"
    print(f"[debug] 경로 계산 상세 출력: {state}")
    print("        (LocalMap 전체 · A* 행렬 · 경로 시각화 · step 로그)")


# auto — LLM plan → Navigation → Exploration 반복 (사람 입력 없음)  plan ─► nav(목표 맵 [→ 목표 NPC]) ─► explore…
_SEP_AUTO = "=" * 64
_SEP_STEP = "-" * 64


def _auto_pos(gmap: GlobalMap, map_id: int, x: int, y: int) -> str:
    return f"{gmap.map_name(int(map_id))} (id={int(map_id)}) ({x},{y})"


def _auto_stage_header(stage: str) -> None:
    """auto 한 단계의 머리말."""
    print(f"\n{_SEP_STEP}")
    print(f" [{_auto['cycle']}] {stage}")
    print(_SEP_STEP)


def _auto_nav_header(gmap: GlobalMap, cur_map: int, x: int, y: int, goal) -> None:
    _auto_stage_header("Skill - Navigation")
    print(f"  현재 위치 : {_auto_pos(gmap, cur_map, x, y)}")
    print(f"  목표 위치 : {goal.nav_map_name or goal.target_map_name or '?'}")
    print(f"  목표 NPC  : {nav_npc_for_goal(goal, _talk_ok_npcs()) or '없음'}")


def _auto_explore_header(gmap: GlobalMap, cur_map: int, x: int, y: int, goal) -> None:
    _auto_stage_header("Skill - Exploration")
    print(f"  현재 위치             : {_auto_pos(gmap, cur_map, x, y)}")
    if goal is None:
        return
    print(f"  찾아야 하는 목표 위치 : {goal.target_map_name or '?'}")
    print(f"  찾아야 하는 목표 NPC  : {goal.target_npc or '없음'}")
    if goal.todo:
        print(f"  다음 해야할 일: {goal.todo}")


# 시연 영상 녹화 (recording.enabled). auto 를 켤 때 시작, 끌 때 끝. main() 이 만든다.
_REC: dict = {"rec": None}
# 패널에 넣는 이벤트·배지는 파일/메모리를 읽으므로 1초에 한 번만 새로 읽는다
_REC_CACHE: dict = {"t": 0.0, "events": [], "badges": []}
_PHASE_LABEL = {
    "plan": "LLM Planning",
    "nav_wait": "Navigation",
    "explore": "Exploration",
    "explore_wait": "Exploration",
}


def _handle_rec(sub: str) -> None:
    """rec on | rec off | rec — 메인 루프에서 부른다 (녹화기 훅은 tick 사이에만 건다)."""
    rec = _REC.get("rec")
    if rec is None:
        print("[rec] 녹화기가 아직 준비되지 않았습니다")
        return
    if sub == "on":
        if rec.active:
            print(rec.status_line())
        else:
            rec.start("play")
    elif sub == "off":
        if not rec.stop("rec off"):
            print("[rec] 녹화 중이 아닙니다")
    else:
        print(rec.status_line())


def _rec_status() -> list:
    """녹화 패널 줄들 (engine/recorder.py)."""
    now = time.time()
    if now - _REC_CACHE["t"] >= 1.0:
        _REC_CACHE["t"] = now
        try:
            _REC_CACHE["events"] = list_event_names()[-3:]
        except Exception:
            pass
        try:
            _REC_CACHE["badges"] = list(badges_from_ref(pyboy_ref))
        except Exception:
            pass
    gmap = gmap_ref.get("gmap")
    pyboy = pyboy_ref.get("pyboy")
    here = "?"
    if _auto.get("on"):
        phase = _PHASE_LABEL.get(str(_auto.get("phase") or ""), str(_auto.get("phase") or "-"))
    else:
        phase = "수동 조작"
    if pyboy is not None:
        try:
            here = gmap.map_name(int(pyboy.memory[ADDR_MAP_ID])) if gmap is not None else "?"
            if is_in_battle(pyboy):
                phase = "Battle"
        except Exception:
            pass
    goal = _auto.get("goal") or _last_goal.get("goal")
    lines = [
        "Pokemon Red HRL Agent",
        f"사이클 {int(_auto.get('cycle') or 0)} · 스텝 {int(_auto.get('steps') or 0)}",
        f"# 단계  {phase}",
        f"현재 맵  {here}",
    ]
    if goal is not None:
        lines += [
            f"# 목표 맵  {getattr(goal, 'target_map_name', '') or '-'}",
            f"목표 NPC  {getattr(goal, 'target_npc', '') or '-'}",
            f"스킬  {getattr(goal, 'skill', '') or '-'}",
            f"할 일  {getattr(goal, 'todo', '') or '-'}",
        ]
    badges = _REC_CACHE["badges"]
    lines.append(f"배지  {len(badges)}/8")
    if _REC_CACHE["events"]:
        lines.append("최근 이벤트")
        lines += [f"  · {name}" for name in _REC_CACHE["events"]]
    return lines


def _auto_start(gmap: GlobalMap, cur_map: int, x: int, y: int) -> None:
    """auto 시작 — 기본 정보를 찍고 상태 기계를 plan 부터 돌린다."""
    _auto.update(
        on=True,
        phase="plan",
        goal=None,
        cycle=0,
        nav_ok=False,
        npc_talked=False,
        plan_fails=0,
        explore_skipped=False,
        obs_before=None,
        max_cycles=int(auto_opt("max_cycles", 100)),
        steps=0,
        max_steps=max(0, int(auto_opt("max_steps", 0) or 0)),
    )
    # 여기서 로그를 비우지 않는다.
    runlog.log("session", note=f"auto 시작 — {gmap.map_name(cur_map)}")
    events = list_event_names()
    last = get_last_event()
    print(f"\n{_SEP_AUTO}")
    print(" AUTO 모드 — LLM plan → Navigation → Exploration 반복")
    print("            스토리 이벤트면 nav/explore 를 끊고 즉시 Planning")
    print(_SEP_AUTO)
    print(f"  LLM 플래너     : {_planner_line()}")
    print(f"  현재 위치      : {_auto_pos(gmap, cur_map, x, y)}")
    print(f"  감지된 이벤트  : {len(events)}개" + (f" · 최근 = {last}" if last else ""))
    print(f"  관측한 맵      : {len(gmap.graph.nodes)}개")
    print(f"  최대 사이클    : {_auto['max_cycles']}  (넘으면 자동 중단)")
    step_limit = _auto["max_steps"] or "제한 없음"
    print(f"  최대 스텝      : {step_limit}  (Nav/Local Nav/Exploration 액션 합계)")
    print(
        f"  LocalMap 출력  : {'on' if _debug_flags['localmap'] else 'off'}"
        "  (debug on/off)"
    )
    print("  기록           : runtime/logging.json (play.py 실행 단위로 누적)")
    rec = _REC.get("rec")
    rec_line = "녹화 중" if rec is not None and rec.active else "꺼짐"
    print(f"  녹화           : {rec_line}  (rec on|off — auto 와 따로)")
    print("  중단           : stop")
    print(_SEP_AUTO)


def _auto_stop(reason: str) -> None:
    if not _auto["on"]:
        return
    _auto["on"] = False
    _auto["goal"] = None
    print(f"\n{_SEP_AUTO}")
    print(
        f" AUTO 모드 종료 — {reason} "
        f"(사이클 {_auto['cycle']}, 스텝 {_auto['steps']})"
    )
    print(f"{_SEP_AUTO}\n")


def _auto_record_step() -> bool:
    """완료된 Nav/Explore 액션을 누적하고 상한 도달 여부를 반환한다."""
    _SESSION["steps"] += 1
    if not _auto["on"]:
        return False
    _auto["steps"] += 1
    limit = int(_auto.get("max_steps") or 0)
    if limit <= 0 or _auto["steps"] < limit:
        return False
    _auto_stop(f"최대 스텝 {limit} 도달")
    _plan_chain.clear()
    return True


def _auto_begin_replan(*, reason: str) -> None:
    """진행 중이던 nav/explore/plan_chain 을 버리고 다음 유휴 때 plan 한다."""
    if not _auto["on"]:
        return
    _plan_chain.clear()
    _cmd["action"] = None
    _cmd["nav_target"] = None
    _cmd["nav_local"] = None
    _cmd.pop("plan_apply", None)
    _auto["phase"] = "plan"
    _auto["goal"] = None
    print(f"[auto] {reason} — 즉시 Planning")


# `explore` 를 콘솔에서 직접 쳤을 때 쓸 "무제한".
_EXPLORE_NO_LIMIT = ExploreRunner.NO_LIMIT_STEPS


def _explore_step_limit(manual: bool) -> int:
    """이번 Exploration 에피소드의 스텝 상한."""
    if not manual:
        limit = int(exploration_opt("max_steps", 900))
        print(f"[explore] 스텝 상한 {limit} (auto — exploration.max_steps)")
        return limit
    raw = int(exploration_opt("max_steps_manual", 0) or 0)
    if raw <= 0:
        print("[explore] 스텝 상한 없음 — stop 을 치거나 최종 종료 조건에 닿으면 끝납니다")
        return _EXPLORE_NO_LIMIT
    print(f"[explore] 스텝 상한 {raw} (exploration.max_steps_manual)")
    return raw


def _obs_size(gmap: GlobalMap) -> tuple:
    """지금까지 관측한 규모 — (아는 맵 수, 관측한 칸 총합)."""
    try:
        nodes = list(gmap.graph.nodes)
    except Exception:
        return (0, 0)
    cells = 0
    for mid in nodes:
        local = gmap.get_local_map(int(mid))
        if local is not None:
            cells += len(getattr(local, "grid", None) or {})
    return (len(nodes), cells)


# 마지막으로 돌린 탐험 스킬 (로그용)
_explore_state: dict = {"kind": None}


def _choose_explore_skill(gmap: GlobalMap, pyboy, cur_map: int, x: int, y: int):
    """도착한 맵에서 무엇을 돌릴지 **코드가** 정한다 (LLM 이 뭘 골랐든)."""
    closed = explore_gate.check(gmap, pyboy, int(cur_map), (int(x), int(y)))

    if not bool(exploration_opt("auto_split", False)):
        # 스킬이 스스로 할 일을 판단할 수 있으면(skill: bfs) 그 답을 쓴다.
        has_work = getattr(_explore_state.get("default"), "has_work", None)
        if callable(has_work):
            try:
                return "exploration" if has_work(gmap, int(cur_map), (int(x), int(y))) else None
            except Exception as exc:
                print(f"[explore] 할 일 판정 실패: {type(exc).__name__}: {exc}")
        # 종료 조건은 안 지나가 본 문이 남아 있으면 성립하지 않으므로, 분할 전에도 "문이 남았으면 계속 탐험한다"가 그대로 지켜진다.
        return None if closed else "exploration"

    can_cut, can_surf = explore_gate.capabilities(pyboy)
    try:
        targets = seek_targets(gmap, int(cur_map), (int(x), int(y)), can_cut=can_cut, can_surf=can_surf)
    except Exception as exc:
        print(f"[explore] 워프 목표 조회 실패: {type(exc).__name__}: {exc}")
        targets = []
    if targets:
        return "seekwarp"
    if closed:
        return None
    return "coverage"


def _explore_skill_for(kind, skills: dict, default):
    """kind → (스킬 객체, 라벨)."""
    if kind in ("seekwarp", "coverage") and kind not in skills:
        maker = (skills.get("_make") or {}).get(kind)
        skills[kind] = None
        if maker is not None:
            try:
                skills[kind] = maker()
            except Exception as exc:
                print(f"[explore] {kind} 초기화 실패: {type(exc).__name__}: {exc}")
    if kind == "seekwarp" and skills.get("seekwarp") is not None:
        return skills["seekwarp"], "seekwarp"
    if kind == "coverage":
        cov = skills.get("coverage")
        path = getattr(cov, "model_path", None) if cov is not None else None
        if cov is not None and path and Path(path).is_file():
            return cov, "coverage"
        print("[explore] Coverage 모델이 없어 기존 Exploration 으로 대신합니다 "
              "(Train/explore 로 학습한 model_*.zip 을 넣으면 바뀝니다)")
    return default, str(_explore_state.get("default_label") or "exploration")


def _auto_step(gmap: GlobalMap, cur_map: int, x: int, y: int) -> None:
    """auto 상태 기계 한 단계."""
    phase = _auto["phase"]

    if phase == "plan":
        _auto["cycle"] += 1
        if _auto["cycle"] > _auto["max_cycles"]:
            _auto_stop(f"최대 사이클 {_auto['max_cycles']} 도달")
            return
        _auto_stage_header("LLM - Planning")
        goal = _run_plan(gmap, cur_map, apply=True, pos=(int(x), int(y)))
        if goal is None:
            # 플래너 실패로는 auto 를 멈추지 않는다.
            _auto["cycle"] -= 1
            _auto["plan_fails"] += 1
            n = _auto["plan_fails"]
            runlog.log(
                "plan_fail",
                cycle=int(_auto["cycle"]) + 1,
                스텝=int(_SESSION["steps"]),
                연속실패=n,
            )
            print(f"[auto] 플래너 실패 {n}회 — 계속 다시 시도합니다 (중단: stop)")
            if n % AUTO_PLAN_FAIL_HINT_EVERY == 0:
                print(
                    "[auto] 계속 실패하고 있습니다 — 백엔드/키 설정을 확인해 "
                    "보세요 (planner.backend / settings/.env)"
                )
            return  # 다음 프레임에 plan 단계로 재진입
        _auto["plan_fails"] = 0
        _auto["goal"] = goal
        if _cmd.get("action") is None and not _plan_chain:
            # nav 를 걸지 못했다(목표 맵이 아직 GlobalMap 에 없음 등) → 관측부터
            _auto["phase"] = "explore"
            _auto_step(gmap, cur_map, x, y)
            return
        _auto["nav_ok"] = False
        _auto["npc_talked"] = False
        _auto["phase"] = "nav_wait"
        _auto_nav_header(gmap, cur_map, x, y, goal)
        return

    if phase == "nav_wait":
        # NPC 대화까지 끝났어도 explore 를 건너뛰지 않는다.
        _auto["phase"] = "explore"
        _auto_step(gmap, cur_map, x, y)
        return

    if phase == "explore":
        kind = _choose_explore_skill(gmap, pyboy_ref["pyboy"], cur_map, x, y)
        if kind is None:
            here = gmap.map_name(cur_map)
            print(
                f'[auto] 탐험 스킬이 "{here}" 에서 할 일이 없다고 합니다 '
                "(더 볼 경계도, 안 지나간 출구도 없음) — 탐험 없이 다시 계획합니다"
            )
            if not _auto.get("npc_talked"):
                # 대화도 없었고 탐험도 못 했다 = 이 사이클은 아무 일도 하지 않았다.
                _note_nav_fail(
                    str(getattr(_auto.get("goal"), "target_map_name", "") or here),
                    here,
                    f'"{here}" 에서 탐험 스킬이 더 볼 것이 없다고 했고, '
                    "말을 걸 NPC 도 지정되지 않아 이번 사이클에는 아무 일도 "
                    "일어나지 않았습니다",
                )
            _auto["explore_skipped"] = True
            _auto["phase"] = "plan"
            _auto_step(gmap, cur_map, x, y)
            return
        _auto_explore_header(gmap, cur_map, x, y, _auto["goal"])
        _KIND_NOTE = {
            "seekwarp": "  (안 가본 워프 통과)",
            "coverage": "  (격자 넓히기)",
        }
        print(f"[auto] 탐험 스킬 = {kind}" + _KIND_NOTE.get(kind, ""))
        # auto 는 한 사이클이 끝나야 다시 계획하므로 상한을 지킨다.
        _cmd["explore_manual"] = False
        _cmd["explore_kind"] = kind
        _cmd["action"] = "explore_start"
        _auto["obs_before"] = _obs_size(gmap)
        _auto["explore_skipped"] = False
        _auto["phase"] = "explore_wait"
        return

    if phase == "explore_wait":
        # **한 번만 굴린다.** 관측이 안 늘었으면 같은 스킬을 또 돌리는 것보다
        # 계획을 다시 세우는 편이 낫다 — 같은 자리에서 맴돌던 이유(닿지 않는
        # 경계 등)는 한 번 더 굴린다고 달라지지 않는다.
        grew = _obs_size(gmap) != _auto.get("obs_before")
        skipped = bool(_auto.get("explore_skipped"))
        if not grew and not skipped:
            print("[auto] 탐험했는데 관측이 안 늘었습니다 — 다시 계획합니다")
        _auto["phase"] = "plan"
        _auto_step(gmap, cur_map, x, y)
        return

    _auto_stop(f"알 수 없는 단계 {phase!r}")


def _start_heal_npc_step(npc_name: Optional[str] = None) -> None:
    """현재 맵에서 Nurse NPC Local Nav (이름 없으면 Navigation이 재해석)."""
    gmap = gmap_ref.get("gmap")
    cur_map = _game.get("map_id")
    nurse = (npc_name or "").strip() if npc_name else ""
    if not nurse and gmap is not None and cur_map is not None:
        nurse = Navigation(gmap).resolve_pokecenter_nurse_name(int(cur_map)) or ""
    if not nurse:
        print("[nav heal] Nurse NPC를 찾지 못함 (맵 스캔/npc_objects 확인)")
        _plan_chain.clear()
        return
    print(f"[nav heal] Nurse → {nurse!r}")
    _cmd["nav_target"] = None
    _cmd["nav_local"] = {"mode": "npc", "npc_name": nurse}
    _cmd["action"] = "nav_local_start"


def _apply_nav_mission(mission: NavMission) -> None:
    """Navigation 스킬이 만든 미션 → plan_chain / _cmd 실행."""
    print(mission.summary)
    _plan_chain.clear()
    for step in mission.steps:
        kind = step.kind
        if kind == "map":
            _plan_chain.append({"kind": "map", "id": int(step.map_id)})
        elif kind == "grass":
            _plan_chain.append({"kind": "grass"})
        elif kind == "heal":
            _plan_chain.append({"kind": "heal", "npc_name": step.npc_name})
        elif kind == "npc":
            _plan_chain.append({"kind": "npc", "name": step.npc_name})
        else:
            print(f"[nav] 알 수 없는 step kind={kind!r}")
            _plan_chain.clear()
            return
    _start_next_plan_step()


def _guard_no_talk(name: str) -> bool:
    """상점 점원이면 안내하고 True."""
    if not is_no_talk_npc(name, _talk_ok_npcs()):
        return False
    print(
        f"[nav] {name!r} 는 상점 점원입니다 — 말을 걸면 구매 메뉴가 열려 "
        "빠져나올 수 없어 대화하지 않습니다"
    )
    return True


def _start_nav_grass() -> None:
    gmap = gmap_ref.get("gmap")
    cur_map = _game.get("map_id")
    if gmap is None or cur_map is None:
        print("[nav grass] GlobalMap/위치 아직 없음")
        return
    mission = Navigation(gmap).plan_to_nearest_grass(
        int(cur_map),
        (int(_game.get("x") or 0), int(_game.get("y") or 0)),
    )
    if mission is None:
        print("[nav grass] Known 맵에 관측된 GRASS 없음")
        return
    _apply_nav_mission(mission)


def _start_nav_heal() -> None:
    gmap = gmap_ref.get("gmap")
    cur_map = _game.get("map_id")
    if gmap is None or cur_map is None:
        print("[nav heal] GlobalMap/위치 아직 없음")
        return
    mission = Navigation(gmap).plan_to_nearest_heal(
        int(cur_map),
        (int(_game.get("x") or 0), int(_game.get("y") or 0)),
    )
    if mission is None:
        print(
            "[nav heal] Known 맵에 Pokemon Center 없음 "
            "(방문해 지도에 올린 뒤 다시 시도)"
        )
        return
    _apply_nav_mission(mission)


def _handle_nav_line(line: str) -> None:
    parts = line.split(maxsplit=2)
    arg = parts[1].strip() if len(parts) > 1 else ""
    rest = parts[2].strip() if len(parts) > 2 else ""

    if arg.lower() == "stop":
        _cmd["action"] = "stop"
        _plan_chain.clear()
        return

    if arg.lower() == "grass":
        _start_nav_grass()
        return

    if arg.lower() == "heal":
        _start_nav_heal()
        return

    if arg.lower() == "npc":
        if not rest:
            print("[nav npc] 사용: nav npc <이름>  (npc 로 목록 확인)")
            return
        gmap = gmap_ref.get("gmap")
        cur_map = _game.get("map_id")
        if gmap is None or cur_map is None:
            print("[nav npc] GlobalMap/위치 아직 없음")
            return
        mission = Navigation(gmap).plan_to_npc(int(cur_map), rest)
        if mission is None:
            print(f"[nav npc] NPC {rest!r} 좌표 없음 (npc 명령으로 목록 확인)")
            return
        _apply_nav_mission(mission)
        return

    gmap = gmap_ref.get("gmap")
    cur_map = _game.get("map_id")
    if gmap is None:
        print("[nav] GlobalMap 아직 없음")
        return

    if not arg:
        targets = print_target_list(gmap, cur_map)
        if not targets:
            return
        try:
            arg = input("목표를 입력해주세요: ").strip()
        except EOFError:
            return
        if not arg:
            print("[nav] 취소")
            return

    map_id = resolve_target(gmap, arg)
    if map_id is None:
        return
    _plan_chain.clear()
    _cmd["nav_target"] = map_id
    _cmd["nav_local"] = None
    _cmd["action"] = "nav_start"


# ===========================================================================
# 메인
# ===========================================================================
def main() -> None:
    pyboy = open_emulator(
        emulation_speed=int(_CFG.get("emulation_speed", 3))
    )

    gmap = GlobalMap()
    gmap_ref["gmap"] = gmap
    # 플래너가 파티 체력/PP 를 읽을 수 있도록 핸들을 공유한다.
    pyboy_ref["pyboy"] = pyboy
    # 시연 영상 — rec on 부터 rec off 까지 (auto 와 따로). 훅은 처음 rec on 할 때 건다.
    _REC["rec"] = DemoRecorder.from_config(pyboy, status=_rec_status)
    if bool(recording_opt("start_on_launch", False)):
        _REC["rec"].start("play")                # recording.start_on_launch

    # 세션마다 current_events.json 리셋 + (설정 시) 이미 켜진 플래그 시드 치트가 켠 이벤트는 current_events.json 에 넣지 않는다
    from engine import cheats as _cheats_mod
    event_watcher = EventWatcher(cheat_bits=lambda: _cheats_mod.CHEAT_EVENT_BITS)
    seed = bool(event_opt("seed_already_set"))
    seeded = event_watcher.reset_session(pyboy, seed_already_set=seed)
    reset_plan_log()
    print(
        f"[event] current_events.json 초기화 "
        f"(seed={seed}, 시드 {len(seeded)}개, catalog={len(event_watcher.catalog)})"
    )
    print("[plan] runtime/logging.json 초기화 (이번 세션)")
    cheats = _CFG.get("cheats") or {}
    expl = _CFG.get("exploration") or {}
    battle_cfg = _CFG.get("battle") or {}
    print(
        f"[config] wild_off={cheats.get('disable_wild_encounters')} "
        f"auto_win={cheats.get('auto_win_battle')} "
        f"fossils_off={cheats.get('skip_mt_moon_fossils')} "
        f"vermilion_puzzle_off={cheats.get('skip_vermilion_gym_puzzle')} "
        f"bill_pc_off={cheats.get('skip_bill_cell_separator')} "
        f"noclip={cheats.get('noclip')} speed={_CFG.get('emulation_speed')}"
    )
    # explore 중에는 학습 파리티를 위해 cheat_profile 이 위 wild_off 보다 우선한다
    print(
        f"[config] explore: skill={exploration_skill_label()} "
        f"max_steps={expl.get('max_steps')} "
        f"device={expl.get('device')} cheats={expl.get('cheat_profile')} "
        f"| auto: max_cycles={(_CFG.get('auto') or {}).get('max_cycles')} "
        f"max_steps={(_CFG.get('auto') or {}).get('max_steps')}"
    )
    print(
        f"[config] battle: enabled={battle_cfg.get('enabled')} "
        f"model={battle_cfg.get('model')} "
        f"device={battle_cfg.get('device')} "
        f"max_turns={battle_cfg.get('max_turns')}"
    )
    party_msg = apply_party_mon_from_config(pyboy)
    if party_msg:
        print(f"[cheat] party_mon: {party_msg}")

    nav_runner = None
    explore_runner = None
    battle_runner = None
    battle_manual = False
    # 한 번 끝난 배틀은 wIsInBattle 이 0 을 찍기 전까지 다시 시작하지 않는다.
    battle_settled = False
    # 배틀인데 아무도 진행시키지 않고 흐른 프레임 수 (아래 감시용)
    battle_idle = 0
    # 콘솔 `explore` 의 기본 스킬은 settings/config.yaml 의 exploration.skill 이 정한다.
    _explore_choice = exploration_skill_choice()
    _rule_kwargs: dict = {}
    if _explore_choice in RULE_BASED:
        _rule_kwargs["capabilities"] = lambda: explore_gate.capabilities(pyboy)
    if _explore_choice == "bfs":
        # 동굴 예외(exploration.bfs_free_roam_tilesets)가 지금 서 있는 맵의 타일셋을 본다.
        _rule_kwargs["tileset"] = lambda: explore_gate.current_tileset(pyboy)
    explore_skill = make_exploration_skill(**_rule_kwargs)
    _explore_state["default"] = explore_skill
    _explore_state["default_label"] = "bfs" if _explore_choice == "bfs" else "exploration"
    # seekwarp / coverage 는 exploration.auto_split 이나 `explore seekwarp|coverage` 에서만 쓴다.
    explore_skills: dict = {
        "_make": {
            "seekwarp": lambda: make_exploration_skill(
                "seekwarp", capabilities=lambda: explore_gate.capabilities(pyboy)
            ),
            "coverage": lambda: make_exploration_skill("coverage"),
        },
    }
    battle_skill = None
    if bool(battle_opt("enabled", True)):
        try:
            battle_skill = make_battle_skill(pyboy)
        except Exception as exc:
            print(
                "[battle] 초기화 실패 - 이번 세션은 수동 배틀: "
                f"{type(exc).__name__}: {exc}"
            )
    # 레벨업 기술 교체 정책.
    try:
        move_learn = MoveLearnResolver(
            battle_opt("on_move_learn", "manual"),
            debug=bool(battle_opt("debug", False)),
        )
    except Exception as exc:
        print(f"[move-learn] 설정 오류 - 수동 위임: {exc}")
        move_learn = MoveLearnResolver("manual")
    # 배틀 밖(진화 직후 학습, 이상한사탕)에서도 같은 정책을 쓰도록 DialogueSkipper 들이 참조하는 세션 기본값으로 등록한다.
    set_default_resolver(move_learn)
    # battle.on_failure — manual: 이번 배틀만 수동 / disable: 이번 세션 자동배틀 끔
    battle_on_failure = str(battle_opt("on_failure", "amash")).strip().lower()
    # explore 를 여러 번 쳐도 같은 프록시를 넘기기 위해 세션당 하나만 만든다
    explore_observed = ObservingPyBoy(pyboy, gmap)
    # auto 단계 전환 직후에는 러너가 하나도 없다.
    auto_dialogue = DialogueSkipper(pyboy, tag="auto")
    auto_battle_paused = False
    # 연출(페이드·스크립트 이동·맵 전환)이 끝나고 가라앉은 뒤에 다음 단계로 (auto.settle_frames, 0=끔)
    settle_gate = SettleGate(int(auto_opt("settle_frames", 60) or 0))
    settle_announced = False
    settle_wait_count = 0

    threading.Thread(target=_input_thread, daemon=True).start()
    _print_skills()

    view_gate = ViewGate(pyboy)
    event_poll_counter = 0

    print("-" * 64)
    print("play.py — 사람(플래너) + Skills + Map")
    print(f"[planner] {_planner_line()}")
    print("스킬 중지: stop   |  종료: Ctrl+C")
    print("-" * 64)

    try:
        while True:
            # Explore/Battle은 내부에서 tick하므로 바깥 tick과 중복하지 않는다.
            exploring_rl = (
                explore_runner is not None and explore_runner.owns_emulator_ticks()
            )
            battling_rl = (
                battle_runner is not None
                and battle_runner.owns_emulator_ticks()
            )
            if not exploring_rl and not battling_rl:
                if not pyboy.tick():
                    break

            disable_wild_encounters(pyboy)
            apply_auto_win_battle(pyboy)     # cheats.auto_win_battle
            apply_noclip_cheat(pyboy)
            apply_skip_mt_moon_fossils(pyboy)
            apply_skip_bill_cell_separator(pyboy)
            apply_wake_snorlax(pyboy)          # cheats.wake_snorlax
            apply_buy_saffron_drink(pyboy)     # cheats.buy_saffron_drink
            if apply_skip_game_corner_poster(pyboy):   # cheats.skip_game_corner_poster
                _mark_new_stairs(gmap, MAP_ID_GAME_CORNER, "ROCKET_HIDEOUT_B1F")
            if apply_skip_vermilion_gym_puzzle(pyboy):
                # 문이 열려 지형이 바뀌었다 — 그 체육관의 "다 봤음" 표시를 지운다
                gym_local = gmap.get_local_map(92)
                if gym_local is not None:
                    gym_local.closed = False
            # 비전머신 능력 동기화 (배지/기술).
            hm.sync(pyboy)
            gmap.update_from_pyboy(pyboy)
            settle_gate.update(pyboy, gmap)

            # wEventFlags 0→1 감지 → runtime/current_events.json (이름만) 관장·라이벌만 기록하고 auto replan.
            event_poll_counter += 1
            poll_now = _auto["on"] or event_poll_counter >= EVENT_POLL_FRAMES
            if poll_now:
                event_poll_counter = 0
                new_events, skipped_events = event_watcher.poll(pyboy)
                for ev_name in skipped_events:
                    print(f"[event] skip {ev_name}")
                # **치트로 켠 이벤트도 스토리 진행이다.** 기록(current_events.json)에는
                # 넣지 않지만, 막혀 있던 길이 열렸을 수 있다는 신호로는 같다.
                if new_events or event_watcher.cheated:
                    _unblock_after_event(gmap, pyboy)
                for ev_name in new_events:
                    steps = int(_SESSION["steps"])
                    print(f"[event] + {ev_name}  ({steps} 스텝째)")
                    _cycle_note(f"이벤트 발생: {ev_name}")
                    runlog.log(
                        "event",
                        이름=ev_name,
                        스텝=steps,
                        cycle=int(_auto["cycle"]) if _auto.get("on") else None,
                        맵=gmap.map_name(int(_game.get("map_id") or 0)),
                    )
                if new_events and _auto["on"]:
                    shown = ", ".join(new_events[:4])
                    extra = (
                        f" 외 {len(new_events) - 4}개"
                        if len(new_events) > 4
                        else ""
                    )
                    if nav_runner is not None:
                        nav_runner.stop("새 이벤트 — replan")
                        nav_runner = None
                    if explore_runner is not None:
                        explore_runner.stop("새 이벤트 — replan")
                        explore_runner = None
                    _auto_begin_replan(
                        reason=f"이벤트 {len(new_events)}개 ({shown}{extra})"
                    )

            cur_map, cur_x, cur_y = read_player(pyboy)
            _game["map_id"] = cur_map
            _game["x"] = cur_x
            _game["y"] = cur_y

            # wIsInBattle: 0=전투 아님 1=야생 2=트레이너 0xFF=전멸 후 블랙아웃.
            battle_raw = battle_phase(pyboy)
            in_battle = battle_raw != BATTLE_NONE
            battle_now = battle_kind(pyboy)

            # 진화는 EndOfBattle 이 wIsInBattle 을 0 으로 만들기 **전에** 일어난다 (end_of_battle.asm 이 EvolutionAfterBattle 를 먼저…
            if battle_raw == BATTLE_NONE:
                battle_settled = False

            # 배틀 제어권은 중앙 BattleRunner가 가진다.
            if (
                battle_runner is None
                and not battle_manual
                and not battle_settled
                and battle_now != "NONE"
                and battle_now != "BLACKOUT"
                # 이겨 놓고 시작하는 치트를 켜면 Battle 스킬은 부르지 않는다.
                and (
                    not cheat_flags.get("auto_win_battle", False)
                    or is_unbeatable_ghost(pyboy)
                )
            ):
                if nav_runner is not None:
                    nav_runner.pause_for_battle()
                if explore_runner is not None:
                    explore_runner.pause_for_battle()
                auto_dialogue.release()

                if battle_now == "GHOST" and cheat_flags.get("ghost_flee", True):
                    # 실프스코프 없는 유령전 — 기술이 하나도 안 통한다.
                    print("[battle] 유령전 — 공격이 통하지 않아 도망칩니다")
                    fled = flee_battle(pyboy, battle_skill)
                    runlog.log(
                        "battle_end",
                        종류="GHOST",
                        맵=gmap.map_name(int(cur_map)),
                        결과="도망" if fled else "도망 실패",
                    )
                    if fled:
                        battle_settled = True
                        continue
                    battle_manual = True
                    print("[battle] 도망치지 못했습니다 — 직접 조작하세요")
                elif battle_now not in PLAYABLE_BATTLE_KINDS:
                    # 사파리(메뉴에 FIGHT 가 없다) / 올드맨(게임이 입력을 강제) / 고스트(모든 기술 무효)는 PPO 액션 공간이 성립하지 않는다.
                    battle_manual = True
                    print(
                        f"[battle] {battle_now} - PPO 액션 공간 밖 전투입니다. "
                        "직접 조작하세요"
                    )
                elif battle_skill is None:
                    battle_manual = True
                    print("[battle] 자동 Battle Skill 없음 - 직접 싸우세요")
                else:
                    candidate = BattleRunner(
                        pyboy,
                        battle_skill,
                        move_learn=move_learn,
                        on_failure=battle_on_failure,
                    )
                    if candidate.start():
                        battle_runner = candidate
                        runlog.log(
                            "battle_start",
                            cycle=(
                                int(_auto["cycle"]) if _auto.get("on") else None
                            ),
                            스텝=int(_SESSION["steps"]),
                            종류=battle_now,
                            맵=gmap.map_name(int(cur_map)),
                        )
                    elif candidate.manual_handoff:
                        battle_manual = True
                        if battle_on_failure == "disable":
                            battle_skill = None
                            print("[battle] on_failure=disable - 이번 세션 자동배틀 끔")
                    else:
                        battle_settled = True
                    # 준비 중 배틀이 끝난 경우에는 다음 루프에서 정상 재개.
                    continue

            if battle_runner is not None:
                battle_runner.tick()
                if battle_runner.finished:
                    runlog.log(
                        "battle_end",
                        cycle=(
                            int(_auto["cycle"]) if _auto.get("on") else None
                        ),
                        스텝=int(_SESSION["steps"]),
                        success=bool(battle_runner.success),
                        턴=int(getattr(battle_runner, "turns", 0) or 0),
                        수동전환=bool(battle_runner.manual_handoff),
                    )
                    if battle_runner.manual_handoff:
                        battle_manual = True
                        if battle_on_failure == "disable":
                            battle_skill = None
                            print("[battle] on_failure=disable - 이번 세션 자동배틀 끔")
                    # 진화 연출이 남아 있을 수 있으므로 같은 전투 플래그 구간에서 다시 시작하지 않는다.
                    battle_settled = True
                    battle_runner = None
                # BattleMacro가 여러 프레임을 진행했으므로 좌표/이벤트를 다음 메인 루프에서 다시 읽기 전에는 overworld runner를 돌리지 않는다.
                continue

            # 러너가 손을 뗐는데 배틀이 그대로면 **아무도 A 를 누르지 않는다**.
            if in_battle and battle_runner is None and not battle_manual:
                battle_idle += 1
                # 치트로 이겨 놓은 전투는 곧바로 A 로 넘긴다 (기다릴 이유가 없다)
                idle_start = (
                    BATTLE_IDLE_A_EVERY
                    if cheat_flags.get("auto_win_battle", False)
                    else BATTLE_IDLE_FRAMES
                )
                if battle_idle == idle_start and not cheat_flags.get("auto_win_battle", False):
                    print("[battle] 진행하는 주체가 없는 배틀 — A 로 마무리합니다")
                if (
                    battle_idle >= idle_start
                    and battle_idle % BATTLE_IDLE_A_EVERY == 0
                ):
                    info = move_learn_menu_open(pyboy)
                    if info is None:
                        pyboy.button("a", 6)
                    elif move_learn is not None and getattr(move_learn, "enabled", False):
                        # 기술 교체 화면에서 A 는 "슬롯 0 삭제" 다 — 여기선 누르지 않는다
                        try:
                            move_learn.resolve(pyboy, info)
                        except Exception as exc:
                            print(f"[battle] 기술 교체 처리 실패: {type(exc).__name__}: {exc}")
            else:
                battle_idle = 0

            if not in_battle and battle_manual:
                battle_manual = False
                print("[battle] 수동 배틀 종료 - 기존 Skill 재개")

            rec_cmd = _cmd.get("rec")
            if rec_cmd:
                _cmd["rec"] = None
                _handle_rec(rec_cmd)

            action = _cmd.get("action")
            if action:
                _cmd["action"] = None

                if action == "stop":
                    _auto_stop("사용자 중단")
                    _plan_chain.clear()
                    if nav_runner is not None:
                        nav_runner.stop()
                        nav_runner = None
                    elif explore_runner is not None:
                        explore_runner.stop()
                        explore_runner = None
                    else:
                        print("[stop] 진행 중인 스킬 없음")

                elif action == "plan":
                    apply = bool(_cmd.pop("plan_apply", False))
                    _run_plan(gmap, cur_map, apply=apply, pos=(int(cur_x), int(cur_y)))

                elif action == "nav_start":
                    if explore_runner is not None:
                        explore_runner.stop("Navigation으로 전환")
                        explore_runner = None
                    if nav_runner is not None and not nav_runner.finished:
                        nav_runner.stop("새 목표")
                    target_id = _cmd.get("nav_target")
                    _cmd["nav_target"] = None
                    if target_id is None:
                        print("[nav] 목표 없음")
                        nav_runner = None
                    else:
                        nav_runner = NavRunner(gmap, pyboy, target_id)
                        if not nav_runner.start(cur_map, cur_x, cur_y):
                            # 이미 목적지 맵이면 즉시 success → NPC 단계로
                            if getattr(nav_runner, "success", False):
                                _continue_plan_chain_after_nav()
                            else:
                                _note_nav_fail(
                                    gmap.map_name(int(target_id)),
                                    gmap.map_name(int(cur_map)),
                                    "관측된 Portal 로 이어지는 길이 없습니다",
                                )
                                _plan_chain.clear()
                            nav_runner = None

                elif action == "nav_local_start":
                    if explore_runner is not None:
                        explore_runner.stop("Local Nav으로 전환")
                        explore_runner = None
                    if nav_runner is not None and not nav_runner.finished:
                        nav_runner.stop("새 Local 목표")
                    spec = _cmd.get("nav_local") or {}
                    _cmd["nav_local"] = None
                    mode = str(spec.get("mode") or "npc")
                    npc_name = spec.get("npc_name")
                    if mode == "npc" and _guard_no_talk(str(npc_name or "")):
                        nav_runner = None
                        _plan_chain.clear()
                    else:
                        if mode == "npc":
                            _talk_begin(cur_map, str(npc_name or ""))
                        nav_runner = LocalNavRunner(
                            gmap,
                            pyboy,
                            mode=mode,
                            npc_name=npc_name,
                        )
                        if not nav_runner.start(cur_map, cur_x, cur_y):
                            if getattr(nav_runner, "success", False):
                                _continue_plan_chain_after_nav()
                            else:
                                _plan_chain.clear()
                            nav_runner = None

                elif action == "explore_start":
                    _plan_chain.clear()
                    if nav_runner is not None:
                        nav_runner.stop("Exploration으로 전환")
                        nav_runner = None
                    if explore_runner is not None:
                        explore_runner.stop("재시작")
                    kind = _cmd.pop("explore_kind", None)
                    manual = bool(_cmd.pop("explore_manual", False))
                    if kind is None and not manual:
                        kind = _choose_explore_skill(gmap, pyboy, cur_map, cur_x, cur_y)
                    skill_obj, label = _explore_skill_for(kind, explore_skills, explore_skill)
                    _explore_state["kind"] = label
                    # 이미 최종 종료 조건을 넘긴 맵이면 격자 탐험은 부르지 않는다.
                    if kind is None and not manual:
                        print(
                            f'[explore] "{gmap.map_name(cur_map)}" 는 더 볼 게 '
                            "없습니다 (폐영역 · 안 가본 문 없음) — 탐험을 건너뜁니다"
                        )
                        _auto["explore_skipped"] = True
                    elif not getattr(skill_obj, "ignore_closure", False) and explore_gate.check(gmap, pyboy, cur_map, (cur_x, cur_y)):
                        print(
                            f'[explore] "{gmap.map_name(cur_map)}" 는 더 볼 게 '
                            "없습니다 (폐영역) — 탐험을 건너뜁니다"
                        )
                        # 건너뛴 것이지 "안 늘어난" 게 아니다. 재시도 대상 아님.
                        _auto["explore_skipped"] = True
                    else:
                        explore_runner = ExploreRunner(
                            gmap, pyboy, skill_obj, observed=explore_observed
                        )
                        print(f"[explore] 스킬 = {label}")
                        if label == "seekwarp":
                            started = explore_runner.start(cur_map)   # 상한은 스킬 설정(seekwarp_max_steps)
                        else:
                            started = explore_runner.start(cur_map, max_steps=_explore_step_limit(manual))
                        if not started:
                            explore_runner = None

                elif action == "auto_start":
                    if nav_runner is not None:
                        nav_runner.stop("auto 시작")
                        nav_runner = None
                    if explore_runner is not None:
                        explore_runner.stop("auto 시작")
                        explore_runner = None
                    _plan_chain.clear()
                    _auto_start(gmap, cur_map, cur_x, cur_y)

                elif action == "global":
                    print_current_node(gmap, cur_map, cur_x, cur_y)
                elif action == "photo":
                    try:
                        photo_path, counts = save_next_play_photo(gmap)
                        print(
                            f"[photo] 저장: {photo_path} "
                            f"(방문={counts['visited']}, 관측={counts['observed']}, "
                            f"벽={counts['wall']}, 미배치={counts['unplaced']})"
                        )
                    except Exception as exc:
                        print(f"[photo] 저장 실패: {exc}")
                else:
                    handle_local_action(action, gmap, cur_map, cur_x, cur_y)

            if nav_runner is not None:
                nav_stepped = nav_runner.tick(cur_map, cur_x, cur_y)
                if nav_stepped and _debug_flags["localmap"]:
                    print_local_map_block(gmap, cur_map, cur_x, cur_y)
                if nav_stepped and _auto_record_step():
                    if not nav_runner.finished:
                        nav_runner.stop("auto 최대 스텝 도달")
                    nav_runner = None
                    continue
                if nav_runner.finished:
                    ok = bool(getattr(nav_runner, "success", False))
                    was_npc = (
                        isinstance(nav_runner, LocalNavRunner)
                        and getattr(nav_runner, "mode", "") == "npc"
                    )
                    blocked = getattr(nav_runner, "blocked_by", None)
                    if blocked:
                        _note_obstacle(blocked)
                    elif not ok and not was_npc:
                        _note_nav_fail(
                            str(getattr(nav_runner, "target_name", "") or "?"),
                            gmap.map_name(int(cur_map)),
                            str(getattr(nav_runner, "reason", "") or "이동 실패"),
                        )
                    if was_npc:
                        # 대화는 항상 "성공"으로 끝난다.
                        talked = bool(
                            getattr(nav_runner, "talked_to_target", ok)
                        )
                        why = _talk_end(bool(ok) and talked)
                        if (not ok or not talked) and why:
                            _note_nav_fail(
                                str(getattr(nav_runner, "target_name", "") or "?"),
                                gmap.map_name(int(cur_map)),
                                why,
                            )
                    runlog.log(
                        "nav_end",
                        cycle=int(_auto["cycle"]) if _auto.get("on") else None,
                        스텝=int(_SESSION["steps"]),
                        종류="npc" if was_npc else "map",
                        맵=gmap.map_name(int(cur_map)),
                        목표=str(getattr(nav_runner, "target_name", "") or ""),
                        success=ok,
                        막은_NPC=blocked or None,
                        사유=str(getattr(nav_runner, "reason", "") or ""),
                    )
                    target_label = str(getattr(nav_runner, "target_name", "") or "")
                    if was_npc and why and (not ok or not talked):
                        pass
                    elif ok:
                        _cycle_note(
                            f"{'대화함' if was_npc else '이동 성공'}"
                            + (f" ({target_label})" if target_label else "")
                        )
                    else:
                        why = str(getattr(nav_runner, "reason", "") or "실패")
                        _cycle_note(f"{'대화' if was_npc else '이동'} 실패: {why}")
                    nav_runner = None
                    _auto["nav_ok"] = ok
                    if ok and was_npc:
                        _auto["npc_talked"] = True
                    if ok:
                        _continue_plan_chain_after_nav()
                    else:
                        _plan_chain.clear()

            if explore_runner is not None:
                explore_stepped = explore_runner.tick(cur_map, cur_x, cur_y)
                if explore_stepped and _debug_flags["localmap"]:
                    print_local_map_block(gmap, cur_map, cur_x, cur_y)
                if explore_stepped and _auto_record_step():
                    if not explore_runner.finished:
                        explore_runner.stop("auto 최대 스텝 도달")
                    explore_runner = None
                    continue
                if explore_runner.finished:
                    runlog.log(
                        "explore_end",
                        cycle=int(_auto["cycle"]) if _auto.get("on") else None,
                        스텝=int(_SESSION["steps"]),
                        스킬=_explore_state.get("kind"),
                        맵=gmap.map_name(cur_map),
                        success=bool(explore_runner.success),
                        achieved=bool(getattr(explore_runner, "achieved", False)),
                        steps=int(getattr(explore_runner.skill, "steps", 0) or 0),
                        사유=str(getattr(explore_runner.skill, "reason", "") or ""),
                    )
                    _cycle_note(
                        f"탐험 {int(getattr(explore_runner.skill, 'steps', 0) or 0)}걸음"
                        f" — {str(getattr(explore_runner.skill, 'reason', '') or '끝')}"
                    )
                    explore_runner = None
                    # 이번 에피소드로 맵이 닫혔는지 기록해 둔다.
                    if explore_gate.check(gmap, pyboy, cur_map, (cur_x, cur_y)):
                        print(
                            f'[explore] "{gmap.map_name(cur_map)}" 최종 종료 조건 충족 '
                            "(폐영역 · 안 지나가 본 출구 없음) "
                            "— 이 맵은 더 탐험하지 않습니다"
                        )

            # auto: 러너가 하나도 안 돌 때만 다음 단계로 넘어간다
            if _auto["on"] and nav_runner is None and explore_runner is None:
                in_battle = battle_phase(pyboy) != BATTLE_NONE
                if in_battle:
                    if not auto_battle_paused:
                        auto_battle_paused = True
                        print("[auto] 배틀 감지 — 일시정지, 직접 싸우세요")
                else:
                    if auto_battle_paused:
                        auto_battle_paused = False
                        print("[auto] 배틀 종료 — 재개")
                    if auto_dialogue.busy():
                        pass
                    elif not _plan_chain and _cmd.get("action") is None:
                        # **연출이 끝나고 가라앉을 때까지 기다린다.** 이벤트 직후의 페이드·스크립트
                        # 이동에는 대화창도 입력 잠금도 꺼진 프레임이 끼어 있어, 그 틈에 Planning 이
                        # 시작되면 도착 전 맵·아직 안 받은 아이템으로 계획을 세운다 (후지 구출 → 집).
                        if not settle_gate.ready():
                            settle_wait_count += 1
                            if settle_gate.waited == settle_gate.frames:
                                # 정착 창보다 오래 기다림 = 진짜 연출. 한 번만 알린다.
                                settle_announced = True
                                print(f"[auto] 연출 중 ({settle_gate.reason}) — 끝나면 다음 단계로 넘어갑니다")
                        else:
                            if settle_gate.forced:
                                print(
                                    f"[auto] {settle_gate.max_wait}프레임을 기다려도 연출이 안 끝났습니다 "
                                    f"({settle_gate.reason}) — 그냥 진행합니다"
                                )
                            if settle_announced or settle_gate.forced:
                                runlog.log(
                                    "settle_wait",
                                    프레임=settle_wait_count,
                                    이유=settle_gate.reason,
                                    강제=bool(settle_gate.forced),
                                    맵=gmap.map_name(cur_map),
                                )
                                if settle_announced:
                                    print(f"[auto] 연출 끝 — {settle_wait_count}프레임 기다렸습니다")
                            settle_announced = False
                            settle_wait_count = 0
                            auto_dialogue.take_finished()
                            _auto_step(gmap, cur_map, cur_x, cur_y)

            if view_gate.ready(pyboy, cur_map, cur_x, cur_y):
                render_views(gmap, pyboy, cur_map, cur_x, cur_y)

    except KeyboardInterrupt:
        pass
    finally:
        if battle_runner is not None and not battle_runner.finished:
            battle_runner.stop("프로그램 종료")
        if nav_runner is not None and not nav_runner.finished:
            nav_runner.stop("프로그램 종료")
        if explore_runner is not None and not explore_runner.finished:
            explore_runner.stop("프로그램 종료")
        if _REC.get("rec") is not None:
            _REC["rec"].stop("프로그램 종료")
        save_play_state(pyboy)
        pyboy.stop()
        print("종료.")


if __name__ == "__main__":
    main()
