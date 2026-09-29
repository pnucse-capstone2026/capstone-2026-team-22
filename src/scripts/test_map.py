"""test_map.py — Map.py 기본 지도 + portal.py 확장 + Navigation Skill 실기 테스트"""

from __future__ import annotations

HELP_TEXT = """
================================================================
  test_map.py 터미널 명령
================================================================

  help              이 도움말

  global            현재 위치한 노드의 연결·관측칸·POI 정보

  summary           전체 GlobalMap 요약 출력 (관측된 맵만)

  portal / warp     Portal 그래프를 터미널 텍스트로 출력
  portal detail     모든 Portal의 warp/arrival/access와 방향 간선 출력
  portal detail current
                    현재 맵의 Portal 상세 정보만 출력

  debug on          다음 nav부터 internal edge 상세 로그 활성화
  debug off         internal edge 상세 로그 비활성화
  debug status      현재 디버그 상태 출력
  debug current     현재 위치·현재 맵만 즉시 진단
                    (START→EXIT 및 ENTRY→EXIT internal 검사)

  arrival status    맵 전환 도착 좌표 안정화 상태 출력
                    (pending/candidate/stable/최근 transition 확인)
  scan status       LocalMap 마지막 스캔 좌표와 현재 정렬/걷기 상태 출력

  local             현재 맵 LocalMap 전체 2D 문자 행렬 (즉시 출력)
  local split       위와 동일, y 10줄씩 분할 (터미널 잘림 완화)
  local file        map_dump/map_<id>.txt 파일로 저장
  local coords      LocalMap.grid 에 있는 (x,y) 좌표 목록 (y별 정렬)

  target            가능한 목표 지점들(Global 노드) 목록 출력
  nav               목표 목록 + "목표를 입력해주세요:" 입력 → 자동 이동
                    (한 걸음마다 전체 맵을 local 형식으로 출력)
  nav stop          자동 이동 중단

  view              scene/block/name/tiles ON·OFF 상태 보기
  view scene on/off [M] 10×9 장면 문자 매트릭스 (. # W P …)
  view block on/off [M] to_matrix('block') 숫자 (화면 영역)
  view name on/off  [N] cell_name POI 매트릭스
  view tiles on/off [T] VRAM 타일 ID 20×18 (hex)
  view all on/off   위 네 가지 한꺼번에 ON/OFF

  view 항목은 한 칸 이동(step)마다 ON인 것만 출력합니다.
  view scene 은 map_id·이름·player(x,y)를 함께 출력합니다.

  noclip on|off     풀베기 나무는 통과, 괴력 바위는 밀기 가능
  noclip            현재 on/off 상태 출력
                    ※ 일반 벽/NPC는 그대로. Map.py와 무관(WRAM 조작)

  Ctrl+C            종료
================================================================
""".strip()


def print_help() -> None:
    print(HELP_TEXT)

import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from core.gb_memory import (
    ADDR_JOY_IGNORE,
    ADDR_SCX,
    ADDR_SCY,
    ADDR_WALK_COUNTER,
)
from core.portal import ARRIVAL_STABLE_FRAMES, GlobalMap


# engine/ 로 옮긴 실행 계층 (play.py 와 공유).
from engine.cheats import (  # noqa: E402
    apply_noclip_cheat as _apply_noclip_cheat,
    apply_skip_mt_moon_fossils as _apply_skip_mt_moon_fossils,
    disable_wild_encounters as _disable_wild_encounters,
    handle_noclip_command as _handle_noclip_command,
)
from engine.debug_view import handle_view_command as _handle_view_command  # noqa: E402
from engine.session import (  # noqa: E402
    ViewGate,
    handle_local_action,
    open_emulator,
    print_local_map_block,
    read_player,
    render_views,
)
from engine.portal_debug import (  # noqa: E402
    debug_current_portals,
    is_portal_key as _is_portal_key,
    portal_sort_key as _portal_sort_key,
    portal_text as _portal_text,
    print_current_node,
    print_target_list,
    resolve_target as _resolve_target,
)
from engine.runners import NavRunner, debug_flags as _debug_flags  # noqa: E402
from engine.state import game_state as _game_state, gmap_ref as _gmap_ref  # noqa: E402

_shared_cmd: dict = {"action": None, "nav_target": None}


def print_portal_graph(gmap: GlobalMap) -> None:
    """gate_graph의 EXIT/ENTRY 노드와 transition/internal 간선을 출력한다."""
    graph = getattr(gmap, "gate_graph", None)
    if graph is None or graph.number_of_nodes() == 0:
        print("\n[portal] 아직 관측된 Portal이 없습니다.\n")
        return

    portal_nodes = [node for node in list(graph.nodes) if _is_portal_key(node)]
    if not portal_nodes:
        print("\n[portal] 표시할 EXIT/ENTRY Portal 노드가 없습니다.\n")
        return

    print("\n" + "=" * 88)
    print("[portal] EXIT/ENTRY Portal 그래프")

    portal_node_set = set(portal_nodes)
    for node in sorted(portal_nodes, key=_portal_sort_key):
        print(f"\n- {_portal_text(gmap, node)}")
        outgoing = [
            (destination, data)
            for _, destination, data in graph.out_edges(node, data=True)
            if destination in portal_node_set
        ]
        if not outgoing:
            print("    → 연결 없음")
            continue
        for destination, data in outgoing:
            print(
                f"    → {_portal_text(gmap, destination)} "
                f"[{data.get('edge_type', 'unknown')}, "
                f"weight={data.get('weight', 1)}]"
            )

    print("=" * 88 + "\n")


def print_portal_detail(
    gmap: GlobalMap,
    only_map_id: int | None = None,
) -> None:
    """EXIT/ENTRY Portal 메타데이터와 방향 간선을 상세 출력한다."""
    graph = getattr(gmap, "gate_graph", None)
    if graph is None or graph.number_of_nodes() == 0:
        print("\n[portal detail] 아직 관측된 Portal이 없습니다.\n")
        return

    nodes = [
        node
        for node in list(graph.nodes)
        if _is_portal_key(node)
        and (only_map_id is None or int(node[0]) == int(only_map_id))
    ]
    if not nodes:
        scope = "전체" if only_map_id is None else f"map_id={only_map_id}"
        print(f"\n[portal detail] {scope}에 표시할 Portal이 없습니다.\n")
        return

    title = (
        "전체 Portal"
        if only_map_id is None
        else f"{gmap.map_name(int(only_map_id))}(id={int(only_map_id)})"
    )
    print("\n" + "=" * 110)
    print(f"[portal detail] {title}")

    for node in sorted(nodes, key=_portal_sort_key):
        portal = gmap.get_portal(node)
        print(f"\n- {_portal_text(gmap, node)}")
        if portal is not None:
            print(
                "    metadata:"
                f" role={portal.role!r},"
                f" kind={portal.kind!r},"
                f" destination_map_id={portal.destination_map_id!r},"
                f" destination_portal={portal.destination_portal!r}"
            )

        incoming = [
            (source, data)
            for source, _, data in graph.in_edges(node, data=True)
            if _is_portal_key(source)
        ]
        outgoing = [
            (destination, data)
            for _, destination, data in graph.out_edges(node, data=True)
            if _is_portal_key(destination)
        ]

        print("    incoming:")
        if not incoming:
            print("      ← 없음")
        for source, data in incoming:
            print(
                f"      ← {_portal_text(gmap, source)} "
                f"[{data.get('edge_type', 'unknown')}, "
                f"weight={data.get('weight', 1)}]"
            )

        print("    outgoing:")
        if not outgoing:
            print("      → 없음")
        for destination, data in outgoing:
            path = data.get("path")
            path_note = f", path_len={len(path)}" if path is not None else ""
            print(
                f"      → {_portal_text(gmap, destination)} "
                f"[{data.get('edge_type', 'unknown')}, "
                f"weight={data.get('weight', 1)}{path_note}]"
            )

    print("=" * 110 + "\n")


def _last_observed_transition(gmap: GlobalMap):
    transitions = list(getattr(gmap, "observed_transitions", ()) or ())
    return transitions[-1] if transitions else None


def _transition_text(gmap: GlobalMap, transition) -> str:
    if transition is None:
        return "없음"
    return (
        f"{gmap.map_name(transition.source_map_id)}"
        f"(id={transition.source_map_id}) "
        f"EXIT{transition.source_exit_coord} -> "
        f"{gmap.map_name(transition.destination_map_id)}"
        f"(id={transition.destination_map_id}) "
        f"ENTRY{transition.destination_arrival_coord}"
    )


def print_arrival_status(gmap: GlobalMap) -> None:
    """portal.GlobalMap의 arrival 후보와 최근 EXIT→ENTRY 전환을 출력한다."""
    pending = getattr(gmap, "_pending_map_change", None)
    candidate = getattr(gmap, "_arrival_candidate", None)
    candidate_frames = int(getattr(gmap, "_arrival_candidate_frames", 0))
    wait_frames = int(getattr(gmap, "_arrival_wait_frames", 0))
    settle = int(getattr(gmap, "_map_settle", 0))
    last_transition = _last_observed_transition(gmap)

    print()
    print("=" * 88)
    print("[arrival status]")
    print(f"  pending          = {pending!r}")
    print(f"  settle remaining = {settle}")
    print(f"  candidate        = {candidate!r}")
    print(
        f"  stable frames    = {candidate_frames}"
        f" / {ARRIVAL_STABLE_FRAMES}"
    )
    print(f"  wait frames      = {wait_frames}")
    print(f"  last transition  = {_transition_text(gmap, last_transition)}")

    if pending is None:
        print("  상태: 대기 중인 맵 전환 없음")
    elif settle > 0:
        print("  상태: 맵 전환 후 실제 프레임 settle 중")
    elif candidate is None:
        print("  상태: 새 맵의 타당한 첫 arrival 후보를 기다리는 중")
    else:
        print(
            "  상태: arrival 후보 확인 중 — 정지하면 stable frames로 확정하고, "
            "계속 걸으면 다음 인접 셀에서 최초 후보를 ENTRY로 확정"
        )
    print("=" * 88)
    print()


def print_scan_status(gmap: GlobalMap, pyboy) -> None:
    """연속 이동/arrival OBS에서 현재 화면을 저장할 수 있는지 출력한다."""
    current_map, px, py = read_player(pyboy)
    current_pos = (px, py)
    walk_counter = int(pyboy.memory[ADDR_WALK_COUNTER])
    joy_locked = int(pyboy.memory[ADDR_JOY_IGNORE]) != 0
    scx = int(pyboy.memory[ADDR_SCX])
    scy = int(pyboy.memory[ADDR_SCY])
    aligned = scx % 8 == 0 and scy % 8 == 0

    pending = getattr(gmap, "_pending_map_change", None)
    settle = int(getattr(gmap, "_map_settle", 0))
    candidate = getattr(gmap, "_arrival_candidate", None)
    candidate_frames = int(getattr(gmap, "_arrival_candidate_frames", 0))
    last_transition = _last_observed_transition(gmap)

    print()
    print("=" * 88)
    print("[scan status]")
    print(f"  current map/pos   = {current_map}, {current_pos}")
    print(f"  wWalkCounter      = {walk_counter}")
    print(f"  wJoyIgnore        = {int(joy_locked)}")
    print(f"  scroll            = ({scx}, {scy})")
    print(f"  scroll aligned    = {aligned}")
    print(
        "  last scanned      = "
        f"{getattr(gmap, '_last_scanned_map_id', None)}, "
        f"{getattr(gmap, '_last_scanned_pos', None)}"
    )
    print(
        "  scanned scroll    = "
        f"{getattr(gmap, '_last_scanned_scroll', None)}"
    )
    print(
        "  last visited      = "
        f"{getattr(gmap, '_last_visited_map_id', None)}, "
        f"{getattr(gmap, '_last_visited_pos', None)}"
    )
    print(f"  pending transition= {pending!r}")
    print(f"  settle remaining  = {settle}")
    print(f"  arrival candidate = {candidate!r}")
    print(f"  candidate frames  = {candidate_frames}")
    print(f"  last transition   = {_transition_text(gmap, last_transition)}")

    if joy_locked:
        print("  판정: EnterMap/대화/스크립트 중 — MapID 변화만 보존하고 VRAM은 스캔하지 않음")
    elif pending is not None and settle > 0:
        print("  판정: 맵 전환 후 실제 프레임 settle 중 — 아직 스캔하지 않음")
    elif not aligned:
        print("  판정: 픽셀 스크롤 중간이라 스캔하지 않음")
    elif pending is not None and candidate is None:
        print("  판정: 새 맵의 타당한 첫 arrival 후보를 기다리는 중")
    elif pending is not None and candidate is not None:
        if current_pos == candidate:
            print(
                "  판정: arrival 후보 화면을 스캔함 — 정지 안정화 또는 "
                "다음 인접 셀을 기다리는 중"
            )
        else:
            distance = (
                abs(current_pos[0] - candidate[0])
                + abs(current_pos[1] - candidate[1])
            )
            print(f"  판정: candidate와 현재 좌표 거리={distance}")
    elif (
        getattr(gmap, "_last_scanned_map_id", None) != current_map
        or getattr(gmap, "_last_scanned_pos", None) != current_pos
    ):
        print("  판정: 새 맵/새 셀 경계 — walking 중이어도 스캔 가능")
    elif walk_counter != 0:
        print("  판정: 같은 좌표에서 walking 중 — 화면 중간 프레임이라 스캔하지 않음")
    else:
        print("  판정: 정지 상태 — NPC/동적 오브젝트 갱신을 위해 반복 스캔 가능")
    print("=" * 88)
    print()


def _handle_debug_command(line: str) -> None:
    """debug on/off/status/current 명령을 처리한다."""
    parts = line.strip().lower().split()

    if len(parts) == 1 or parts[1] == "status":
        state = "ON" if _debug_flags["nav_internal"] else "OFF"
        print(f"[debug] nav internal edge 로그: {state}")
        print(
            "[debug] 사용: debug on | debug off | "
            "debug status | debug current"
        )
        return

    if parts[1] == "on":
        _debug_flags["nav_internal"] = True
        print(
            "[debug] ON — 다음 nav부터 모든 맵의 "
            "[internal-debug] 로그를 출력합니다."
        )
        return

    if parts[1] == "off":
        _debug_flags["nav_internal"] = False
        print("[debug] OFF")
        return

    if parts[1] == "current":
        _shared_cmd["action"] = "debug_current"
        return

    print(
        "[debug] 사용: debug on | debug off | "
        "debug status | debug current"
    )

def _handle_nav_command(line: str) -> None:
    parts = line.strip().lower().split()
    if len(parts) >= 2 and parts[1] in ("stop", "off", "cancel"):
        _shared_cmd["action"] = "nav_stop"
        return

    gmap = _gmap_ref.get("gmap")
    if gmap is None:
        print("[nav] 아직 게임이 준비되지 않았습니다.")
        return

    cur_map = _game_state.get("map_id")
    targets = print_target_list(gmap, cur_map)
    if not targets:
        return
    try:
        answer = input("목표를 입력해주세요: ").strip()
    except EOFError:
        return
    if not answer:
        print("[nav] 취소")
        return

    map_id = _resolve_target(gmap, answer)
    if map_id is None:
        return
    _shared_cmd["nav_target"] = map_id
    _shared_cmd["action"] = "nav_start"


def _input_thread() -> None:
    print(
        "\n[명령] help | global | portal | portal detail | "
        "debug current | debug on/off | local | target | nav | "
        "view scene|block|name|tiles on|off | noclip on|off"
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
            print_help()
        elif key.startswith("noclip"):
            _handle_noclip_command(line)
        elif key.startswith("view"):
            _handle_view_command(line)
        elif key.startswith("debug"):
            _handle_debug_command(line)
        elif key.startswith("nav"):
            _handle_nav_command(line)
        elif key in ("arrival status", "arrival", "arrival debug"):
            _shared_cmd["action"] = "arrival_status"
        elif key in ("scan status", "scan", "scan debug"):
            _shared_cmd["action"] = "scan_status"
        elif key == "global":
            _shared_cmd["action"] = "global"
        elif key == "summary":
            _shared_cmd["action"] = "summary"
        elif key in ("portal detail current", "portal current detail"):
            _shared_cmd["action"] = "portal_detail_current"
        elif key in ("portal detail", "warp detail"):
            _shared_cmd["action"] = "portal_detail_all"
        elif key in ("portal", "warp"):
            _shared_cmd["action"] = "portal"
        elif key == "target":
            _shared_cmd["action"] = "targets"
        elif key == "local":
            _shared_cmd["action"] = "local"
        elif key == "local split":
            _shared_cmd["action"] = "local_split"
        elif key == "local file":
            _shared_cmd["action"] = "local_file"
        elif key == "local coords":
            _shared_cmd["action"] = "local_coords"


# ===========================================================================
# 메인 루프
# ===========================================================================
def main() -> None:
    pyboy = open_emulator()

    gmap = GlobalMap()
    _gmap_ref["gmap"] = gmap
    nav_runner = None

    threading.Thread(target=_input_thread, daemon=True).start()

    view_gate = ViewGate(pyboy)

    print("-" * 64)
    print("게임 창에서 이동하세요. 지도는 조용히 쌓입니다.")
    print(
        "야생 전투 비활성. 터미널: "
        "help | global | portal | portal detail | "
        "debug current | debug on/off | local | target | nav"
    )
    print("종료: Ctrl+C")
    print("-" * 64)

    try:
        while pyboy.tick():
            _disable_wild_encounters(pyboy)
            _apply_noclip_cheat(pyboy)
            _apply_skip_mt_moon_fossils(pyboy)
            gmap.update_from_pyboy(pyboy)

            cur_map, cur_x, cur_y = read_player(pyboy)
            _game_state["map_id"] = cur_map
            _game_state["x"] = cur_x
            _game_state["y"] = cur_y

            action = _shared_cmd.get("action")
            if action:
                _shared_cmd["action"] = None
                if action == "global":
                    print_current_node(gmap, cur_map, cur_x, cur_y)
                elif action == "arrival_status":
                    print_arrival_status(gmap)
                elif action == "scan_status":
                    print_scan_status(gmap, pyboy)
                elif action == "portal":
                    print_portal_graph(gmap)
                elif action == "portal_detail_all":
                    print_portal_detail(gmap)
                elif action == "portal_detail_current":
                    print_portal_detail(gmap, only_map_id=cur_map)
                elif action == "debug_current":
                    debug_current_portals(
                        gmap,
                        cur_map,
                        cur_x,
                        cur_y,
                    )
                elif action == "summary":
                    print("[summary] GlobalMap 요약")
                    print(gmap.get_summary_text())
                elif action == "targets":
                    print_target_list(gmap, cur_map)
                elif action == "nav_start":
                    if nav_runner is not None and not nav_runner.finished:
                        nav_runner.stop("새 목표 입력")
                    target_id = _shared_cmd.get("nav_target")
                    _shared_cmd["nav_target"] = None
                    if target_id is None:
                        print("[nav] 목표가 지정되지 않았습니다.")
                        nav_runner = None
                    else:
                        nav_runner = NavRunner(gmap, pyboy, target_id)
                        if not nav_runner.start(cur_map, cur_x, cur_y):
                            nav_runner = None
                elif action == "nav_stop":
                    if nav_runner is not None:
                        nav_runner.stop()
                        nav_runner = None
                    else:
                        print("[nav] 진행 중인 자동 이동이 없습니다.")
                else:
                    handle_local_action(action, gmap, cur_map, cur_x, cur_y)

            # nav 자동 이동: 한 걸음 끝날 때마다 전체 맵(local) 출력
            if nav_runner is not None:
                if nav_runner.tick(cur_map, cur_x, cur_y):
                    print_local_map_block(gmap, cur_map, cur_x, cur_y)
                if nav_runner.finished:
                    nav_runner = None

            if view_gate.ready(pyboy, cur_map, cur_x, cur_y):
                render_views(gmap, pyboy, cur_map, cur_x, cur_y)

    except KeyboardInterrupt:
        pass
    finally:
        if nav_runner is not None and not nav_runner.finished:
            nav_runner.stop("프로그램 종료")
        pyboy.stop()


if __name__ == "__main__":
    main()
