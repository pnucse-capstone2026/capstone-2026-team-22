"""Portal 그래프 진단 출력 + 목표(맵) 목록/이름 해석."""

from __future__ import annotations

import networkx as nx

from core.portal import GlobalMap, PortalKey, ROLE_ENTRY, ROLE_EXIT
from Skills.Navigation import Navigation

def is_portal_key(node: object) -> bool:
    """현재 EXIT/ENTRY PortalKey 형식인지 확인한다."""
    return (
        isinstance(node, tuple)
        and len(node) == 4
        and str(node[1]).upper() in (ROLE_EXIT, ROLE_ENTRY)
    )


def portal_sort_key(node: PortalKey) -> tuple:
    """(map_id, role, direction, coords) 순서로 안정적으로 정렬한다."""
    return (
        int(node[0]),
        str(node[1]),
        str(node[2]),
        repr(node[3]),
    )


# ===========================================================================
# GlobalMap — 현재 노드만
# ===========================================================================
def print_current_node(
    gmap: GlobalMap,
    cur_map: int,
    cur_x: int | None = None,
    cur_y: int | None = None,
) -> None:
    """현재 위치한 Global 노드의 연결·grid·POI·Portal을 출력한다."""
    name = gmap.map_name(cur_map)
    pos = f"  pos=({cur_x},{cur_y})" if cur_x is not None and cur_y is not None else ""
    print(f'\n[global] 현재 노드: "{name}" (id={cur_map}){pos}')

    local = gmap.get_local_map(cur_map)
    if local is None:
        print("  (이 맵은 아직 GlobalMap에 없습니다)")
        print()
        return

    outs = [gmap.graph.nodes[n]["name"] for n in gmap.graph.successors(cur_map)]
    ins = [gmap.graph.nodes[n]["name"] for n in gmap.graph.predecessors(cur_map)]
    pois = sorted(local.name_index.keys())
    n_cells = len(local.grid)
    n_visited = sum(1 for cell in local.grid.values() if cell.visited)

    # grid에는 화면 관측 셀 + 밟아서/전환으로 확정된 W가 포함된다.
    print(f"  grid {n_cells}칸 / 방문 {n_visited}칸")
    print(f"  실제 출구→ [{', '.join(outs) if outs else '없음'}]")
    print(f"  실제 입구← [{', '.join(ins) if ins else '없음'}]")
    print(f"  POI   [{', '.join(pois) if pois else '없음'}]")
    if pois:
        for pname in pois:
            for x, y in local.name_index[pname]:
                print(f'    · "{pname}" → ({x},{y})')

    portals = list(local.iter_portals())
    exit_count = sum(1 for portal in portals if portal.is_exit)
    entry_count = sum(1 for portal in portals if portal.is_entry)
    print(f"  Portal {len(portals)}개 (EXIT={exit_count}, ENTRY={entry_count})")
    for portal in sorted(portals, key=lambda p: portal_sort_key(p.key)):
        print(
            f"    · role={portal.role}, direction={portal.direction}, "
            f"coords={list(portal.warp_coords)}, "
            f"arrival={list(portal.arrival_coords)}, "
            f"access={list(portal.access_coords)}, "
            f"label={portal.label!r}, kind={portal.kind}"
        )
    print()


# ===========================================================================
# Portal 그래프 조회/미리보기
# ===========================================================================
def portal_text(gmap: GlobalMap, node: PortalKey) -> str:
    """role-aware PortalKey를 사람이 읽기 쉬운 문자열로 바꾼다."""
    map_id = int(node[0])
    map_name = gmap.map_name(map_id)
    portal = gmap.get_portal(node)

    if portal is None:
        role = str(node[1]) if is_portal_key(node) else "UNKNOWN"
        direction = str(node[2]) if is_portal_key(node) else "UNKNOWN"
        coords = list(node[3]) if is_portal_key(node) else []
        return (
            f"{map_name}(id={map_id}) role={role}, "
            f"direction={direction}, coords={coords}"
        )

    return (
        f"{map_name}(id={map_id}) "
        f"role={portal.role}, "
        f"direction={portal.direction}, "
        f"coords={list(portal.warp_coords)}, "
        f"arrival={list(portal.arrival_coords)}, "
        f"access={list(portal.access_coords)}, "
        f"label={portal.label!r}"
    )


def debug_current_portals(
    gmap: GlobalMap,
    cur_map: int,
    cur_x: int,
    cur_y: int,
) -> None:
    """현재 위치의 START→EXIT와 같은 맵 ENTRY→EXIT 연결을 진단한다."""
    cur_map = int(cur_map)
    start = (int(cur_x), int(cur_y))
    local = gmap.get_local_map(cur_map)

    print("\n" + "#" * 110)
    print(
        f"[debug current] map={gmap.map_name(cur_map)}"
        f"(id={cur_map}), player={start}"
    )
    if local is None:
        print("[debug current] 현재 LocalMap이 없습니다.")
        print("#" * 110 + "\n")
        return

    exit_nodes = list(gmap.exit_portal_nodes(cur_map))
    entry_nodes = list(gmap.entry_portal_nodes(cur_map))
    if not exit_nodes and not entry_nodes:
        print("[debug current] 현재 맵에 EXIT/ENTRY Portal이 없습니다.")
        print("#" * 110 + "\n")
        return

    nav = Navigation(gmap)
    nav.set_internal_edge_debug(True)
    nav.update_internal_edges(cur_map, is_surfing=False, can_cut=False)

    print("\n[1] 현재 위치 → EXIT")
    if not exit_nodes:
        print("  EXIT 없음")
    for node in sorted(exit_nodes, key=portal_sort_key):
        portal = gmap.get_portal(node)
        raw_coords = list(gmap.portal_coords(node))
        activation = nav._portal_activation_coords(node, local)
        final_action = nav._portal_final_exit_action(node, local)
        try:
            result = nav._a_star(
                local,
                start,
                activation,
                is_surfing=False,
                can_cut=False,
                debug=False,
                force_goal_passable=True,
            )
        except Exception as exc:
            result = None
            result_text = f"예외 {type(exc).__name__}: {exc}"
        else:
            result_text = (
                "도달 불가"
                if result is None
                else f"도달 가능, cost={result.cost}, path_len={len(result.path)}"
            )
        print(f"\n  - {portal_text(gmap, node)}")
        print(f"      raw EXIT       = {raw_coords}")
        print(f"      activation     = {activation}")
        print(f"      final action   = {final_action}")
        print(
            "      stored access  = "
            f"{list(portal.access_coords) if portal is not None else []}"
        )
        print(f"      START -> EXIT  = {result_text}")

    print("\n[2] ENTRY 시작 좌표")
    if not entry_nodes:
        print("  ENTRY 없음")
    for node in sorted(entry_nodes, key=portal_sort_key):
        portal = gmap.get_portal(node)
        selected = nav._portal_internal_start_coords(
            node,
            local,
            is_surfing=False,
            can_cut=False,
        )
        print(f"\n  - {portal_text(gmap, node)}")
        print(
            "      arrival stored = "
            f"{list(portal.arrival_coords) if portal is not None else []}"
        )
        print(
            "      access stored  = "
            f"{list(portal.access_coords) if portal is not None else []}"
        )
        print(f"      internal start = {selected}")

    print("\n[3] 같은 맵 ENTRY → EXIT internal edge")
    graph = gmap.gate_graph
    if not entry_nodes or not exit_nodes:
        print("  ENTRY 또는 EXIT가 없어 검사할 쌍이 없습니다.")
    for source in sorted(entry_nodes, key=portal_sort_key):
        for destination in sorted(exit_nodes, key=portal_sort_key):
            data = graph.get_edge_data(source, destination)
            if data is not None and data.get("edge_type") == "internal":
                print(
                    f"  [OK] {portal_text(gmap, source)}"
                    f"\n       -> {portal_text(gmap, destination)}"
                    f"\n       weight={data.get('weight')}, "
                    f"path_length={data.get('path_length')}, "
                    f"exit_action={data.get('destination_exit_action')}"
                )
            else:
                print(
                    f"  [--] {portal_text(gmap, source)}"
                    f"\n       -> {portal_text(gmap, destination)}"
                    "\n       internal edge 없음"
                )

    print("\n[판정]")
    print("  - START는 현재 맵의 EXIT에만 연결되어야 합니다.")
    print("  - internal edge는 ENTRY → EXIT 방향만 생성되어야 합니다.")
    print("  - 분리된 구역은 ENTRY → EXIT가 [--]로 남는 것이 정상입니다.")
    print("  - raw EXIT가 맵 밖이면 activation은 경계 안쪽, final action은 방향키여야 합니다.")
    print("#" * 110 + "\n")


def portal_route_preview(
    gmap: GlobalMap,
    current_map_id: int,
    target_map_id: int,
) -> list[int] | None:
    """현재 맵 EXIT에서 목표 맵 ENTRY까지의 관측 Portal 경로 미리보기."""
    current_map_id = int(current_map_id)
    target_map_id = int(target_map_id)
    if current_map_id == target_map_id:
        return [current_map_id]

    starts = list(gmap.exit_portal_nodes(current_map_id))
    targets = set(gmap.entry_portal_nodes(target_map_id))
    graph = gmap.gate_graph.copy()
    if not starts or not targets:
        return None

    best_path = None
    best_cost = None
    for source in starts:
        try:
            lengths, paths = nx.single_source_dijkstra(
                graph, source, weight="weight"
            )
        except nx.NodeNotFound:
            continue
        for destination in targets:
            if destination not in lengths:
                continue
            cost = lengths[destination]
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_path = paths[destination]

    if not best_path:
        return None

    map_route: list[int] = []
    for node in best_path:
        if not is_portal_key(node):
            continue
        map_id = int(node[0])
        if not map_route or map_route[-1] != map_id:
            map_route.append(map_id)
    return map_route or None


def available_targets(gmap: GlobalMap) -> list:
    """목표 후보 = GlobalMap 노드 → [(map_id, name), ...]."""
    try:
        node_ids = list(gmap.graph.nodes)
    except RuntimeError:
        return []
    return sorted(((int(m), gmap.map_name(m)) for m in node_ids), key=lambda t: t[1])


def print_target_list(gmap: GlobalMap, cur_map=None) -> list:
    """'가능한 목표 지점들: [...]' 한 줄만 찍는다."""
    targets = available_targets(gmap)
    if not targets:
        print("\n[target] 가능한 목표 지점들: []  (아직 관측된 맵이 없습니다)")
        return targets
    print(f"\n[target] 가능한 목표 지점들: {[name for _, name in targets]}")
    return targets


def resolve_target(gmap: GlobalMap, text: str):
    """입력한 맵 이름 또는 map_id → map_id."""
    text = text.strip()
    if not text:
        return None
    targets = available_targets(gmap)

    if text.isdigit():
        mid = int(text)
        if any(t_mid == mid for t_mid, _ in targets):
            return mid
        print(f"[nav] id={mid} 는 아직 관측되지 않은 맵입니다.")
        return None

    low = text.lower()
    for mid, name in targets:
        if name.lower() == low:
            return mid
    partial = [(mid, name) for mid, name in targets if low in name.lower()]
    if len(partial) == 1:
        print(f'[nav] "{text}" → "{partial[0][1]}" 로 해석')
        return partial[0][0]
    if len(partial) > 1:
        print(f"[nav] '{text}' 후보가 여러 개입니다: {[n for _, n in partial]}")
        return None
    print(f"[nav] '{text}' 는 가능한 목표 지점에 없습니다. (target 으로 목록 확인)")
    return None
