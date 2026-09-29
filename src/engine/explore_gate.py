"""explore_gate.py — "이 맵은 더 볼 게 없다" 판정과 설정 (탐험 최종 종료 조건)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from core import runlog
from core.closure import (
    GATE_CUT_TREE,
    GATE_HM_NAMES,
    GATE_SILPH_SCOPE,
    GATE_SNORLAX,
    GATE_WATER,
    ClosureResult,
    closure_for,
    experienced_warp_arrival,
    is_exit_free_map,
    observed_in_map_warps,
    pick_seed,
    region_key,
    seed_inside_map,
    STORY_GATES,
    pokecenter_done,
    snorlax_blocking,
    static_warp_arrival,
    story_gate_closed,
    unpassed_static_exits,
)
from core.config import exploration_opt
from core.gb_memory import ADDR_TILESET
from engine.hm import can_cut as _hm_can_cut
from engine.hm import can_surf as _hm_can_surf

Coord = Tuple[int, int]
PortalStatus = Tuple[bool, str]

# 판정에서 제외할 타일셋의 폴백.
_DEFAULT_SKIP_TILESETS: Tuple[int, ...] = ()


def _opt(name: str, default: Any) -> Any:
    return exploration_opt(name, default)


def stop_on_closed() -> bool:
    """탐험 도중 최종 종료 조건에 닿으면 에피소드를 끊을지."""
    return bool(_opt("close_stop_explore", True))


def check_every() -> int:
    """몇 액션마다 닫힘을 볼지."""
    try:
        n = int(_opt("close_check_every", 25))
    except (TypeError, ValueError):
        n = 25
    return max(1, n)


def close_portals() -> bool:
    """포탈(관측한 워프) 단위로 너머 구역을 볼지 (``exploration.close_portals``)."""
    return bool(_opt("close_portals", True))


def capabilities(pyboy: Any) -> Tuple[bool, bool]:
    """현재 비전머신 능력 (can_cut, can_surf)."""
    return (bool(_hm_can_cut(pyboy)), bool(_hm_can_surf(pyboy)))


def current_tileset(pyboy: Any) -> Optional[int]:
    try:
        return int(pyboy.memory[ADDR_TILESET])
    except Exception:
        return None


def skip_tilesets() -> frozenset:
    raw = _opt("close_skip_tilesets", list(_DEFAULT_SKIP_TILESETS))
    if isinstance(raw, (int, float)):
        raw = [raw]
    try:
        return frozenset(int(t) for t in (raw or ()))
    except (TypeError, ValueError):
        return frozenset(_DEFAULT_SKIP_TILESETS)


def _region(
    gmap: Any,
    pyboy: Any,
    map_id: int,
    pos: Coord,
    *,
    can_cut: Optional[bool] = None,
    mark: bool = True,
) -> ClosureResult:
    """``pos`` 에서 걸어갈 수 있는 **구역 하나**의 폐영역 판정 (능력·설정 반영)."""
    return closure_for(
        gmap,
        int(map_id),
        (int(pos[0]), int(pos[1])),
        is_surfing=_hm_can_surf(pyboy),
        can_cut=_hm_can_cut(pyboy) if can_cut is None else bool(can_cut),
        npc_blocks=bool(_opt("close_npc_blocks", False)),
        mark=mark,
    )


def evaluate(
    gmap: Any,
    pyboy: Any,
    map_id: int,
    pos: Coord,
    *,
    can_cut: Optional[bool] = None,
) -> ClosureResult:
    """현재 상태로 폐영역을 판정하고 Cell.frontier 를 갱신한다."""
    # 제외 타일셋은 **판정 대상 맵**의 타일셋으로 본다 (스캔 때 LocalMap 에 남긴 값).
    local = gmap.get_local_map(int(map_id))
    tileset = getattr(local, "tileset", None) if local is not None else None
    if tileset is None:
        tileset = current_tileset(pyboy)
    if tileset is not None and int(tileset) in skip_tilesets():
        return ClosureResult(False, "tileset_skipped")
    return _region(gmap, pyboy, map_id, pos, can_cut=can_cut, mark=True)


def is_done(result: ClosureResult) -> bool:
    """폐영역 + 부조건 둘."""
    if not result.closed:
        return False
    # 다 봤는데 나갈 문을 하나도 못 찾았으면 플래너에게 줄 게 없다.
    if result.exit_count < int(_opt("close_min_warps", 1)):
        return False
    if bool(_opt("close_require_all_warps", True)) and result.unresolved:
        return False
    return True


def warp_destination(gmap: Any, map_id: int, coord: Coord) -> Optional[int]:
    """그 워프 칸이 어느 맵으로 이어지는가 (Portal 관측 기준)."""
    local = gmap.get_local_map(int(map_id))
    if local is None or not hasattr(local, "iter_portals"):
        return None
    target = (int(coord[0]), int(coord[1]))
    for portal in local.iter_portals():
        if target in tuple(portal.warp_coords or ()):
            dest = getattr(portal, "destination_map_id", None)
            if dest is not None:
                return int(dest)
    return None


def resolve_warp(gmap: Any, map_id: int, coord: Coord) -> Optional[Tuple[int, Optional[Coord]]]:
    """워프 칸 → (목적지 맵, 도착 칸)."""
    got = experienced_warp_arrival(gmap, map_id, coord)
    if got is not None:
        return got
    return static_warp_arrival(gmap, map_id, coord)


def _seed_pos(gmap: Any, pyboy: Any, map_id: int) -> Optional[Coord]:
    """그 맵에서 폐영역 BFS 를 시작할 칸 (지금 서 있지 않은 맵용)."""
    local = gmap.get_local_map(int(map_id))
    size = (getattr(gmap, "map_size_cells", {}) or {}).get(int(map_id))
    if local is None or not size:
        return None
    can_cut, can_surf = capabilities(pyboy)
    return pick_seed(local, size, is_surfing=can_surf, can_cut=can_cut)


def is_passage_map(name: Any, const: Any = None) -> bool:
    """안쪽 포탈을 따라 내려갈 필요가 없는 맵인가 (도로·게이트)."""
    marks = _opt("close_no_recurse_marks", ["route"]) or []
    source = const if const else name
    low = str(source or "").lower().replace("_", " ")
    return any(str(m).lower() in low for m in marks)


def _passage_map(gmap: Any, map_id: int) -> bool:
    consts = getattr(gmap, "map_consts", None) or {}
    name = gmap.map_name(int(map_id)) if hasattr(gmap, "map_name") else ""
    return is_passage_map(name, consts.get(int(map_id)))


def _map_label(gmap: Any, map_id: int) -> str:
    return gmap.map_name(int(map_id)) if hasattr(gmap, "map_name") else str(map_id)


class _PortalCtx:
    """포탈 재귀의 작업 상태."""

    __slots__ = ("stack", "memo", "cells")

    def __init__(self) -> None:
        self.stack: Dict[int, set] = {}
        self.memo: Dict[Any, PortalStatus] = {}
        self.cells: Dict[int, Dict[Coord, Any]] = {}


def portal_done(
    gmap: Any,
    pyboy: Any,
    map_id: int,
    coord: Coord,
    *,
    _ctx: Optional[_PortalCtx] = None,
    _depth: int = 0,
) -> PortalStatus:
    """``map_id`` 의 워프 칸 ``coord`` **너머 구역**이 끝났는가."""
    ctx = _ctx
    if ctx is None:
        ctx = _PortalCtx()
        origin = _region(gmap, pyboy, map_id, coord, mark=False)
        ctx.stack[int(map_id)] = set(origin.reachable) | {(int(coord[0]), int(coord[1]))}
    if _depth > int(_opt("close_max_depth", 4)):
        return False, "재귀 한도 초과"

    got = resolve_warp(gmap, map_id, coord)
    if got is None:
        return False, "목적지를 모름 (아직 안 들어가 봄)"
    dest, arrival = int(got[0]), got[1]
    name = _map_label(gmap, dest)
    dest_local = gmap.get_local_map(dest)
    if dest_local is None or not (getattr(dest_local, "grid", None) or {}):
        return False, f'"{name}" 아직 안 들어가 봄'
    if pokecenter_done(dest_local, name):
        return True, f'"{name}" 포켓몬센터 — 간호사를 봤다'
    if arrival is None:
        return False, f'"{name}" 도착 칸을 모름'
    size = (getattr(gmap, "map_size_cells", {}) or {}).get(dest)
    if not size:
        return False, f'"{name}" 맵 크기를 모름'
    # 정적 좌표는 아래쪽 변의 문이 맵 밖(y=height)일 수 있다 — 판정 시드만 안쪽 칸으로.
    arrival = seed_inside_map(arrival, size)
    known = ctx.cells.get(dest, {}).get(arrival)
    if known is not None and known in ctx.memo:
        return ctx.memo[known]                  # 이미 끝난 구역 (2칸 문의 다른 칸 등)
    if arrival in ctx.stack.get(dest, ()):
        return True, "순환 — 위에서 보고 있는 구역"
    if arrival not in dest_local.grid:
        return False, f'"{name}" 도착 칸 {arrival} 미관측 (아직 안 들어가 봄)'

    res = _region(gmap, pyboy, dest, arrival, mark=False)
    if res.reason not in ("closed", "frontier"):
        return False, f'"{name}" 판정 불가 ({res.reason})'
    key = region_key(dest, res)
    if key in ctx.memo:
        return ctx.memo[key]
    frame = ctx.stack.setdefault(dest, set())
    added = set(res.reachable) - frame
    frame |= added

    out: PortalStatus
    try:
        if not res.closed:
            out = (False, f'"{name}" 구역 폐영역 아님 (미관측 경계 {len(res.frontier)}칸)')
        else:
            unpassed = unpassed_static_exits(gmap, dest, res)
            if unpassed:
                out = (False, f'"{name}" 안 지나가 본 맵 밖 문 {len(unpassed)}개')
            elif _passage_map(gmap, dest):
                out = (True, f'"{name}" 폐영역 (도로·게이트 — 안쪽 포탈은 안 따라감)')
            else:
                out = (True, f'"{name}" 폐영역 + 안쪽 포탈 완료')
                inner = sorted(res.warps & observed_in_map_warps(dest_local, size))
                for w in inner:
                    ok, why = portal_done(gmap, pyboy, dest, w, _ctx=ctx, _depth=_depth + 1)
                    if not ok:
                        out = (False, f'"{name}" {w} → {why}')
                        break
    finally:
        frame -= added
    ctx.memo[key] = out
    cellmap = ctx.cells.setdefault(dest, {})
    for c in res.reachable:
        cellmap[c] = key
    return out


def map_done(
    gmap: Any,
    pyboy: Any,
    map_id: int,
    pos: Optional[Coord] = None,
    *,
    recurse: bool = True,
    _seen: Optional[set] = None,
    _depth: int = 0,
    _statuses_out: Optional[Dict[Coord, PortalStatus]] = None,
) -> Tuple[bool, str]:
    """이 맵의 탐험이 끝났는가."""
    map_id = int(map_id)
    seen = set(_seen or ())
    if map_id in seen:
        return True, "순환"          # 집 -> 마을 -> 집. 이미 위에서 보고 있다
    seen.add(map_id)

    if _depth > int(_opt("close_max_depth", 4)):
        return False, "재귀 한도 초과"

    name = _map_label(gmap, map_id)
    if pokecenter_done(gmap.get_local_map(map_id), name):
        return True, "포켓몬센터 — 간호사를 봤다 (구석은 안 본다)"

    if pos is None:
        pos = _seed_pos(gmap, pyboy, map_id)
        if pos is None:
            return False, "판정할 칸이 없다"

    result = evaluate(gmap, pyboy, map_id, pos)
    if result.reason == "tileset_skipped":
        # 제외 타일셋은 판정을 하지 않는다 — **완료가 아니라 미완료**다.
        return False, "제외 타일셋 — 폐영역 판정 안 함 (스텝으로 탐험)"
    if not result.closed:
        return False, f"폐영역 아님 (미관측 경계 {len(result.frontier)}칸)"
    if result.exit_count < int(_opt("close_min_warps", 1)):
        return False, "지나가 본 출구가 없음"
    if not bool(_opt("close_require_all_warps", True)):
        return True, "폐영역"

    if is_exit_free_map(name):
        # 포켓몬센터는 **폐영역만 보면 된다.** 안의 워프는 들어온 입구뿐이라 출구 조건을 걸면 회복하러 들를 때마다 영영 미완료로 남는다.
        return True, "폐영역 (포켓몬센터 — 출구는 안 따진다)"

    if result.unresolved:
        return False, f"안 지나가 본 출구 {len(result.unresolved)}개"
    # 화면 타일로는 문으로 안 잡히는 맵 밖 문 (지하통로 입구 아래 문 등)
    unpassed = unpassed_static_exits(gmap, map_id, result)
    if unpassed:
        return False, f"안 지나가 본 출구 {len(unpassed)}개 (맵 밖 문)"

    if not recurse:
        return True, "이 맵은 다 봤다 (하위 맵은 안 봄)"

    if _passage_map(gmap, map_id):
        # 도로·게이트는 안쪽 포탈을 보지 않는다.
        return True, "폐영역 (도로·게이트 — 포탈은 안 봄)"

    # 보이는 것만으로 조건에 들어간다 — 안 들어가 봤으면 목적지 구역을 몰라 아래에서 "미완료"로 걸린다.
    statuses: Dict[Coord, PortalStatus] = {}
    if close_portals():
        ctx = _PortalCtx()
        ctx.stack[map_id] = set(result.reachable)     # 서 있는 구역 — 되돌아오는 워프는 순환
        for coord in sorted(result.observed_warps):
            statuses[coord] = portal_done(gmap, pyboy, map_id, coord, _ctx=ctx, _depth=_depth + 1)
    else:
        # 예전 방식 — 목적지 **맵 전체**를 밟아 본 칸 하나에서 판정한다.
        for coord in sorted(result.observed_warps):
            dest = warp_destination(gmap, map_id, coord)
            if dest is None:
                statuses[coord] = (False, "출구의 목적지를 모름")
                continue
            if dest in seen:
                statuses[coord] = (True, "순환")
                continue
            ok, why = map_done(gmap, pyboy, dest, None, _seen=seen, _depth=_depth + 1)
            statuses[coord] = (ok, why if ok else f'하위 맵 "{_map_label(gmap, dest)}" 미완료 ({why})')
    if _statuses_out is not None:
        _statuses_out.clear()
        _statuses_out.update(statuses)
    for coord in sorted(statuses):
        ok, why = statuses[coord]
        if not ok:
            return False, f"{coord} 포탈 미완료 ({why})"
    if close_portals():
        return True, "폐영역 + 관측한 포탈 전부 완료"
    return True, "폐영역 + 하위 맵 전부 완료"


def describe(gmap: Any, pyboy: Any, map_id: int, pos: Coord) -> list:
    """`local` 명령용 사람이 읽는 요약."""
    name = _map_label(gmap, map_id)
    result = evaluate(gmap, pyboy, int(map_id), pos)
    need = int(_opt("close_min_warps", 1))
    lines = []

    if result.reason == "tileset_skipped":
        lines.append(
            f'[closure] map="{name}"  폐영역 판정 안 함 — 제외 타일셋'
            "(close_skip_tilesets) · 스텝 예산으로만 탐험합니다"
        )
        return lines
    if result.reason in ("no_map_size", "no_local_map", "no_grid", "seed_off_map"):
        why = {
            "no_map_size": "맵 크기를 아직 못 읽었다",
            "no_local_map": "LocalMap 이 없다",
            "no_grid": "관측된 칸이 없다",
            "seed_off_map": "현재 좌표가 맵 사각형 밖이다",
        }[result.reason]
        lines.append(f'[closure] map="{name}"  판정 불가 — {why}')
        return lines

    # 두 판정을 나눠서 보여 준다.
    done, why = map_done(gmap, pyboy, int(map_id), pos, recurse=False)
    statuses: Dict[Coord, PortalStatus] = {}
    final_done, final_why = map_done(gmap, pyboy, int(map_id), pos, _statuses_out=statuses)
    unresolved = sorted(result.unresolved | unpassed_static_exits(gmap, int(map_id), result))
    head = ("충족 — 이 맵에서는 Exploration 을 부르지 않습니다" if done
            else f"미충족 — {why}")
    lines.append(f'[closure] map="{name}"  이 맵 탐험 = {head}')
    lines.append(
        f"          폐영역={'예' if result.closed else '아니오'} · "
        f"도달 {len(result.reachable)}칸 · 미관측 경계 {len(result.frontier)}칸 · "
        f"지나가 본 출구 {result.exit_count}개(필요 {need}, "
        f"안 지나가 본 {len(unresolved)}) · 밟은 칸 {result.visited}"
    )
    if result.edge_exits:
        lines.append(
            f"          맵 밖 출구(게이트·도로)={sorted(result.edge_exits)[:8]}"
        )
    lines.append(
        "          최종(포탈까지) = "
        + ("완료" if final_done else f"미완료 — {final_why}")
        + "   (참고용 · 탐험 여부와 무관)"
    )
    for coord in sorted(statuses):
        ok, status_why = statuses[coord]
        got = resolve_warp(gmap, int(map_id), coord)
        dest_name = _map_label(gmap, got[0]) if got is not None else "?"
        lines.append(
            f"          포탈 {coord} → {dest_name}: "
            + ("완료" if ok else f"미완료 — {status_why}")
        )
    if unresolved:
        shown = unresolved[:12]
        more = "" if len(unresolved) == len(shown) else f" 외 {len(unresolved)-len(shown)}개"
        lines.append(f"          안 지나가 본 출구={shown}{more}")
    if result.capability_gated:
        have = {GATE_CUT_TREE: _hm_can_cut(pyboy), GATE_WATER: _hm_can_surf(pyboy)}
        parts = [
            f"{GATE_HM_NAMES[g]}({'있음' if have.get(g) else '없음'})" if g in have
            else GATE_HM_NAMES[g]
            for g in sorted(result.gated_by)
            if g in GATE_HM_NAMES
        ]
        lines.append(
            "          경계에 " + " · ".join(parts) + " 로 막힌 칸이 있습니다 — "
            "그 능력을 얻으면 이 맵은 다시 열립니다"
        )
    if 0 < len(result.frontier) <= 12:
        lines.append(f"          미관측 경계 좌표={sorted(result.frontier)}")
    if not bool(_opt("close_on_map_closed", True)):
        lines.append("          (close_on_map_closed=false — 판정만 하고 건너뛰지는 않습니다)")

    local = gmap.get_local_map(int(map_id))
    if local is not None and bool(getattr(local, "closed", False)) != done:
        lines.append(
            f"          ※ 기록된 값은 closed={getattr(local, 'closed')} 입니다 "
            "(Explore 에피소드가 끝날 때 갱신됩니다)"
        )
    return lines


# 스토리 경계를 여는 조건 (core.closure.STORY_GATES 의 이름 → 게임 상태).
ITEM_SILPH_SCOPE = 0x48
EVENT_BEAT_GHOST_MAROWAK = (0xD768, 7)     # data/events.json "Beat Ghost Marowak"


def story_gate_open(pyboy: Any, gate: str) -> bool:
    """그 스토리 경계가 지금 열렸는가."""
    if pyboy is None:
        return True
    from core.gb_memory import has_bag_item
    from engine.hm import has_event
    if gate == GATE_SILPH_SCOPE:
        from core.config import cheat
        if cheat("auto_win_battle") and not cheat("ghost_marowak_needs_scope", True):
            return True               # 치트가 유령 마로왁을 이겨 버리는 설정 — 막힌 게 아니다
        return has_bag_item(pyboy, ITEM_SILPH_SCOPE) or has_event(pyboy, EVENT_BEAT_GHOST_MAROWAK)
    return True


def is_marked_done(gmap: Any, map_id: int, pyboy: Any = None) -> bool:
    """기록된 닫힘을 믿어도 되는가."""
    local = gmap.get_local_map(int(map_id))
    if local is None or not bool(getattr(local, "closed", False)):
        return False
    gated_by = set(getattr(local, "closed_gated_by", ()) or ())
    if not gated_by:
        return True   # 물·나무와 무관하게 닫힌 맵 (실내 등)
    if GATE_SNORLAX in gated_by and not snorlax_blocking(local):
        return False  # 잠만보가 사라졌다 (깨워서 싸웠다) — 그 너머를 다시 본다
    for (mid, spot), gate in STORY_GATES.items():
        if int(mid) == int(map_id) and gate in gated_by and story_gate_closed(mid, spot) is None:
            return False  # 스토리 조건이 갖춰졌다 (실프스코프 등) — 그 너머를 다시 본다
    if pyboy is None:
        return True
    was = getattr(local, "closed_caps", None)
    if not was:
        return True
    now = capabilities(pyboy)          # (can_cut, can_surf)
    for reason, idx in ((GATE_CUT_TREE, 0), (GATE_WATER, 1)):
        if reason in gated_by and bool(now[idx]) != bool(was[idx]):
            return False               # 이 맵을 막던 능력이 바뀌었다 — 다시 본다
    return True


def check(
    gmap: Any,
    pyboy: Any,
    map_id: int,
    pos: Coord,
    *,
    can_cut: Optional[bool] = None,
    tag: str = "explore",
) -> bool:
    """판정 + 로그 + LocalMap 에 결과 기록."""
    result = evaluate(gmap, pyboy, map_id, pos, can_cut=can_cut)
    # **Exploration 을 부를지는 이 맵만 보고 정한다.** 너머 구역이 안 끝났다는
    # 이유로 이미 다 본 맵을 다시 탐험해 봐야 새로 관측되는 게 없다.
    done, why = map_done(gmap, pyboy, map_id, pos, recurse=False)
    # 포탈까지 끝났는지는 기록만 해 둔다 (플래너·`local` 명령이 쓴다).
    statuses: Dict[Coord, PortalStatus] = {}
    final_done, final_why = map_done(gmap, pyboy, map_id, pos, _statuses_out=statuses)

    if bool(_opt("close_debug", False)):
        name = _map_label(gmap, map_id)
        print(
            f"[{tag}] closure map={name} reach={len(result.reachable)} "
            f"미관측경계={len(result.frontier)} 출구={result.exit_count} "
            f"미진입={len(result.unresolved)} 방문={result.visited} "
            f"done={done} ({why}) final={final_done} ({final_why})"
        )
        if 0 < len(result.frontier) <= 12:
            print(f"[{tag}]   미관측 경계 좌표={sorted(result.frontier)}")

    runlog.log(
        "closure",
        맵=_map_label(gmap, map_id),
        map_id=int(map_id),
        종료조건=bool(done),
        사유=why,
        지역완료=bool(final_done),
        지역사유=final_why,
        포탈완료=f"{sum(1 for ok, _ in statuses.values() if ok)}/{len(statuses)}",
        폐곡선=bool(result.closed),
        도달=len(result.reachable),
        미관측경계=len(result.frontier),
        출구=result.exit_count,
        미진입=len(result.unresolved),
        능력=list(capabilities(pyboy)),
    )

    local = gmap.get_local_map(int(map_id))
    if local is not None:
        # LocalMap 에 결과를 남긴다. 폐영역이어도 부조건을 못 넘기면 미완료다.
        local.closed = bool(done)
        # 이 닫힘이 **무엇에** 걸려 있었는지도 같이 남긴다 (물/나무).
        local.closed_gated_by = set(result.gated_by) if done else set()
        local.closed_caps = capabilities(pyboy) if done else None
        # 포탈까지의 최종 완료와 워프별 상태 — LLM 의 "포탈 n곳 미완료" 와 `local` 이 읽는다.
        local.final_done = bool(final_done)
        local.portal_done = dict(statuses)

    if not bool(_opt("close_on_map_closed", True)):
        return False
    return done


def reopen_all(gmap: Any, why: str = "") -> None:
    """모든 맵의 닫힘 표시를 즉시 지운다 (포탈 판정 기록도 같이)."""
    n = 0
    for map_id in list(getattr(gmap, "graph", {}).nodes or ()):
        local = gmap.get_local_map(int(map_id))
        if local is None:
            continue
        if getattr(local, "closed", False):
            local.closed = False
            n += 1
        local.final_done = False
        portal_done_map = getattr(local, "portal_done", None)
        if isinstance(portal_done_map, dict):
            portal_done_map.clear()
    if n and why:
        print(f"[explore] {why} — 닫힌 맵 {n}개를 다시 엽니다")


__all__ = [
    "check",
    "close_portals",
    "describe",
    "evaluate",
    "is_done",
    "is_marked_done",
    "is_passage_map",
    "map_done",
    "portal_done",
    "reopen_all",
    "resolve_warp",
    "skip_tilesets",
]
