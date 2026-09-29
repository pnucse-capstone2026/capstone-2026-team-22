"""runtime/logging.json → 실험 요약 표 + 그래프용 CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 읽기
# ---------------------------------------------------------------------------
def load_entries(path: Path) -> List[Dict[str, Any]]:
    """logging.json 을 번호 순서대로 편 리스트로."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[warn] {path} 를 읽지 못했습니다: {exc}")
        return []
    if not isinstance(data, dict):
        return []

    def order(key: str) -> int:
        try:
            return int(key)
        except ValueError:
            return 0

    out: List[Dict[str, Any]] = []
    for key in sorted(data, key=order):
        item = data[key]
        if isinstance(item, dict):
            out.append(item)
    return out


def _seconds(entry: Dict[str, Any]) -> Optional[int]:
    """기록된 시각(HH:MM:SS)을 자정 기준 초로."""
    raw = str(entry.get("time") or "")
    try:
        t = datetime.strptime(raw, "%H:%M:%S").time()
    except ValueError:
        return None
    return t.hour * 3600 + t.minute * 60 + t.second


def elapsed_minutes(entries: List[Dict[str, Any]]) -> Optional[float]:
    """첫 기록부터 마지막 기록까지의 분."""
    stamps = [s for s in (_seconds(e) for e in entries) if s is not None]
    if len(stamps) < 2:
        return None
    backward = sum(1 for a, b in zip(stamps, stamps[1:]) if b < a)
    if backward > 1:
        return (max(stamps) - min(stamps)) / 60.0
    span = stamps[-1] - stamps[0] + (24 * 3600 if backward else 0)
    return span / 60.0


# ---------------------------------------------------------------------------
# 1. 진행 곡선
# ---------------------------------------------------------------------------
def progress_rows(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """계획 사이클마다 한 줄."""
    rows: List[Dict[str, Any]] = []
    seen_cycles = set()
    for entry in entries:
        if entry.get("kind") != "planning":
            continue
        cycle = entry.get("cycle")
        if cycle is None:
            cycle = len(rows) + 1
        cycle = int(cycle)
        if cycle in seen_cycles:
            continue          # 같은 사이클이 두 번 찍혔으면 첫 것만
        seen_cycles.add(cycle)
        badges = entry.get("배지") or []
        maps = entry.get("관측_맵") or []
        status = entry.get("맵_상태") or {}
        closed = sum(1 for v in status.values() if "완료" in str(v)
                     and "미완료" not in str(v))
        rows.append({
            "cycle": cycle,
            "steps": int(entry.get("스텝") or 0),
            "badges": len(badges),
            "maps_observed": len(maps),
            "maps_closed": closed,
            "events": int(entry.get("이벤트수") or 0),
            "current_map": entry.get("현재_맵") or "",
        })
    rows.sort(key=lambda r: r["cycle"])
    return rows


# ---------------------------------------------------------------------------
# 2. 스킬별 시도 횟수
# ---------------------------------------------------------------------------
def skill_counts(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """스킬마다 (시도, 성공, 실패)."""
    plan_calls = sum(1 for e in entries if e.get("kind") == "planning")
    plan_cycles = len({e.get("cycle") for e in entries
                       if e.get("kind") == "planning"} - {None})

    nav: Dict[str, List[int]] = defaultdict(lambda: [0, 0])   # [성공, 실패]
    for e in entries:
        if e.get("kind") != "nav_end":
            continue
        kind = "Navigation (맵 이동)" if e.get("종류") == "map" else "Navigation (NPC 대화)"
        nav[kind][0 if e.get("success") else 1] += 1

    # nav_end 가 없던 시절의 기록도 최소한 실패는 센다.
    legacy_fail = sum(1 for e in entries if e.get("kind") == "nav_fail")

    explore: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for e in _explore_episodes(entries):
        name = str(e.get("스킬") or "Exploration")
        explore[name][0 if e.get("success") else 1] += 1

    battles = [e for e in entries if e.get("kind") == "battle_end"]
    battle_ok = sum(1 for e in battles if e.get("success"))
    battle_manual = sum(1 for e in battles if e.get("수동전환"))

    rows: List[Dict[str, Any]] = [{
        "skill": "Planning",
        "attempts": plan_calls,
        "success": plan_calls,
        "fail": 0,
        "note": f"사이클 {plan_cycles}회",
    }]
    for name, (ok, bad) in sorted(nav.items()):
        rows.append({
            "skill": name,
            "attempts": ok + bad,
            "success": ok,
            "fail": bad,
            "note": "",
        })
    if not nav and legacy_fail:
        rows.append({
            "skill": "Navigation",
            "attempts": legacy_fail,
            "success": 0,
            "fail": legacy_fail,
            "note": "실패만 기록된 옛 로그",
        })
    for name, (ok, bad) in sorted(explore.items()):
        rows.append({
            "skill": f"Exploration ({name})" if name != "Exploration" else name,
            "attempts": ok + bad,
            "success": ok,
            "fail": bad,
            "note": "",
        })
    if battles:
        rows.append({
            "skill": "Battle",
            "attempts": len(battles),
            "success": battle_ok,
            "fail": len(battles) - battle_ok,
            "note": f"수동 전환 {battle_manual}회" if battle_manual else "",
        })
    return rows


def _explore_episodes(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """탐험 한 번 = 한 항목."""
    out: List[Dict[str, Any]] = []
    for e in entries:
        if e.get("kind") != "explore_end":
            continue
        if e.get("조기종료") or "success" not in e:
            continue
        out.append(e)
    return out


def explore_per_map(entries: List[Dict[str, Any]], top: int = 10) -> List[Tuple[str, int]]:
    """어느 맵에서 탐험이 몇 번 불렸는가."""
    counter: Counter = Counter()
    for e in _explore_episodes(entries):
        counter[str(e.get("맵") or "?")] += 1
    return counter.most_common(top)


def event_rows(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """스토리 이벤트가 몇 스텝째에 났는가."""
    rows: List[Dict[str, Any]] = []
    prev = 0
    for e in entries:
        if e.get("kind") != "event":
            continue
        steps = int(e.get("스텝") or 0)
        rows.append({
            "event": str(e.get("이름") or "?"),
            "steps": steps,
            "count": len(rows) + 1,       # 이때까지 달성한 이벤트 수 (누적)
            "delta": steps - prev,        # 직전 이벤트 이후 걸린 스텝
            "cycle": e.get("cycle") if e.get("cycle") is not None else "",
            "map": str(e.get("맵") or ""),
            "time": str(e.get("time") or ""),
        })
        prev = steps
    return rows


def step_rows(
    events: List[Dict[str, Any]],
    prog: List[Dict[str, Any]],
    bin_size: int,
) -> List[Dict[str, Any]]:
    """스텝 구간마다 그때까지의 이벤트·배지·관측 맵 수."""
    marks = [int(e.get("steps") or 0) for e in events]
    marks += [int(r.get("steps") or 0) for r in prog]
    last = max(marks) if marks else 0
    if last <= 0:
        return []

    rows: List[Dict[str, Any]] = []
    for step in range(0, last + bin_size, bin_size):
        done = sum(1 for e in events if int(e.get("steps") or 0) <= step)
        # 그 스텝 이전의 마지막 계획 기록에서 배지·관측 맵을 읽는다.
        seen = [r for r in prog if int(r.get("steps") or 0) <= step]
        latest = seen[-1] if seen else {}
        rows.append({
            "steps": step,
            "events": done,
            "badges": latest.get("badges", 0),
            "maps_observed": latest.get("maps_observed", 0),
            "maps_closed": latest.get("maps_closed", 0),
            "cycle": latest.get("cycle", 0),
        })
    return rows


_SERIES_KINDS = (
    ("planning", "Planning"),
    ("nav_end", "Navigation"),
    ("explore_end", "Exploration"),
    ("battle_end", "Battle"),
)


def skill_series(
    entries: List[Dict[str, Any]], bin_size: int
) -> List[Dict[str, Any]]:
    """스텝 구간마다 **그때까지 실행된 누적 횟수**를 스킬별로."""
    marks: Dict[str, List[int]] = {}
    for kind, label in _SERIES_KINDS:
        if kind == "explore_end":
            picked = _explore_episodes(entries)
        else:
            picked = [e for e in entries if e.get("kind") == kind]
        marks[label] = sorted(int(e.get("스텝") or 0) for e in picked)

    last = max((m[-1] for m in marks.values() if m), default=0)
    if last <= 0:
        return []

    rows: List[Dict[str, Any]] = []
    for step in range(0, last + bin_size, bin_size):
        row: Dict[str, Any] = {"steps": step}
        for _, label in _SERIES_KINDS:
            row[label] = sum(1 for m in marks[label] if m <= step)
        rows.append(row)
    return rows


def plan_quality(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """계획 품질 — 사이클이 실제로 무엇을 이뤘는지."""
    cycles = {e.get("cycle") for e in entries
              if e.get("kind") == "planning"} - {None}
    nav_ok: set = set()
    talk_ok: set = set()
    explore_ok: set = set()
    event_hit: set = set()
    for e in entries:
        cyc = e.get("cycle")
        if cyc is None:
            continue
        kind = e.get("kind")
        if kind == "nav_end" and e.get("success"):
            nav_ok.add(cyc)
            if e.get("종류") == "npc":
                talk_ok.add(cyc)
        elif kind == "explore_end" and e.get("success"):
            explore_ok.add(cyc)
        elif kind == "event":
            event_hit.add(cyc)
    empty = cycles - (nav_ok | explore_ok | event_hit)
    total = len(cycles)
    return {
        "cycles": total,
        "nav_ok": len(nav_ok & cycles),
        "talk_ok": len(talk_ok & cycles),
        "event_hit": len(event_hit & cycles),
        "empty": len(empty),
        "ratio": ((total - len(empty)) / total) if total else 0.0,
        "format_retries": sum(1 for e in entries
                              if e.get("kind") == "plan_fail"),
    }


def closure_summary(entries: List[Dict[str, Any]]) -> Tuple[int, int]:
    """폐영역 판정 횟수와 그중 '완료'로 나온 횟수."""
    checks = [e for e in entries if e.get("kind") == "closure"]
    done = sum(1 for e in checks if e.get("종료조건"))
    return len(checks), done


# ---------------------------------------------------------------------------
# 출력
# ---------------------------------------------------------------------------
def _table(headers: List[str], rows: List[List[Any]]) -> str:
    cols = len(headers)
    width = [len(str(headers[i])) for i in range(cols)]
    for row in rows:
        for i in range(cols):
            width[i] = max(width[i], len(str(row[i])))
    line = "  ".join(str(headers[i]).ljust(width[i]) for i in range(cols))
    out = [line, "  ".join("-" * width[i] for i in range(cols))]
    for row in rows:
        out.append("  ".join(str(row[i]).ljust(width[i]) for i in range(cols)))
    return "\n".join(out)


def report(
    path: Path, entries: List[Dict[str, Any]], bin_size: int = 100
) -> Dict[str, Any]:
    prog = progress_rows(entries)
    skills = skill_counts(entries)
    last = prog[-1] if prog else {}
    minutes = elapsed_minutes(entries)
    checks, closed_hits = closure_summary(entries)

    print(f"\n=== {path} ===")
    if not entries:
        print("  (기록 없음)")
        return {"path": str(path), "progress": prog, "skills": skills,
                "events": [], "steps": [], "series": [],
                "quality": plan_quality(entries)}

    print("\n[진행 정도]")
    summary = [
        ["계획 사이클", len(prog)],
        ["배지", last.get("badges", 0)],
        ["관측한 맵", last.get("maps_observed", 0)],
        ["탐험 완료 맵", last.get("maps_closed", 0)],
        ["스토리 이벤트", last.get("events", 0)],
        ["마지막 위치", last.get("current_map", "?")],
        ["실행 시간(분)", f"{minutes:.1f}" if minutes is not None else "?"],
    ]
    print(_table(["항목", "값"], summary))

    print("\n[스킬별 시도]")
    print(_table(
        ["스킬", "시도", "성공", "실패", "비고"],
        [[r["skill"], r["attempts"], r["success"], r["fail"], r["note"]]
         for r in skills],
    ))

    if prog:
        total = sum(r["attempts"] for r in skills if r["skill"] != "Planning")
        print(f"\n  사이클당 스킬 호출 = {total / len(prog):.2f}회")
    if checks:
        print(f"  폐영역 판정 {checks}회 중 완료 {closed_hits}회")

    per_map = explore_per_map(entries)
    if per_map:
        print("\n[맵별 Exploration 호출]")
        print(_table(["맵", "횟수"], [[m, n] for m, n in per_map]))

    events = event_rows(entries)
    if events:
        print("\n[스토리 이벤트 — 세션 시작부터의 스텝]")
        print(_table(
            ["스텝", "간격", "누적", "사이클", "맵", "이벤트"],
            [[e["steps"], e["delta"], e["count"], e["cycle"], e["map"],
              e["event"]] for e in events],
        ))

    steps = step_rows(events, prog, bin_size)
    if steps:
        print(f"\n[스텝별 진행 — {bin_size} 스텝 간격]")
        print(_table(
            ["스텝", "이벤트", "배지", "관측 맵", "폐영역 맵", "사이클"],
            [[r["steps"], r["events"], r["badges"], r["maps_observed"],
              r["maps_closed"], r["cycle"]] for r in steps],
        ))

    series = skill_series(entries, bin_size)
    if series:
        labels = [label for _, label in _SERIES_KINDS]
        print(f"\n[스텝별 스킬 실행 (누적) — {bin_size} 스텝 간격]")
        print(_table(
            ["스텝"] + labels,
            [[r["steps"]] + [r[k] for k in labels] for r in series],
        ))

    quality = plan_quality(entries)
    if quality["cycles"]:
        total = quality["cycles"]

        def pct(n: int) -> str:
            return f"{n}  ({n / total * 100:.1f}%)" if total else str(n)

        print("\n[계획 품질]")
        print(_table(["항목", "값"], [
            ["전체 사이클", total],
            ["목표 맵 도달", pct(quality["nav_ok"])],
            ["그중 대화까지", pct(quality["talk_ok"])],
            ["이벤트가 난 사이클", pct(quality["event_hit"])],
            ["빈 사이클", pct(quality["empty"])],
            ["형식 오류 재호출", quality["format_retries"]],
        ]))
        print("  (빈 사이클 = 이동도 탐험도 성공 못 하고 이벤트도 없던 사이클)")

    return {"path": str(path), "progress": prog, "skills": skills,
            "events": events, "steps": steps, "series": series,
            "quality": quality}


def write_csv(rows: List[Dict[str, Any]], fields: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    print(f"  → {path}")


# 그래프 (matplotlib 이 있을 때만) 보고서에 넣을 PNG.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
_INK = "#0b0b0b"
_INK_SOFT = "#52514e"
_GRID = "#dcdbd6"
_SURFACE = "#fcfcfb"


# 한글이 네모로 깨지지 않도록 실제로 있는 폰트 파일을 찾아 등록한다.
_KO_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumSquareRoundR.ttf",
    "/mnt/c/Windows/Fonts/malgun.ttf",          # WSL 에서 윈도우 폰트
    "C:/Windows/Fonts/malgun.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
)


def _use_korean_font(plt) -> bool:
    from matplotlib import font_manager
    for path in _KO_FONT_CANDIDATES:
        f = Path(path)
        if not f.is_file():
            continue
        try:
            font_manager.fontManager.addfont(str(f))
            name = font_manager.FontProperties(fname=str(f)).get_name()
        except Exception:
            continue
        plt.rcParams["font.family"] = name
        return True
    return False


def _plt():
    """matplotlib 을 늦게 부른다."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    if not _use_korean_font(plt):
        print("  (한글 폰트를 못 찾아 그래프 글자가 깨질 수 있습니다 — "
              "나눔고딕이나 맑은 고딕을 설치하세요)")
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _line_chart(plt, path: Path, rows, x_key, series, title, ylabel):
    """스텝을 가로축으로 하는 꺾은선 하나."""
    xs = [r[x_key] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)

    ends = []
    for i, (key, label) in enumerate(series):
        ys = [r.get(key, 0) for r in rows]
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        # 개수는 구간 사이에서 연속으로 늘지 않는다.
        ax.plot(xs, ys, color=color, linewidth=2.0, label=label,
                drawstyle="steps-post", solid_capstyle="round")
        if xs:
            ends.append([float(ys[-1]), f"{label} {ys[-1]}", color])

    # 선 끝 직접 라벨 — 색 대비가 낮은 슬롯을 위한 보조 표시.
    if ends:
        span = max(
            max(r.get(k, 0) for r in rows for k, _ in series) or 1, 1
        )
        gap = span * 0.055
        ends.sort(key=lambda e: e[0])
        for j in range(1, len(ends)):
            if ends[j][0] - ends[j - 1][0] < gap:
                ends[j][0] = ends[j - 1][0] + gap
        for y, text, color in ends:
            ax.annotate(
                text, (xs[-1], y), color=color, fontsize=8.5,
                va="center", ha="left", clip_on=False,
                xytext=(6, 0), textcoords="offset points",
            )

    ax.set_title(title, color=_INK, fontsize=12, pad=12, loc="left")
    ax.set_xlabel("스텝 (누적 이동·탐험 액션)", color=_INK_SOFT, fontsize=9.5)
    # 세로로 눕힌 한글은 읽기 나쁘다 — 축 위쪽에 가로로 붙인다.
    ax.set_ylabel(ylabel, color=_INK_SOFT, fontsize=9.5,
                  rotation=0, loc="top", labelpad=-14)
    ax.grid(True, axis="y", color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_INK_SOFT, labelsize=9)
    # 개수 축에 0.5 같은 눈금이 생기면 안 된다.
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    # 위로 밀어낸 끝 라벨까지 들어가게 위쪽을 넉넉히 잡는다.
    top = max([e[0] for e in ends] or [1]) * 1.12 + 0.5
    ax.set_ylim(0, top)
    ax.set_xlim(left=0)                      # 스텝에 음수는 없다
    ax.margins(x=0.20)                       # 끝 라벨이 잘리지 않게
    ax.legend(frameon=False, fontsize=9, labelcolor=_INK_SOFT, loc="upper left")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=_SURFACE)
    plt.close(fig)
    print(f"  → {path}")


def write_plots(res: Dict[str, Any], out: Path, stem: str) -> None:
    """한 판의 그래프 세 장."""
    plt = _plt()
    if plt is None:
        print("  (matplotlib 이 없어 그래프는 건너뜁니다 — "
              "pip install 'matplotlib==3.7.5' 로 설치하면 나옵니다)")
        return

    steps = res.get("steps") or []
    if steps:
        _line_chart(
            plt, out / f"fig_progress_{stem}.png", steps, "steps",
            [("events", "스토리 이벤트"), ("badges", "배지")],
            "진행도", "개수",
        )
        _line_chart(
            plt, out / f"fig_observation_{stem}.png", steps, "steps",
            [("maps_observed", "관측한 맵"), ("maps_closed", "탐험 완료 맵")],
            "관측 확장", "맵 수",
        )

    series = res.get("series") or []
    if series:
        _line_chart(
            plt, out / f"fig_skills_{stem}.png", series, "steps",
            [(label, label) for _, label in _SERIES_KINDS],
            "스킬별 실행 횟수 (누적)", "실행 횟수",
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="*", default=[], help="logging.json 경로들")
    ap.add_argument("--out", default="", help="CSV 를 쓸 폴더 (없으면 안 씀)")
    ap.add_argument("--bin", type=int, default=100,
                    help="스텝별 진행 표의 구간 크기 (기본 100)")
    ap.add_argument("--no-plot", action="store_true",
                    help="--out 을 줘도 PNG 는 만들지 않는다 (CSV 만)")
    args = ap.parse_args()

    paths = [Path(p) for p in args.logs] or [ROOT / "runtime" / "logging.json"]
    results = []
    for path in paths:
        entries = load_entries(path)
        results.append(report(path, entries, max(1, int(args.bin))))

    if not args.out:
        return
    out = Path(args.out)
    print("\n[CSV]")
    for i, res in enumerate(results, start=1):
        tag = Path(res["path"]).stem
        stem = f"{i:02d}_{tag}" if len(results) > 1 else tag
        write_csv(
            res["progress"],
            ["cycle", "steps", "badges", "maps_observed", "maps_closed",
             "events", "current_map"],
            out / f"progress_{stem}.csv",
        )
        write_csv(
            res["skills"],
            ["skill", "attempts", "success", "fail", "note"],
            out / f"skills_{stem}.csv",
        )
        if res["events"]:
            write_csv(
                res["events"],
                ["steps", "count", "delta", "cycle", "map", "event", "time"],
                out / f"events_{stem}.csv",
            )
        if res["steps"]:
            write_csv(
                res["steps"],
                ["steps", "events", "badges", "maps_observed", "maps_closed",
                 "cycle"],
                out / f"steps_{stem}.csv",
            )
        if res["series"]:
            write_csv(
                res["series"],
                ["steps"] + [label for _, label in _SERIES_KINDS],
                out / f"skill_series_{stem}.csv",
            )
        if not args.no_plot:
            write_plots(res, out, stem)


if __name__ == "__main__":
    main()
