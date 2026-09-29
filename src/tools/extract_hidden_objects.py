"""pokered hidden_events.asm + bg_events -> hidden_objects.json"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POKERED = ROOT / "third_party" / "pokered-master"
OUT = ROOT / "data" / "hidden_objects.json"

NPC_FUNCS = {"PrintBenchGuyText"}
SKIP_FUNCS = {"HiddenItems", "HiddenCoins"}


def parse_map_ids() -> dict:
    """constants/map_constants.asm -> {MAP_NAME: id}."""
    ids = {}
    idx = 0
    pat = re.compile(r"^\s*map_const\s+(\w+),")
    for line in (POKERED / "constants" / "map_constants.asm").read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        m = pat.match(line)
        if m:
            ids[m.group(1)] = idx
            idx += 1
    return ids


def parse_hidden_events() -> dict:
    """hidden_events.asm -> {MAP_NAME: [[x, y, kind], ...]}."""
    per_map = {}
    current = None
    ev_pat = re.compile(
        r"^\s*(?:hidden_event|hidden_text_predef)\s+"
        r"(-?\d+)\s*,\s*(-?\d+)\s*,\s*(\w+)"
    )
    for line in (POKERED / "data" / "events" / "hidden_events.asm").read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        m = re.match(r"^\s*hidden_events_for\s+(\w+)", line)
        if m:
            current = m.group(1)
            per_map.setdefault(current, [])
            continue
        if current is None:
            continue
        m = ev_pat.match(line)
        if m:
            x, y, func = int(m.group(1)), int(m.group(2)), m.group(3)
            if func in SKIP_FUNCS:
                continue
            kind = "npc" if func in NPC_FUNCS else "object"
            per_map[current].append([x, y, kind, func])
    return per_map


def parse_bg_events() -> dict:
    """data/maps/objects/*.asm 의 bg_event(표지판 등) -> {MAP_NAME: [[x,y,kind,func]]}."""
    per_map = {}
    bg_pat = re.compile(r"^\s*bg_event\s+(-?\d+)\s*,\s*(-?\d+)\s*,\s*(\w+)")
    warps_pat = re.compile(r"^\s*def_warps_to\s+(\w+)")
    for asm in sorted((POKERED / "data" / "maps" / "objects").glob("*.asm")):
        text = asm.read_text(encoding="utf-8", errors="replace")
        m = warps_pat.search(text, re.MULTILINE)
        map_name = None
        events = []
        for line in text.splitlines():
            bm = bg_pat.match(line)
            if bm:
                events.append(
                    [int(bm.group(1)), int(bm.group(2)), "object", bm.group(3)]
                )
                continue
            wm = warps_pat.match(line)
            if wm:
                map_name = wm.group(1)
        if map_name and events:
            per_map.setdefault(map_name, []).extend(events)
    return per_map


def main() -> None:
    map_ids = parse_map_ids()
    events = parse_hidden_events()

    # bg_event(표지판) 병합 — 같은 좌표에 hidden event가 있으면 그쪽 우선
    for name, evs in parse_bg_events().items():
        existing = events.setdefault(name, [])
        taken = {(e[0], e[1]) for e in existing}
        for ev in evs:
            if (ev[0], ev[1]) not in taken:
                existing.append(ev)

    out = {}
    missing = []
    n_events = 0
    for name, evs in events.items():
        if not evs:
            continue
        if name not in map_ids:
            missing.append(name)
            continue
        out[str(map_ids[name])] = {"name": name, "events": evs}
        n_events += len(evs)

    OUT.write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"maps: {len(out)}, events: {n_events} -> {OUT.name}")
    if missing:
        print(f"WARNING unmatched map names: {missing}")


if __name__ == "__main__":
    main()
