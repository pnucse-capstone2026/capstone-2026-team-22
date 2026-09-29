"""pokered warp tile IDs + warp_event (x,y) + connection 경계 칸 -> warps.json"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
POKERED = ROOT / "third_party" / "pokered-master"
OUT = ROOT / "data" / "warps.json"

LAST_MAP = "LAST_MAP"


def _parse_int_token(tok: str) -> int:
    tok = tok.strip()
    if tok.startswith("$"):
        return int(tok[1:], 16)
    return int(tok)


def parse_tileset_ids() -> Dict[str, int]:
    ids: Dict[str, int] = {}
    idx = 0
    started = False
    for line in (POKERED / "constants" / "tileset_constants.asm").read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        if re.match(r"^\s*const_def\b", line):
            started = True
            idx = 0
            continue
        if not started:
            continue
        m = re.match(r"^\s*const\s+(\w+)", line)
        if m:
            ids[m.group(1)] = idx
            idx += 1
            continue
        if re.match(r"^\s*DEF\s+NUM_TILESETS\b", line):
            break
    return ids


def parse_map_meta() -> Dict[str, Dict]:
    """map_constants.asm -> {CONST: {id, width_blocks, height_blocks}}."""
    meta: Dict[str, Dict] = {}
    idx = 0
    pat = re.compile(r"^\s*map_const\s+(\w+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)")
    for line in (POKERED / "constants" / "map_constants.asm").read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        m = pat.match(line)
        if m:
            meta[m.group(1)] = {
                "id": idx,
                "w_blocks": int(m.group(2)),
                "h_blocks": int(m.group(3)),
            }
            idx += 1
    return meta


def _collect_tile_bytes(lines: List[str], start: int) -> Tuple[List[int], int]:
    tiles: List[int] = []
    i = start
    while i < len(lines):
        line = lines[i].split(";")[0].strip()
        if not line:
            i += 1
            continue
        if line.endswith(":") or line.startswith("WarpTileIDPointers"):
            break
        if line.startswith("MACRO ") or line.startswith("ENDM") or line.startswith(
            "INCLUDE "
        ):
            i += 1
            continue
        if line.startswith("assert_table_length") or line.startswith("table_width"):
            i += 1
            continue

        m = re.match(r"^(?:warp_tiles|db)\s+(.*)$", line)
        if not m:
            if line.startswith(";") or line == "ENDC" or line.startswith("IF "):
                i += 1
                continue
            break

        args = m.group(1).strip()
        if not args:
            i += 1
            break

        for part in args.split(","):
            part = part.strip()
            if not part:
                continue
            val = _parse_int_token(part)
            if val in (-1, 0) and len(args.split(",")) == 1:
                i += 1
                return tiles, i
            if val == -1:
                continue
            tiles.append(val)
        i += 1
        if line.startswith("warp_tiles"):
            break
        if re.search(r"\b-1\b", args):
            break
    return tiles, i


def parse_warp_tile_ids() -> Dict[int, List[int]]:
    text = (POKERED / "data" / "tilesets" / "warp_tile_ids.asm").read_text(
        encoding="utf-8", errors="replace"
    )
    lines = text.splitlines()

    ptrs: List[str] = []
    in_ptrs = False
    for line in lines:
        if line.startswith("WarpTileIDPointers:"):
            in_ptrs = True
            continue
        if in_ptrs:
            if re.match(r"^\s*assert_table_length", line):
                break
            m = re.match(r"^\s*dw\s+\.(\w+)", line)
            if m:
                ptrs.append(m.group(1))

    label_tiles: Dict[str, List[int]] = {}
    pending_labels: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].split(";")[0].strip()
        lm = re.match(r"^\.(\w+):\s*$", line)
        if lm:
            pending_labels.append(lm.group(1))
            i += 1
            continue
        if pending_labels and (line.startswith("warp_tiles") or line.startswith("db ")):
            tiles, ni = _collect_tile_bytes(lines, i)
            for lab in pending_labels:
                label_tiles[lab] = list(tiles)
            pending_labels = []
            i = ni
            continue
        i += 1

    out: Dict[int, List[int]] = {}
    for ts_id, label in enumerate(ptrs):
        out[ts_id] = sorted(set(label_tiles.get(label, [])))
    return out


def parse_warp_events(map_meta: Dict[str, Dict]) -> Dict[str, List[dict]]:
    """objects/*.asm → {MAP_CONST: [{x,y,dest_const,dest_warp}, ...]}."""
    warp_pat = re.compile(
        r"^\s*warp_event\s+(-?\d+)\s*,\s*(-?\d+)\s*,\s*(\w+)\s*,\s*(\d+)"
    )
    warps_to_pat = re.compile(r"^\s*def_warps_to\s+(\w+)")
    per_map: Dict[str, List[dict]] = {}

    for asm in sorted((POKERED / "data" / "maps" / "objects").glob("*.asm")):
        text = asm.read_text(encoding="utf-8", errors="replace")
        map_const: Optional[str] = None
        warps: List[dict] = []
        for line in text.splitlines():
            wm = warps_to_pat.match(line)
            if wm:
                map_const = wm.group(1)
                continue
            m = warp_pat.match(line)
            if m:
                warps.append(
                    {
                        "x": int(m.group(1)),
                        "y": int(m.group(2)),
                        "dest_const": m.group(3),
                        "dest_warp": int(m.group(4)),
                        "kind": "warp",
                    }
                )
        if map_const and warps:
            if map_const not in map_meta:
                continue
            per_map[map_const] = warps
    return per_map


def parse_connections(map_meta: Dict[str, Dict]) -> Dict[str, List[dict]]:
    """headers/*.asm connection → {MAP_CONST: [{dir, dest_const, offset}, ...]}."""
    header_pat = re.compile(r"^\s*map_header\s+\w+\s*,\s*(\w+)\s*,")
    conn_pat = re.compile(
        r"^\s*connection\s+(north|south|west|east)\s*,\s*\w+\s*,\s*(\w+)\s*,\s*(-?\d+)"
    )
    per_map: Dict[str, List[dict]] = {}

    for asm in sorted((POKERED / "data" / "maps" / "headers").glob("*.asm")):
        text = asm.read_text(encoding="utf-8", errors="replace")
        map_const: Optional[str] = None
        for line in text.splitlines():
            hm = header_pat.match(line)
            if hm:
                map_const = hm.group(1)
                per_map.setdefault(map_const, [])
                continue
            cm = conn_pat.match(line)
            if cm and map_const:
                per_map[map_const].append(
                    {
                        "dir": cm.group(1),
                        "dest_const": cm.group(2),
                        "offset": int(cm.group(3)),
                        "kind": "connection",
                    }
                )
    return per_map


def connection_cells(
    direction: str,
    cur_w_blocks: int,
    cur_h_blocks: int,
    dest_w_blocks: int,
    dest_h_blocks: int,
    offset: int,
) -> List[Tuple[int, int]]:
    """CheckMapConnections 트리거 칸 (맵 밖 한 칸) + 겹치는 스트립."""
    cw, ch = cur_w_blocks * 2, cur_h_blocks * 2
    dw, dh = dest_w_blocks * 2, dest_h_blocks * 2
    cells: List[Tuple[int, int]] = []

    if direction == "east":
        # new_y = old_y + (-2*offset); valid when new_y ∈ [0, dh)
        align = -2 * offset
        y_lo = max(0, -align)
        y_hi = min(ch, dh - align)
        for y in range(y_lo, y_hi):
            cells.append((cw, y))
    elif direction == "west":
        # x = -1. new_y = old_y + (-2*offset)
        align = -2 * offset
        y_lo = max(0, -align)
        y_hi = min(ch, dh - align)
        for y in range(y_lo, y_hi):
            cells.append((-1, y))
    elif direction == "south":
        # y = ch. new_x = old_x + (-2*offset)
        align = -2 * offset
        x_lo = max(0, -align)
        x_hi = min(cw, dw - align)
        for x in range(x_lo, x_hi):
            cells.append((x, ch))
    elif direction == "north":
        align = -2 * offset
        x_lo = max(0, -align)
        x_hi = min(cw, dw - align)
        for x in range(x_lo, x_hi):
            cells.append((x, -1))
    return cells


def main() -> None:
    tileset_ids = parse_tileset_ids()
    warp_tiles = parse_warp_tile_ids()
    map_meta = parse_map_meta()
    id_to_tileset = {v: k for k, v in tileset_ids.items()}
    id_to_map = {v["id"]: k for k, v in map_meta.items()}

    warp_events = parse_warp_events(map_meta)
    connections = parse_connections(map_meta)

    tilesets_out = {}
    for ts_id in sorted(warp_tiles):
        tilesets_out[str(ts_id)] = {
            "name": id_to_tileset.get(ts_id, f"TILESET_{ts_id}"),
            "warp_tiles": warp_tiles[ts_id],
        }

    maps_out: Dict[str, dict] = {}
    n_warp_pts = 0
    n_conn_pts = 0

    all_consts = set(warp_events) | set(connections) | set(map_meta)
    for const in sorted(all_consts, key=lambda c: map_meta.get(c, {}).get("id", 10**9)):
        if const not in map_meta:
            continue
        info = map_meta[const]
        mid = info["id"]
        w_b, h_b = info["w_blocks"], info["h_blocks"]
        if w_b <= 0 or h_b <= 0:
            continue

        points: List[dict] = []

        for w in warp_events.get(const, []):
            dest_c = w["dest_const"]
            dest_id: Optional[int]
            if dest_c == LAST_MAP:
                dest_id = None
            elif dest_c in map_meta:
                dest_id = map_meta[dest_c]["id"]
            else:
                dest_id = None
            points.append(
                {
                    "x": w["x"],
                    "y": w["y"],
                    "kind": "warp",
                    "dest_id": dest_id,
                    "dest_const": dest_c,
                    "dest_warp": w["dest_warp"],
                }
            )
            n_warp_pts += 1

        for c in connections.get(const, []):
            dest_c = c["dest_const"]
            if dest_c not in map_meta:
                continue
            dinfo = map_meta[dest_c]
            for x, y in connection_cells(
                c["dir"], w_b, h_b, dinfo["w_blocks"], dinfo["h_blocks"], c["offset"]
            ):
                points.append(
                    {
                        "x": x,
                        "y": y,
                        "kind": "connection",
                        "dir": c["dir"],
                        "dest_id": dinfo["id"],
                        "dest_const": dest_c,
                    }
                )
                n_conn_pts += 1

        if not points:
            continue

        maps_out[str(mid)] = {
            "const": const,
            "width_cells": w_b * 2,
            "height_cells": h_b * 2,
            "points": points,
        }

    payload = {
        "_comment": (
            "타일셋별 warp 타일 ID + 맵별 고정 워프/connection 칸 (x,y). "
            "dest_id=null 은 LAST_MAP(실내→야외). Map.py 가 LocalMap 생성 시 시드."
        ),
        "tilesets": tilesets_out,
        "maps": maps_out,
    }
    OUT.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    n_tile = sum(len(v["warp_tiles"]) for v in tilesets_out.values())
    print(
        f"tilesets: {len(tilesets_out)} (tile_ids={n_tile}), "
        f"maps: {len(maps_out)} (warp_pts={n_warp_pts}, conn_pts={n_conn_pts}) "
        f"-> {OUT.name}"
    )
    # sanity: Pewter east → Route 3
    pewter = maps_out.get("2")
    if pewter:
        east = [
            p
            for p in pewter["points"]
            if p.get("kind") == "connection" and p.get("dir") == "east"
        ]
        print(f"  Pewter east connection cells: {len(east)} "
              f"(sample {east[:3]}...)")


if __name__ == "__main__":
    main()
