"""세션 누적 GlobalMap을 Kanto 기준 이미지 위에 색으로 합성한다."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Tuple

from PIL import Image, ImageDraw

from core.map_types import BlockType

ROOT = Path(__file__).resolve().parents[1]
BACKGROUND_PATH = ROOT / "runtime" / "map" / "kanto_map_dsv.png"
OUTPUT_DIR = ROOT / "runtime" / "playphoto"
MAP_DATA_PATH = ROOT / "data" / "map_data.json"

# map_data 좌표 한 칸은 기준 PNG의 16×16 픽셀이다.
CELL_PIXELS = 16
OVERLAY_ALPHA = 184  # 약 72%
VISITED_RGBA = (0, 255, 0, OVERLAY_ALPHA)
OBSERVED_RGBA = (255, 255, 0, OVERLAY_ALPHA)
WALL_RGBA = (255, 0, 0, OVERLAY_ALPHA)

_NUMBERED_PNG = re.compile(r"^(\d+)\.png$", re.IGNORECASE)


def _map_placements() -> Dict[int, Tuple[int, int, int, int]]:
    with MAP_DATA_PATH.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    result: Dict[int, Tuple[int, int, int, int]] = {}
    for region in raw.get("regions", []):
        try:
            map_id = int(region["id"])
            x, y = map(int, region["coordinates"])
            width, height = map(int, region["tileSize"])
        except (KeyError, TypeError, ValueError):
            continue
        if map_id >= 0:
            result[map_id] = (x, y, width, height)
    return result


def _blend_cell(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    global_x: int,
    global_y: int,
    color: Tuple[int, int, int, int],
) -> None:
    x0 = int(global_x) * CELL_PIXELS
    y0 = int(global_y) * CELL_PIXELS
    x1 = min(x0 + CELL_PIXELS, image.width)
    y1 = min(y0 + CELL_PIXELS, image.height)
    if x0 < 0 or y0 < 0 or x0 >= x1 or y0 >= y1:
        return
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=color)


def render_play_photo(gmap: Any, output_path: Path) -> Dict[str, int]:
    """현재 세션의 관측/벽/방문 셀을 합성하고 각 셀 수를 반환한다."""
    if not BACKGROUND_PATH.is_file():
        raise FileNotFoundError(f"기준 지도 이미지가 없습니다: {BACKGROUND_PATH}")

    with Image.open(BACKGROUND_PATH) as source:
        image = source.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    placements = _map_placements()
    counts = {"visited": 0, "observed": 0, "wall": 0, "unplaced": 0}

    for map_id in getattr(gmap, "graph", {}).nodes:
        map_id = int(map_id)
        placement = placements.get(map_id)
        local = gmap.get_local_map(map_id)
        if placement is None or local is None:
            counts["unplaced"] += len(getattr(local, "grid", {}) or {})
            continue
        base_x, base_y, map_width, map_height = placement
        for (x, y), cell in local.grid.items():
            x, y = int(x), int(y)
            # 화면 가장자리 스캔으로 생긴 맵 밖 좌표는 이 맵 그림에 칠하지 않는다.
            if not (0 <= x < map_width and 0 <= y < map_height):
                continue
            if bool(getattr(cell, "visited", False)):
                color = VISITED_RGBA
                counts["visited"] += 1
            elif getattr(cell, "block_type", None) is BlockType.WALL:
                color = WALL_RGBA
                counts["wall"] += 1
            else:
                color = OBSERVED_RGBA
                counts["observed"] += 1
            _blend_cell(overlay, draw, base_x + x, base_y + y, color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(image, overlay).save(output_path)
    return counts


def save_next_play_photo(gmap: Any) -> Tuple[Path, Dict[str, int]]:
    """`runtime/playphoto/<다음 번호>.png`로 누적 지도 스냅샷을 저장한다."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    numbers = []
    for path in OUTPUT_DIR.iterdir():
        match = _NUMBERED_PNG.match(path.name) if path.is_file() else None
        if match:
            numbers.append(int(match.group(1)))
    output = OUTPUT_DIR / f"{max(numbers, default=0) + 1}.png"
    return output, render_play_photo(gmap, output)
