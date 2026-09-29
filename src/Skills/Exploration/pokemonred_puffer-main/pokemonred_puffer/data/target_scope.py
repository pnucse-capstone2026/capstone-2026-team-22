"""Target-map scope helpers.

The episode's starting map is the Target. A cave-like start expands Target to
its warp-connected cave floors. Buildings reached from Target, including their
interior-only floor connections, are Connected maps. Everything else is Outside.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from enum import IntEnum
from functools import lru_cache
from pathlib import Path

from pokemonred_puffer.data.map_categories import is_cave_like_map, is_indoor_map


class TargetRegion(IntEnum):
    TARGET = 0
    CONNECTED = 1
    OUTSIDE = 2


_WARPS_PATH = Path(__file__).resolve().parent.parent / "warps.json"


def _is_indoor(map_id: int) -> bool:
    """Unknown/unused map IDs are never treated as a connected building."""
    try:
        return is_indoor_map(map_id)
    except ValueError:
        return False


@lru_cache(maxsize=1)
def _outgoing_warps() -> dict[int, frozenset[int]]:
    with _WARPS_PATH.open(encoding="utf-8") as file:
        data = json.load(file)

    outgoing: dict[int, set[int]] = {}
    for source_id, entry in data.get("maps", {}).items():
        try:
            source = int(source_id)
        except ValueError:
            continue

        for point in entry.get("points", []):
            destination = point.get("dest_id")
            if point.get("kind") != "warp" or destination is None:
                continue
            outgoing.setdefault(source, set()).add(int(destination))

    return {source: frozenset(destinations) for source, destinations in outgoing.items()}


def target_map_ids(target_map_id: int) -> frozenset[int]:
    """Return all map IDs that belong to the episode Target.

    Towns and routes use one map ID. Cave-like targets include every cave-like
    floor reachable through warps, stopping before exits and buildings.
    """
    target = int(target_map_id)
    if not _is_cave_like(target):
        return frozenset({target})

    outgoing = _outgoing_warps()
    reverse: dict[int, set[int]] = {}
    for source, destinations in outgoing.items():
        for destination in destinations:
            reverse.setdefault(destination, set()).add(source)

    target_maps = {target}
    pending = [target]
    while pending:
        source = pending.pop()
        neighbors = set(outgoing.get(source, ())) | reverse.get(source, set())
        for destination in neighbors:
            if _is_cave_like(destination) and destination not in target_maps:
                target_maps.add(destination)
                pending.append(destination)
    return frozenset(target_maps)


def _is_cave_like(map_id: int) -> bool:
    try:
        return is_cave_like_map(map_id)
    except ValueError:
        return False


def connected_map_ids(
    target_map_id: int,
    target_maps: Iterable[int] | frozenset[int] | None = None,
) -> frozenset[int]:
    """Return indoor maps belonging directly to the target map.

    A direct outgoing warp from any Target map may seed the Connected set only
    when its destination is indoors. The search then follows indoor-to-indoor
    warps (e.g. department-store floors), but never routes, towns, or exits.
    """
    outgoing = _outgoing_warps()
    target_map_id = int(target_map_id)
    target_set = frozenset(target_maps or target_map_ids(target_map_id))
    connected: set[int] = {
        destination
        for source in target_set
        for destination in outgoing.get(source, ())
        if _is_indoor(destination) and destination not in target_set
    }
    pending = list(connected)

    while pending:
        source = pending.pop()
        for destination in outgoing.get(source, ()):
            if _is_indoor(destination) and destination not in connected:
                connected.add(destination)
                pending.append(destination)

    return frozenset(connected)


def classify_target_region(
    map_n: int,
    target_map_id: int | None,
    connected: Iterable[int] | frozenset[int],
    target_maps: Iterable[int] | frozenset[int] | None = None,
) -> TargetRegion:
    """Classify ``map_n`` relative to a fixed episode target map."""
    if target_map_id is not None and (
        int(map_n) == int(target_map_id)
        or (target_maps is not None and int(map_n) in target_maps)
    ):
        return TargetRegion.TARGET
    if int(map_n) in connected:
        return TargetRegion.CONNECTED
    return TargetRegion.OUTSIDE
