"""portal_types.py — Portal 그래프용 타입·상수."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .gb_memory import FACING_DOWN, FACING_LEFT, FACING_RIGHT, FACING_UP
from .map_types import BlockType, Cell as BaseCell, EntityType

Coord = Tuple[int, int]

ARRIVAL_STABLE_FRAMES = 6
ARRIVAL_CONFIRM_TIMEOUT = 180

DIR_NORTH = "NORTH"
DIR_SOUTH = "SOUTH"
DIR_WEST = "WEST"
DIR_EAST = "EAST"
DIR_INTERNAL = "INTERNAL"

FACING_TO_DIRECTION: Dict[int, str] = {
    FACING_DOWN: DIR_SOUTH,
    FACING_UP: DIR_NORTH,
    FACING_LEFT: DIR_WEST,
    FACING_RIGHT: DIR_EAST,
}

OPPOSITE_DIRECTION: Dict[str, str] = {
    DIR_NORTH: DIR_SOUTH,
    DIR_SOUTH: DIR_NORTH,
    DIR_WEST: DIR_EAST,
    DIR_EAST: DIR_WEST,
    DIR_INTERNAL: DIR_INTERNAL,
}

ROLE_EXIT = "EXIT"
ROLE_ENTRY = "ENTRY"

TRANSITION_OBSERVED = "observed"
TRANSITION_INFERRED_REVERSE = "inferred_reverse"

# (map_id, role, direction, coords) EXIT과 ENTRY가 같은 좌표여도 role로 다른 노드가 된다.
PortalKey = Tuple[int, str, str, Tuple[Coord, ...]]


@dataclass
class Portal:
    """PortalGraph의 한 노드."""

    map_id: int
    role: str
    direction: str
    warp_coords: Tuple[Coord, ...]
    label: str
    arrival_coords: Tuple[Coord, ...] = ()
    access_coords: Tuple[Coord, ...] = ()
    kind: str = "observed"
    destination_map_id: Optional[int] = None
    destination_portal: Optional[PortalKey] = None

    @property
    def key(self) -> PortalKey:
        return (
            int(self.map_id),
            str(self.role).upper(),
            str(self.direction).upper(),
            tuple(self.warp_coords),
        )

    @property
    def is_exit(self) -> bool:
        return str(self.role).upper() == ROLE_EXIT

    @property
    def is_entry(self) -> bool:
        return str(self.role).upper() == ROLE_ENTRY


@dataclass(frozen=True)
class ObservedTransition:
    """실제 MapID 변경으로 확인된 EXIT -> ENTRY 전환."""

    source_map_id: int
    source_exit_coord: Coord
    destination_map_id: int
    destination_arrival_coord: Coord
    source_portal: PortalKey
    destination_portal: PortalKey
    source_access_coord: Optional[Coord] = None
    reverse_confirmed: bool = False


@dataclass(frozen=True)
class ReverseExitGroup:
    """목적지→출발 reverse EXIT 묶음 (JSON canonical + arrival 후보)."""

    map_id: int
    raw_coords: Tuple[Coord, ...]
    direction: str
    kind: str
    destination_map_id: int
    arrival_candidates: Tuple[Coord, ...]
    source: str


@dataclass
class Cell(BaseCell):
    """기본 Cell + EXIT/ENTRY Portal 키."""

    exit_portal_key: Optional[PortalKey] = None
    entry_portal_keys: Tuple[PortalKey, ...] = ()


# re-export for convenience
__all__ = [
    "ARRIVAL_CONFIRM_TIMEOUT",
    "ARRIVAL_STABLE_FRAMES",
    "BlockType",
    "Cell",
    "Coord",
    "DIR_EAST",
    "DIR_INTERNAL",
    "DIR_NORTH",
    "DIR_SOUTH",
    "DIR_WEST",
    "EntityType",
    "FACING_TO_DIRECTION",
    "OPPOSITE_DIRECTION",
    "ObservedTransition",
    "Portal",
    "PortalKey",
    "ROLE_ENTRY",
    "ROLE_EXIT",
    "ReverseExitGroup",
    "TRANSITION_INFERRED_REVERSE",
    "TRANSITION_OBSERVED",
]
