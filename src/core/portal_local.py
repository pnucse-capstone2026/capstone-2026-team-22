"""portal_local.py — Portal을 담는 LocalMap 확장."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .Map import LocalMap as BaseLocalMap
from .map_types import BlockType, Cell as BaseCell
from .portal_types import (
    DIR_INTERNAL,
    ROLE_ENTRY,
    ROLE_EXIT,
    Cell,
    Coord,
    Portal,
    PortalKey,
)


class LocalMap(BaseLocalMap):
    def __init__(self, map_id: int, map_name: str) -> None:
        super().__init__(map_id, map_name)
        self.portals: Dict[PortalKey, Portal] = {}
        self.portal_at: Dict[Coord, PortalKey] = {}
        self.exit_portal_at = self.portal_at
        self.entry_portal_at: Dict[Coord, Tuple[PortalKey, ...]] = {}

    @staticmethod
    def _upgrade_base_cell(cell: BaseCell) -> Cell:
        if isinstance(cell, Cell):
            return cell
        # 기본 Cell 의 필드를 **전부** 옮긴다.
        return Cell(
            cell_name=cell.cell_name,
            npc_name=getattr(cell, "npc_name", None),
            block_type=cell.block_type,
            warp_candidate=bool(getattr(cell, "warp_candidate", False)),
            entity=cell.entity,
            frontier=bool(getattr(cell, "frontier", False)),
            visited=cell.visited,
            visited_count=cell.visited_count,
        )

    def ensure_cell(self, x: int, y: int) -> Cell:
        """Portal 확장 Cell을 생성하고 Base Cell이면 안전하게 승격한다."""
        key = (int(x), int(y))
        cell = self.grid.get(key)
        if cell is None:
            cell = Cell()
            self.grid[key] = cell
        elif not isinstance(cell, Cell):
            cell = self._upgrade_base_cell(cell)
            self.grid[key] = cell
        return cell

    @staticmethod
    def _normalize_portal_coords(
        coords: List[Coord],
    ) -> Tuple[Coord, ...]:
        return tuple(sorted({(int(x), int(y)) for x, y in coords}))

    def register_portal(
        self,
        direction: str,
        warp_coords: List[Coord],
        label: str,
        *,
        role: str = ROLE_EXIT,
        kind: str = "observed",
        destination_map_id: Optional[int] = None,
    ) -> Portal:
        """EXIT 또는 ENTRY Portal을 등록한다."""
        normalized = self._normalize_portal_coords(warp_coords)
        if not normalized:
            raise ValueError("warp_coords must contain at least one coordinate")

        role = str(role).upper()
        if role not in (ROLE_EXIT, ROLE_ENTRY):
            raise ValueError(f"unsupported portal role: {role}")

        direction = str(direction).upper()
        key: PortalKey = (self.map_id, role, direction, normalized)
        portal = self.portals.get(key)

        if portal is None:
            portal = Portal(
                map_id=self.map_id,
                role=role,
                direction=direction,
                warp_coords=normalized,
                label=str(label),
                kind=str(kind),
                destination_map_id=(
                    None
                    if destination_map_id is None
                    else int(destination_map_id)
                ),
            )
            self.portals[key] = portal
        else:
            if label:
                portal.label = str(label)
            if destination_map_id is not None:
                portal.destination_map_id = int(destination_map_id)
            if role == ROLE_ENTRY and kind:
                if str(kind) == "arrival":
                    portal.kind = "arrival"
                elif portal.kind not in ("arrival",):
                    portal.kind = str(kind)
            elif portal.kind == "observed" and kind:
                portal.kind = str(kind)

        for coord in normalized:
            cell = self.ensure_cell(*coord)
            cell.block_type = BlockType.WARP
            if role == ROLE_EXIT:
                self.exit_portal_at[coord] = key
                cell.exit_portal_key = key
                cell.cell_name = str(label)
                positions = self.name_index.setdefault(str(label), [])
                if coord not in positions:
                    positions.append(coord)
            else:
                keys = set(self.entry_portal_at.get(coord, ()))
                keys.add(key)
                ordered = tuple(sorted(keys))
                self.entry_portal_at[coord] = ordered
                cell.entry_portal_keys = ordered

        return portal

    def register_exit_portal(
        self,
        direction: str,
        warp_coords: List[Coord],
        label: str,
        *,
        kind: str,
        destination_map_id: Optional[int],
    ) -> Portal:
        return self.register_portal(
            direction,
            warp_coords,
            label,
            role=ROLE_EXIT,
            kind=kind,
            destination_map_id=destination_map_id,
        )

    def register_entry_portal(
        self,
        direction: str,
        canonical_coords: List[Coord] | Tuple[Coord, ...] | Coord,
        label: str,
        *,
        source_map_id: int,
        arrival_coord: Optional[Coord] = None,
    ) -> Portal:
        """ENTRY를 reverse EXIT canonical 좌표로 등록한다."""
        if (
            isinstance(canonical_coords, tuple)
            and len(canonical_coords) == 2
            and all(isinstance(v, int) for v in canonical_coords)
        ):
            coords = [(int(canonical_coords[0]), int(canonical_coords[1]))]
        else:
            coords = [
                (int(x), int(y))
                for x, y in canonical_coords  # type: ignore[misc]
            ]

        if not coords:
            raise ValueError("canonical_coords must not be empty")

        actual = (
            (int(arrival_coord[0]), int(arrival_coord[1]))
            if arrival_coord is not None
            else coords[0]
        )
        portal = self.register_portal(
            direction,
            coords,
            label,
            role=ROLE_ENTRY,
            kind="arrival",
            destination_map_id=int(source_map_id),
        )
        self.record_portal_arrival(portal.key, actual)
        return portal

    def record_portal_access(
        self,
        key: PortalKey,
        coord: Coord,
    ) -> Optional[Portal]:
        portal = self.portals.get(key)
        if portal is None:
            return None
        access = {(int(x), int(y)) for x, y in portal.access_coords}
        access.add((int(coord[0]), int(coord[1])))
        portal.access_coords = tuple(sorted(access))
        return portal

    def record_portal_arrival(
        self,
        key: PortalKey,
        coord: Coord,
    ) -> Optional[Portal]:
        """실제 도착 좌표를 ENTRY 메타데이터에만 기록 (바닥을 WARP로 덮지 않음)."""
        portal = self.portals.get(key)
        if portal is None:
            return None

        normalized = (int(coord[0]), int(coord[1]))
        arrivals = {(int(x), int(y)) for x, y in portal.arrival_coords}
        arrivals.add(normalized)
        portal.arrival_coords = tuple(sorted(arrivals))
        self.record_portal_access(key, normalized)

        if portal.is_entry:
            keys = set(self.entry_portal_at.get(normalized, ()))
            keys.add(key)
            ordered = tuple(sorted(keys))
            self.entry_portal_at[normalized] = ordered
            cell = self.ensure_cell(*normalized)
            cell.entry_portal_keys = ordered

        return portal

    def set_cell_name(
        self,
        x: int,
        y: int,
        name: str,
        *,
        direction: str = DIR_INTERNAL,
        destination_map_id: Optional[int] = None,
        kind: str = "legacy",
    ) -> Portal:
        """호환용."""
        return self.register_exit_portal(
            direction,
            [(int(x), int(y))],
            name,
            kind=kind,
            destination_map_id=destination_map_id,
        )

    def get_portal(self, key: PortalKey) -> Optional[Portal]:
        return self.portals.get(key)

    def iter_portals(self) -> List[Portal]:
        return list(self.portals.values())
