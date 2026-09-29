"""Policy-facing map knowledge kept separate from the internal GlobalMap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from pokemonred_puffer.Map import (
    CELL_H,
    CELL_W,
    PLAYER_SCREEN_CELL_X,
    PLAYER_SCREEN_CELL_Y,
    BlockType,
    LocalMap,
)


@dataclass
class ObservationCell:
    """Map facts the policy has earned through observation or interaction."""

    observed: bool = False
    confirmed_blocked: bool = False
    confirmed_water: bool = False
    confirmed_warp: bool = False


class ObservationKnowledge:
    """Episode-local policy knowledge indexed by ``(map_id, x, y)``."""

    def __init__(self) -> None:
        self._cells: dict[tuple[int, int, int], ObservationCell] = {}
        self._initialized_maps: set[int] = set()

    def _ensure_cell(self, map_id: int, x: int, y: int) -> ObservationCell:
        key = (int(map_id), int(x), int(y))
        if key not in self._cells:
            self._cells[key] = ObservationCell()
        return self._cells[key]

    def get_cell(self, map_id: int, x: int, y: int) -> ObservationCell | None:
        return self._cells.get((int(map_id), int(x), int(y)))

    def observed_cells(self, map_id: int) -> Iterator[tuple[int, int]]:
        map_id = int(map_id)
        for (cell_map_id, x, y), cell in self._cells.items():
            if cell_map_id == map_id and cell.observed:
                yield x, y

    def observe_visible_map(
        self,
        map_id: int,
        player_x: int,
        player_y: int,
        local_map: LocalMap | None,
    ) -> int:
        """Record visible cells and return newly observed cells eligible for reward."""
        if local_map is None:
            return 0
        map_id = int(map_id)
        new_tiles = 0
        min_y = int(player_y) - PLAYER_SCREEN_CELL_Y
        min_x = int(player_x) - PLAYER_SCREEN_CELL_X
        for y in range(min_y, min_y + CELL_H):
            for x in range(min_x, min_x + CELL_W):
                if local_map.get_cell(x, y) is not None:
                    cell = self._ensure_cell(map_id, x, y)
                    if not cell.observed:
                        cell.observed = True
                        new_tiles += 1

        if map_id not in self._initialized_maps:
            self._initialized_maps.add(map_id)
            return 0
        return new_tiles

    def confirm_blocked(
        self,
        map_id: int,
        x: int,
        y: int,
        block_type: BlockType | None,
    ) -> None:
        """Record a structural obstacle after a confirmed collision."""
        cell = self._ensure_cell(map_id, x, y)
        cell.observed = True
        cell.confirmed_blocked = True
        cell.confirmed_water = block_type is BlockType.WATER

    def confirm_warp(
        self,
        old_map_id: int,
        old_pos: tuple[int, int],
        new_map_id: int,
        new_pos: tuple[int, int],
    ) -> None:
        """Record source and arrival cells after an observed map transition."""
        for map_id, (x, y) in (
            (old_map_id, old_pos),
            (new_map_id, new_pos),
        ):
            cell = self._ensure_cell(map_id, x, y)
            cell.observed = True
            cell.confirmed_warp = True
