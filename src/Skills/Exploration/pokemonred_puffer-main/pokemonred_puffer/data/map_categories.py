"""탐험 보상/obs용 맵 카테고리 헬퍼."""

from __future__ import annotations

from enum import Enum, auto

from pokemonred_puffer.data.map import MapIds


class MapCategory(Enum):
    HUB = auto()
    TRANSIT = auto()
    CAVE_LIKE = auto()
    INTERIOR = auto()
    SKIP = auto()


# 이름 규칙보다 명시 오버라이드가 우선.
_CATEGORY_OVERRIDES: dict[MapIds, MapCategory] = {
    MapIds.INDIGO_PLATEAU: MapCategory.HUB,
    MapIds.INDIGO_PLATEAU_LOBBY: MapCategory.HUB,
    MapIds.CINNABAR_ISLAND: MapCategory.HUB,
    MapIds.VERMILION_DOCK: MapCategory.TRANSIT,
    MapIds.VIRIDIAN_FOREST: MapCategory.CAVE_LIKE,
    MapIds.POWER_PLANT: MapCategory.CAVE_LIKE,
    MapIds.DIGLETTS_CAVE: MapCategory.CAVE_LIKE,
    MapIds.DIGLETTS_CAVE_ROUTE_2: MapCategory.CAVE_LIKE,
    MapIds.DIGLETTS_CAVE_ROUTE_11: MapCategory.CAVE_LIKE,
    MapIds.UNDERGROUND_PATH_NORTH_SOUTH: MapCategory.CAVE_LIKE,
    MapIds.UNDERGROUND_PATH_WEST_EAST: MapCategory.CAVE_LIKE,
    MapIds.UNDERGROUND_PATH_ROUTE_5: MapCategory.CAVE_LIKE,
    MapIds.UNDERGROUND_PATH_ROUTE_6: MapCategory.CAVE_LIKE,
    MapIds.UNDERGROUND_PATH_ROUTE_6_COPY: MapCategory.SKIP,
    MapIds.UNDERGROUND_PATH_ROUTE_7: MapCategory.CAVE_LIKE,
    MapIds.UNDERGROUND_PATH_ROUTE_7_COPY: MapCategory.SKIP,
    MapIds.UNDERGROUND_PATH_ROUTE_8: MapCategory.CAVE_LIKE,
    MapIds.SAFARI_ZONE_EAST: MapCategory.TRANSIT,
    MapIds.SAFARI_ZONE_NORTH: MapCategory.TRANSIT,
    MapIds.SAFARI_ZONE_WEST: MapCategory.TRANSIT,
    MapIds.SAFARI_ZONE_CENTER: MapCategory.TRANSIT,
    MapIds.SAFARI_ZONE_GATE: MapCategory.INTERIOR,
    MapIds.SAFARI_ZONE_SECRET_HOUSE: MapCategory.INTERIOR,
    MapIds.SAFARI_ZONE_CENTER_REST_HOUSE: MapCategory.INTERIOR,
    MapIds.SAFARI_ZONE_WEST_REST_HOUSE: MapCategory.INTERIOR,
    MapIds.SAFARI_ZONE_EAST_REST_HOUSE: MapCategory.INTERIOR,
    MapIds.SAFARI_ZONE_NORTH_REST_HOUSE: MapCategory.INTERIOR,
    MapIds.FIGHTING_DOJO: MapCategory.INTERIOR,
    MapIds.HALL_OF_FAME: MapCategory.INTERIOR,
    MapIds.LANCES_ROOM: MapCategory.CAVE_LIKE,
    MapIds.LORELEIS_ROOM: MapCategory.CAVE_LIKE,
    MapIds.BRUNOS_ROOM: MapCategory.CAVE_LIKE,
    MapIds.AGATHAS_ROOM: MapCategory.CAVE_LIKE,
    MapIds.CHAMPIONS_ROOM: MapCategory.CAVE_LIKE,
    MapIds.TRADE_CENTER: MapCategory.SKIP,
    MapIds.COLOSSEUM: MapCategory.SKIP,
    MapIds.CERULEAN_TRASHED_HOUSE_COPY: MapCategory.SKIP,
    MapIds.CINNABAR_MART_COPY: MapCategory.SKIP,
}

_CAVE_LIKE_TOKENS = (
    "CAVE",
    "MT_MOON",
    "ROCK_TUNNEL",
    "SEAFOAM",
    "VICTORY_ROAD",
    "POKEMON_TOWER",
    "POKEMON_MANSION",
    "ROCKET_HIDEOUT",
    "SILPH_CO",
    "CERULEAN_CAVE",
    "SS_ANNE",
)


def _as_map_id(map_id: int | MapIds) -> MapIds:
    if isinstance(map_id, MapIds):
        return map_id
    return MapIds(map_id)


def is_pokecenter_or_gym(map_id: int | MapIds) -> bool:
    name = _as_map_id(map_id).name
    return name.endswith("_POKECENTER") or name.endswith("_GYM")


def is_indoor_map(map_id: int | MapIds) -> bool:
    """new_tile용 건물: INTERIOR + 센터/체육관 (HUB이지만 야외 아님)."""
    if categorize_map(map_id) == MapCategory.INTERIOR:
        return True
    return is_pokecenter_or_gym(map_id)


def is_cave_like_map(map_id: int | MapIds) -> bool:
    """Return whether a map is one floor of a cave-like area."""
    return categorize_map(map_id) is MapCategory.CAVE_LIKE


def is_route_gate(map_id: int | MapIds) -> bool:
    """루트·숲 연결 게이트. Connected 진입 보상 대상에서 제외.

    ``ROUTE_*_GATE*`` 와 ``VIRIDIAN_FOREST_*_GATE`` 는 통로용이다.
    ``SAFARI_ZONE_GATE`` 는 마을 건물(사파리 입구)이라 제외하지 않는다.
    """
    name = _as_map_id(map_id).name
    if name == "SAFARI_ZONE_GATE":
        return False
    return "_GATE" in name


def categorize_map(map_id: int | MapIds) -> MapCategory:
    mid = _as_map_id(map_id)
    if mid in _CATEGORY_OVERRIDES:
        return _CATEGORY_OVERRIDES[mid]

    name = mid.name
    if name.startswith("UNUSED_"):
        return MapCategory.SKIP

    if name.endswith("_POKECENTER") or name.endswith("_GYM"):
        return MapCategory.HUB
    if name.endswith("_TOWN") or name.endswith("_CITY"):
        return MapCategory.HUB

    if name.startswith("ROUTE_"):
        # ROUTE_N_GATE / HOUSE 등은 INTERIOR, 순수 ROUTE_N만 TRANSIT.
        rest = name[len("ROUTE_") :]
        if rest.isdigit() or (len(rest) <= 2 and rest[:1].isdigit()):
            return MapCategory.TRANSIT
        return MapCategory.INTERIOR

    for token in _CAVE_LIKE_TOKENS:
        if token in name:
            if name.endswith("_ELEVATOR"):
                return MapCategory.INTERIOR
            return MapCategory.CAVE_LIKE

    return MapCategory.INTERIOR
