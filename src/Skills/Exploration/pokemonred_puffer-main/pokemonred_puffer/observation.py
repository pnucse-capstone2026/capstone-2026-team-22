"""Observation construction helpers for the RedGymEnv façade."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from pokemonred_puffer.data.map_categories import categorize_map

PIXEL_VALUES = np.array([0, 85, 153, 255], dtype=np.uint8)
_SCREEN_CELL_WIDTH = 10
_SCREEN_CELL_HEIGHT = 9
_PIXELS_PER_CELL = 16


def observation_space(screen_shape):
    return spaces.Dict(
        {
            "screen": spaces.Box(low=0, high=255, shape=screen_shape, dtype=np.uint8),
            "visited_mask": spaces.Box(low=0, high=255, shape=screen_shape, dtype=np.uint8),
            "map_id": spaces.Box(low=0, high=0xFF, shape=(1,), dtype=np.uint8),
            # 0=Target map, 1=Connected map, 2=Outside map.
            "map_role": spaces.Box(low=0, high=2, shape=(1,), dtype=np.uint8),
            "direction": spaces.Box(low=0, high=3, shape=(1,), dtype=np.uint8),
            "x": spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8),
            "y": spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8),
            "moved": spaces.Box(low=0, high=2, shape=(1,), dtype=np.uint8),
            "map_category": spaces.Box(low=0, high=4, shape=(1,), dtype=np.uint8),
        }
    )


def render(env) -> npt.NDArray[np.uint8]:
    return env.screen.ndarray[:, :, 1]


def screen_obs(env):
    game_pixels_render = np.expand_dims(env.screen.ndarray[:, :, 1], axis=-1)
    visited_mask = _visited_mask(env)
    if env.reduce_res:
        game_pixels_render = game_pixels_render[::2, ::2, :]
        visited_mask = visited_mask[::2, ::2, :]
    if env.two_bit:
        game_pixels_render = _pack_two_bit(game_pixels_render)
        visited_mask = _pack_two_bit(visited_mask)
    return {"screen": game_pixels_render, "visited_mask": visited_mask}


def _pack_two_bit(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    return (
        (
            np.digitize(image.reshape((-1, 4)), PIXEL_VALUES, right=True).astype(np.uint8)
            << np.array([6, 4, 2, 0], dtype=np.uint8)
        )
        .sum(axis=1, dtype=np.uint8)
        .reshape((-1, image.shape[1] // 4, 1))
    )


def _visited_mask(env) -> npt.NDArray[np.uint8]:
    """Render observation-only episode visits in the same pixel geometry as the screen."""
    output = np.zeros((_SCREEN_CELL_HEIGHT, _SCREEN_CELL_WIDTH), dtype=np.uint8)
    if env.read_m("wIsInBattle") != 0:
        return np.zeros((144, 160, 1), dtype=np.uint8)

    player_x, player_y, map_n = env.get_game_coords()
    visited_tiles = env.visited_mask_tiles.get(int(map_n), set())
    for row, y in enumerate(range(player_y - 4, player_y + 5)):
        for column, x in enumerate(range(player_x - 4, player_x + 6)):
            output[row, column] = 255 if (x, y) in visited_tiles else 0
    # Keep full 144×160 geometry so visited_mask matches screen through reduce_res/two_bit.
    return np.repeat(
        np.repeat(output, _PIXELS_PER_CELL, axis=0), _PIXELS_PER_CELL, axis=1
    )[:, :, np.newaxis]


def refresh_moved(env) -> None:
    x, y, map_n = env.get_game_coords()
    if env._prev_coords is None:
        env._moved = 0
    else:
        px, py, pm = env._prev_coords
        env._moved = 2 if map_n != pm else 1 if (x != px or y != py) else 0
    env._prev_coords = (x, y, map_n)


def record_visited_mask_tile(env, player_x: int, player_y: int, map_n: int) -> None:
    """Record policy-only visit history; it intentionally does not update LocalMap."""
    if env.read_m("wIsInBattle") == 0:
        env.visited_mask_tiles.setdefault(int(map_n), set()).add((int(player_x), int(player_y)))


def get_obs(env):
    refresh_moved(env)
    player_x, player_y, map_n = env.get_game_coords()
    record_visited_mask_tile(env, player_x, player_y, map_n)
    target_region = getattr(env, "target_region", None)
    if callable(target_region):
        map_role = int(target_region(map_n))
    else:
        # RedGymEnv test stubs do not use ExplorationReward.
        target = getattr(env, "target_map_id", None)
        map_role = 0 if target is not None and map_n == target else 2
    # Call through the façade so unit-test stubs and downstream subclasses can
    # override screen construction without providing a PyBoy screen object.
    return env.screen_obs() | {
        "map_id": np.array([map_n], dtype=np.uint8),
        "map_role": np.array([map_role], dtype=np.uint8),
        "direction": np.array(
            [env.read_m("wSpritePlayerStateData1FacingDirection") // 4], dtype=np.uint8
        ),
        "x": np.array([player_x], dtype=np.uint8),
        "y": np.array([player_y], dtype=np.uint8),
        "moved": np.array([env._moved], dtype=np.uint8),
        "map_category": np.array([categorize_map(map_n).value - 1], dtype=np.uint8),
    }
