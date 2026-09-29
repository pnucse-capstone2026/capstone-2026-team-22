"""Observation construction helpers for the RedGymEnv facade."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from pokemonred_puffer.Map import (
    CELL_H,
    CELL_W,
    PLAYER_SCREEN_CELL_X,
    PLAYER_SCREEN_CELL_Y,
    SCREEN_TILE_H,
    SCREEN_TILE_W,
    BlockType,
    EntityType,
)
from pokemonred_puffer.actions import DIRECTION_DELTA
from pokemonred_puffer.observation_knowledge import ObservationKnowledge

PIXEL_VALUES = np.array([0, 85, 153, 255], dtype=np.uint8)

TRACKING_HEIGHT = SCREEN_TILE_H
TRACKING_WIDTH = SCREEN_TILE_W
TRACKING_CH_OBSERVED = 0
TRACKING_CH_BLOCKED = 1
TRACKING_CH_WATER = 2
TRACKING_CH_WARP = 3
TRACKING_CH_NPC = 4
TRACKING_CH_OBJECT = 5
TRACKING_CH_VISITED = 6
TRACKING_CHANNELS = 7
_TRACKING_CELL_HEIGHT = SCREEN_TILE_H // CELL_H
_TRACKING_CELL_WIDTH = SCREEN_TILE_W // CELL_W

STILL = 0
MOVED = 1
COLLISION = 2
MAP_TRANSITION = 3
NO_PREVIOUS_ACTION = 4

_STRUCTURAL_BLOCKS = {
    BlockType.WALL,
    BlockType.WATER,
    BlockType.LEDGE_U,
    BlockType.LEDGE_D,
    BlockType.LEDGE_L,
    BlockType.LEDGE_R,
}

SCALAR_OBS_KEYS = ("map_scope", "direction", "last_action", "movement_result")


def observation_space(screen_shape):
    return spaces.Dict(
        {
            "screen": spaces.Box(low=0, high=255, shape=screen_shape, dtype=np.uint8),
            "tracking_mask": spaces.Box(
                low=0,
                high=255,
                shape=(TRACKING_HEIGHT, TRACKING_WIDTH, TRACKING_CHANNELS),
                dtype=np.uint8,
            ),
            # 0=Target, 1=Connected/attached, 2=Outside/external.
            "map_scope": spaces.Box(low=0, high=2, shape=(1,), dtype=np.uint8),
            "direction": spaces.Box(low=0, high=3, shape=(1,), dtype=np.uint8),
            "last_action": spaces.Box(low=0, high=4, shape=(1,), dtype=np.uint8),
            "movement_result": spaces.Box(low=0, high=3, shape=(1,), dtype=np.uint8),
        }
    )


def render(env) -> npt.NDArray[np.uint8]:
    return env.screen.ndarray[:, :, 1]


def screen_obs(env):
    screen = np.expand_dims(env.screen.ndarray[:, :, 1], axis=-1)
    if env.reduce_res:
        screen = screen[::2, ::2, :]
    if env.two_bit:
        screen = _pack_two_bit(screen)
    return {"screen": screen}


def _pack_two_bit(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    return (
        (
            np.digitize(image.reshape((-1, 4)), PIXEL_VALUES, right=True).astype(np.uint8)
            << np.array([6, 4, 2, 0], dtype=np.uint8)
        )
        .sum(axis=1, dtype=np.uint8)
        .reshape((-1, image.shape[1] // 4, 1))
    )


def reset_observation_state(env) -> None:
    env.observation_knowledge = ObservationKnowledge()
    env.new_observed_tiles = 0
    reset_observation_temporal_state(env)


def reset_observation_temporal_state(env) -> None:
    """Reset action history without discarding accumulated map knowledge."""
    env._last_action = NO_PREVIOUS_ACTION
    env._movement_result = STILL
    env._collision_target: tuple[int, int, int] | None = None
    env._collision_count = 0


def _tracking_mask(env, player_x: int, player_y: int, map_n: int) -> npt.NDArray[np.uint8]:
    output = np.zeros((TRACKING_HEIGHT, TRACKING_WIDTH, TRACKING_CHANNELS), dtype=np.uint8)
    if env.read_m("wIsInBattle") != 0:
        return output

    local = env.map_state.get_local_map(int(map_n))
    if local is None:
        return output

    for cell_row in range(CELL_H):
        y = int(player_y) + cell_row - PLAYER_SCREEN_CELL_Y
        row = slice(
            cell_row * _TRACKING_CELL_HEIGHT,
            (cell_row + 1) * _TRACKING_CELL_HEIGHT,
        )
        for cell_column in range(CELL_W):
            x = int(player_x) + cell_column - PLAYER_SCREEN_CELL_X
            column = slice(
                cell_column * _TRACKING_CELL_WIDTH,
                (cell_column + 1) * _TRACKING_CELL_WIDTH,
            )
            cell = local.get_cell(x, y)
            if cell is None:
                continue
            knowledge = env.observation_knowledge.get_cell(int(map_n), x, y)
            output[row, column, TRACKING_CH_OBSERVED] = (
                255 if knowledge is not None and knowledge.observed else 0
            )
            output[row, column, TRACKING_CH_BLOCKED] = (
                255 if knowledge is not None and knowledge.confirmed_blocked else 0
            )
            output[row, column, TRACKING_CH_WATER] = (
                255 if knowledge is not None and knowledge.confirmed_water else 0
            )
            output[row, column, TRACKING_CH_WARP] = (
                255 if knowledge is not None and knowledge.confirmed_warp else 0
            )
            entity = cell.entity
            output[row, column, TRACKING_CH_NPC] = 255 if entity is EntityType.NPC else 0
            output[row, column, TRACKING_CH_OBJECT] = (
                255
                if entity
                in (
                    EntityType.OBJECT,
                    EntityType.CUT_TREE,
                )
                else 0
            )
            output[row, column, TRACKING_CH_VISITED] = 255 if cell.visited else 0
    return output


def _reset_collision_candidate(env) -> None:
    env._collision_target = None
    env._collision_count = 0


def record_action_result(
    env,
    action: int,
    pre_coords: tuple[int, int, int],
    post_coords: tuple[int, int, int],
) -> None:
    """Record direct movement experience for the next policy observation."""
    action = int(action)
    env._last_action = action
    pre_x, pre_y, pre_map = map(int, pre_coords)
    post_x, post_y, post_map = map(int, post_coords)

    if pre_map != post_map:
        dx, dy = DIRECTION_DELTA[action]
        env.observation_knowledge.confirm_warp(
            pre_map,
            (pre_x + dx, pre_y + dy),
            post_map,
            (post_x, post_y),
        )
        env._movement_result = MAP_TRANSITION
        _reset_collision_candidate(env)
        return

    if (pre_x, pre_y) != (post_x, post_y):
        env._movement_result = MOVED
        _reset_collision_candidate(env)
        return

    env._movement_result = STILL
    if env.read_m("wIsInBattle") != 0 or env.read_m("wJoyIgnore") != 0:
        _reset_collision_candidate(env)
        return

    dx, dy = DIRECTION_DELTA[action]
    target = (pre_x + dx, pre_y + dy, pre_map)
    if target == env._collision_target:
        env._collision_count += 1
    else:
        env._collision_target = target
        env._collision_count = 1
    if env._collision_count < 2:
        return

    local = env.map_state.get_local_map(pre_map)
    cell = None if local is None else local.get_cell(target[0], target[1])
    if cell is not None and cell.entity in (EntityType.NPC, EntityType.OBJECT):
        env._movement_result = COLLISION
        return
    if cell is None or cell.block_type in _STRUCTURAL_BLOCKS:
        block_type = None if cell is None else cell.block_type
        env.observation_knowledge.confirm_blocked(
            pre_map,
            target[0],
            target[1],
            block_type,
        )
        env._movement_result = COLLISION


def _map_scope(env, map_n: int) -> int:
    classifier = getattr(env, "map_scope", None)
    if callable(classifier):
        return int(classifier(map_n))
    target = getattr(env, "target_map_id", None)
    return 0 if target is not None and int(map_n) == int(target) else 2


def get_obs(env):
    player_x, player_y, map_n = env.get_game_coords()
    return env.screen_obs() | {
        "tracking_mask": _tracking_mask(env, player_x, player_y, map_n),
        "map_scope": np.array([_map_scope(env, map_n)], dtype=np.uint8),
        "direction": np.array(
            [env.read_m("wSpritePlayerStateData1FacingDirection") // 4], dtype=np.uint8
        ),
        "last_action": np.array([env._last_action], dtype=np.uint8),
        "movement_result": np.array([env._movement_result], dtype=np.uint8),
    }
