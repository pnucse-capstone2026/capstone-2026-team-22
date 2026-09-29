"""Shared action-space semantics used by observations and exploration rewards.

Single source of truth for discrete agent actions (no A/B). Index order here
defines policy logits, DIRECTION_DELTA, and PyBoy press/release mapping.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ActionSpec:
    name: str
    # Local-map delta (dx, dy) when the move succeeds.
    delta: tuple[int, int]


# Discrete actions: down, left, right, up (no A/B).
ACTIONS: tuple[ActionSpec, ...] = (
    ActionSpec("DOWN", (0, 1)),
    ActionSpec("LEFT", (-1, 0)),
    ActionSpec("RIGHT", (1, 0)),
    ActionSpec("UP", (0, -1)),
)

ACTION_NAMES: tuple[str, ...] = tuple(spec.name for spec in ACTIONS)
DIRECTION_DELTA: dict[int, tuple[int, int]] = {
    index: spec.delta for index, spec in enumerate(ACTIONS)
}
