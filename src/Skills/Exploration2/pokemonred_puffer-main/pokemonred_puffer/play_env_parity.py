"""play.py PyBoy 에 학습 env 전투 스킵 훅을 붙이는 어댑터.

설정은 config.yaml env.* 를 그대로 쓴다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from pokemonred_puffer.emulator_runtime import get_game_coords, read_m, symbol_address
from pokemonred_puffer.exploration_cheats import (
    TrainerBattleCheats,
    WildEncounterCheats,
    parse_disable_wild_encounters,
    register_battle_skip_hooks,
    resolve_cheat_profile,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "config.yaml"


def load_env_config(config_path: Path | str | None = None) -> DictConfig:
    path = Path(config_path) if config_path else _DEFAULT_CONFIG
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    cfg = OmegaConf.load(path)
    return cfg.env


class TrainCheatBridge:
    """play PyBoy 용 전투 스킵 세션."""

    def __init__(self, pyboy: Any, env_config: DictConfig) -> None:
        self.pyboy = pyboy
        self._symbol_addresses: dict[str, int] = {}
        self._hooks_registered = False

        self.disable_ai_actions = bool(env_config.get("disable_ai_actions", False))

        enabled, exception_maps = parse_disable_wild_encounters(env_config)
        self.disable_wild_encounters = enabled
        self.disable_wild_encounters_maps = exception_maps
        self.auto_win_trainer_battles = bool(env_config.get("auto_win_trainer_battles", False))
        self.wild_cheats = WildEncounterCheats(self)
        self.trainer_cheats = TrainerBattleCheats(self)

    def read_m(self, addr: str | int) -> int:
        return read_m(self, addr)

    def _symbol_address(self, symbol: str) -> int:
        return symbol_address(self, symbol)

    def get_game_coords(self) -> tuple[int, int, int]:
        return get_game_coords(self)

    def register_hooks(self) -> None:
        if self._hooks_registered:
            return
        register_battle_skip_hooks(self, self.pyboy)
        self._hooks_registered = True

    def apply_episode_start(self) -> None:
        """explore start — 전투 스킵만 재적용."""
        self.before_action()

    def before_action(self) -> None:
        self.wild_cheats.ensure_repel()

    def after_action(self) -> None:
        if self.auto_win_trainer_battles:
            self.trainer_cheats.force_finish_if_active()


def _env_config_for_profile(env_config: DictConfig, profile: str) -> DictConfig:
    """play 치트 프로필에 맞게 env 플래그를 덮어쓴다."""
    cfg = OmegaConf.create(OmegaConf.to_container(env_config, resolve=True))
    key = resolve_cheat_profile(profile)
    if key == "off":
        cfg.disable_wild_encounters = False
        cfg.auto_win_trainer_battles = False
    elif key == "wild":
        cfg.disable_wild_encounters = True
        cfg.auto_win_trainer_battles = False
    return cfg


def apply_train_env_cheats(
    pyboy: Any,
    env_config: DictConfig | None = None,
    *,
    config_path: Path | str | None = None,
    profile: str = "full",
) -> TrainCheatBridge:
    """훅 등록 후 브리지를 반환."""
    base = env_config if env_config is not None else load_env_config(config_path)
    cfg = _env_config_for_profile(base, profile)
    bridge = TrainCheatBridge(pyboy, cfg)
    bridge.register_hooks()
    bridge.before_action()
    return bridge
