"""탐험 학습용 전투 스킵.

RedGymEnv / play bridge는 훅 등록·step 위임만 하고, 실제 로직은 여기 클래스가 담당한다.
"""

from __future__ import annotations

from typing import Any

from omegaconf import ListConfig

from pokemonred_puffer.data.map import MapIds, map_display_name
from pokemonred_puffer.emulator_runtime import read_m, symbol_address

# pret/pokered wStatusFlags* bit indices.
_BIT_TALKED_TO_TRAINER = 6  # wStatusFlags3
_BIT_PRINT_END_BATTLE_TEXT = 7  # wStatusFlags3
_BIT_UNKNOWN_4_1 = 1  # wStatusFlags4
_BIT_TRAINER_BATTLE = 3  # wStatusFlags7


def parse_disable_wild_encounters(env_config: Any) -> tuple[bool, set[str]]:
    """Return ``(enabled, exception_map_names)`` from env config."""
    value = env_config.disable_wild_encounters
    if isinstance(value, bool):
        return bool(value), set()
    if isinstance(value, (list, ListConfig)):
        return len(value) > 0, {MapIds[item].name for item in value}
    raise ValueError("disable_wild_encounters must be a boolean or a list of MapIds")


def resolve_cheat_profile(
    cheat_profile: str = "full",
) -> str:
    """Normalize play cheat profile: ``off`` | ``wild`` | ``full``."""
    key = str(cheat_profile).strip().lower() or "full"
    if key in ("off", "none", "false", "0"):
        return "off"
    if key in ("wild", "wild_only", "no_wild"):
        return "wild"
    if key in ("full", "train", "all"):
        return "full"
    raise ValueError(f"Unknown cheat profile: {cheat_profile!r} (off|wild|full)")


def _clear_status_bit(pyboy: Any, symbol: str, bit: int) -> None:
    addr = pyboy.symbol_lookup(symbol)[1]
    pyboy.memory[addr] = int(pyboy.memory[addr]) & ~(1 << bit)


class WildEncounterCheats:
    """야생 조우 OFF (리펠 + TryDoWildEncounter 훅)."""

    def __init__(self, env: Any) -> None:
        self.env = env

    def _enabled_here(self) -> bool:
        env = self.env
        if not env.disable_wild_encounters:
            return False
        return map_display_name(read_m(env, "wCurMap")) not in env.disable_wild_encounters_maps

    def _exception_map(self) -> bool:
        env = self.env
        return (
            env.disable_wild_encounters
            and map_display_name(read_m(env, "wCurMap")) in env.disable_wild_encounters_maps
        )

    def register(self, pyboy: Any) -> None:
        pyboy.hook_register(None, "SetLastBlackoutMap.done", self.on_blackout, None)
        if not self.env.disable_wild_encounters:
            return
        bank, addr = pyboy.symbol_lookup("TryDoWildEncounter.gotWildEncounterType")
        pyboy.hook_register(bank, addr + 8, self.on_wild_attempt, None)

    def on_blackout(self, *args: Any, **kwargs: Any) -> None:
        env = self.env
        if self._exception_map():
            env.pyboy.memory[symbol_address(env, "wRepelRemainingSteps")] = 0x01

    def on_wild_attempt(self, *args: Any, **kwargs: Any) -> None:
        if not self._enabled_here():
            return
        env = self.env
        env.pyboy.memory[symbol_address(env, "wRepelRemainingSteps")] = 0xFF
        env.pyboy.memory[symbol_address(env, "wCurEnemyLevel")] = 0x01

    def ensure_repel(self) -> None:
        """step/before_action: keep repel active outside exception maps."""
        if not self._enabled_here():
            return
        env = self.env
        env.pyboy.memory[symbol_address(env, "wRepelRemainingSteps")] = 0xFF


class TrainerBattleCheats:
    """맵 트레이너 전투 자동 승리 / 강제 종료."""

    def __init__(self, env: Any) -> None:
        self.env = env

    def register(self, pyboy: Any) -> None:
        if not self.env.auto_win_trainer_battles:
            return
        pyboy.hook_register(None, "StartTrainerBattle", self.on_start_trainer_battle, None)

    def on_start_trainer_battle(self, *args: Any, **kwargs: Any) -> None:
        """StartTrainerBattle 진입 시 EndTrainerBattle로 리다이렉트."""
        if not self.env.auto_win_trainer_battles:
            return

        env = self.env
        engaged = read_m(env, "wEngagedTrainerClass")
        env.pyboy.memory[symbol_address(env, "wEnemyMonOrTrainerClass")] = engaged
        # 오버월드 NewBattle이 싸움을 시작하지 못하게.
        env.pyboy.memory[symbol_address(env, "wCurOpponent")] = 0
        # EndTrainerBattle은 wIsInBattle==$ff를 패배로 봄 → 0으로 두면 승리 경로.
        env.pyboy.memory[symbol_address(env, "wIsInBattle")] = 0
        env.pyboy.memory[symbol_address(env, "wBattleResult")] = 0

        _clear_status_bit(env.pyboy, "wStatusFlags4", _BIT_UNKNOWN_4_1)
        _clear_status_bit(env.pyboy, "wStatusFlags7", _BIT_TRAINER_BATTLE)
        _clear_status_bit(env.pyboy, "wStatusFlags3", _BIT_TALKED_TO_TRAINER)
        _clear_status_bit(env.pyboy, "wStatusFlags3", _BIT_PRINT_END_BATTLE_TEXT)

        env.pyboy.register_file.PC = env.pyboy.symbol_lookup("EndTrainerBattle")[1]

    def force_finish_if_active(self) -> None:
        """스크립트 트레이너 전투 안전망: 적 HP를 0으로."""
        env = self.env
        if read_m(env, "wIsInBattle") <= 1:
            return
        env.pyboy.memory[symbol_address(env, "wBattleResult")] = 0
        enemy_count = min(6, max(0, read_m(env, "wEnemyPartyCount")))
        for i in range(enemy_count):
            hp_addr = symbol_address(env, f"wEnemyMon{i+1}HP")
            env.pyboy.memory[hp_addr] = 0
            env.pyboy.memory[hp_addr + 1] = 0
        battle_hp = symbol_address(env, "wEnemyMonHP")
        env.pyboy.memory[battle_hp] = 0
        env.pyboy.memory[battle_hp + 1] = 0


def register_battle_skip_hooks(env: Any, pyboy: Any) -> None:
    """Register wild + trainer battle-skip hooks on ``pyboy``."""
    if not hasattr(env, "wild_cheats"):
        env.wild_cheats = WildEncounterCheats(env)
    if not hasattr(env, "trainer_cheats"):
        env.trainer_cheats = TrainerBattleCheats(env)
    env.wild_cheats.register(pyboy)
    env.trainer_cheats.register(pyboy)
