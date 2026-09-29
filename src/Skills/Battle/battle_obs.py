"""
battle_obs_3.py

Pokemon Red(PyBoy) Battle RAM -> 현재 PPO가 학습한 Compact Gen-1
276차원 observation 변환기.

학습 env(Battle_RL_r2의 gen1-hierarchical-safe-finisher-v8)와 같은 순서
------------------------------------------------
[0:28]    global
[28:132]  own moves      4 * 26
[132:156] revealed opp moves 4 * 6
[156:228] own party      6 * 12
[228:276] revealed opp party 6 * 8

중요
----
- species id / move id 자체는 observation에 넣지 않는다.
  id는 species.json / moves.json lookup 용도로만 사용한다.
- opponent party / move는 학습환경과 맞추기 위해 "공개된 것만" 넣는다.
- action mask는 observation과 별도다.
- RL native action:
      0..5 = switch party 0..5
      6..9 = move 0..3
  BattleMacro native action:
      0..3 = move 0..3
      4..9 = switch party 0..5
  따라서 rl_action_to_macro()로 변환한다.

필요 파일
---------
프로젝트 루트 기준:
    data/species.json
    data/moves.json

BattleRAMReader는 gb_memory.py의 현재 객체를 그대로 넘기면 된다.

사용 예
-------
    battle_ram = BattleRAMReader(pyboy)
    battle_obs = BattleObs(battle_ram)

    battle_obs.reset_battle()

    # RL 결정 시점: 학습 정책이 받은 Dict observation과 동일한 형태
    model_obs = battle_obs.model_input()

    # model_obs:
    # {
    #     "observation": np.ndarray(shape=(276,), dtype=float32),
    #     "action_mask": np.ndarray(shape=(10,), dtype=int64),
    # }

    # model -> rl_action (0..9)
    macro_action = rl_action_to_macro(rl_action)
    battle_macro.execute_action(macro_action)

    # 다음 메인 배틀 메뉴까지 macro가 진행한 뒤
    battle_obs.finish_turn()

주의
----
moves.json의 effect는 poke-env의 Move 객체보다 정보가 압축되어 있다.
따라서 move candidate의
  debuff / self-boost / status / volatile / side-condition / heal
feature는 Gen-1 effect 이름을 이용해 가능한 한 동일하게 복원한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


# ===========================================================================
# Observation layout
# ===========================================================================

GEN1_BOOST_DIM = 6

SWITCH_CONTEXT_FEATURE_NAMES = (
    "current_matchup_score",
    "best_switch_matchup_score",
    "best_switch_gain_ratio",
    "matchup_flip_available",
    "severe_debuff_escape_available",
    "switch_need",
)
SWITCH_CONTEXT_DIM = len(SWITCH_CONTEXT_FEATURE_NAMES)
SWITCH_CONTEXT_START_INDEX = 10 + GEN1_BOOST_DIM * 2
SWITCH_NEED_GLOBAL_INDEX = (
    SWITCH_CONTEXT_START_INDEX
    + SWITCH_CONTEXT_FEATURE_NAMES.index("switch_need")
)
GLOBAL_DIM = SWITCH_CONTEXT_START_INDEX + SWITCH_CONTEXT_DIM
MOVE_FEATURE_NAMES = (
    "power_norm",
    "accuracy",
    "pp_ratio",
    "effectiveness_norm",
    "category",
    "priority",
    "stab",
    "opponent_debuff",
    "self_boost",
    "major_status",
    "volatile_status",
    "side_condition",
    "heal",
    "persistent_effect",
    "persistent_effect_active",
    "attack_quality_ratio",
    "best_attack",
    "attack_score_norm",
    "effective_accuracy",
    "min_damage_ratio",
    "ko_on_hit",
    "one_turn_move",
    "acts_first",
    "safe_immediate_ko",
    "status_utility",
    "debuff_remaining_capacity",
)
PERSISTENT_EFFECT_INDEX = MOVE_FEATURE_NAMES.index("persistent_effect")
PERSISTENT_EFFECT_ACTIVE_INDEX = MOVE_FEATURE_NAMES.index(
    "persistent_effect_active"
)
OPPONENT_DEBUFF_INDEX = MOVE_FEATURE_NAMES.index("opponent_debuff")
ATTACK_QUALITY_RATIO_INDEX = MOVE_FEATURE_NAMES.index("attack_quality_ratio")
BEST_ATTACK_INDEX = MOVE_FEATURE_NAMES.index("best_attack")
ATTACK_SCORE_NORM_INDEX = MOVE_FEATURE_NAMES.index("attack_score_norm")
EFFECTIVENESS_INDEX = MOVE_FEATURE_NAMES.index("effectiveness_norm")
EFFECTIVE_ACCURACY_INDEX = MOVE_FEATURE_NAMES.index("effective_accuracy")
MIN_DAMAGE_RATIO_INDEX = MOVE_FEATURE_NAMES.index("min_damage_ratio")
KO_ON_HIT_INDEX = MOVE_FEATURE_NAMES.index("ko_on_hit")
ONE_TURN_MOVE_INDEX = MOVE_FEATURE_NAMES.index("one_turn_move")
ACTS_FIRST_INDEX = MOVE_FEATURE_NAMES.index("acts_first")
SAFE_IMMEDIATE_KO_INDEX = MOVE_FEATURE_NAMES.index("safe_immediate_ko")
STATUS_UTILITY_INDEX = MOVE_FEATURE_NAMES.index("status_utility")
MOVE_CATEGORY_INDEX = MOVE_FEATURE_NAMES.index("category")
DEBUFF_REMAINING_CAPACITY_INDEX = MOVE_FEATURE_NAMES.index(
    "debuff_remaining_capacity"
)

MOVE_SLOT_DIM = len(MOVE_FEATURE_NAMES)
MOVE_DIM = MOVE_SLOT_DIM * 4

OPP_MOVE_SLOT_DIM = 6
OPP_MOVE_DIM = OPP_MOVE_SLOT_DIM * 4

OWN_TEAM_FEATURE_NAMES = (
    "active",
    "fainted",
    "hp",
    "level",
    "status",
    "offensive_score",
    "incoming_threat",
    "speed_diff",
    "matchup_score",
    "matchup_gain",
    "switch_quality_ratio",
    "best_switch",
)
OWN_TEAM_MATCHUP_SCORE_INDEX = OWN_TEAM_FEATURE_NAMES.index("matchup_score")
OWN_TEAM_MATCHUP_GAIN_INDEX = OWN_TEAM_FEATURE_NAMES.index("matchup_gain")
OWN_TEAM_SWITCH_QUALITY_INDEX = OWN_TEAM_FEATURE_NAMES.index("switch_quality_ratio")
OWN_TEAM_BEST_SWITCH_INDEX = OWN_TEAM_FEATURE_NAMES.index("best_switch")
OWN_TEAM_SLOT_DIM = len(OWN_TEAM_FEATURE_NAMES)
OWN_TEAM_DIM = OWN_TEAM_SLOT_DIM * 6

OPP_TEAM_SLOT_DIM = 8
OPP_TEAM_DIM = OPP_TEAM_SLOT_DIM * 6

N_FEATURES = (
    GLOBAL_DIM
    + MOVE_DIM
    + OPP_MOVE_DIM
    + OWN_TEAM_DIM
    + OPP_TEAM_DIM
)

assert N_FEATURES == 276

OBSERVATION_LAYOUT_VERSION = "gen1-hierarchical-safe-finisher-v8"

FEATURE_SLICES = {
    "global": slice(0, 28),
    "own_moves": slice(28, 132),
    "opponent_moves": slice(132, 156),
    "own_team": slice(156, 228),
    "opponent_team": slice(228, 276),
}

SWITCH_SEVERE_DEBUFF_STAGE = -2
SWITCH_CLEAR_MATCHUP_GAIN = 0.100
SWITCH_FLIP_MIN_GAIN = 0.050
SWITCH_DEBUFF_ESCAPE_MAX_MATCHUP_LOSS = 0.000
SWITCH_FAVORABLE_SCORE_MARGIN = 0.0
SWITCH_LOW_HP_RELIEF_THRESHOLD = 0.250


# ===========================================================================
# Gen-1 party struct
# ===========================================================================

# pret/pokered PARTYMON_STRUCT_LENGTH
PARTYMON_STRUCT_LENGTH = 0x2C  # 44 bytes

# party_struct offsets
MON_SPECIES = 0
MON_HP = 1                   # 2-byte big endian
MON_STATUS = 4
MON_MOVES = 8                # 4 bytes
MON_PP = 29                  # 4 bytes
MON_LEVEL = 33
MON_MAX_HP = 34              # 2-byte big endian

MAX_PARTY = 6


# ===========================================================================
# Gen-1 battle status bits used by the observation/mask
# ===========================================================================

# w*BattleStatus1
BIT_STORING_ENERGY = 0
BIT_THRASHING_ABOUT = 1
BIT_ATTACKING_MULTIPLE_TIMES = 2
BIT_CHARGING_UP = 4
BIT_USING_TRAPPING_MOVE = 5
BIT_CONFUSED = 7

# w*BattleStatus2
BIT_PROTECTED_BY_MIST = 1
BIT_GETTING_PUMPED = 2
BIT_HAS_SUBSTITUTE_UP = 4
BIT_NEEDS_TO_RECHARGE = 5
BIT_USING_RAGE = 6
BIT_SEEDED = 7

# w*BattleStatus3
BIT_BADLY_POISONED = 0
BIT_HAS_LIGHT_SCREEN_UP = 1
BIT_HAS_REFLECT_UP = 2


# ===========================================================================
# Type chart
# ===========================================================================

# Gen-1 Red type chart.
# Omitted combinations are neutral (1.0).
# Ghost -> Psychic is intentionally 0.0 in Gen 1.
_TYPE_ROWS = """
WATER FIRE 2
FIRE GRASS 2
FIRE ICE 2
GRASS WATER 2
ELECTRIC WATER 2
WATER ROCK 2
GROUND FLYING 0
WATER WATER 0.5
FIRE FIRE 0.5
ELECTRIC ELECTRIC 0.5
ICE ICE 0.5
GRASS GRASS 0.5
PSYCHIC PSYCHIC 0.5
FIRE WATER 0.5
GRASS FIRE 0.5
WATER GRASS 0.5
ELECTRIC GRASS 0.5
NORMAL ROCK 0.5
NORMAL GHOST 0
GHOST GHOST 2
FIRE BUG 2
FIRE ROCK 0.5
WATER GROUND 2
ELECTRIC GROUND 0
ELECTRIC FLYING 2
GRASS GROUND 2
GRASS BUG 0.5
GRASS POISON 0.5
GRASS ROCK 2
GRASS FLYING 0.5
ICE WATER 0.5
ICE GRASS 2
ICE GROUND 2
ICE FLYING 2
ICE DRAGON 2
FIGHTING NORMAL 2
FIGHTING POISON 0.5
FIGHTING FLYING 0.5
FIGHTING PSYCHIC 0.5
FIGHTING BUG 0.5
FIGHTING ROCK 2
FIGHTING ICE 2
FIGHTING GHOST 0
POISON GRASS 2
POISON POISON 0.5
POISON GROUND 0.5
POISON BUG 2
POISON ROCK 0.5
POISON GHOST 0.5
GROUND FIRE 2
GROUND ELECTRIC 2
GROUND GRASS 0.5
GROUND BUG 0.5
GROUND ROCK 2
GROUND POISON 2
FLYING ELECTRIC 0.5
FLYING FIGHTING 2
FLYING BUG 2
FLYING GRASS 2
FLYING ROCK 0.5
PSYCHIC FIGHTING 2
PSYCHIC POISON 2
BUG FIRE 0.5
BUG GRASS 2
BUG FIGHTING 0.5
BUG FLYING 0.5
BUG PSYCHIC 2
BUG GHOST 0.5
BUG POISON 2
ROCK FIRE 2
ROCK FIGHTING 0.5
ROCK GROUND 0.5
ROCK FLYING 2
ROCK BUG 2
ROCK ICE 2
GHOST NORMAL 0
GHOST PSYCHIC 0
FIRE DRAGON 0.5
WATER DRAGON 0.5
ELECTRIC DRAGON 0.5
GRASS DRAGON 0.5
DRAGON DRAGON 2
""".strip()

TYPE_CHART: dict[tuple[str, str], float] = {}
for _line in _TYPE_ROWS.splitlines():
    _atk, _def, _mult = _line.split()
    TYPE_CHART[(_atk, _def)] = float(_mult)


# ===========================================================================
# Effect -> compact move semantics
# ===========================================================================

# 상대 stage 하락량.
_OPP_DEBUFF_STAGE = {
    "ATTACK_DOWN1_EFFECT": 1,
    "ATTACK_DOWN_SIDE_EFFECT": 1,
    "DEFENSE_DOWN1_EFFECT": 1,
    "DEFENSE_DOWN2_EFFECT": 2,
    "DEFENSE_DOWN_SIDE_EFFECT": 1,
    "SPEED_DOWN1_EFFECT": 1,
    "SPEED_DOWN_SIDE_EFFECT": 1,
    "SPECIAL_DOWN_SIDE_EFFECT": 1,
    "ACCURACY_DOWN1_EFFECT": 1,
}

# 자기 stage 상승량.
_SELF_BOOST_STAGE = {
    "ATTACK_UP1_EFFECT": 1,
    "ATTACK_UP2_EFFECT": 2,
    "DEFENSE_UP1_EFFECT": 1,
    "DEFENSE_UP2_EFFECT": 2,
    "SPEED_UP2_EFFECT": 2,
    "SPECIAL_UP1_EFFECT": 1,
    "SPECIAL_UP2_EFFECT": 2,
    "EVASION_UP1_EFFECT": 1,
}

_MAJOR_STATUS_EFFECTS = {
    "BURN_SIDE_EFFECT1",
    "BURN_SIDE_EFFECT2",
    "FREEZE_SIDE_EFFECT1",
    "PARALYZE_EFFECT",
    "PARALYZE_SIDE_EFFECT1",
    "PARALYZE_SIDE_EFFECT2",
    "POISON_EFFECT",
    "POISON_SIDE_EFFECT1",
    "POISON_SIDE_EFFECT2",
    "SLEEP_EFFECT",
    "TWINEEDLE_EFFECT",
}

_VOLATILE_EFFECTS = {
    "CONFUSION_EFFECT",
    "CONFUSION_SIDE_EFFECT",
    "TRAPPING_EFFECT",
    "LEECH_SEED_EFFECT",
    "DISABLE_EFFECT",
    "FOCUS_ENERGY_EFFECT",
    "SUBSTITUTE_EFFECT",
    "RAGE_EFFECT",
    "BIDE_EFFECT",
}

_SIDE_CONDITION_EFFECTS = {
    "REFLECT_EFFECT",
    "LIGHT_SCREEN_EFFECT",
    "MIST_EFFECT",
}

# Showdown/poke-env에서 basePower 대신 damage로 표현되는 계열.
_FIXED_DAMAGE_EFFECTS = {
    "SPECIAL_DAMAGE_EFFECT",  # Dragon Rage / SonicBoom / Seismic Toss etc.
    "SUPER_FANG_EFFECT",
    "OHKO_EFFECT",
}

# Battle_RL_r2 env.py와 같은 순수 디버프 선택 상한.
OPPONENT_DEBUFF_MASK_STAGE = -2

# BattleMon stage 순서: atk, def, special, spe, accuracy, evasion.
_DEBUFF_STAT_INDEX = {
    "ATTACK_DOWN1_EFFECT": 0,
    "DEFENSE_DOWN1_EFFECT": 1,
    "DEFENSE_DOWN2_EFFECT": 1,
    "SPEED_DOWN1_EFFECT": 3,
    "ACCURACY_DOWN1_EFFECT": 4,
}

# 한 번 적용된 동안에는 다시 쌓이지 않는 Gen-1 status move.
_PERSISTENT_EFFECT_KIND = {
    "SLEEP_EFFECT": "major_status",
    "PARALYZE_EFFECT": "major_status",
    "POISON_EFFECT": "major_status",
    "CONFUSION_EFFECT": "confusion",
    "LEECH_SEED_EFFECT": "leech_seed",
    "DISABLE_EFFECT": "disable",
    "SUBSTITUTE_EFFECT": "substitute",
    "FOCUS_ENERGY_EFFECT": "focus_energy",
    "REFLECT_EFFECT": "reflect",
    "LIGHT_SCREEN_EFFECT": "light_screen",
    "MIST_EFFECT": "mist",
}


class BattleObsError(RuntimeError):
    pass


# ===========================================================================
# Lightweight RAM snapshots
# ===========================================================================

@dataclass
class PartyMonSnapshot:
    slot: int
    species_id: int
    level: int
    hp: int
    max_hp: int
    status_raw: int
    move_ids: tuple[int, int, int, int]
    pp_raw: tuple[int, int, int, int]
    active: bool = False
    toxic: bool = False

    @property
    def fainted(self) -> bool:
        return self.hp <= 0

    @property
    def hp_ratio(self) -> float:
        if self.max_hp <= 0:
            return 0.0
        return _clip01(self.hp / self.max_hp)


@dataclass
class RevealedEnemy:
    key: tuple[int, int]        # (enemy_party_index, species_id)
    reveal_order: int

    party_index: int
    species_id: int
    level: int
    hp: int
    max_hp: int
    status_raw: int

    fainted: bool = False
    active: bool = False
    toxic: bool = False

    # reveal 순서를 보존한다.
    move_ids: list[int] = field(default_factory=list)

    @property
    def hp_ratio(self) -> float:
        if self.max_hp <= 0:
            return 0.0
        return _clip01(self.hp / self.max_hp)


# ===========================================================================
# Numeric helpers
# ===========================================================================

def _finite(value, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _clip01(value) -> float:
    return float(np.clip(_finite(value), 0.0, 1.0))


def _clip11(value) -> float:
    return float(np.clip(_finite(value), -1.0, 1.0))


def _u16be(memory, address: int) -> int:
    hi = int(memory[address]) & 0xFF
    lo = int(memory[address + 1]) & 0xFF
    return (hi << 8) | lo


def _bytes(memory, address: int, count: int) -> tuple[int, ...]:
    return tuple(
        int(memory[address + i]) & 0xFF
        for i in range(count)
    )


# ===========================================================================
# Public action mapping
# ===========================================================================

def rl_action_to_macro(action: int) -> int:
    """
    현재 PPO action -> BattleMacro action.

    PPO:
      0..5 switch0..5
      6..9 move0..3

    BattleMacro:
      0..3 move0..3
      4..9 switch0..5
    """
    action = int(action)

    if 0 <= action <= 5:
        return action + 4

    if 6 <= action <= 9:
        return action - 6

    raise ValueError(f"RL action must be 0..9, got {action}")


# ===========================================================================
# Main encoder
# ===========================================================================

class BattleObs:
    """
    BattleRAMReader + PyBoy memory -> PPO 276-feature observation.
    """

    def __init__(
        self,
        battle_ram,
        *,
        species_path: str | Path | None = None,
        moves_path: str | Path | None = None,
        debug: bool = False,
    ):
        self.battle_ram = battle_ram
        self.pyboy = battle_ram.pyboy
        self.sym = battle_ram.sym
        self.debug = bool(debug)

        root = Path(__file__).resolve().parent
        data_dir = root / "data"

        if species_path is None:
            species_path = data_dir / "species.json"
        if moves_path is None:
            moves_path = data_dir / "moves.json"

        self.species_path = Path(species_path)
        self.moves_path = Path(moves_path)

        self.species_db = self._load_json_dict(self.species_path)
        self.moves_db = self._load_json_dict(self.moves_path)

        # Showdown battle.turn에 대응하는 emulator-side decision turn counter.
        self.turn = 1

        # 상대 공개 정보 tracker.
        self._enemy_by_key: dict[tuple[int, int], RevealedEnemy] = {}
        self._enemy_reveal_order: list[tuple[int, int]] = []

        # 현재 RL 결정 시점의 상대. 다음 finish_turn()에서 enemy_used_move를
        # 이 포켓몬의 공개 move로 기록한다.
        self._turn_enemy_key: tuple[int, int] | None = None

        self._last_obs: np.ndarray | None = None

        self._warned_missing_symbols: set[str] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset_battle(self) -> None:
        self.turn = 1
        self._enemy_by_key.clear()
        self._enemy_reveal_order.clear()
        self._turn_enemy_key = None
        self._last_obs = None

    def set_turn(self, turn: int) -> None:
        self.turn = max(0, int(turn))

    def finish_turn(self, state=None) -> None:
        """
        BattleMacro가 한 action을 처리하고 다음 decision menu까지 온 뒤 호출.

        - 직전 decision 시점 상대에게 enemy_used_move를 공개 move로 기록
        - 이미 공개된 상대 party 상태 갱신
        - 새 active 상대를 공개
        - turn += 1

        wEnemyUsedMove는 lazy RAM이므로 build()에서 무작정 기록하지 않고,
        action boundary인 finish_turn()에서만 사용한다.
        """
        if state is None:
            state = self.battle_ram.read()

        # 직전 턴에 행동한 상대의 move를 직전 active에게 귀속.
        move_id = int(getattr(state, "enemy_used_move", 0) or 0)

        if move_id != 0 and self._turn_enemy_key is not None:
            enemy = self._enemy_by_key.get(self._turn_enemy_key)

            if enemy is not None and move_id not in enemy.move_ids:
                enemy.move_ids.append(move_id)
                enemy.move_ids[:] = enemy.move_ids[:4]

                if self.debug:
                    print(
                        "[battle_obs] revealed enemy move "
                        f"party={enemy.party_index} "
                        f"species={enemy.species_id} move={move_id}"
                    )

        # 이미 공개된 party slot만 hidden party RAM을 이용해 상태 갱신.
        # unrevealed slot을 새로 observation에 노출하지 않는다.
        self._refresh_revealed_enemy_party()

        # 교체/기절 후 새 active가 있다면 이 시점에 공개.
        self._observe_enemy_active(state)

        self.turn += 1

    def record_enemy_move(
        self,
        move_id: int,
        *,
        party_index: int | None = None,
        species_id: int | None = None,
    ) -> None:
        """
        별도 battle log에서 상대가 사용한 기술을 정확히 잡았을 때 직접 기록 가능.
        """
        move_id = int(move_id)

        if move_id <= 0:
            return

        key = None

        if party_index is not None and species_id is not None:
            candidate = (int(party_index), int(species_id))
            if candidate in self._enemy_by_key:
                key = candidate

        if key is None:
            key = self._turn_enemy_key

        if key is None:
            return

        enemy = self._enemy_by_key.get(key)

        if enemy is None:
            return

        if move_id not in enemy.move_ids:
            enemy.move_ids.append(move_id)
            enemy.move_ids[:] = enemy.move_ids[:4]

    # ------------------------------------------------------------------
    # Public outputs
    # ------------------------------------------------------------------

    def build(self, state=None) -> np.ndarray:
        """
        현재 decision state -> (276,) np.float32.
        """
        if state is None:
            state = self.battle_ram.read()

        if not bool(getattr(state, "in_battle", False)):
            raise BattleObsError("battle_obs.build(): 현재 전투 중이 아닙니다.")

        our_active = getattr(state, "player", None)
        opp_active = getattr(state, "enemy", None)

        if our_active is None or opp_active is None:
            raise BattleObsError(
                "battle_obs.build(): active Battle RAM이 아직 준비되지 않았습니다."
            )

        # 현재 상대 active는 이 decision 시점에서 이미 공개된 정보.
        enemy_entry = self._observe_enemy_active(state)
        self._turn_enemy_key = enemy_entry.key if enemy_entry is not None else None

        # player party는 action 0..5와 정확히 맞춰야 하므로 RAM slot order 유지.
        own_team = self._read_player_party(state)

        # 이미 공개된 상대 party 상태만 갱신.
        self._refresh_revealed_enemy_party()

        player_status1 = self._state_ram_u8(
            state,
            "player_battle_status1",
            "wPlayerBattleStatus1",
        )
        player_status2 = self._state_ram_u8(
            state,
            "player_battle_status2",
            "wPlayerBattleStatus2",
        )
        player_status3 = self._state_ram_u8(
            state,
            "player_battle_status3",
            "wPlayerBattleStatus3",
        )

        enemy_status1 = self._state_ram_u8(
            state,
            "enemy_battle_status1",
            "wEnemyBattleStatus1",
        )
        enemy_status3 = self._state_ram_u8(
            state,
            "enemy_battle_status3",
            "wEnemyBattleStatus3",
        )

        trapped = bool(enemy_status1 & (1 << BIT_USING_TRAPPING_MOVE))
        forced_switch = bool(getattr(our_active, "fainted", False))
        must_recharge = bool(player_status2 & (1 << BIT_NEEDS_TO_RECHARGE))
        preparing = bool(player_status1 & (1 << BIT_CHARGING_UP))

        own_fainted = sum(mon.fainted for mon in own_team)
        opponent_fainted = sum(
            enemy.fainted
            for enemy in self._revealed_enemy_list()
        )

        revealed_moves: list[int] = []
        if enemy_entry is not None:
            revealed_moves = list(enemy_entry.move_ids[:4])

        switch_context = self._switch_decision_context(
            our_active=our_active,
            opponent_active=opp_active,
            own_team=own_team,
            opponent_known_moves=revealed_moves,
            # A trapping flag can remain set while the active Pokemon is
            # fainted.  Showdown exposes the living bench as available_switches
            # for this request, so forced replacement must ignore trapping.
            trapped=(trapped and not forced_switch),
        )

        features: list[float] = [
            # 0..9 global scalars
            _clip01(own_fainted / 6.0),
            _clip01(opponent_fainted / 6.0),
            _clip01(self.turn / 100.0),
            float(trapped),
            self._hp_fraction_active(our_active),
            self._hp_fraction_active(opp_active),
            _clip01(int(getattr(our_active, "level", 0)) / 100.0),
            _clip01(int(getattr(opp_active, "level", 0)) / 100.0),
            float(must_recharge),
            float(preparing),

            # 10..15 own boosts
            *self._boost_vector(our_active),

            # 16..21 opponent boosts
            *self._boost_vector(opp_active),

            # 22..27 tactical switch-family context
            _clip11(switch_context["current_score"]),
            _clip11(switch_context["best_switch_score"]),
            _clip11(
                switch_context["best_switch_gain"]
                / max(SWITCH_CLEAR_MATCHUP_GAIN, 1.0e-9)
            ),
            float(switch_context["matchup_flip"]),
            float(switch_context["debuff_escape_available"]),
            _clip01(switch_context["switch_need"]),
        ]

        # --------------------------------------------------------------
        # Own active moves: 4 * 26
        # --------------------------------------------------------------
        own_move_ids = list(getattr(our_active, "move_ids", ()) or ())[:4]
        own_pp_raw = list(getattr(our_active, "pp_raw", ()) or ())[:4]

        own_move_ids += [0] * (4 - len(own_move_ids))
        own_pp_raw += [0] * (4 - len(own_pp_raw))

        disabled_slot = self._player_disabled_slot(state)
        legal_move_ids = {
            int(move_id)
            for slot, (move_id, pp_raw) in enumerate(
                zip(own_move_ids, own_pp_raw)
            )
            if (
                int(move_id) > 0
                and self._current_pp(pp_raw) > 0
                and slot != disabled_slot
            )
        }
        best_legal_attack_score = max(
            (
                self._attack_choice_score(
                    move_id,
                    defender_species=int(
                        getattr(opp_active, "species_id", 0)
                    ),
                )
                for move_id in legal_move_ids
            ),
            default=0.0,
        )

        for i in range(4):
            features.extend(
                self._own_move_features(
                    move_id=own_move_ids[i],
                    pp_raw=own_pp_raw[i],
                    attacker_species=int(getattr(our_active, "species_id", 0)),
                    defender_species=int(getattr(opp_active, "species_id", 0)),
                    attacker=our_active,
                    opponent=opp_active,
                    state=state,
                    legal_move_ids=legal_move_ids,
                    best_legal_attack_score=best_legal_attack_score,
                )
            )

        # --------------------------------------------------------------
        # Opponent active revealed moves: 4 * 6
        # --------------------------------------------------------------
        for i in range(4):
            move_id = revealed_moves[i] if i < len(revealed_moves) else 0
            features.extend(
                self._opponent_move_features(
                    move_id,
                    our_species=int(getattr(our_active, "species_id", 0)),
                )
            )

        # --------------------------------------------------------------
        # Own party: 6 * 12
        # --------------------------------------------------------------
        own_team_by_slot = {
            int(mon.slot): mon
            for mon in own_team
            if 0 <= int(mon.slot) < 6
        }

        for i in range(6):
            mon = own_team_by_slot.get(i)

            features.extend(
                self._own_party_features(
                    mon,
                    opponent_species=int(getattr(opp_active, "species_id", 0)),
                    opponent_known_moves=(
                        revealed_moves
                        if enemy_entry is not None
                        else []
                    ),
                    current_matchup_score=switch_context["current_score"],
                    best_legal_switch_gain=switch_context["best_switch_gain"],
                    legal_switch_slots=switch_context["legal_switch_slots"],
                )
            )

        # --------------------------------------------------------------
        # Revealed opponent party: 6 * 8
        # --------------------------------------------------------------
        revealed_team = self._revealed_enemy_list()

        for i in range(6):
            mon = revealed_team[i] if i < len(revealed_team) else None

            features.extend(
                self._opponent_party_features(
                    mon,
                    our_species=int(getattr(our_active, "species_id", 0)),
                )
            )

        obs = np.asarray(features, dtype=np.float32)

        if obs.shape != (N_FEATURES,):
            raise BattleObsError(
                f"Observation-size bug: got {obs.shape}, "
                f"expected {(N_FEATURES,)}"
            )

        obs = np.nan_to_num(
            obs,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        obs = np.clip(obs, -1.0, 1.0).astype(
            np.float32,
            copy=False,
        )

        self._last_obs = obs

        if self.debug:
            print(
                "[battle_obs] built "
                f"shape={obs.shape} dtype={obs.dtype} "
                f"turn={self.turn} "
                f"own_fainted={own_fainted} "
                f"opp_fainted={opponent_fainted} "
                f"revealed_opp={len(revealed_team)} "
                f"revealed_moves={revealed_moves}"
            )

        return obs

    def action_mask(self, state=None) -> np.ndarray:
        """
        RL native action mask, shape=(10,).

        index:
          0..5 switch party0..5
          6..9 move0..3

        - switch: slot 존재, HP>0, 현재 active 아님, trapping 아님
        - move: move id 존재 + current PP > 0
        """
        if state is None:
            state = self.battle_ram.read()

        if not bool(getattr(state, "in_battle", False)):
            return np.zeros(10, dtype=np.int64)

        player = getattr(state, "player", None)
        if player is None:
            return np.zeros(10, dtype=np.int64)

        mask = np.zeros(10, dtype=np.int64)

        enemy_status1 = self._state_ram_u8(
            state,
            "enemy_battle_status1",
            "wEnemyBattleStatus1",
        )
        trapped = bool(enemy_status1 & (1 << BIT_USING_TRAPPING_MOVE))

        own_team = self._read_player_party(state)
        active_index = int(getattr(state, "player_party_index", 0))

        forced_switch = bool(getattr(player, "fainted", False))

        if forced_switch or not trapped:
            for mon in own_team:
                i = int(mon.slot)
                if (
                    0 <= i < 6
                    and mon.species_id != 0
                    and not mon.fainted
                    and i != active_index
                ):
                    mask[i] = 1.0

        # A fainted active Pokemon creates the same request as poke-env's
        # forceSwitch: only living bench slots are legal.  The separate Dict
        # action_mask tells the hierarchical policy that SWITCH is the sole
        # valid family; the 276 observation layout stays unchanged.
        if forced_switch:
            return mask

        move_ids = list(getattr(player, "move_ids", ()) or ())[:4]
        pp_raw = list(getattr(player, "pp_raw", ()) or ())[:4]

        move_ids += [0] * (4 - len(move_ids))
        pp_raw += [0] * (4 - len(pp_raw))

        # Gen 1 wPlayerDisabledMove packs the disabled slot (1..4) in the
        # high nibble and the remaining duration in the low nibble.  Disable
        # is not a major status byte, so it must be represented through the
        # action mask just like poke-env removes a disabled move from
        # available_moves.  This keeps the trained 276-feature layout intact.
        disabled_slot = self._player_disabled_slot(state)

        for i in range(4):
            if (
                move_ids[i] != 0
                and self._current_pp(pp_raw[i]) > 0
                and i != disabled_slot
            ):
                mask[6 + i] = 1.0

        base_mask = mask.copy()
        defender_species = int(
            getattr(getattr(state, "enemy", None), "species_id", 0)
        )

        # Battle_RL_r2 RedBattleEnv.get_action_mask()의 의미 기반 필터.
        for i, move_id in enumerate(move_ids):
            action_index = 6 + i
            if mask[action_index] <= 0.5:
                continue

            move = self._move(move_id)
            if not move:
                continue

            if (
                self._persistent_effect_active(move, state)
                or self._known_immune_to_target(move, defender_species)
                or (
                    not self._is_status_move(move)
                    and self._effectiveness_raw(move_id, defender_species) <= 1.0e-9
                )
                or self._debuff_policy_cap_reached(
                    move,
                    getattr(state, "enemy", None),
                )
            ):
                mask[action_index] = 0.0

        # forced/degenerate request에서 유효 행동을 전부 없애지는 않는다.
        if not np.any(mask) and np.any(base_mask):
            return base_mask

        return mask

    def _player_disabled_slot(self, state) -> int | None:
        """Return the zero-based disabled move slot encoded by Gen-1 RAM."""
        disabled_raw = self._state_ram_u8(
            state,
            "player_disabled_move",
            "wPlayerDisabledMove",
        )
        slot = ((disabled_raw >> 4) & 0x0F) - 1
        return slot if disabled_raw != 0 and 0 <= slot < 4 else None

    def decision(self, state=None) -> tuple[np.ndarray, np.ndarray]:
        """
        한 번의 decision input을 같이 얻는다.
        """
        if state is None:
            state = self.battle_ram.read()

        obs = self.build(state)
        mask = self.action_mask(state)

        if not np.any(mask):
            raise BattleObsError(
                "action mask가 전부 0입니다. "
                "forced move/Struggle 상태인지 별도 확인이 필요합니다."
            )

        return obs, mask

    @staticmethod
    def pack_model_input(
        observation: np.ndarray,
        action_mask: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        BattleObs의 두 출력값을 PPO가 학습 때 받은 Dict observation으로 포장한다.

        학습 policy:
            obs["observation"] -> 276 feature vector
            obs["action_mask"] -> 10 action legality mask

        반환 shape:
            {
                "observation": (276,) float32,
                "action_mask": (10,) int64,
            }

        주의:
        여기서는 batch 차원을 붙이지 않는다.
        stable-baselines3의 policy.obs_to_tensor()가 단일 observation을
        자동으로 batch 형태로 바꾼다.
        """
        observation = np.asarray(
            observation,
            dtype=np.float32,
        ).reshape(-1)

        action_mask = np.asarray(
            action_mask,
            dtype=np.int64,
        ).reshape(-1)

        if observation.shape != (N_FEATURES,):
            raise BattleObsError(
                "model observation shape mismatch: "
                f"got={observation.shape}, expected={(N_FEATURES,)}"
            )

        if action_mask.shape != (10,):
            raise BattleObsError(
                "action mask shape mismatch: "
                f"got={action_mask.shape}, expected={(10,)}"
            )

        if not np.isfinite(observation).all():
            raise BattleObsError(
                "model observation contains NaN/Inf"
            )

        if not np.isfinite(action_mask).all():
            raise BattleObsError(
                "action mask contains NaN/Inf"
            )

        observation = np.clip(
            observation,
            -1.0,
            1.0,
        ).astype(np.float32, copy=False)

        # 최종 model.zip의 action_mask Box dtype이 int64이다.
        # policy는 mask > 0.5를 legal로 처리하므로 값도 명확히 0/1로 맞춘다.
        action_mask = (
            action_mask > 0.5
        ).astype(np.int64, copy=False)

        return {
            "observation": observation,
            "action_mask": action_mask,
        }

    def model_input(self, state=None) -> dict[str, np.ndarray]:
        """
        현재 Battle RAM -> PPO가 실제 학습 때 받은 입력 형식.

        기존 decision():
            (obs_276, mask_10)

        model_input():
            {
                "observation": obs_276,
                "action_mask": mask_10,
            }

        즉 276 feature의 내용/순서를 바꾸는 함수가 아니라,
        학습 policy의 Dict observation 구조까지 동일하게 맞추는 함수다.
        """
        observation, action_mask = self.decision(state)

        return self.pack_model_input(
            observation,
            action_mask,
        )

    # 의미가 더 분명한 alias.
    policy_input = model_input

    def validate_model_input(
        self,
        model_input: dict[str, np.ndarray],
        *,
        require_legal_action: bool = True,
    ) -> None:
        """
        controller/model 호출 직전 사용할 수 있는 빠른 검증.
        """
        if not isinstance(model_input, dict):
            raise BattleObsError(
                "model input must be dict"
            )

        expected_keys = {
            "observation",
            "action_mask",
        }

        if set(model_input.keys()) != expected_keys:
            raise BattleObsError(
                "model input keys mismatch: "
                f"got={sorted(model_input.keys())}, "
                f"expected={sorted(expected_keys)}"
            )

        observation = np.asarray(
            model_input["observation"]
        )
        action_mask = np.asarray(
            model_input["action_mask"]
        )

        if observation.shape != (N_FEATURES,):
            raise BattleObsError(
                "model observation shape mismatch: "
                f"{observation.shape}"
            )

        if action_mask.shape != (10,):
            raise BattleObsError(
                "model action mask shape mismatch: "
                f"{action_mask.shape}"
            )

        if observation.dtype != np.float32:
            raise BattleObsError(
                "model observation dtype mismatch: "
                f"{observation.dtype}, expected float32"
            )

        if action_mask.dtype != np.int64:
            raise BattleObsError(
                "model action mask dtype mismatch: "
                f"{action_mask.dtype}, expected int64"
            )

        if require_legal_action and not np.any(
            action_mask > 0.5
        ):
            raise BattleObsError(
                "ALL-ZERO ACTION MASK: PPO에 넘길 legal action이 없습니다."
            )

    def forced_turn(self, state=None) -> bool:
        """
        recharge / charge / thrash / bide / trapping / rage 등
        게임이 선택 메뉴를 생략할 수 있는 강제 턴 상태.

        BattleMacro가 실제 메인 메뉴 READY를 기준으로 controller를 호출한다면
        보통 이 함수가 True일 때는 PPO를 호출하지 않게 된다.
        """
        if state is None:
            state = self.battle_ram.read()

        s1 = self._state_ram_u8(
            state,
            "player_battle_status1",
            "wPlayerBattleStatus1",
        )
        s2 = self._state_ram_u8(
            state,
            "player_battle_status2",
            "wPlayerBattleStatus2",
        )

        forced1 = (
            (1 << BIT_STORING_ENERGY)
            | (1 << BIT_THRASHING_ABOUT)
            | (1 << BIT_ATTACKING_MULTIPLE_TIMES)
            | (1 << BIT_CHARGING_UP)
            | (1 << BIT_USING_TRAPPING_MOVE)
        )

        forced2 = (
            (1 << BIT_NEEDS_TO_RECHARGE)
            | (1 << BIT_USING_RAGE)
        )

        return bool((s1 & forced1) or (s2 & forced2))

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    @staticmethod
    def _load_json_dict(path: Path) -> dict[str, dict]:
        if not path.exists():
            raise FileNotFoundError(f"battle data가 없습니다: {path}")

        with path.open("r", encoding="utf-8") as fp:
            value = json.load(fp)

        if not isinstance(value, dict):
            raise TypeError(f"{path} 최상위 값은 dict여야 합니다.")

        return value

    def _species(self, species_id: int) -> dict:
        value = self.species_db.get(str(int(species_id)))
        return value if isinstance(value, dict) else {}

    def _move(self, move_id: int) -> dict:
        value = self.moves_db.get(str(int(move_id)))
        return value if isinstance(value, dict) else {}

    # ------------------------------------------------------------------
    # Symbol / RAM helpers
    # ------------------------------------------------------------------

    def _symbol(self, name: str) -> int | None:
        try:
            return self.sym.optional(name)
        except Exception:
            return None

    def _sym_u8(self, name: str, default: int = 0) -> int:
        address = self._symbol(name)

        if address is None:
            if self.debug and name not in self._warned_missing_symbols:
                self._warned_missing_symbols.add(name)
                print(f"[battle_obs] optional symbol missing: {name}")
            return int(default)

        return int(self.pyboy.memory[address]) & 0xFF

    def _state_ram_u8(
        self,
        state,
        attribute: str,
        symbol: str,
        default: int = 0,
    ) -> int:
        """Prefer the value captured in BattleRAMState's decision snapshot."""
        if state is not None and hasattr(state, attribute):
            return int(getattr(state, attribute)) & 0xFF
        return self._sym_u8(symbol, default)

    # ------------------------------------------------------------------
    # Party RAM
    # ------------------------------------------------------------------

    def _read_party_slot(
        self,
        *,
        base_symbol: str,
        slot: int,
        active: bool,
        toxic: bool = False,
    ) -> PartyMonSnapshot | None:
        base = self._symbol(base_symbol)

        if base is None:
            return None

        slot = int(slot)
        address = int(base) + slot * PARTYMON_STRUCT_LENGTH
        mem = self.pyboy.memory

        species_id = int(mem[address + MON_SPECIES]) & 0xFF

        if species_id == 0:
            return None

        return PartyMonSnapshot(
            slot=slot,
            species_id=species_id,
            level=int(mem[address + MON_LEVEL]) & 0xFF,
            hp=_u16be(mem, address + MON_HP),
            max_hp=_u16be(mem, address + MON_MAX_HP),
            status_raw=int(mem[address + MON_STATUS]) & 0xFF,
            move_ids=tuple(
                int(v)
                for v in _bytes(mem, address + MON_MOVES, 4)
            ),
            pp_raw=tuple(
                int(v)
                for v in _bytes(mem, address + MON_PP, 4)
            ),
            active=bool(active),
            toxic=bool(toxic),
        )

    def _read_player_party(self, state) -> list[PartyMonSnapshot]:
        count = min(
            MAX_PARTY,
            self._sym_u8("wPartyCount", MAX_PARTY),
        )

        active_index = int(getattr(state, "player_party_index", 0))
        active = getattr(state, "player", None)

        active_toxic = bool(
            self._state_ram_u8(
                state,
                "player_battle_status3",
                "wPlayerBattleStatus3",
            )
            & (1 << BIT_BADLY_POISONED)
        )

        result: list[PartyMonSnapshot] = []

        for i in range(count):
            mon = self._read_party_slot(
                base_symbol="wPartyMons",
                slot=i,
                active=(i == active_index),
                toxic=(active_toxic if i == active_index else False),
            )

            if mon is None:
                continue

            # active BattleMon RAM이 가장 현재값이므로 현재 slot은 덮어쓴다.
            if i == active_index and active is not None:
                active_moves = list(
                    getattr(active, "move_ids", ()) or ()
                )[:4]
                active_pp = list(
                    getattr(active, "pp_raw", ()) or ()
                )[:4]

                active_moves += [0] * (4 - len(active_moves))
                active_pp += [0] * (4 - len(active_pp))

                mon = PartyMonSnapshot(
                    slot=i,
                    species_id=int(
                        getattr(active, "species_id", mon.species_id)
                    ),
                    level=int(
                        getattr(active, "level", mon.level)
                    ),
                    hp=int(
                        getattr(active, "hp", mon.hp)
                    ),
                    max_hp=int(
                        getattr(active, "max_hp", mon.max_hp)
                    ),
                    status_raw=int(
                        getattr(active, "status_raw", mon.status_raw)
                    ),
                    move_ids=tuple(active_moves[:4]),
                    pp_raw=tuple(active_pp[:4]),
                    active=True,
                    toxic=active_toxic,
                )

            result.append(mon)

        # RAM count가 없거나 구조체 읽기가 실패했어도 active만은 보존.
        if not result and active is not None:
            moves = list(getattr(active, "move_ids", ()) or ())[:4]
            pp = list(getattr(active, "pp_raw", ()) or ())[:4]
            moves += [0] * (4 - len(moves))
            pp += [0] * (4 - len(pp))

            result.append(
                PartyMonSnapshot(
                    slot=active_index,
                    species_id=int(getattr(active, "species_id", 0)),
                    level=int(getattr(active, "level", 0)),
                    hp=int(getattr(active, "hp", 0)),
                    max_hp=int(getattr(active, "max_hp", 0)),
                    status_raw=int(getattr(active, "status_raw", 0)),
                    move_ids=tuple(moves[:4]),
                    pp_raw=tuple(pp[:4]),
                    active=True,
                    toxic=active_toxic,
                )
            )

        # Action alignment: slot index 자체를 보존해야 한다.
        by_slot = {mon.slot: mon for mon in result}
        aligned: list[PartyMonSnapshot] = []

        for i in range(6):
            mon = by_slot.get(i)
            if mon is not None:
                aligned.append(mon)
            else:
                # 없는 slot은 build 단계에서 None으로 다루기 위해
                # 여기서는 끊지 않고 placeholder를 만들지 않는다.
                pass

        # count 안에서 hole이 없어야 정상. 실제 feature 생성은 helper에서
        # slot 번호를 다시 맞춘다.
        return result

    def _read_enemy_party_slot(
        self,
        party_index: int,
    ) -> PartyMonSnapshot | None:
        return self._read_party_slot(
            base_symbol="wEnemyMons",
            slot=int(party_index),
            active=False,
            toxic=False,
        )

    # ------------------------------------------------------------------
    # Opponent reveal tracker
    # ------------------------------------------------------------------

    def _observe_enemy_active(self, state) -> RevealedEnemy | None:
        enemy = getattr(state, "enemy", None)

        if enemy is None:
            return None

        species_id = int(getattr(enemy, "species_id", 0))
        if species_id == 0:
            return None

        party_index = int(getattr(state, "enemy_party_index", 0))
        key = (party_index, species_id)

        entry = self._enemy_by_key.get(key)

        toxic = bool(
            self._state_ram_u8(
                state,
                "enemy_battle_status3",
                "wEnemyBattleStatus3",
            )
            & (1 << BIT_BADLY_POISONED)
        )

        if entry is None:
            entry = RevealedEnemy(
                key=key,
                reveal_order=len(self._enemy_reveal_order),
                party_index=party_index,
                species_id=species_id,
                level=int(getattr(enemy, "level", 0)),
                hp=int(getattr(enemy, "hp", 0)),
                max_hp=int(getattr(enemy, "max_hp", 0)),
                status_raw=int(getattr(enemy, "status_raw", 0)),
                fainted=bool(
                    int(getattr(enemy, "hp", 0)) <= 0
                ),
                active=True,
                toxic=toxic,
            )
            self._enemy_by_key[key] = entry
            self._enemy_reveal_order.append(key)

            if self.debug:
                print(
                    "[battle_obs] revealed enemy "
                    f"order={entry.reveal_order} "
                    f"party={party_index} species={species_id}"
                )

        # 모두 inactive -> 현재만 active.
        for other in self._enemy_by_key.values():
            other.active = False

        entry.active = True
        entry.level = int(getattr(enemy, "level", entry.level))
        entry.hp = int(getattr(enemy, "hp", entry.hp))
        entry.max_hp = int(getattr(enemy, "max_hp", entry.max_hp))
        entry.status_raw = int(
            getattr(enemy, "status_raw", entry.status_raw)
        )
        entry.fainted = entry.fainted or entry.hp <= 0
        entry.toxic = toxic

        return entry

    def _refresh_revealed_enemy_party(self) -> None:
        """
        이미 공개된 enemy만 wEnemyMons에서 갱신.
        unrevealed slot은 읽더라도 observation에 추가하지 않는다.
        """
        for key in list(self._enemy_reveal_order):
            entry = self._enemy_by_key.get(key)
            if entry is None:
                continue

            party_mon = self._read_enemy_party_slot(
                entry.party_index
            )

            if party_mon is None:
                continue

            # species가 같을 때만 해당 공개 entry에 반영.
            # duplicate species도 party index와 함께 key로 구분한다.
            if party_mon.species_id != entry.species_id:
                continue

            entry.level = party_mon.level
            entry.hp = party_mon.hp
            entry.max_hp = party_mon.max_hp
            entry.status_raw = party_mon.status_raw
            entry.fainted = entry.fainted or party_mon.fainted

    def _revealed_enemy_list(self) -> list[RevealedEnemy]:
        result = []

        for key in self._enemy_reveal_order[:6]:
            enemy = self._enemy_by_key.get(key)
            if enemy is not None:
                result.append(enemy)

        return result

    # ------------------------------------------------------------------
    # Common Pokemon helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hp_fraction_active(mon) -> float:
        if mon is None:
            return 0.0

        max_hp = int(getattr(mon, "max_hp", 0))
        hp = int(getattr(mon, "hp", 0))

        if max_hp <= 0:
            return 0.0

        return _clip01(hp / max_hp)

    @staticmethod
    def _boost_vector(mon) -> list[float]:
        if mon is None:
            return [0.0] * 6

        values = [
            int(getattr(mon, "atk_stage", 0)),
            int(getattr(mon, "def_stage", 0)),
            int(getattr(mon, "special_stage", 0)),
            int(getattr(mon, "spe_stage", 0)),
            int(getattr(mon, "accuracy_stage", 0)),
            int(getattr(mon, "evasion_stage", 0)),
        ]

        return [
            _clip11(value / 6.0)
            for value in values
        ]

    @staticmethod
    def _status_scalar(
        status_raw: int,
        *,
        toxic: bool = False,
    ) -> float:
        status_raw = int(status_raw) & 0xFF

        # Training order:
        # none / BRN / FRZ / PAR / PSN / SLP / TOX / OTHER
        if status_raw == 0:
            return 0.0

        if status_raw & 0x07:
            return 5.0 / 7.0

        if toxic and (status_raw & 0x08):
            return 6.0 / 7.0

        if status_raw & 0x10:
            return 1.0 / 7.0

        if status_raw & 0x20:
            return 2.0 / 7.0

        if status_raw & 0x40:
            return 3.0 / 7.0

        if status_raw & 0x08:
            return 4.0 / 7.0

        return 1.0

    def _base_speed(self, species_id: int) -> float:
        return _finite(
            self._species(species_id).get("speed", 0.0)
        )

    def _speed_diff(
        self,
        a_species: int,
        b_species: int,
    ) -> float:
        return _clip11(
            (
                self._base_speed(a_species)
                - self._base_speed(b_species)
            )
            / 255.0
        )

    def _switch_matchup_score(
        self,
        *,
        our_species: int,
        our_moves: Iterable[int],
        opponent_species: int,
        opponent_moves: Iterable[int],
    ) -> float:
        offensive = self._best_known_attack_score(
            attacker_species=our_species,
            attacker_moves=our_moves,
            defender_species=opponent_species,
        )
        incoming = self._best_known_attack_score(
            attacker_species=opponent_species,
            attacker_moves=opponent_moves,
            defender_species=our_species,
        )
        return float(offensive - 0.5 * incoming)

    def _switch_decision_context(
        self,
        *,
        our_active,
        opponent_active,
        own_team: Iterable[PartyMonSnapshot],
        opponent_known_moves: Iterable[int],
        trapped: bool,
    ) -> dict:
        own_species = int(getattr(our_active, "species_id", 0))
        opponent_species = int(getattr(opponent_active, "species_id", 0))
        current_score = self._switch_matchup_score(
            our_species=own_species,
            our_moves=getattr(our_active, "move_ids", ()) or (),
            opponent_species=opponent_species,
            opponent_moves=opponent_known_moves,
        )

        candidates = [
            mon
            for mon in own_team
            if (
                not trapped
                and mon is not None
                and not mon.active
                and not mon.fainted
            )
        ]
        legal_switch_slots = frozenset(int(mon.slot) for mon in candidates)
        switch_scores = [
            self._switch_matchup_score(
                our_species=mon.species_id,
                our_moves=mon.move_ids,
                opponent_species=opponent_species,
                opponent_moves=opponent_known_moves,
            )
            for mon in candidates
        ]
        best_switch_score = max(switch_scores, default=current_score)
        best_switch_gain = (
            float(best_switch_score - current_score) if candidates else 0.0
        )
        matchup_flip = bool(
            candidates
            and current_score < -SWITCH_FAVORABLE_SCORE_MARGIN
            and best_switch_score > SWITCH_FAVORABLE_SCORE_MARGIN
            and best_switch_gain >= SWITCH_FLIP_MIN_GAIN
        )
        worst_stage = min(self._raw_boost_stages(our_active), default=0)
        severe_debuff = worst_stage <= SWITCH_SEVERE_DEBUFF_STAGE
        debuff_escape_available = bool(
            candidates
            and severe_debuff
            and best_switch_score
            >= current_score - SWITCH_DEBUFF_ESCAPE_MAX_MATCHUP_LOSS
        )
        low_hp_escape_available = bool(
            self._hp_fraction_active(our_active)
            <= SWITCH_LOW_HP_RELIEF_THRESHOLD
            and candidates
            and best_switch_gain >= SWITCH_CLEAR_MATCHUP_GAIN
            and best_switch_score > SWITCH_FAVORABLE_SCORE_MARGIN
        )
        switch_need = max(
            1.0 if matchup_flip else 0.0,
            1.0 if debuff_escape_available else 0.0,
            0.75 if low_hp_escape_available else 0.0,
        )
        return {
            "current_score": float(current_score),
            "best_switch_score": float(best_switch_score),
            "best_switch_gain": float(best_switch_gain),
            "matchup_flip": matchup_flip,
            "debuff_escape_available": debuff_escape_available,
            "low_hp_escape_available": low_hp_escape_available,
            "switch_need": _clip01(switch_need),
            "legal_switch_slots": legal_switch_slots,
        }

    # ------------------------------------------------------------------
    # Move helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _current_pp(pp_raw: int) -> int:
        # Gen1 PP byte: upper two bits are PP Up count.
        return int(pp_raw) & 0x3F

    @staticmethod
    def _pp_up_count(pp_raw: int) -> int:
        return (int(pp_raw) >> 6) & 0x03

    def _max_pp(self, move_id: int, pp_raw: int) -> int:
        base_pp = int(
            self._move(move_id).get("pp", 0) or 0
        )

        if base_pp <= 0:
            return 0

        pp_ups = self._pp_up_count(pp_raw)
        return base_pp + (base_pp * pp_ups) // 5

    def _pp_ratio(self, move_id: int, pp_raw: int) -> float:
        max_pp = self._max_pp(move_id, pp_raw)

        if max_pp <= 0:
            return 0.0

        return _clip01(
            self._current_pp(pp_raw) / max_pp
        )

    def _species_types(self, species_id: int) -> tuple[str, ...]:
        info = self._species(species_id)

        t1 = str(info.get("type1") or "").upper()
        t2 = str(info.get("type2") or "").upper()

        result = []

        if t1:
            result.append(t1)

        if t2 and t2 != t1:
            result.append(t2)

        return tuple(result)

    @staticmethod
    def _move_type(move: dict) -> str:
        return str(move.get("type") or "").upper()

    def _effectiveness_raw(
        self,
        move_id: int,
        defender_species: int,
    ) -> float:
        move = self._move(move_id)

        if not move:
            return 0.0

        move_type = self._move_type(move)
        defender_types = self._species_types(defender_species)

        if not move_type or not defender_types:
            return 0.0

        multiplier = 1.0

        for defender_type in defender_types:
            multiplier *= TYPE_CHART.get(
                (move_type, defender_type),
                1.0,
            )

        return float(multiplier)

    def _effectiveness(
        self,
        move_id: int,
        defender_species: int,
    ) -> float:
        return _clip01(
            self._effectiveness_raw(
                move_id,
                defender_species,
            )
            / 4.0
        )

    def _stab(
        self,
        attacker_species: int,
        move_id: int,
    ) -> bool:
        move = self._move(move_id)

        if not move:
            return False

        move_type = self._move_type(move)

        return move_type in self._species_types(
            attacker_species
        )

    @staticmethod
    def _effect_name(move: dict) -> str:
        return str(move.get("effect") or "").upper()

    def _base_power_for_observation(self, move_id: int) -> float:
        move = self._move(move_id)

        if not move:
            return 0.0

        effect = self._effect_name(move)

        # Showdown fixed damage move의 basePower는 0이고 damage가 별도다.
        if effect in _FIXED_DAMAGE_EFFECTS:
            return 0.0

        return _finite(move.get("power", 0.0))

    def _is_fixed_damage(self, move_id: int) -> bool:
        move = self._move(move_id)

        if not move:
            return False

        return self._effect_name(move) in _FIXED_DAMAGE_EFFECTS

    @staticmethod
    def _accuracy(move: dict) -> float:
        # local moves.json은 0..100.
        return _clip01(
            _finite(move.get("accuracy", 100.0))
            / 100.0
        )

    @staticmethod
    def _category(move: dict) -> float:
        """
        학습 env와 동일한 damaging-vs-status 표현.

        moves.json 원본 값:
          physical = -1
          status   =  0
          special  = +1

        PPO observation:
          physical = 1
          status   = 0
          special  = 1

        원본 moves.json은 수정하지 않고 observation 생성 시점에만
        절대값을 적용한다.
        """
        raw_category = _clip11(
            _finite(move.get("category", 0.0))
        )
        return abs(raw_category)

    @staticmethod
    def _priority(move: dict) -> float:
        return _clip11(
            _finite(move.get("priority", 0.0))
            / 5.0
        )

    def _heal_fraction(self, move: dict) -> float:
        name = str(move.get("name") or "").upper()
        effect = self._effect_name(move)

        if effect != "HEAL_EFFECT":
            return 0.0

        if name == "REST":
            return 1.0

        # Recover / Softboiled
        return 0.5

    @staticmethod
    def _is_status_move(move: dict) -> bool:
        return int(_finite(move.get("category", 0.0))) == 0

    def _attack_choice_score(
        self,
        move_id: int,
        *,
        defender_species: int,
    ) -> float:
        """Battle_RL_r2: ordinary attack quality = power * effectiveness."""
        move = self._move(move_id)
        if not move or self._is_status_move(move):
            return 0.0

        power = self._base_power_for_observation(move_id)
        if power <= 0.0:
            return 0.0

        return float(
            power
            * max(
                0.0,
                self._effectiveness_raw(move_id, defender_species),
            )
        )

    @staticmethod
    def _gen1_stage_ratio(stage: int) -> tuple[int, int]:
        stage = max(-6, min(6, int(stage)))
        return ((2 + stage, 2) if stage >= 0 else (2, 2 - stage))

    def _gen1_conservative_stat(
        self,
        species_id: int,
        level: int,
        stat: str,
        *,
        upper: bool,
    ) -> int:
        species = self._species(species_id)
        key = {
            "atk": "attack",
            "def": "defense",
            "special": "special",
            "spe": "speed",
            "hp": "hp",
        }.get(stat, stat)
        base = max(0.0, _finite(species.get(key, 0.0)))
        if base <= 0.0:
            return 1
        level = max(1, int(level))
        dv = 15 if upper else 0
        stat_exp_term = 63 if upper else 0
        core = int((((base + dv) * 2.0 + stat_exp_term) * level) // 100)
        if stat == "hp":
            return max(1, core + level + 10)
        return max(1, core + 5)

    @classmethod
    def _gen1_apply_stage(cls, stat: int, stage: int) -> int:
        numerator, denominator = cls._gen1_stage_ratio(stage)
        return max(1, int(stat) * numerator // denominator)

    def _fixed_damage_amount(self, move: dict, attacker) -> int | None:
        name = str(move.get("name") or "").upper().replace("_", "")
        if name == "SONICBOOM":
            return 20
        if name == "DRAGONRAGE":
            return 40
        if name in {"NIGHTSHADE", "SEISMICTOSS"}:
            return max(1, int(getattr(attacker, "level", 1)))
        return None

    @staticmethod
    def _gen1_accuracy_stage_ratio(stage: int) -> tuple[int, int]:
        stage = max(-6, min(6, int(stage)))
        return ((3 + stage, 3) if stage >= 0 else (3, 3 - stage))

    @staticmethod
    def _normalized_move_name(move: dict) -> str:
        return str(move.get("name") or "").lower().replace("_", "").replace(" ", "")

    def _effective_move_accuracy(self, move: dict, attacker, opponent) -> float:
        if not move:
            return 0.0
        if self._normalized_move_name(move) == "swift":
            return 1.0
        own_stage = self._raw_boost_stages(attacker)[4]
        evasion_stage = self._raw_boost_stages(opponent)[5]
        acc_num, acc_den = self._gen1_accuracy_stage_ratio(own_stage)
        eva_num, eva_den = self._gen1_accuracy_stage_ratio(evasion_stage)
        return _clip01(
            self._accuracy(move) * acc_num * eva_den / (acc_den * eva_num)
        )

    def _move_is_one_turn(self, move: dict) -> bool:
        return self._normalized_move_name(move) not in {
            "dig",
            "fly",
            "razorwind",
            "solarbeam",
            "skullbash",
            "skyattack",
        }

    def _move_execution_condition_met(self, move: dict, opponent) -> bool:
        if self._normalized_move_name(move) == "dreameater":
            return bool(int(getattr(opponent, "status_raw", 0)) & 0x07)
        return True

    def _acts_first_estimate(self, move: dict, attacker, opponent) -> bool:
        priority = int(round(_finite(move.get("priority", 0.0))))
        if priority > 0:
            return True
        if priority < 0:
            return False
        own_speed = self._gen1_apply_stage(
            self._gen1_conservative_stat(
                int(getattr(attacker, "species_id", 0)),
                int(getattr(attacker, "level", 1)),
                "spe",
                upper=False,
            ),
            self._raw_boost_stages(attacker)[3],
        )
        opponent_speed = self._gen1_apply_stage(
            self._gen1_conservative_stat(
                int(getattr(opponent, "species_id", 0)),
                int(getattr(opponent, "level", 1)),
                "spe",
                upper=True,
            ),
            self._raw_boost_stages(opponent)[3],
        )
        if int(getattr(attacker, "status_raw", 0)) & 0x40:
            own_speed = max(1, own_speed // 4)
        if int(getattr(opponent, "status_raw", 0)) & 0x40:
            opponent_speed = max(1, opponent_speed // 4)
        return own_speed >= opponent_speed

    def _finisher_features(
        self,
        move_id: int,
        *,
        attacker,
        opponent,
    ) -> tuple[float, float, float, float, float, float]:
        move = self._move(move_id)
        if (
            not move
            or attacker is None
            or opponent is None
            or self._is_status_move(move)
        ):
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        effective_accuracy = self._effective_move_accuracy(move, attacker, opponent)
        one_turn = float(self._move_is_one_turn(move))
        acts_first = float(self._acts_first_estimate(move, attacker, opponent))

        effectiveness = self._effectiveness_raw(
            move_id, int(getattr(opponent, "species_id", 0))
        )
        if effectiveness <= 0.0:
            return effective_accuracy, 0.0, 0.0, one_turn, acts_first, 0.0

        min_damage = self._fixed_damage_amount(move, attacker)
        if min_damage is None:
            power = max(0, int(round(self._base_power_for_observation(move_id))))
            raw_category = int(_finite(move.get("category", 0.0)))
            if power <= 0 or raw_category == 0:
                return effective_accuracy, 0.0, 0.0, one_turn, acts_first, 0.0

            if raw_category < 0:
                attack_name, defense_name = "atk", "def"
                attack_stage_index, defense_stage_index = 0, 1
            else:
                attack_name = defense_name = "special"
                attack_stage_index = defense_stage_index = 2

            attack_stat = self._gen1_conservative_stat(
                int(getattr(attacker, "species_id", 0)),
                int(getattr(attacker, "level", 1)),
                attack_name,
                upper=False,
            )
            defense_stat = self._gen1_conservative_stat(
                int(getattr(opponent, "species_id", 0)),
                int(getattr(opponent, "level", 1)),
                defense_name,
                upper=True,
            )
            attack_stat = self._gen1_apply_stage(
                attack_stat, self._raw_boost_stages(attacker)[attack_stage_index]
            )
            defense_stat = self._gen1_apply_stage(
                defense_stat, self._raw_boost_stages(opponent)[defense_stage_index]
            )
            if raw_category < 0 and (int(getattr(attacker, "status_raw", 0)) & 0x10):
                attack_stat = max(1, attack_stat // 2)

            level = max(1, int(getattr(attacker, "level", 1)))
            min_damage = (
                ((2 * level // 5 + 2) * power * attack_stat // defense_stat)
                // 50
                + 2
            )
            if self._stab(int(getattr(attacker, "species_id", 0)), move_id):
                min_damage = min_damage * 3 // 2
            min_damage = int(min_damage * effectiveness)
            min_damage = min_damage * 217 // 255

        min_damage = max(0, int(min_damage))
        if min_damage <= 0:
            return effective_accuracy, 0.0, 0.0, one_turn, acts_first, 0.0

        target_max_hp = self._gen1_conservative_stat(
            int(getattr(opponent, "species_id", 0)),
            int(getattr(opponent, "level", 1)),
            "hp",
            upper=True,
        )
        remaining_hp = max(
            1,
            int(np.ceil(
                self._hp_fraction_active(opponent) * target_max_hp - 1.0e-9
            )),
        )
        coverage = _clip01(min_damage / remaining_hp)
        ko_on_hit = float(min_damage >= remaining_hp)
        safe_immediate = float(
            ko_on_hit > 0.5
            and effective_accuracy >= 1.0 - 1.0e-9
            and one_turn > 0.5
            and acts_first > 0.5
            and self._move_execution_condition_met(move, opponent)
            and self._normalized_move_name(move) not in {"explosion", "selfdestruct"}
        )
        return (
            effective_accuracy,
            coverage,
            ko_on_hit,
            one_turn,
            acts_first,
            safe_immediate,
        )

    def _status_move_utility(self, move: dict, *, attacker, opponent, state) -> float:
        if not move or not self._is_status_move(move):
            return 0.0
        if self._persistent_effect_active(move, state):
            return 0.0

        effect = self._effect_name(move)
        utility = 0.0
        if self._persistent_effect_kind(move) is not None:
            utility = max(utility, 0.80)
        if effect in _MAJOR_STATUS_EFFECTS and not int(
            getattr(opponent, "status_raw", 0)
        ):
            utility = max(utility, 0.90)
        if effect in _VOLATILE_EFFECTS:
            utility = max(utility, 0.75)
        if effect in _SIDE_CONDITION_EFFECTS:
            utility = max(utility, 0.65)

        heal = self._heal_fraction(move)
        if heal > 0.0:
            missing_hp = _clip01(1.0 - self._hp_fraction_active(attacker))
            utility = max(utility, _clip01(0.25 + 0.75 * missing_hp))
        if _OPP_DEBUFF_STAGE.get(effect, 0) > 0:
            utility = max(
                utility,
                0.10 * self._debuff_remaining_capacity(move, opponent),
            )
        if _SELF_BOOST_STAGE.get(effect, 0) > 0:
            utility = max(utility, 0.18)
        return _clip01(utility)

    def _persistent_effect_kind(self, move: dict) -> str | None:
        if not move or not self._is_status_move(move):
            return None
        return _PERSISTENT_EFFECT_KIND.get(self._effect_name(move))

    def _persistent_effect_active(self, move: dict, state) -> bool:
        kind = self._persistent_effect_kind(move)
        if kind is None:
            return False

        player_status2 = self._state_ram_u8(
            state,
            "player_battle_status2",
            "wPlayerBattleStatus2",
        )
        player_status3 = self._state_ram_u8(
            state,
            "player_battle_status3",
            "wPlayerBattleStatus3",
        )
        enemy_status1 = self._state_ram_u8(
            state,
            "enemy_battle_status1",
            "wEnemyBattleStatus1",
        )
        enemy_status2 = self._state_ram_u8(
            state,
            "enemy_battle_status2",
            "wEnemyBattleStatus2",
        )

        if kind == "major_status":
            opponent = getattr(state, "enemy", None)
            return bool(int(getattr(opponent, "status_raw", 0)) & 0xFF)
        if kind == "confusion":
            return bool(enemy_status1 & (1 << BIT_CONFUSED))
        if kind == "leech_seed":
            return bool(enemy_status2 & (1 << BIT_SEEDED))
        if kind == "disable":
            return self._state_ram_u8(
                state,
                "enemy_disabled_move",
                "wEnemyDisabledMove",
            ) != 0
        if kind == "substitute":
            return bool(
                player_status2 & (1 << BIT_HAS_SUBSTITUTE_UP)
                or self._state_ram_u8(
                    state,
                    "player_substitute_hp",
                    "wPlayerSubstituteHP",
                ) > 0
            )
        if kind == "focus_energy":
            return bool(player_status2 & (1 << BIT_GETTING_PUMPED))
        if kind == "reflect":
            return bool(player_status3 & (1 << BIT_HAS_REFLECT_UP))
        if kind == "light_screen":
            return bool(player_status3 & (1 << BIT_HAS_LIGHT_SCREEN_UP))
        if kind == "mist":
            return bool(player_status2 & (1 << BIT_PROTECTED_BY_MIST))
        return False

    def _debuff_target_index(self, move: dict) -> int | None:
        if not move or not self._is_status_move(move):
            return None
        return _DEBUFF_STAT_INDEX.get(self._effect_name(move))

    @staticmethod
    def _raw_boost_stages(mon) -> tuple[int, int, int, int, int, int]:
        if mon is None:
            return (0, 0, 0, 0, 0, 0)
        return (
            int(getattr(mon, "atk_stage", 0)),
            int(getattr(mon, "def_stage", 0)),
            int(getattr(mon, "special_stage", 0)),
            int(getattr(mon, "spe_stage", 0)),
            int(getattr(mon, "accuracy_stage", 0)),
            int(getattr(mon, "evasion_stage", 0)),
        )

    def _debuff_remaining_capacity(self, move: dict, opponent) -> float:
        index = self._debuff_target_index(move)
        if index is None:
            return 0.0
        stage = self._raw_boost_stages(opponent)[index]
        return _clip01((stage + 6.0) / 6.0)

    def _debuff_policy_cap_reached(self, move: dict, opponent) -> bool:
        index = self._debuff_target_index(move)
        if index is None or opponent is None:
            return False
        return (
            self._raw_boost_stages(opponent)[index]
            <= OPPONENT_DEBUFF_MASK_STAGE
        )

    def _known_immune_to_target(
        self,
        move: dict,
        defender_species: int,
    ) -> bool:
        defender_types = self._species_types(defender_species)
        effect = self._effect_name(move)
        name = self._normalized_move_name(move)
        if effect == "LEECH_SEED_EFFECT" and "GRASS" in defender_types:
            return True
        if (
            (effect in {"POISON_EFFECT", "TOXIC_EFFECT"} or name in {"poisonpowder", "toxic"})
            and "POISON" in defender_types
        ):
            return True
        if name == "thunderwave" and "GROUND" in defender_types:
            return True
        return False

    def _own_move_features(
        self,
        *,
        move_id: int,
        pp_raw: int,
        attacker_species: int,
        defender_species: int,
        attacker,
        opponent,
        state,
        legal_move_ids: set[int],
        best_legal_attack_score: float,
    ) -> list[float]:
        move_id = int(move_id)

        if move_id <= 0:
            return [0.0] * MOVE_SLOT_DIM

        move = self._move(move_id)

        if not move:
            return [0.0] * MOVE_SLOT_DIM

        effect = self._effect_name(move)
        is_status = self._is_status_move(move)
        persistent_effect = self._persistent_effect_kind(move) is not None
        attack_score = (
            self._attack_choice_score(
                move_id,
                defender_species=defender_species,
            )
            if not is_status and move_id in legal_move_ids
            else 0.0
        )
        attack_quality_ratio = (
            _clip01(attack_score / best_legal_attack_score)
            if best_legal_attack_score > 1.0e-9
            else 0.0
        )
        best_attack = float(
            attack_score > 1.0e-9
            and abs(attack_score - best_legal_attack_score) <= 1.0e-9
        )
        attack_score_norm = _clip01(attack_score / 200.0)
        (
            effective_accuracy,
            min_damage_ratio,
            ko_on_hit,
            one_turn_move,
            acts_first,
            safe_immediate_ko,
        ) = (
            self._finisher_features(
                move_id,
                attacker=attacker,
                opponent=opponent,
            )
            if not is_status and move_id in legal_move_ids
            else (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        )
        status_utility = self._status_move_utility(
            move,
            attacker=attacker,
            opponent=opponent,
            state=state,
        )

        return [
            # 0 power
            _clip01(
                self._base_power_for_observation(move_id)
                / 200.0
            ),

            # 1 accuracy
            self._accuracy(move),

            # 2 pp
            self._pp_ratio(move_id, pp_raw),

            # 3 effectiveness
            0.0 if is_status else self._effectiveness(
                move_id, defender_species
            ),

            # 4 category
            self._category(move),

            # 5 priority
            self._priority(move),

            # 6 STAB
            0.0 if is_status else float(
                self._stab(attacker_species, move_id)
            ),

            # 7 opponent debuff magnitude
            _clip01(
                _OPP_DEBUFF_STAGE.get(effect, 0)
                / 6.0
            ),

            # 8 self boost magnitude
            _clip01(
                _SELF_BOOST_STAGE.get(effect, 0)
                / 6.0
            ),

            # 9 major status
            float(effect in _MAJOR_STATUS_EFFECTS),

            # 10 volatile status
            float(effect in _VOLATILE_EFFECTS),

            # 11 side condition
            float(effect in _SIDE_CONDITION_EFFECTS),

            # 12 heal
            self._heal_fraction(move),

            # 13 persistent effect semantics
            float(persistent_effect),

            # 14 requested persistent effect is already active
            float(self._persistent_effect_active(move, state)),

            # 15..24 legal damaging-move quality / finisher signal
            attack_quality_ratio,
            best_attack,
            attack_score_norm,
            effective_accuracy,
            min_damage_ratio,
            ko_on_hit,
            one_turn_move,
            acts_first,
            safe_immediate_ko,
            status_utility,

            # 25 pure opponent-debuff room before stage -6
            self._debuff_remaining_capacity(move, opponent),
        ]

    def _opponent_move_features(
        self,
        move_id: int,
        *,
        our_species: int,
    ) -> list[float]:
        move_id = int(move_id)

        if move_id <= 0:
            return [0.0] * OPP_MOVE_SLOT_DIM

        move = self._move(move_id)

        if not move:
            return [0.0] * OPP_MOVE_SLOT_DIM

        return [
            1.0,
            _clip01(
                self._base_power_for_observation(move_id)
                / 200.0
            ),
            self._accuracy(move),
            self._effectiveness(move_id, our_species),
            self._category(move),
            self._priority(move),
        ]

    def _best_known_attack_score(
        self,
        *,
        attacker_species: int,
        attacker_moves: Iterable[int],
        defender_species: int,
    ) -> float:
        scores: list[float] = []

        for move_id in list(attacker_moves)[:4]:
            move_id = int(move_id)

            if move_id <= 0:
                continue

            move = self._move(move_id)

            if not move:
                continue

            power = self._base_power_for_observation(
                move_id
            )
            fixed_damage = self._is_fixed_damage(move_id)

            if power <= 0.0 and not fixed_damage:
                continue

            power_norm = (
                0.5
                if fixed_damage
                else _clip01(power / 200.0)
            )

            accuracy = self._accuracy(move)
            effectiveness = self._effectiveness(
                move_id,
                defender_species,
            )

            # training:
            # stab_multiplier / 1.5
            stab_norm = (
                1.0
                if self._stab(attacker_species, move_id)
                else (1.0 / 1.5)
            )

            scores.append(
                power_norm
                * accuracy
                * effectiveness
                * stab_norm
            )

        return _clip01(max(scores, default=0.0))

    # ------------------------------------------------------------------
    # Party feature helpers
    # ------------------------------------------------------------------

    def _own_party_features(
        self,
        mon: PartyMonSnapshot | None,
        *,
        opponent_species: int,
        opponent_known_moves: Iterable[int],
        current_matchup_score: float,
        best_legal_switch_gain: float,
        legal_switch_slots: frozenset[int],
    ) -> list[float]:
        if mon is None:
            return [0.0] * OWN_TEAM_SLOT_DIM

        best_attack = self._best_known_attack_score(
            attacker_species=mon.species_id,
            attacker_moves=mon.move_ids,
            defender_species=opponent_species,
        )

        incoming = self._best_known_attack_score(
            attacker_species=opponent_species,
            attacker_moves=opponent_known_moves,
            defender_species=mon.species_id,
        )
        matchup_score = float(best_attack - 0.5 * incoming)
        matchup_gain = float(matchup_score - current_matchup_score)
        is_legal_switch = int(mon.slot) in legal_switch_slots
        if is_legal_switch and matchup_gain > 0.0:
            switch_quality_ratio = _clip01(
                matchup_gain / max(SWITCH_CLEAR_MATCHUP_GAIN, 1.0e-9)
            )
            best_switch = float(
                matchup_gain >= SWITCH_CLEAR_MATCHUP_GAIN
                and abs(matchup_gain - best_legal_switch_gain) <= 1.0e-9
            )
        else:
            switch_quality_ratio = 0.0
            best_switch = 0.0

        return [
            float(mon.active),
            float(mon.fainted),
            mon.hp_ratio,
            _clip01(mon.level / 100.0),
            self._status_scalar(
                mon.status_raw,
                toxic=mon.toxic,
            ),
            best_attack,
            incoming,
            self._speed_diff(
                mon.species_id,
                opponent_species,
            ),
            _clip11(matchup_score),
            _clip11(
                matchup_gain / max(SWITCH_CLEAR_MATCHUP_GAIN, 1.0e-9)
            ),
            switch_quality_ratio,
            best_switch,
        ]

    def _opponent_party_features(
        self,
        mon: RevealedEnemy | None,
        *,
        our_species: int,
    ) -> list[float]:
        if mon is None:
            return [0.0] * OPP_TEAM_SLOT_DIM

        threat = self._best_known_attack_score(
            attacker_species=mon.species_id,
            attacker_moves=mon.move_ids,
            defender_species=our_species,
        )

        return [
            1.0,
            float(mon.active),
            float(mon.fainted),
            mon.hp_ratio,
            _clip01(mon.level / 100.0),
            self._status_scalar(
                mon.status_raw,
                toxic=mon.toxic,
            ),
            threat,
            self._speed_diff(
                mon.species_id,
                our_species,
            ),
        ]

    # ------------------------------------------------------------------
    # Layout/debug
    # ------------------------------------------------------------------

    @staticmethod
    def feature_slices() -> dict[str, slice]:
        return dict(FEATURE_SLICES)

    def debug_summary(
        self,
        obs: np.ndarray | None = None,
    ) -> str:
        if obs is None:
            obs = self._last_obs

        if obs is None:
            return "[battle_obs] 아직 observation이 없습니다."

        lines = [
            "[battle_obs]",
            f"shape={obs.shape}",
            f"dtype={obs.dtype}",
            f"range=({float(obs.min()):.4f}, {float(obs.max()):.4f})",
            f"turn={self.turn}",
            (
                "revealed_enemy="
                f"{len(self._enemy_reveal_order)}"
            ),
        ]

        for name, sl in FEATURE_SLICES.items():
            part = obs[sl]
            lines.append(
                f"{name:16s} "
                f"[{sl.start:3d}:{sl.stop:3d}] "
                f"nonzero={int(np.count_nonzero(part)):3d}"
            )

        return "\n".join(lines)


__all__ = [
    "BattleObs",
    "BattleObsError",
    "PartyMonSnapshot",
    "RevealedEnemy",
    "N_FEATURES",
    "OBSERVATION_LAYOUT_VERSION",
    "MOVE_FEATURE_NAMES",
    "GLOBAL_DIM",
    "SWITCH_CONTEXT_START_INDEX",
    "SWITCH_NEED_GLOBAL_INDEX",
    "MOVE_SLOT_DIM",
    "MOVE_CATEGORY_INDEX",
    "ATTACK_QUALITY_RATIO_INDEX",
    "BEST_ATTACK_INDEX",
    "ATTACK_SCORE_NORM_INDEX",
    "EFFECTIVENESS_INDEX",
    "EFFECTIVE_ACCURACY_INDEX",
    "MIN_DAMAGE_RATIO_INDEX",
    "KO_ON_HIT_INDEX",
    "ONE_TURN_MOVE_INDEX",
    "ACTS_FIRST_INDEX",
    "SAFE_IMMEDIATE_KO_INDEX",
    "STATUS_UTILITY_INDEX",
    "OPP_MOVE_SLOT_DIM",
    "OWN_TEAM_SLOT_DIM",
    "OWN_TEAM_MATCHUP_GAIN_INDEX",
    "OWN_TEAM_SWITCH_QUALITY_INDEX",
    "OWN_TEAM_BEST_SWITCH_INDEX",
    "OPP_TEAM_SLOT_DIM",
    "PERSISTENT_EFFECT_INDEX",
    "PERSISTENT_EFFECT_ACTIVE_INDEX",
    "OPPONENT_DEBUFF_INDEX",
    "DEBUFF_REMAINING_CAPACITY_INDEX",
    "FEATURE_SLICES",
    "rl_action_to_macro",
]
