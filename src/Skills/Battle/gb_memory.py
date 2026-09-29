"""
gb_memory.py — Pokémon Red Battle RL 전용 RAM reader

지도/VRAM/Navigation 관련 주소는 전혀 다루지 않는다.
Battle RL에 필요한 전투 RAM snapshot만 제공한다.

- 실제 전투 주소는 assets/roms/pokered.sym에서 심볼 이름으로 읽는다.
- ADDR_IN_BATTLE만 기존 프로젝트 호환용으로 0xD057을 유지한다.
- 프레임마다 print하지 않는다.
- BattleRAMReader는 현재 상태만 읽고,
  BattleActionLogger는 controller가 액션 경계에서 호출할 때만 출력한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SYM_PATH = PROJECT_ROOT / "assets" / "roms" / "pokered.sym"

# 기존 프로젝트에서 사용하던 wIsInBattle 주소.
ADDR_IN_BATTLE = 0xD057

_SYM_RE = re.compile(r"^([0-9A-Fa-f]{2}):([0-9A-Fa-f]{4})\s+(\S+)$")


# 276-feature observation이 직접 사용하는 Pokemon Red RAM 심볼.
#
# battle_obs가 누락된 심볼을 0으로 조용히 대체하면 shape는 276으로 맞아도
# boost / 상태기 / 교체 판단이 전부 중립값으로 오염될 수 있다. 최종 모델용
# reader는 시작할 때 이 목록을 검증해 잘못된 .sym/ROM 조합을 즉시 막는다.
OBSERVATION_RAM_SYMBOLS = (
    "wIsInBattle",
    "wBattleType",
    "wBattleResult",
    "wWhichPokemon",
    "wPlayerMonNumber",
    "wBattleMonPartyPos",
    "wEnemyMonPartyPos",
    "wBattleMonSpecies2",
    "wBattleMonLevel",
    "wBattleMonHP",
    "wBattleMonMaxHP",
    "wBattleMonStatus",
    "wBattleMonMoves",
    "wBattleMonPP",
    "wEnemyMonSpecies2",
    "wEnemyMonLevel",
    "wEnemyMonHP",
    "wEnemyMonMaxHP",
    "wEnemyMonStatus",
    "wEnemyMonMoves",
    "wEnemyMonPP",
    "wPlayerMonAttackMod",
    "wPlayerMonDefenseMod",
    "wPlayerMonSpeedMod",
    "wPlayerMonSpecialMod",
    "wPlayerMonAccuracyMod",
    "wPlayerMonEvasionMod",
    "wEnemyMonAttackMod",
    "wEnemyMonDefenseMod",
    "wEnemyMonSpeedMod",
    "wEnemyMonSpecialMod",
    "wEnemyMonAccuracyMod",
    "wEnemyMonEvasionMod",
    "wPlayerBattleStatus1",
    "wPlayerBattleStatus2",
    "wPlayerBattleStatus3",
    "wEnemyBattleStatus1",
    "wEnemyBattleStatus2",
    "wEnemyBattleStatus3",
    "wPlayerSubstituteHP",
    "wPlayerDisabledMove",
    "wEnemyDisabledMove",
    "wPartyCount",
    "wPartyMons",
    "wEnemyMons",
)


def load_symbols(sym_path: Path = DEFAULT_SYM_PATH) -> dict[str, int]:
    """RGBDS .sym -> {symbol_name: 16-bit address}."""
    sym_path = Path(sym_path)

    if not sym_path.exists():
        raise FileNotFoundError(
            f"pokered.sym이 없습니다: {sym_path}\n"
            "Battle RAM 주소는 추측하지 않고 .sym을 기준으로 읽습니다."
        )

    result: dict[str, int] = {}

    with sym_path.open("r", encoding="utf-8", errors="replace") as fp:
        for raw_line in fp:
            m = _SYM_RE.match(raw_line.strip())
            if not m:
                continue
            _, address_hex, name = m.groups()
            result[name] = int(address_hex, 16)

    return result


class BattleSymbols:
    def __init__(self, sym_path: Path = DEFAULT_SYM_PATH):
        self.sym_path = Path(sym_path)
        self.symbols = load_symbols(self.sym_path)

        sym_in_battle = self.symbols.get("wIsInBattle")
        if sym_in_battle is None:
            raise KeyError("pokered.sym에 wIsInBattle 심볼이 없습니다.")

        if sym_in_battle != ADDR_IN_BATTLE:
            raise RuntimeError(
                "wIsInBattle 주소 불일치: "
                f"gb_memory=0x{ADDR_IN_BATTLE:04X}, "
                f"pokered.sym=0x{sym_in_battle:04X}"
            )

    def require(self, name: str) -> int:
        try:
            return self.symbols[name]
        except KeyError as exc:
            raise KeyError(
                f"필수 Battle RAM 심볼이 pokered.sym에 없습니다: {name}"
            ) from exc

    def optional(self, name: str) -> int | None:
        return self.symbols.get(name)

    def first(self, *names: str) -> int | None:
        for name in names:
            addr = self.optional(name)
            if addr is not None:
                return addr
        return None

    def require_all(self, names: tuple[str, ...]) -> None:
        missing = [name for name in names if name not in self.symbols]
        if missing:
            raise KeyError(
                "276-feature observation에 필요한 Battle RAM 심볼이 "
                f"누락됐습니다: {', '.join(missing)}"
            )


@dataclass(frozen=True)
class ActiveMonRAM:
    species_id: int
    level: int

    hp: int
    max_hp: int
    status_raw: int

    move_ids: tuple[int, int, int, int]
    pp_raw: tuple[int, int, int, int]

    atk_stage: int = 0
    def_stage: int = 0
    spe_stage: int = 0
    special_stage: int = 0
    accuracy_stage: int = 0
    evasion_stage: int = 0

    @property
    def fainted(self) -> bool:
        return self.hp <= 0

    @property
    def hp_ratio(self) -> float:
        if self.max_hp <= 0:
            return 0.0
        return max(0.0, min(1.0, self.hp / self.max_hp))


@dataclass(frozen=True)
class BattleRAMState:
    in_battle: bool
    battle_type: int
    battle_result: int

    player_party_index: int
    enemy_party_index: int

    player: ActiveMonRAM | None
    enemy: ActiveMonRAM | None

    player_selected_move: int
    enemy_selected_move: int
    player_used_move: int
    enemy_used_move: int

    player_move_index: int
    enemy_move_index: int

    action_result_or_took_turn: int
    whose_turn_raw: int

    # observation/action-mask에 쓰는 임시 전투 상태도 active/party RAM과
    # 같은 snapshot에서 읽는다. 관측 생성 중 프레임이 넘어가도 한 decision의
    # recharge/trap/persistent-effect 값이 서로 달라지지 않는다.
    player_battle_status1: int = 0
    player_battle_status2: int = 0
    player_battle_status3: int = 0
    enemy_battle_status1: int = 0
    enemy_battle_status2: int = 0
    enemy_battle_status3: int = 0
    player_substitute_hp: int = 0
    player_disabled_move: int = 0
    enemy_disabled_move: int = 0

    def compact_key(self) -> tuple:
        return (
            self.in_battle,
            self.battle_type,
            self.battle_result,
            self.player_party_index,
            self.enemy_party_index,
            self.player,
            self.enemy,
            self.player_selected_move,
            self.enemy_selected_move,
            self.player_used_move,
            self.enemy_used_move,
            self.player_move_index,
            self.enemy_move_index,
            self.action_result_or_took_turn,
            self.whose_turn_raw,
            self.player_battle_status1,
            self.player_battle_status2,
            self.player_battle_status3,
            self.enemy_battle_status1,
            self.enemy_battle_status2,
            self.enemy_battle_status3,
            self.player_substitute_hp,
            self.player_disabled_move,
            self.enemy_disabled_move,
        )


class BattleRAMReader:
    """PyBoy memory -> BattleRAMState. 이 클래스는 print하지 않는다."""

    def __init__(self, pyboy, sym_path: Path = DEFAULT_SYM_PATH):
        self.pyboy = pyboy
        self.sym = BattleSymbols(sym_path)
        self.sym.require_all(OBSERVATION_RAM_SYMBOLS)

        # Several Gen-1 faint/party routines reuse or temporarily restore the
        # BattleMon work area.  Keep the last slot observed while HP was
        # positive so a fainted Pidgey cannot be reported as the old lead.
        self._last_player_party_index: int | None = None

    def _u8_addr(self, address: int | None, default: int = 0) -> int:
        if address is None:
            return int(default)
        return int(self.pyboy.memory[address]) & 0xFF

    def _u8(self, symbol: str, default: int = 0) -> int:
        return self._u8_addr(self.sym.optional(symbol), default)

    def _u16be_addr(self, address: int | None, default: int = 0) -> int:
        if address is None:
            return int(default)
        hi = int(self.pyboy.memory[address]) & 0xFF
        lo = int(self.pyboy.memory[address + 1]) & 0xFF
        return (hi << 8) | lo

    def _u16be(self, symbol: str, default: int = 0) -> int:
        return self._u16be_addr(self.sym.optional(symbol), default)

    def _bytes_addr(
        self,
        address: int | None,
        count: int,
        default: int = 0,
    ) -> tuple[int, ...]:
        if address is None:
            return tuple(int(default) for _ in range(count))
        return tuple(
            int(self.pyboy.memory[address + i]) & 0xFF
            for i in range(count)
        )

    @staticmethod
    def _modifier_to_stage(raw: int) -> int:
        # Gen1 battle modifier: 7 = neutral, 1..13 = -6..+6.
        if raw <= 0:
            return 0
        return max(-6, min(6, int(raw) - 7))

    def _read_stage(self, symbol: str) -> int:
        addr = self.sym.optional(symbol)
        if addr is None:
            return 0
        return self._modifier_to_stage(self._u8_addr(addr))

    def _party_count(self) -> int:
        return min(6, self._u8("wPartyCount", 0))

    def _party_mon_address(self, party_index: int) -> int | None:
        base = self.sym.optional("wPartyMons")
        party_index = int(party_index)
        if base is None or not 0 <= party_index < self._party_count():
            return None
        return int(base) + party_index * 0x2C

    def _party_species(self, party_index: int) -> int:
        address = self._party_mon_address(party_index)
        return self._u8_addr(address) if address is not None else 0

    def note_player_party_index(self, party_index: int) -> None:
        """Record a macro-confirmed switch target for transient RAM recovery."""
        party_index = int(party_index)
        if 0 <= party_index < self._party_count():
            self._last_player_party_index = party_index

    def _resolve_player_party_index(self, battle_hp: int) -> int:
        # wPlayerMonNumber is the battle engine's authoritative active-party
        # slot. SwitchPlayerMon/ChooseNextMon write it before loading the new
        # BattleMon. wBattleMonPartyPos can remain stale (observed as 0 while
        # party slot 1 Pidgey was active), so retain it only as a fallback for
        # unusual symbol builds.
        player_number_addr = self.sym.optional("wPlayerMonNumber")
        if player_number_addr is not None:
            raw_index = self._u8_addr(player_number_addr)
        else:
            raw_index = self._u8("wBattleMonPartyPos")
        valid_raw = 0 <= raw_index < self._party_count()

        if int(battle_hp) > 0:
            battle_species_addr = self.sym.first(
                "wBattleMonSpecies2",
                "wBattleMonSpecies",
            )
            battle_species = self._u8_addr(battle_species_addr)

            # wBattleMonPartyPos can remain on the fainted slot after a forced
            # replacement.  Accept it only when its party species agrees with
            # the live BattleMon species.
            if (
                valid_raw
                and self._party_species(raw_index) == battle_species
            ):
                self._last_player_party_index = raw_index
                return raw_index

            # Recover the new slot by matching the live BattleMon species to a
            # living party entry.  Prefer the macro-confirmed index for duplicate
            # species, otherwise require a unique match.
            matches = [
                i
                for i in range(self._party_count())
                if self._party_species(i) == battle_species
                and (
                    (addr := self._party_mon_address(i)) is not None
                    and self._u16be_addr(addr + 1) > 0
                )
            ]
            if (
                self._last_player_party_index is not None
                and self._last_player_party_index in matches
            ):
                return self._last_player_party_index
            if len(matches) == 1:
                self._last_player_party_index = matches[0]
                return matches[0]
            if valid_raw:
                self._last_player_party_index = raw_index
                return raw_index

        # During faint handling wBattleMonPartyPos/BattleMon data can be reused.
        # The last slot seen alive is the Pokemon that just fainted.
        if self._last_player_party_index is not None:
            return self._last_player_party_index

        return raw_index if valid_raw else 0

    def _read_player(
        self,
        *,
        party_index: int,
        battle_hp: int,
    ) -> ActiveMonRAM:
        # Gen 1 battle flow keeps the current player's internal species id
        # in wBattleMonSpecies2 before/while loading the active battle mon.
        # Prefer Species2; wBattleMonSpecies can be zero/stale at the snapshot
        # timing used by our logger.
        species_addr = self.sym.first(
            "wBattleMonSpecies2",
            "wBattleMonSpecies",
        )

        # At HP=0 the BattleMon work area can already contain the previous lead
        # while the forced party menu is displayed.  Rebuild the fainted active
        # from its real 44-byte party struct so species, level, moves and PP all
        # describe the Pokemon that actually fainted.
        party_addr = self._party_mon_address(party_index)
        if int(battle_hp) <= 0 and party_addr is not None:
            return ActiveMonRAM(
                species_id=self._u8_addr(party_addr),
                level=self._u8_addr(party_addr + 33),
                hp=0,
                max_hp=self._u16be_addr(party_addr + 34),
                status_raw=self._u8_addr(party_addr + 4),
                move_ids=self._bytes_addr(party_addr + 8, 4),
                pp_raw=self._bytes_addr(party_addr + 29, 4),
                atk_stage=self._read_stage("wPlayerMonAttackMod"),
                def_stage=self._read_stage("wPlayerMonDefenseMod"),
                spe_stage=self._read_stage("wPlayerMonSpeedMod"),
                special_stage=self._read_stage("wPlayerMonSpecialMod"),
                accuracy_stage=self._read_stage("wPlayerMonAccuracyMod"),
                evasion_stage=self._read_stage("wPlayerMonEvasionMod"),
            )

        return ActiveMonRAM(
            species_id=self._u8_addr(species_addr),
            level=self._u8("wBattleMonLevel"),
            hp=int(battle_hp),
            max_hp=self._u16be("wBattleMonMaxHP"),
            status_raw=self._u8("wBattleMonStatus"),
            move_ids=self._bytes_addr(
                self.sym.optional("wBattleMonMoves"),
                4,
            ),
            pp_raw=self._bytes_addr(
                self.sym.optional("wBattleMonPP"),
                4,
            ),
            atk_stage=self._read_stage("wPlayerMonAttackMod"),
            def_stage=self._read_stage("wPlayerMonDefenseMod"),
            spe_stage=self._read_stage("wPlayerMonSpeedMod"),
            special_stage=self._read_stage("wPlayerMonSpecialMod"),
            accuracy_stage=self._read_stage("wPlayerMonAccuracyMod"),
            evasion_stage=self._read_stage("wPlayerMonEvasionMod"),
        )

    def _read_enemy(self) -> ActiveMonRAM:
        # wEnemyMonSpecies2 is the species id the Gen 1 battle intro itself
        # uses for the wild-mon cry/display and for loading enemy data.
        # Prefer Species2 so a previous encounter's wEnemyMonSpecies value
        # cannot leak into the next BATTLE_START snapshot.
        species_addr = self.sym.first(
            "wEnemyMonSpecies2",
            "wEnemyMonSpecies",
        )

        return ActiveMonRAM(
            species_id=self._u8_addr(species_addr),
            level=self._u8("wEnemyMonLevel"),
            hp=self._u16be("wEnemyMonHP"),
            max_hp=self._u16be("wEnemyMonMaxHP"),
            status_raw=self._u8("wEnemyMonStatus"),
            move_ids=self._bytes_addr(
                self.sym.optional("wEnemyMonMoves"),
                4,
            ),
            pp_raw=self._bytes_addr(
                self.sym.optional("wEnemyMonPP"),
                4,
            ),
            atk_stage=self._read_stage("wEnemyMonAttackMod"),
            def_stage=self._read_stage("wEnemyMonDefenseMod"),
            spe_stage=self._read_stage("wEnemyMonSpeedMod"),
            special_stage=self._read_stage("wEnemyMonSpecialMod"),
            accuracy_stage=self._read_stage("wEnemyMonAccuracyMod"),
            evasion_stage=self._read_stage("wEnemyMonEvasionMod"),
        )

    def is_in_battle(self) -> bool:
        return self._u8_addr(ADDR_IN_BATTLE) != 0

    def reset_battle(self) -> None:
        """새 배틀에서 이전 active party 추적값을 버린다."""
        self._last_player_party_index = None

    def read(self) -> BattleRAMState:
        in_battle = self.is_in_battle()

        if not in_battle:
            self._last_player_party_index = None
            return BattleRAMState(
                in_battle=False,
                battle_type=0,
                battle_result=0,
                player_party_index=0,
                enemy_party_index=0,
                player=None,
                enemy=None,
                player_selected_move=0,
                enemy_selected_move=0,
                player_used_move=0,
                enemy_used_move=0,
                player_move_index=0,
                enemy_move_index=0,
                action_result_or_took_turn=0,
                whose_turn_raw=0,
                player_battle_status1=0,
                player_battle_status2=0,
                player_battle_status3=0,
                enemy_battle_status1=0,
                enemy_battle_status2=0,
                enemy_battle_status3=0,
                player_substitute_hp=0,
                player_disabled_move=0,
                enemy_disabled_move=0,
            )

        battle_hp = self._u16be("wBattleMonHP")
        player_party_index = self._resolve_player_party_index(battle_hp)

        return BattleRAMState(
            in_battle=True,
            battle_type=self._u8("wBattleType"),
            battle_result=self._u8("wBattleResult"),
            player_party_index=player_party_index,
            enemy_party_index=self._u8("wEnemyMonPartyPos"),
            player=self._read_player(
                party_index=player_party_index,
                battle_hp=battle_hp,
            ),
            enemy=self._read_enemy(),
            player_selected_move=self._u8("wPlayerSelectedMove"),
            enemy_selected_move=self._u8("wEnemySelectedMove"),
            player_used_move=self._u8("wPlayerUsedMove"),
            enemy_used_move=self._u8("wEnemyUsedMove"),
            player_move_index=self._u8("wPlayerMoveListIndex"),
            enemy_move_index=self._u8("wEnemyMoveListIndex"),
            action_result_or_took_turn=self._u8(
                "wActionResultOrTookBattleTurn"
            ),
            whose_turn_raw=self._u8("hWhoseTurn"),
            player_battle_status1=self._u8("wPlayerBattleStatus1"),
            player_battle_status2=self._u8("wPlayerBattleStatus2"),
            player_battle_status3=self._u8("wPlayerBattleStatus3"),
            enemy_battle_status1=self._u8("wEnemyBattleStatus1"),
            enemy_battle_status2=self._u8("wEnemyBattleStatus2"),
            enemy_battle_status3=self._u8("wEnemyBattleStatus3"),
            player_substitute_hp=self._u8("wPlayerSubstituteHP"),
            player_disabled_move=self._u8("wPlayerDisabledMove"),
            enemy_disabled_move=self._u8("wEnemyDisabledMove"),
        )


def _format_mon(prefix: str, mon: ActiveMonRAM | None) -> str:
    if mon is None:
        return f"{prefix}=None"

    return (
        f"{prefix}(species={mon.species_id}, "
        f"Lv={mon.level}, "
        f"HP={mon.hp}/{mon.max_hp}, "
        f"status=0x{mon.status_raw:02X}, "
        f"moves={list(mon.move_ids)}, "
        f"pp_raw={list(mon.pp_raw)}, "
        f"pp={[value & 0x3F for value in mon.pp_raw]}, "
        f"boosts=[atk {mon.atk_stage:+d}, def {mon.def_stage:+d}, "
        f"spe {mon.spe_stage:+d}, spc {mon.special_stage:+d}, "
        f"acc {mon.accuracy_stage:+d}, eva {mon.evasion_stage:+d}])"
    )


def format_battle_state(
    state: BattleRAMState,
    *,
    label: str | None = None,
) -> str:
    title = f"[BattleRAM:{label}]" if label else "[BattleRAM]"

    if not state.in_battle:
        return f"{title} not in battle"

    return "\n".join(
        [
            (
                f"{title} type={state.battle_type} "
                f"result={state.battle_result} "
                f"party={state.player_party_index} "
                f"enemy_party={state.enemy_party_index}"
            ),
            "  " + _format_mon("PLAYER", state.player),
            "  " + _format_mon("ENEMY ", state.enemy),
            (
                "  action_ram: "
                f"player_selected={state.player_selected_move} "
                f"enemy_selected={state.enemy_selected_move} | "
                f"player_used={state.player_used_move} "
                f"enemy_used={state.enemy_used_move} | "
                f"player_move_index={state.player_move_index} "
                f"enemy_move_index={state.enemy_move_index} | "
                f"took_turn={state.action_result_or_took_turn} "
                f"whose_turn={state.whose_turn_raw} | "
                f"player_disable=0x{state.player_disabled_move:02X} "
                f"enemy_disable=0x{state.enemy_disabled_move:02X}"
            ),
        ]
    )


class BattleActionLogger:
    """
    액션 단위 로그 helper.

    프레임마다 poll해서 출력하지 않는다.
    controller가 원하는 경계에서만 log()를 호출한다.

    예:
        logger.log("BATTLE_START")

        action = policy(...)
        logger.log(f"BEFORE_AGENT action={action}")
        executor.execute(action)

        wait_until_next_decision(...)
        logger.log("AFTER_TURN")
    """

    def __init__(self, reader: BattleRAMReader):
        self.reader = reader
        self._last_key: tuple | None = None

    def reset(self) -> None:
        self._last_key = None

    def log(
        self,
        label: str,
        *,
        only_if_changed: bool = False,
    ) -> BattleRAMState:
        state = self.reader.read()
        key = state.compact_key()

        if not only_if_changed or key != self._last_key:
            print(format_battle_state(state, label=label))

        self._last_key = key
        return state
