"""
battle_macro_v6.py

Pokemon Red Battle-RL용 battle macro.

핵심 원칙
---------
1) wPlayerSelectedMove / wPlayerUsedMove / hWhoseTurn /
   wActionResultOrTookBattleTurn 같은 lazy RAM을
   "현재 플레이어가 선택할 차례인가" 판정에 사용하지 않는다.

2) 같은 wMenuCursorLocation 값이 서로 다른 배틀 장면에서 재사용될 수 있으므로
   cursor 값 하나만으로 메뉴 의미를 판단하지 않는다.

   예:
       0xCC30~31 == A9 C4

   이 값은 기술 메뉴 move0 위치에서도 보일 수 있고,
   크리티컬/미스 등의 전투 진행 중에도 보일 수 있다.

   따라서:
       phase == OPEN_FIGHT / MOVE_MENU
           -> A9는 기술 메뉴로 해석

       phase == TURN_RESOLUTION
           -> A9가 보여도 기술 메뉴로 해석하지 않음
           -> 전투 텍스트를 A로 진행

3) 한 action이 끝난 뒤 RL에게 주도권을 돌려주는 조건:
       실제 메인 배틀 메뉴 cursor
       C1 / C7 / E9 / EF
   가 입력 없이 연속 stable_frames 동안 다시 나타난 경우.

4) 교체의 경우 추가로:
       선택한 party species == 현재 BattleMon species
   가 확인된 뒤 메인 메뉴가 돌아와야 완료.

사용자가 확인한 RAM
-------------------
0xD11D
    first-mon flag: 배틀 첫 등장 과정에서 0 -> 1 -> 0

0xCC30
    wMenuCursorLocation LOW BYTE

    MAIN:
        FIGHT C1
        PKMN  C7
        ITEM  E9
        RUN   EF

    FIGHT move:
        0 A9
        1 BD
        2 D1
        3 E5

    PKMN:
        0 B4
        1 DC
        2 04
        3 2C
        4 54
        5 7C

    상대 포켓몬 기절 후 SHIFT 질문:
        YES 41
        NO  69

    Showdown 학습 환경과 동일하게 이 질문에서는 항상 NO를 선택한다.

0xD05E ~ 0xD05F
    사용자가 크리티컬/미스 등 전투 진행 중 변화하는 것을 관찰한 보조 RAM.
    정확한 의미를 여기서 새로 추측하지 않고,
    TURN_RESOLUTION 중 non-zero이면 A 진행을 우선하는 힌트로만 사용한다.

Action
------
0~3 : move 0~3
4~9 : switch party 0~5
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Observed RAM addresses
# ---------------------------------------------------------------------------

ADDR_FIRST_MON_YET = 0xD11D
ADDR_LAST_MENU_ITEM = 0xCC2A

# wMenuCursorLocation pointer bytes.
ADDR_MENU_CURSOR_LO = 0xCC30
ADDR_MENU_CURSOR_HI = 0xCC31

# 사용자가 관찰한 크리티컬/미스 관련 보조 RAM.
# 정확한 심볼 의미를 이 코드에서 임의로 단정하지 않는다.
ADDR_RESOLUTION_FLAG_0 = 0xD05E
ADDR_RESOLUTION_FLAG_1 = 0xD05F

# lazy RAM — 디버그용으로만 유지.
ADDR_PLAYER_SELECTED_MOVE = 0xCCDC
ADDR_PLAYER_USED_MOVE = 0xCCF1


# ---------------------------------------------------------------------------
# Observed cursor LOW BYTE values
# ---------------------------------------------------------------------------

MAIN_FIGHT = 0xC1
MAIN_PKMN = 0xC7
MAIN_ITEM = 0xE9
MAIN_RUN = 0xEF

MAIN_CURSORS = (
    MAIN_FIGHT,
    MAIN_PKMN,
    MAIN_ITEM,
    MAIN_RUN,
)

MAIN_CURSOR_TO_COORD = {
    MAIN_FIGHT: (0, 0),
    MAIN_PKMN:  (1, 0),
    MAIN_ITEM:  (0, 1),
    MAIN_RUN:   (1, 1),
}

MAIN_COORD_TO_CURSOR = {
    coord: cursor
    for cursor, coord in MAIN_CURSOR_TO_COORD.items()
}

MOVE_CURSORS = (
    0xA9,
    0xBD,
    0xD1,
    0xE5,
)

MOVE_CURSOR_PAIRS = tuple((cursor, 0xC4) for cursor in MOVE_CURSORS)

# Level-up "Which move should be forgotten?" menu. learn_move.asm places
# the first cursor at tilemap (x=5, y=8), then HandleMenuInput advances one
# SCREEN_WIDTH (0x14) per move while BIT_DOUBLE_SPACED_MENU is set.
MOVE_LEARN_CURSOR_PAIRS = (
    (0x45, 0xC4),
    (0x59, 0xC4),
    (0x6D, 0xC4),
    (0x81, 0xC4),
)

# Cursor seen immediately before the four-choice battle menu becomes ready.
INITIAL_BATTLE_CURSOR_PAIR = (0xC3, 0xC4)

PARTY_CURSORS = (
    0xB4,
    0xDC,
    0x04,
    0x2C,
    0x54,
    0x7C,
)

# Full wMenuCursorLocation pointer pairs for the forced/normal party screen.
# LOW alone is not sufficient because the pointer crosses C3FF between slots
# 1 and 2 and the same LOW values can be reused by other menus.
PARTY_CURSOR_PAIRS = (
    (0xB4, 0xC3),
    (0xDC, 0xC3),
    (0x04, 0xC4),
    (0x2C, 0xC4),
    (0x54, 0xC4),
    (0x7C, 0xC4),
)

MAIN_CURSOR_PAIRS = (
    (MAIN_FIGHT, 0xC4),
    (MAIN_PKMN, 0xC4),
    (MAIN_ITEM, 0xC4),
    (MAIN_RUN, 0xC4),
)

MENU_INITIAL = "INITIAL"
MENU_MAIN = "MAIN"
MENU_MOVE = "MOVE"
MENU_PARTY = "PARTY"
MENU_UNKNOWN = "UNKNOWN"

# TrainerAboutToUseText 뒤의 YES/NO 메뉴.
# wMenuCursorLocation은 VRAM 주소이므로 HIGH BYTE까지 함께 확인해
# 다른 장면에서 LOW BYTE 41/69가 우연히 재사용되는 경우를 피한다.
SHIFT_PROMPT_CURSOR_HI = 0xC4
SHIFT_YES = 0x41
SHIFT_NO = 0x69
SHIFT_CURSORS = (
    SHIFT_YES,
    SHIFT_NO,
)


# ---------------------------------------------------------------------------
# Phase
# ---------------------------------------------------------------------------

PHASE_IDLE = "IDLE"
PHASE_FIRST_POKEMON = "FIRST_POKEMON"
PHASE_READY = "READY"

PHASE_OPEN_FIGHT = "OPEN_FIGHT"
PHASE_MOVE_MENU = "MOVE_MENU"
PHASE_CONFIRM_MOVE = "CONFIRM_MOVE"

PHASE_OPEN_PKMN = "OPEN_PKMN"
PHASE_PARTY_MENU = "PARTY_MENU"
PHASE_CONFIRM_SWITCH = "CONFIRM_SWITCH"
PHASE_FORCED_SWITCH_READY = "FORCED_SWITCH_READY"
PHASE_CONFIRM_FORCED_SWITCH = "CONFIRM_FORCED_SWITCH"
PHASE_OPEN_RUN = "OPEN_RUN"

PHASE_TURN_RESOLUTION = "TURN_RESOLUTION"
PHASE_DECLINE_SHIFT = "DECLINE_SHIFT"
PHASE_MOVE_LEARN_HANDOFF = "MOVE_LEARN_HANDOFF"
PHASE_BATTLE_END = "BATTLE_END"


# ---------------------------------------------------------------------------
# Exceptions / debug
# ---------------------------------------------------------------------------

class BattleMacroError(RuntimeError):
    pass


class BattleMacroTimeout(BattleMacroError):
    pass


class BattleMacroStopped(BattleMacroError):
    pass


class BattleManualHandoff(BattleMacroError):
    """Intentional stop at a menu that must be controlled by the user."""

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        party_index: int,
        new_move_id: int,
        current_move_ids: tuple[int, int, int, int],
        selected_slot: int,
        cursor_pair: tuple[int, int],
    ):
        super().__init__(message)
        self.reason = str(reason)
        self.party_index = int(party_index)
        self.new_move_id = int(new_move_id)
        self.current_move_ids = tuple(int(v) for v in current_move_ids)
        self.selected_slot = int(selected_slot)
        self.cursor_pair = (
            int(cursor_pair[0]) & 0xFF,
            int(cursor_pair[1]) & 0xFF,
        )


@dataclass(frozen=True)
class MacroDebugState:
    phase: str
    first_mon_yet: int
    last_menu_item: int
    menu_cursor_lo: int
    menu_cursor_hi: int
    resolution_flag_0: int
    resolution_flag_1: int
    selected_move: int
    used_move: int


class BattleMacro:
    """
    PyBoy 버튼 입력 + 관찰된 battle RAM을 이용한 상태 기반 macro.
    """

    def __init__(
        self,
        pyboy,
        battle_ram=None,
        *,
        menu_timeout_frames: int = 300,
        decision_stable_frames: int = 3,
        direction_hold_max_frames: int = 24,
        confirm_hold_frames: int = 8,
        move_tap_hold_frames: int = 1,
        move_tap_release_frames: int = 4,
        move_cursor_stable_frames: int = 2,
        move_step_retries: int = 4,
        resolution_a_hold_frames: int = 6,
        resolution_a_gap_frames: int = 10,
        resolution_event_hold_timeout: int = 120,
        resolution_release_frames: int = 3,
        turn_timeout_frames: int = 2400,
        debug: bool = False,
    ):
        self.pyboy = pyboy
        self.battle_ram = battle_ram

        self.menu_timeout_frames = max(1, int(menu_timeout_frames))
        self.decision_stable_frames = max(1, int(decision_stable_frames))
        self.direction_hold_max_frames = max(
            1, int(direction_hold_max_frames)
        )
        self.confirm_hold_frames = max(1, int(confirm_hold_frames))

        # --------------------------------------------------------------
        # MOVE menu 전용 방향 입력.
        #
        # move0/1은 기존 hold 방식에서도 잘 동작했지만 move2/3처럼
        # 연속으로 2~3칸 이동할 때 다음 DOWN edge가 누락되거나
        # cursor RAM의 순간값을 너무 빨리 잡는 경우를 피하기 위해
        # 짧은 tap + 충분한 release + stable 확인을 사용한다.
        # PKMN 메뉴는 기존 로직을 그대로 유지한다.
        # --------------------------------------------------------------
        self.move_tap_hold_frames = max(
            1, int(move_tap_hold_frames)
        )
        self.move_tap_release_frames = max(
            2, int(move_tap_release_frames)
        )
        self.move_cursor_stable_frames = max(
            1, int(move_cursor_stable_frames)
        )
        self.move_step_retries = max(
            1, int(move_step_retries)
        )

        # test_map.py Navigation의 battle A 처리와 비슷한 pulse 구조.
        self.resolution_a_hold_frames = max(
            1, int(resolution_a_hold_frames)
        )
        self.resolution_a_gap_frames = max(
            1, int(resolution_a_gap_frames)
        )
        self.resolution_event_hold_timeout = max(
            1, int(resolution_event_hold_timeout)
        )
        self.resolution_release_frames = max(
            1, int(resolution_release_frames)
        )
        self.turn_timeout_frames = max(1, int(turn_timeout_frames))

        self.debug = bool(debug)
        self.phase = PHASE_IDLE
        self._confirmed_player_party_index: int | None = None
        # Same A9 C4 cursor is reused for MOVE and battle text.  Do not B-out
        # of an apparent MOVE/PARTY menu until this battle has shown MAIN.
        self._seen_main_this_battle = False

        # 주소는 기존 BattleRAMReader가 쓰는 .sym에서만 찾는다.
        self.addr_action_result = self._symbol_address(
            "wActionResultOrTookBattleTurn"
        )
        self.addr_battle_species = (
            self._symbol_address("wBattleMonSpecies2")
            or self._symbol_address("wBattleMonSpecies")
        )
        self.addr_party_species = self._symbol_address("wPartySpecies")
        self.addr_party_count = self._symbol_address("wPartyCount")
        self.addr_party_mons = self._symbol_address("wPartyMons")
        self.addr_battle_hp = self._symbol_address("wBattleMonHP")
        self.addr_battle_party_pos = self._symbol_address(
            "wBattleMonPartyPos"
        )
        self.addr_player_mon_number = self._symbol_address(
            "wPlayerMonNumber"
        )
        self.addr_in_battle = self._symbol_address("wIsInBattle")
        self.addr_move_num = self._symbol_address("wMoveNum")
        self.addr_moves_buffer = self._symbol_address("wMoves")
        self.addr_which_pokemon = self._symbol_address("wWhichPokemon")
        self.addr_num_moves_minus_one = self._symbol_address(
            "wNumMovesMinusOne"
        )
        self.addr_top_menu_y = self._symbol_address("wTopMenuItemY")
        self.addr_top_menu_x = self._symbol_address("wTopMenuItemX")
        self.addr_current_menu_item = self._symbol_address(
            "wCurrentMenuItem"
        )
        self.addr_max_menu_item = self._symbol_address("wMaxMenuItem")
        self.addr_menu_watched_keys = self._symbol_address(
            "wMenuWatchedKeys"
        )

        if self.debug:
            print(
                "[macro-v5] symbols "
                f"action_result={self._addr_text(self.addr_action_result)} "
                f"battle_species={self._addr_text(self.addr_battle_species)} "
                f"party_species={self._addr_text(self.addr_party_species)} "
                f"in_battle={self._addr_text(self.addr_in_battle)}"
            )

    def reset_battle(self) -> None:
        """새 배틀 전에 매크로의 이전 메뉴/파티 상태를 초기화한다."""
        for button in ("up", "down", "left", "right", "a", "b"):
            self._release(button)
        self.phase = PHASE_IDLE
        self._confirmed_player_party_index = None
        self._seen_main_this_battle = False

    # ------------------------------------------------------------------
    # RAM
    # ------------------------------------------------------------------

    def _read_u8(self, address: int) -> int:
        return int(self.pyboy.memory[address]) & 0xFF

    def _read_u16be(self, address: int) -> int:
        return (
            (self._read_u8(address) << 8)
            | self._read_u8(address + 1)
        )

    def _symbol_address(self, name: str) -> int | None:
        if self.battle_ram is None:
            return None
        try:
            address = self.battle_ram.sym.optional(name)
        except Exception:
            return None
        if address is None:
            return None
        return int(address) & 0xFFFF

    @staticmethod
    def _addr_text(address: int | None) -> str:
        if address is None:
            return "None"
        return f"0x{address:04X}"

    def first_mon_value(self) -> int:
        return self._read_u8(ADDR_FIRST_MON_YET)

    def last_menu_item(self) -> int:
        return self._read_u8(ADDR_LAST_MENU_ITEM)

    def menu_cursor(self) -> int:
        """wMenuCursorLocation LOW byte."""
        return self._read_u8(ADDR_MENU_CURSOR_LO)

    def menu_cursor_pair(self) -> tuple[int, int]:
        """터미널에서 보던 A9 C4 형태 그대로 반환."""
        return (
            self._read_u8(ADDR_MENU_CURSOR_LO),
            self._read_u8(ADDR_MENU_CURSOR_HI),
        )

    @staticmethod
    def _party_cursor_index(pair: tuple[int, int]) -> int | None:
        """Return party slot for a verified B4/DC/04/... cursor pair."""
        lo, hi = (int(pair[0]) & 0xFF, int(pair[1]) & 0xFF)
        if lo not in PARTY_CURSORS or hi not in (0xC3, 0xC4):
            return None

        index = PARTY_CURSORS.index(lo)
        # Slots 0/1 are known C3 pointers.  Later slots normally carry into C4;
        # accepting C3 too keeps compatibility with ROM builds that update the
        # menu row byte separately, while the unique LOW still identifies slot.
        if index < 2 and hi != 0xC3:
            return None
        return index

    def menu_context(self) -> str:
        """Classify the current battle menu using both cursor bytes."""
        pair = self.menu_cursor_pair()
        if pair == INITIAL_BATTLE_CURSOR_PAIR:
            return MENU_INITIAL
        if pair in MAIN_CURSOR_PAIRS:
            return MENU_MAIN
        if pair in MOVE_CURSOR_PAIRS:
            return MENU_MOVE
        if self._party_cursor_index(pair) is not None:
            return MENU_PARTY
        return MENU_UNKNOWN

    def resolution_flags(self) -> tuple[int, int]:
        return (
            self._read_u8(ADDR_RESOLUTION_FLAG_0),
            self._read_u8(ADDR_RESOLUTION_FLAG_1),
        )

    def player_selected_move(self) -> int:
        return self._read_u8(ADDR_PLAYER_SELECTED_MOVE)

    def player_used_move(self) -> int:
        return self._read_u8(ADDR_PLAYER_USED_MOVE)

    def action_result_or_took_turn(self) -> int | None:
        # lazy RAM이므로 debug에서만 사용.
        if self.addr_action_result is None:
            return None
        return self._read_u8(self.addr_action_result)

    def active_player_species(self) -> int:
        if self.addr_battle_species is None:
            raise BattleMacroError(
                "wBattleMonSpecies/wBattleMonSpecies2 주소를 "
                "battle_ram.sym에서 찾지 못했습니다."
            )
        return self._read_u8(self.addr_battle_species)

    def active_player_hp(self) -> int:
        if self.addr_battle_hp is None:
            raise BattleMacroError(
                "wBattleMonHP 주소를 battle_ram.sym에서 찾지 못했습니다."
            )
        return self._read_u16be(self.addr_battle_hp)

    def active_player_party_index(self) -> int:
        # The battle engine writes wPlayerMonNumber whenever the active party
        # member changes. wBattleMonPartyPos may be stale after a switch, so it
        # is only a compatibility fallback.
        if self.addr_player_mon_number is not None:
            return self._read_u8(self.addr_player_mon_number)
        if self.addr_battle_party_pos is None:
            raise BattleMacroError(
                "wPlayerMonNumber/wBattleMonPartyPos 주소를 "
                "battle_ram.sym에서 찾지 못했습니다."
            )
        return self._read_u8(self.addr_battle_party_pos)

    def resolved_active_player_party_index(self) -> int:
        """Resolve a stale forced-switch party position by live species."""
        raw_index = self.active_player_party_index()
        active_species = self.active_player_species()

        if (
            0 <= raw_index < self.party_count()
            and self.party_species(raw_index) == active_species
        ):
            return raw_index

        confirmed = self._confirmed_player_party_index
        if (
            confirmed is not None
            and 0 <= confirmed < self.party_count()
            and self.party_species(confirmed) == active_species
            and self.party_hp(confirmed) > 0
        ):
            return confirmed

        matches = [
            i
            for i in range(self.party_count())
            if self.party_species(i) == active_species
            and self.party_hp(i) > 0
        ]
        if len(matches) == 1:
            return matches[0]
        return raw_index

    def _note_player_party_index(self, party_index: int) -> None:
        self._confirmed_player_party_index = int(party_index)
        note = getattr(self.battle_ram, "note_player_party_index", None)
        if callable(note):
            note(int(party_index))

    def party_species(self, party_index: int) -> int:
        party_index = int(party_index)
        if not 0 <= party_index <= 5:
            raise ValueError(
                f"party_index must be 0..5, got {party_index}"
            )

        if self.addr_party_species is None:
            raise BattleMacroError(
                "wPartySpecies 주소를 battle_ram.sym에서 찾지 못했습니다."
            )

        return self._read_u8(self.addr_party_species + party_index)

    def party_hp(self, party_index: int) -> int:
        party_index = int(party_index)
        if not 0 <= party_index <= 5:
            raise ValueError(
                f"party_index must be 0..5, got {party_index}"
            )
        if self.addr_party_mons is None:
            raise BattleMacroError(
                "wPartyMons 주소를 battle_ram.sym에서 찾지 못했습니다."
            )
        # Gen-1 PartyMon: 44-byte struct, HP at offset +1 (big endian).
        return self._read_u16be(
            self.addr_party_mons + party_index * 0x2C + 1
        )

    def party_count(self) -> int:
        if self.addr_party_count is None:
            return 6
        return min(6, self._read_u8(self.addr_party_count))

    def has_available_party_switch(self) -> bool:
        active_index = self.active_player_party_index()
        return any(
            i != active_index
            and self.party_species(i) not in (0x00, 0xFF)
            and self.party_hp(i) > 0
            for i in range(self.party_count())
        )

    def in_battle(self) -> bool | None:
        if self.addr_in_battle is None:
            return None
        return self._read_u8(self.addr_in_battle) != 0

    def debug_state(self) -> MacroDebugState:
        lo, hi = self.menu_cursor_pair()
        f0, f1 = self.resolution_flags()

        return MacroDebugState(
            phase=self.phase,
            first_mon_yet=self.first_mon_value(),
            last_menu_item=self.last_menu_item(),
            menu_cursor_lo=lo,
            menu_cursor_hi=hi,
            resolution_flag_0=f0,
            resolution_flag_1=f1,
            selected_move=self.player_selected_move(),
            used_move=self.player_used_move(),
        )

    # ------------------------------------------------------------------
    # PyBoy helpers
    # ------------------------------------------------------------------

    def _tick(self, count: int = 1) -> None:
        for _ in range(max(0, int(count))):
            if not self.pyboy.tick():
                raise BattleMacroStopped("PyBoy emulation stopped")

            # The replacement menu can appear in the middle of a held-A text
            # advance (level-up/EXP sequence). Check after every macro-owned
            # frame so the next input edge can never confirm move0 before the
            # outer resolution loop gets another iteration.
            if hasattr(self, "addr_move_num"):
                move_learn_info = self._move_learn_handoff_info()
                if move_learn_info is not None:
                    self._handoff_move_learning_to_user(move_learn_info)

    def _release(self, button: str) -> None:
        try:
            self.pyboy.button_release(button)
        except Exception:
            pass

    def _hold_fixed(
        self,
        button: str,
        frames: int,
        *,
        label: str,
    ) -> None:
        """
        Navigation의 A 입력처럼 명시적으로 press -> hold -> release.
        """
        frames = max(1, int(frames))

        if self.debug:
            lo, hi = self.menu_cursor_pair()
            print(
                f"[macro-v5] {label}: press {button} "
                f"cursor={lo:02X} {hi:02X} hold={frames}f"
            )

        self.pyboy.button_press(button)
        try:
            for _ in range(frames):
                self._tick(1)
        finally:
            self._release(button)

        # release를 게임에 반영하는 중립 1 frame.
        self._tick(1)

    def _press_until_cursor(
        self,
        button: str,
        expected_cursor: int,
        *,
        timeout_frames: int | None = None,
        label: str,
    ) -> int:
        """
        Navigation 방식과 동일한 핵심:
            button_press
            -> 매 frame RAM 확인
            -> 실제 cursor가 목표값이 된 순간 바로 release
        """
        if timeout_frames is None:
            timeout_frames = self.direction_hold_max_frames

        expected_cursor &= 0xFF

        self.pyboy.button_press(button)
        try:
            for frame in range(max(1, int(timeout_frames))):
                self._tick(1)
                cursor = self.menu_cursor()

                if cursor == expected_cursor:
                    if self.debug:
                        lo, hi = self.menu_cursor_pair()
                        print(
                            f"[macro-v5] {label}: accepted "
                            f"after={frame + 1}f cursor={lo:02X} {hi:02X}"
                        )
                    return cursor
        finally:
            self._release(button)

        raise BattleMacroTimeout(
            f"{label} timeout: "
            f"current=0x{self.menu_cursor():02X}, "
            f"expected=0x{expected_cursor:02X}"
        )

    def _press_until_cursor_pair(
        self,
        button: str,
        expected_pair: tuple[int, int],
        *,
        timeout_frames: int | None = None,
        label: str,
    ) -> tuple[int, int]:
        """Move a menu cursor and verify both pointer bytes."""
        if timeout_frames is None:
            timeout_frames = self.direction_hold_max_frames

        expected_pair = (
            int(expected_pair[0]) & 0xFF,
            int(expected_pair[1]) & 0xFF,
        )

        self.pyboy.button_press(button)
        try:
            for frame in range(max(1, int(timeout_frames))):
                self._tick(1)
                pair = self.menu_cursor_pair()
                if pair == expected_pair:
                    if self.debug:
                        print(
                            f"[macro-v6] {label}: accepted "
                            f"after={frame + 1}f "
                            f"cursor={pair[0]:02X} {pair[1]:02X}"
                        )
                    return pair
        finally:
            self._release(button)

        lo, hi = self.menu_cursor_pair()
        raise BattleMacroTimeout(
            f"{label} timeout: current={lo:02X} {hi:02X}, "
            f"expected={expected_pair[0]:02X} {expected_pair[1]:02X}"
        )

    def _wait_party_cursor_index(
        self,
        *,
        timeout_frames: int | None = None,
        stable_frames: int | None = None,
        label: str = "party menu",
    ) -> int:
        if timeout_frames is None:
            timeout_frames = self.menu_timeout_frames
        if stable_frames is None:
            stable_frames = self.decision_stable_frames

        previous = None
        stable = 0
        for _ in range(max(1, int(timeout_frames))):
            pair = self.menu_cursor_pair()
            party_index = self._party_cursor_index(pair)
            if party_index is not None:
                if pair == previous:
                    stable += 1
                else:
                    stable = 1
                if stable >= max(1, int(stable_frames)):
                    if self.debug:
                        print(
                            f"[macro-v6] {label}: ready "
                            f"cursor={pair[0]:02X} {pair[1]:02X}"
                        )
                    return party_index
            else:
                stable = 0
            previous = pair
            self._tick(1)

        lo, hi = self.menu_cursor_pair()
        raise BattleMacroTimeout(
            f"{label} timeout: cursor={lo:02X} {hi:02X}"
        )

    def _press_until_cursor_in(
        self,
        button: str,
        allowed,
        *,
        timeout_frames: int | None = None,
        label: str,
    ) -> int:
        if timeout_frames is None:
            timeout_frames = self.menu_timeout_frames

        allowed = {int(v) & 0xFF for v in allowed}

        self.pyboy.button_press(button)
        try:
            for frame in range(max(1, int(timeout_frames))):
                self._tick(1)
                cursor = self.menu_cursor()

                if cursor in allowed:
                    if self.debug:
                        lo, hi = self.menu_cursor_pair()
                        print(
                            f"[macro-v5] {label}: accepted "
                            f"after={frame + 1}f cursor={lo:02X} {hi:02X}"
                        )
                    return cursor
        finally:
            self._release(button)

        raise BattleMacroTimeout(
            f"{label} timeout: "
            f"current=0x{self.menu_cursor():02X}, "
            f"allowed={[f'0x{x:02X}' for x in sorted(allowed)]}"
        )

    def _wait_cursor_in(
        self,
        allowed,
        *,
        timeout_frames: int | None = None,
        stable_frames: int = 1,
        label: str,
    ) -> int:
        allowed = {int(v) & 0xFF for v in allowed}

        if timeout_frames is None:
            timeout_frames = self.menu_timeout_frames

        stable = 0
        previous = None

        for _ in range(max(1, int(timeout_frames))):
            cursor = self.menu_cursor()

            if cursor in allowed:
                if cursor == previous:
                    stable += 1
                else:
                    stable = 1

                if stable >= max(1, int(stable_frames)):
                    if self.debug:
                        lo, hi = self.menu_cursor_pair()
                        print(
                            f"[macro-v5] {label}: ready "
                            f"cursor={lo:02X} {hi:02X}"
                        )
                    return cursor
            else:
                stable = 0

            previous = cursor
            self._tick(1)

        raise BattleMacroTimeout(
            f"{label} timeout: current=0x{self.menu_cursor():02X}"
        )

    # ------------------------------------------------------------------
    # MOVE menu robust cursor helpers
    # ------------------------------------------------------------------

    def _wait_move_cursor_index(
        self,
        *,
        timeout_frames: int | None = None,
        stable_frames: int | None = None,
        label: str = "move menu",
    ) -> int:
        if timeout_frames is None:
            timeout_frames = self.menu_timeout_frames
        if stable_frames is None:
            stable_frames = self.move_cursor_stable_frames

        stable = 0
        previous = None
        for _ in range(max(1, int(timeout_frames))):
            pair = self.menu_cursor_pair()
            if pair in MOVE_CURSOR_PAIRS:
                if pair == previous:
                    stable += 1
                else:
                    stable = 1
                if stable >= max(1, int(stable_frames)):
                    if self.debug:
                        print(
                            f"[move-v6] {label}: ready "
                            f"cursor={pair[0]:02X} {pair[1]:02X}"
                        )
                    return MOVE_CURSOR_PAIRS.index(pair)
            else:
                stable = 0
            previous = pair
            self._tick(1)

        lo, hi = self.menu_cursor_pair()
        raise BattleMacroTimeout(
            f"{label} timeout: cursor={lo:02X} {hi:02X}"
        )

    def _wait_exact_cursor_stable(
        self,
        expected_cursor: int,
        *,
        stable_frames: int | None = None,
        timeout_frames: int | None = None,
        label: str,
    ) -> int:
        """
        입력을 누르지 않은 상태에서 expected_cursor가 연속으로
        stable_frames 유지되는지 확인한다.

        MOVE menu의 D1/E5를 순간적으로 한 frame 본 것만으로
        이동 완료라고 판단하지 않기 위한 검증용.
        """
        expected_cursor &= 0xFF

        if stable_frames is None:
            stable_frames = self.move_cursor_stable_frames

        if timeout_frames is None:
            timeout_frames = self.direction_hold_max_frames

        stable_frames = max(1, int(stable_frames))
        timeout_frames = max(stable_frames, int(timeout_frames))

        stable = 0

        for frame in range(timeout_frames):
            cursor, cursor_hi = self.menu_cursor_pair()

            if cursor == expected_cursor and cursor_hi == 0xC4:
                stable += 1

                if stable >= stable_frames:
                    if self.debug:
                        lo, hi = self.menu_cursor_pair()
                        print(
                            f"[move-v6] {label}: stable "
                            f"after={frame + 1}f "
                            f"cursor={lo:02X} {hi:02X} "
                            f"stable={stable}"
                        )
                    return cursor
            else:
                stable = 0

            self._tick(1)

        lo, hi = self.menu_cursor_pair()
        raise BattleMacroTimeout(
            f"{label} stable timeout: "
            f"current={lo:02X} {hi:02X}, "
            f"expected=0x{expected_cursor:02X}"
        )

    def _move_menu_step(
        self,
        button: str,
        expected_cursor: int,
        *,
        label: str,
    ) -> int:
        """
        MOVE menu 한 칸 이동.

        기존 _press_until_cursor()는 방향키를 목표 RAM이 보일 때까지
        계속 hold했다. v6에서는 MOVE menu만 다음처럼 처리한다.

            방향키 완전 release
            -> neutral frames
            -> 1-frame 수준의 짧은 tap
            -> release
            -> neutral frames
            -> expected cursor stable 확인

        한 번에 정확히 한 칸만 이동시키므로 move2/move3의
        연속 DOWN 입력도 각각 독립적인 key edge가 된다.
        """
        expected_cursor &= 0xFF

        if button not in ("up", "down"):
            raise ValueError(
                f"MOVE menu direction must be up/down, got {button!r}"
            )

        opposite = "up" if button == "down" else "down"

        for attempt in range(1, self.move_step_retries + 1):
            # 이전 방향 입력이 다음 step으로 이어지지 않도록 둘 다 release.
            self._release(button)
            self._release(opposite)

            for _ in range(self.move_tap_release_frames):
                self._tick(1)

            before_lo, before_hi = self.menu_cursor_pair()

            if before_lo == expected_cursor and before_hi == 0xC4:
                return self._wait_exact_cursor_stable(
                    expected_cursor,
                    label=f"{label} already-target",
                )

            if self.debug:
                print(
                    f"[move-v6] {label}: attempt={attempt} "
                    f"tap={button} "
                    f"before={before_lo:02X} {before_hi:02X} "
                    f"expected={expected_cursor:02X}"
                )

            self.pyboy.button_press(button)
            try:
                for _ in range(self.move_tap_hold_frames):
                    self._tick(1)
            finally:
                self._release(button)

            # 반드시 새 key edge가 생기도록 release 구간 확보.
            for _ in range(self.move_tap_release_frames):
                self._tick(1)

            # 짧은 tap이 실제 cursor 이동으로 반영될 시간을 주고
            # 목표 cursor가 안정됐는지 확인한다.
            stable = 0
            for frame in range(self.direction_hold_max_frames):
                cursor, cursor_hi = self.menu_cursor_pair()

                if cursor == expected_cursor and cursor_hi == 0xC4:
                    stable += 1
                    if stable >= self.move_cursor_stable_frames:
                        if self.debug:
                            lo, hi = self.menu_cursor_pair()
                            print(
                                f"[move-v6] {label}: accepted "
                                f"attempt={attempt} "
                                f"after={frame + 1}f "
                                f"cursor={lo:02X} {hi:02X}"
                            )
                        return cursor
                else:
                    stable = 0

                self._tick(1)

            if self.debug:
                lo, hi = self.menu_cursor_pair()
                print(
                    f"[move-v6] {label}: retry "
                    f"attempt={attempt} "
                    f"current={lo:02X} {hi:02X}"
                )

        lo, hi = self.menu_cursor_pair()
        raise BattleMacroTimeout(
            f"{label} failed after {self.move_step_retries} taps: "
            f"current={lo:02X} {hi:02X}, "
            f"expected=0x{expected_cursor:02X}"
        )

    def _move_menu_cursor_to(
        self,
        target_index: int,
        *,
        current_index: int | None = None,
    ) -> int:
        """
        MOVE_CURSORS:
            0 -> A9
            1 -> BD
            2 -> D1
            3 -> E5

        현재 위치에서 target까지 반드시 한 칸씩 독립 tap으로 이동하고
        매 step마다 실제 RAM cursor를 검증한다.
        """
        target_index = int(target_index)

        if not 0 <= target_index <= 3:
            raise ValueError(
                f"target move index must be 0..3, got {target_index}"
            )

        if current_index is None:
            current_index = self._wait_move_cursor_index(
                stable_frames=self.move_cursor_stable_frames,
                label="move menu current",
            )
        else:
            current_index = int(current_index)

        while current_index < target_index:
            next_index = current_index + 1
            self._move_menu_step(
                "down",
                MOVE_CURSORS[next_index],
                label=f"move {current_index}->{next_index}",
            )
            current_index = next_index

        while current_index > target_index:
            next_index = current_index - 1
            self._move_menu_step(
                "up",
                MOVE_CURSORS[next_index],
                label=f"move {current_index}->{next_index}",
            )
            current_index = next_index

        expected = MOVE_CURSORS[target_index]

        # A를 누르기 직전에 최종 위치를 다시 한번 무입력 상태로 검증.
        self._wait_exact_cursor_stable(
            expected,
            stable_frames=self.move_cursor_stable_frames,
            label=f"move target {target_index}",
        )

        actual, actual_hi = self.menu_cursor_pair()
        if actual != expected or actual_hi != 0xC4:
            lo, hi = self.menu_cursor_pair()
            raise BattleMacroError(
                f"move target verification failed: "
                f"target={target_index} expected=0x{expected:02X} "
                f"actual={lo:02X} {hi:02X}"
            )

        return current_index

    def _confirm_move_fresh_a(
        self,
        move_index: int,
    ) -> None:
        """
        MOVE menu에서 최종 cursor를 검증한 뒤 fresh A edge로 확정.

        move2/3에서 방향키 release와 A confirm이 너무 붙지 않도록
        A를 먼저 명시적으로 release하고 neutral frames를 둔다.
        """
        move_index = int(move_index)
        expected = MOVE_CURSORS[move_index]

        self._release("up")
        self._release("down")
        self._release("a")

        for _ in range(self.move_tap_release_frames):
            self._tick(1)

        self._wait_exact_cursor_stable(
            expected,
            stable_frames=self.move_cursor_stable_frames,
            label=f"pre-confirm move {move_index}",
        )

        if self.debug:
            lo, hi = self.menu_cursor_pair()
            print(
                "[move-v6] CONFIRM fresh-A "
                f"index={move_index} "
                f"cursor={lo:02X} {hi:02X} "
                f"selected(old)=0x{self.player_selected_move():02X} "
                f"used(old)=0x{self.player_used_move():02X}"
            )

        self._hold_fixed(
            "a",
            self.confirm_hold_frames,
            label=f"confirm move {move_index}",
        )


    # ------------------------------------------------------------------
    # First Pokemon
    # ------------------------------------------------------------------

    def first_pokemon(
        self,
        *,
        timeout_frames: int = 1200,
        b_hold_frames: int = 4,
        b_gap_frames: int = 6,
    ) -> None:
        """
        0xD11D를 매 frame 확인하면서 정확히 0 -> 1 -> 0을 기다린다.

        B는:
            hold b_hold_frames
            release
            gap b_gap_frames
        형태로 반복한다.

        D11D 확인 자체는 매 frame 수행하므로 짧은 1 상태를 놓치지 않는다.
        """
        self.phase = PHASE_FIRST_POKEMON

        transition_phase = 0
        # 0: 첫 0 대기
        # 1: 1 대기
        # 2: 마지막 0 대기

        b_pressed = False
        hold_left = 0
        gap_left = 0

        try:
            for frame in range(max(1, int(timeout_frames))):
                value = self.first_mon_value()

                if transition_phase == 0 and value == 0:
                    transition_phase = 1

                elif transition_phase == 1 and value == 1:
                    transition_phase = 2

                elif transition_phase == 2 and value == 0:
                    if b_pressed:
                        self._release("b")
                        b_pressed = False

                    # B가 다음 메뉴로 새어 들어가지 않게 neutral 1 frame.
                    self._tick(1)
                    self.phase = PHASE_READY

                    if self.debug:
                        print(
                            "[macro-v5] first pokemon "
                            "0 -> 1 -> 0 complete"
                        )
                    return

                if self.debug and frame % 30 == 0:
                    print(
                        "[first] "
                        f"frame={frame:04d} "
                        f"D11D={value} "
                        f"transition_phase={transition_phase}"
                    )

                if b_pressed:
                    hold_left -= 1

                    if hold_left <= 0:
                        self._release("b")
                        b_pressed = False
                        gap_left = max(1, int(b_gap_frames))

                else:
                    if gap_left > 0:
                        gap_left -= 1
                    else:
                        self.pyboy.button_press("b")
                        b_pressed = True
                        hold_left = max(1, int(b_hold_frames))

                self._tick(1)

        finally:
            if b_pressed:
                self._release("b")

        raise BattleMacroTimeout(
            "first_pokemon timeout: "
            f"flag={self.first_mon_value()} "
            f"transition_phase={transition_phase}"
        )


    # ------------------------------------------------------------------
    # Main menu
    # ------------------------------------------------------------------

    def wait_main_menu(self) -> int:
        stable = 0
        previous = None

        for _ in range(self.menu_timeout_frames):
            pair = self.menu_cursor_pair()
            if pair in MAIN_CURSOR_PAIRS:
                if pair == previous:
                    stable += 1
                else:
                    stable = 1
                if stable >= self.decision_stable_frames:
                    self.phase = PHASE_READY
                    self._seen_main_this_battle = True
                    if self.debug:
                        print(
                            "[menu] MAIN ready "
                            f"cursor={pair[0]:02X} {pair[1]:02X}"
                        )
                    return pair[0]
            else:
                stable = 0
            previous = pair
            self._tick(1)

        lo, hi = self.menu_cursor_pair()
        raise BattleMacroTimeout(
            f"main battle menu timeout: cursor={lo:02X} {hi:02X}"
        )

    def _probe_main_menu_interactive(
        self,
        pair: tuple[int, int] | None = None,
        *,
        timeout_frames: int = 60,
    ) -> bool:
        """Prove MAIN is real by opening FIGHT and backing out with B.

        Directional input alone is not sufficient: while repeated sleep text
        is visible, stale menu RAM can still change from C1 to C7.  A genuine
        decision menu is therefore accepted only when A reaches one of the
        full A9/BD/D1/E5 C4 move-cursor pairs.  If A is consumed by battle
        text, that page has been safely advanced and False is returned so the
        resolution loop can continue without invoking PPO on a stale state.
        """
        if pair is None:
            pair = self.menu_cursor_pair()
        if pair not in MAIN_CURSOR_PAIRS:
            return False

        # Move the apparent cursor to FIGHT without confirming any option.
        current_x, current_y = MAIN_CURSOR_TO_COORD[pair[0]]
        try:
            if current_y != 0:
                expected = (MAIN_COORD_TO_CURSOR[(current_x, 0)], 0xC4)
                self._press_until_cursor_pair(
                    "up",
                    expected,
                    timeout_frames=12,
                    label="probe MAIN up to FIGHT row",
                )
                self._tick(1)
                current_y = 0

            if current_x != 0:
                self._press_until_cursor_pair(
                    "left",
                    (MAIN_FIGHT, 0xC4),
                    timeout_frames=12,
                    label="probe MAIN left to FIGHT",
                )
                self._tick(1)
        except BattleMacroTimeout:
            # A non-interactive text page may ignore direction completely.
            # Advancing it is safe; a real menu should have accepted the
            # direction within the bounded probe above.
            self._tick(1)
            self._resolution_a_pulse()
            return False

        self._neutral_a_release()
        opened_move_menu = False
        self.pyboy.button_press("a")
        try:
            for _ in range(max(1, int(timeout_frames))):
                self._tick(1)
                if self.menu_cursor_pair() in MOVE_CURSOR_PAIRS:
                    opened_move_menu = True
                    break
        finally:
            self._release("a")

        for _ in range(self.resolution_release_frames):
            self._tick(1)

        if not opened_move_menu:
            # The fresh A edge was consumed by sleep/wake/attack text.
            return False

        # This was a real decision menu. Return to MAIN without selecting a
        # move, then hand the unchanged decision to PPO.
        self.phase = PHASE_MOVE_MENU
        self._hold_fixed(
            "b",
            3,
            label="probe FIGHT menu -> back to MAIN",
        )
        self.wait_main_menu()
        return True

    def prepare_decision_menu(
        self,
        *,
        timeout_frames: int | None = None,
    ) -> bool:
        """Normalize a manual battle/menu state into a PPO decision boundary.

        MAIN        -> start immediately
        MOVE/PARTY  -> B back to MAIN only after this battle has shown MAIN
                      (except an HP=0 forced party request)
        INITIAL     -> one A press, then advance remaining battle text
        UNKNOWN / stale MOVE before MAIN
                    -> let the normal resolution state machine find MAIN
        """
        if timeout_frames is None:
            timeout_frames = self.turn_timeout_frames

        for attempt in range(6):
            if self.in_battle() is False:
                return False

            pair = self.menu_cursor_pair()
            context = self.menu_context()

            if self.debug:
                print(
                    "[menu] prepare "
                    f"attempt={attempt} context={context} "
                    f"cursor={pair[0]:02X} {pair[1]:02X} "
                    f"seen_main={self._seen_main_this_battle} "
                    f"HP={self.active_player_hp()}"
                )

            if context == MENU_MAIN:
                self.wait_main_menu()
                pair = self.menu_cursor_pair()
                if self._probe_main_menu_interactive(pair):
                    self.phase = PHASE_READY
                    self._seen_main_this_battle = True
                    return True

                if self.debug:
                    print(
                        "[menu] manual-entry MAIN was text overlay "
                        f"cursor={pair[0]:02X} {pair[1]:02X} -> continue"
                    )
                return self.wait_next_decision(
                    timeout_frames=timeout_frames,
                )

            if context == MENU_PARTY and self.active_player_hp() == 0:
                # Already at the forced replacement request.  PPO must choose
                # a living slot; B would incorrectly try to leave the request.
                self._wait_party_cursor_index(
                    stable_frames=self.decision_stable_frames,
                    label="manual-entry forced party menu",
                )
                self.phase = PHASE_FORCED_SWITCH_READY
                return True

            if (
                context in (MENU_MOVE, MENU_PARTY)
                and self._seen_main_this_battle
            ):
                self._release("a")
                self._hold_fixed(
                    "b",
                    3,
                    label=f"leave manual {context} menu",
                )
                continue

            if context == MENU_INITIAL:
                self._neutral_a_release()
                self._hold_fixed(
                    "a",
                    self.resolution_a_hold_frames,
                    label="advance initial C3 C4 cursor",
                )
                if self.menu_context() == MENU_MAIN:
                    self.wait_main_menu()
                    return True
                return self.wait_next_decision(
                    timeout_frames=timeout_frames,
                )

            # Stale MOVE/PARTY (same A9 C4 as intro text) or UNKNOWN:
            # advance with A until the real command menu appears.
            return self.wait_next_decision(
                timeout_frames=timeout_frames,
            )

        lo, hi = self.menu_cursor_pair()
        raise BattleMacroTimeout(
            "failed to normalize manual battle menu: "
            f"context={self.menu_context()} cursor={lo:02X} {hi:02X}"
        )

    def _move_main_cursor_to(self, target_cursor: int) -> None:
        if target_cursor not in MAIN_CURSOR_TO_COORD:
            raise ValueError(
                f"unknown main cursor: 0x{target_cursor:02X}"
            )

        last_error = None
        for attempt in range(1, 5):
            current = self.wait_main_menu()
            if current == target_cursor:
                return

            current_x, current_y = MAIN_CURSOR_TO_COORD[current]
            target_x, target_y = MAIN_CURSOR_TO_COORD[target_cursor]

            try:
                if current_y != target_y:
                    button = "down" if target_y > current_y else "up"
                    expected = MAIN_COORD_TO_CURSOR[(current_x, target_y)]

                    self._press_until_cursor(
                        button,
                        expected,
                        label=f"main {button}",
                    )
                    self._tick(2)
                    current_y = target_y

                if current_x != target_x:
                    button = "right" if target_x > current_x else "left"
                    expected = MAIN_COORD_TO_CURSOR[(target_x, current_y)]

                    self._press_until_cursor(
                        button,
                        expected,
                        label=f"main {button}",
                    )
                    self._tick(2)
                return

            except BattleMacroTimeout as exc:
                last_error = exc
                if self.menu_context() != MENU_MAIN:
                    raise

                if self.debug:
                    lo, hi = self.menu_cursor_pair()
                    print(
                        "[menu] MAIN cursor was a text overlay -> A clear "
                        f"attempt={attempt} cursor={lo:02X} {hi:02X}"
                    )

                # Direction input could not move the apparent MAIN cursor, so
                # it is a text overlay.  Clear exactly one page, fully release,
                # then retry cursor navigation with a fresh direction edge.
                self._resolution_a_pulse()

        raise BattleMacroTimeout(
            "main menu stayed non-interactive after text clears: "
            f"target=0x{target_cursor:02X}, last={last_error}"
        )

    # ------------------------------------------------------------------
    # Turn resolution
    # ------------------------------------------------------------------

    def _neutral_a_release(self, frames: int | None = None) -> None:
        """
        A를 확실히 뗀 상태를 실제 게임 프레임에 반영한다.

        release 직후 같은 프레임에 다시 press하면 Game Boy 쪽에서
        새로운 A 입력 edge를 못 볼 수 있으므로 neutral frame을 보장한다.
        """
        if frames is None:
            frames = self.resolution_release_frames

        self._release("a")
        for _ in range(max(1, int(frames))):
            self._tick(1)

    def _resolution_event_hold(self) -> bool:
        """
        D05E/D05F 이벤트 전용.

        release 상태 보장
        -> A press
        -> D05E/D05F 둘 다 0이 될 때까지 hold
        -> 즉시 release
        -> 다시 release 상태 몇 frame 보장
        """
        before = self.resolution_flags()

        self._neutral_a_release()

        if self.debug:
            lo, hi = self.menu_cursor_pair()
            print(
                "[resolution-event] A HOLD start "
                f"cursor={lo:02X} {hi:02X} "
                f"D05E={before[0]} D05F={before[1]}"
            )

        self.pyboy.button_press("a")
        cleared = False

        try:
            for held in range(self.resolution_event_hold_timeout):
                self._tick(1)

                cursor = self.menu_cursor()
                flags = self.resolution_flags()

                if self.menu_cursor_pair() in MAIN_CURSOR_PAIRS:
                    if self.debug:
                        lo, hi = self.menu_cursor_pair()
                        print(
                            "[resolution-event] main menu appeared "
                            f"after={held + 1}f cursor={lo:02X} {hi:02X}"
                        )
                    cleared = not any(flags)
                    break

                if not any(flags):
                    cleared = True
                    if self.debug:
                        print(
                            "[resolution-event] flags cleared "
                            f"{before} -> {flags} after={held + 1}f"
                        )
                    break
        finally:
            self._release("a")

        for _ in range(self.resolution_release_frames):
            self._tick(1)

        if not cleared and self.debug:
            flags = self.resolution_flags()
            print(
                "[resolution-event] hold timeout "
                f"D05E={flags[0]} D05F={flags[1]} "
                "(A released; next retry gets a fresh press)"
            )

        return cleared

    def _resolution_a_pulse(self) -> None:
        """
        일반 전투 텍스트용 A pulse.

        pulse 전후에 반드시 release frame을 넣어서 다음 A가
        새로운 press로 인식되게 한다.
        """
        self._neutral_a_release()

        self.pyboy.button_press("a")
        try:
            for _ in range(self.resolution_a_hold_frames):
                self._tick(1)

                cursor = self.menu_cursor()
                if self.menu_cursor_pair() in MAIN_CURSOR_PAIRS:
                    if self.debug:
                        lo, hi = self.menu_cursor_pair()
                        print(
                            "[resolution] main-menu candidate -> A release "
                            f"cursor={lo:02X} {hi:02X}"
                        )
                    break
        finally:
            self._release("a")

        for _ in range(self.resolution_release_frames):
            self._tick(1)

    def _move_learn_handoff_info(self) -> dict | None:
        """Return verified level-up move replacement menu data, if active.

        wMoveNum is a shared scratch byte and can remain stale, so it is never
        accepted alone.  The exact learn_move.asm menu geometry, A/B watched
        keys, four copied old moves and the live cursor must all agree.
        """
        required_addresses = (
            self.addr_move_num,
            self.addr_moves_buffer,
            self.addr_which_pokemon,
            self.addr_num_moves_minus_one,
            self.addr_top_menu_y,
            self.addr_top_menu_x,
            self.addr_current_menu_item,
            self.addr_max_menu_item,
            self.addr_menu_watched_keys,
            self.addr_party_mons,
        )
        if any(address is None for address in required_addresses):
            return None

        party_index = self._read_u8(self.addr_which_pokemon)
        if not 0 <= party_index < self.party_count():
            return None

        selected_slot = self._read_u8(self.addr_current_menu_item)
        if not 0 <= selected_slot < 4:
            return None

        menu_signature = (
            self._read_u8(self.addr_top_menu_y) == 8
            and self._read_u8(self.addr_top_menu_x) == 5
            and self._read_u8(self.addr_max_menu_item) == 3
            and self._read_u8(self.addr_num_moves_minus_one) == 3
            and self._read_u8(self.addr_menu_watched_keys) == 0x03
        )
        if not menu_signature:
            return None

        cursor_pair = self.menu_cursor_pair()
        if cursor_pair != MOVE_LEARN_CURSOR_PAIRS[selected_slot]:
            return None

        new_move_id = self._read_u8(self.addr_move_num)
        if new_move_id == 0:
            return None

        copied_moves = tuple(
            self._read_u8(self.addr_moves_buffer + i)
            for i in range(4)
        )
        party_moves_address = (
            self.addr_party_mons
            + party_index * 0x2C
            + 8
        )
        party_moves = tuple(
            self._read_u8(party_moves_address + i)
            for i in range(4)
        )

        # TryingToLearn copies the four current moves into wMoves immediately
        # before this menu. Requiring the exact copy prevents stale wMoveNum or
        # another four-item menu from triggering a false manual handoff.
        if (
            any(move_id == 0 for move_id in party_moves)
            or copied_moves != party_moves
            or new_move_id in party_moves
        ):
            return None

        return {
            "party_index": party_index,
            "new_move_id": new_move_id,
            "current_move_ids": party_moves,
            "selected_slot": selected_slot,
            "cursor_pair": cursor_pair,
        }

    def _handoff_move_learning_to_user(self, info: dict) -> None:
        """Stop all macro input at the verified replacement-slot menu."""
        for button in ("a", "b", "up", "down", "left", "right"):
            self._release(button)
        self.phase = PHASE_MOVE_LEARN_HANDOFF

        cursor_pair = info["cursor_pair"]
        if self.debug:
            print(
                "[move-learn] manual handoff "
                f"party={info['party_index']} "
                f"new_move={info['new_move_id']} "
                f"moves={list(info['current_move_ids'])} "
                f"selected=move{info['selected_slot']} "
                f"cursor={cursor_pair[0]:02X} {cursor_pair[1]:02X}"
            )

        raise BattleManualHandoff(
            "레벨업 기술 교체 슬롯 선택 화면을 감지했습니다.",
            reason="move_learn_replace_slot",
            party_index=info["party_index"],
            new_move_id=info["new_move_id"],
            current_move_ids=info["current_move_ids"],
            selected_slot=info["selected_slot"],
            cursor_pair=cursor_pair,
        )

    @staticmethod
    def _is_shift_prompt_cursor(lo: int, hi: int) -> bool:
        """상대의 다음 포켓몬 전 SHIFT YES/NO 메뉴인지 확인한다."""
        return (
            (int(lo) & 0xFF) in SHIFT_CURSORS
            and (int(hi) & 0xFF) == SHIFT_PROMPT_CURSOR_HI
        )

    def _decline_shift_prompt(self) -> None:
        """SHIFT 질문에서 반드시 NO를 선택하고 질문 화면을 벗어난다.

        Showdown의 강제 교체 요청에는 상대를 쓰러뜨린 쪽의 무료 교체가
        없으므로, 카트리지의 SHIFT 질문을 거절해야 실행 환경도 같은
        턴 흐름을 갖는다.
        """
        self.phase = PHASE_DECLINE_SHIFT
        self._neutral_a_release()

        lo, hi = self.menu_cursor_pair()
        if not self._is_shift_prompt_cursor(lo, hi):
            raise BattleMacroError(
                "SHIFT prompt disappeared before NO selection: "
                f"cursor={lo:02X} {hi:02X}"
            )

        if self.debug:
            selected = "YES" if lo == SHIFT_YES else "NO"
            print(
                "[shift] prompt detected "
                f"cursor={lo:02X} {hi:02X} selected={selected}"
            )

        # YES에 있으면 한 칸 내려 NO로 이동한다. 이미 NO면 이동하지 않는다.
        if lo == SHIFT_YES:
            self._press_until_cursor(
                "down",
                SHIFT_NO,
                label="SHIFT YES -> NO",
            )
            self._tick(2)

        lo, hi = self.menu_cursor_pair()
        if lo != SHIFT_NO or hi != SHIFT_PROMPT_CURSOR_HI:
            raise BattleMacroError(
                "failed to select NO in SHIFT prompt: "
                f"cursor={lo:02X} {hi:02X}"
            )

        # 방향 입력과 확정 A 사이에 중립 프레임을 둬 fresh A edge를 만든다.
        self._neutral_a_release()
        self.pyboy.button_press("a")
        left_prompt = False

        try:
            for frame in range(self.menu_timeout_frames):
                self._tick(1)
                current_lo, current_hi = self.menu_cursor_pair()

                if not self._is_shift_prompt_cursor(
                    current_lo,
                    current_hi,
                ):
                    left_prompt = True

                    if self.debug:
                        print(
                            "[shift] NO confirmed "
                            f"after={frame + 1}f "
                            f"cursor={current_lo:02X} {current_hi:02X}"
                        )
                    break
        finally:
            self._release("a")

        for _ in range(self.resolution_release_frames):
            self._tick(1)

        self.phase = PHASE_TURN_RESOLUTION

        if not left_prompt:
            lo, hi = self.menu_cursor_pair()
            raise BattleMacroTimeout(
                "SHIFT NO confirm timeout: "
                f"cursor={lo:02X} {hi:02X}"
            )


    def wait_next_decision(
        self,
        *,
        required_species: int | None = None,
        required_party_index: int | None = None,
        require_positive_hp: bool = False,
        timeout_frames: int | None = None,
    ) -> bool:
        """
        한 action 처리 후 다음 RL 주도권을 기다린다.

        lazy battle RAM은 턴 판정에 사용하지 않는다.

        완료:
          - 메인 cursor(C1/C7/E9/EF)가
            decision_stable_frames 연속으로 입력 없이 유지
          - 교체라면 species / party slot / positive HP 검증까지 만족

        플레이어 BattleMon HP가 0인 채 B4 C3 계열의 파티 커서가
        안정적으로 나타나면 강제교체 decision으로 PPO에 즉시 반환한다.
        이때 A를 더 누르지 않으므로 기본 커서의 기절 포켓몬이 잘못
        선택되는 것을 막는다.

        상대 포켓몬 기절 후 SHIFT YES/NO 질문이 나타나면 항상 NO를
        확정한 뒤 상대의 다음 포켓몬 등장 과정까지 계속 진행한다.

        TURN_RESOLUTION에서 A9 C4가 보여도 MOVE_MENU로 해석하지 않는다.
        phase가 TURN_RESOLUTION이므로 전투 텍스트 상태로 처리한다.
        """
        self.phase = PHASE_TURN_RESOLUTION

        if timeout_frames is None:
            timeout_frames = self.turn_timeout_frames

        if required_species is not None:
            required_species = int(required_species) & 0xFF
        if required_party_index is not None:
            required_party_index = int(required_party_index)

        stable_main = 0
        stable_cursor = None
        stable_forced = 0
        stable_forced_pair = None
        gap = 0
        last_same_a9_log = False

        for frame in range(max(1, int(timeout_frames))):
            in_battle = self.in_battle()
            if in_battle is False:
                self._release("a")
                self.phase = PHASE_BATTLE_END

                if self.debug:
                    print("[resolution] battle ended")
                return False

            cursor = self.menu_cursor()
            lo, hi = self.menu_cursor_pair()
            flags = self.resolution_flags()

            # Level-up move replacement is not part of the PPO action space.
            # Detect it before SHIFT-menu handling or generic A text pulses so
            # move0 cannot be forgotten accidentally by the macro.
            move_learn_info = self._move_learn_handoff_info()
            if move_learn_info is not None:
                self._handoff_move_learning_to_user(move_learn_info)

            species_ok = True
            if required_species is not None:
                species_ok = (
                    self.active_player_species() == required_species
                )

            party_ok = True
            if required_party_index is not None:
                party_ok = (
                    self.resolved_active_player_party_index()
                    == required_party_index
                )

            hp_ok = True
            if require_positive_hp:
                hp_ok = self.active_player_hp() > 0

            # ----------------------------------------------------------
            # Player fainted -> forced replacement request.
            # A9 C4 is still the faint text; only the stable full party cursor
            # pair transfers control back to PPO.
            # ----------------------------------------------------------
            forced_pair = (lo, hi)
            forced_candidate = (
                required_species is None
                and required_party_index is None
                and not require_positive_hp
                and self.active_player_hp() == 0
                and self._party_cursor_index(forced_pair) is not None
                and self.has_available_party_switch()
            )

            if forced_candidate:
                self._release("a")
                if forced_pair == stable_forced_pair:
                    stable_forced += 1
                else:
                    stable_forced_pair = forced_pair
                    stable_forced = 1

                if stable_forced >= self.decision_stable_frames:
                    self.phase = PHASE_FORCED_SWITCH_READY
                    if self.debug:
                        print(
                            "[forced-switch] PPO READY "
                            f"cursor={lo:02X} {hi:02X} "
                            f"fainted_slot={self.active_player_party_index()} "
                            f"HP={self.active_player_hp()}"
                        )
                    return True

                self._tick(1)
                gap += 1
                continue

            stable_forced = 0
            stable_forced_pair = None

            # ----------------------------------------------------------
            # 다음 의사결정 메뉴 후보.
            # 이 상태에서는 A를 누르지 않고 neutral frame으로 안정 확인.
            #
            # 과거에는 먼저 non-MAIN cursor를 봐야만 후보로 인정했지만,
            # sleep 문구에서는 cursor RAM이 C1 C4로 계속 남아 교착됐다.
            # 현재는 아래의 FIGHT-open/B-back probe가 실제 메뉴임을 직접
            # 증명하므로 species/party/HP 검증만으로 probe를 시작한다.
            # ----------------------------------------------------------
            if (lo, hi) in MAIN_CURSOR_PAIRS:
                self._release("a")

                verified_main = bool(
                    species_ok
                    and party_ok
                    and hp_ok
                )

                if verified_main and cursor == stable_cursor:
                    stable_main += 1
                elif verified_main:
                    stable_cursor = cursor
                    stable_main = 1
                else:
                    stable_main = 0
                    stable_cursor = None

                if self.debug and (
                    stable_main == 1
                    or (not verified_main and frame % 30 == 0)
                ):
                    print(
                        "[resolution] main-menu neutral wait "
                        f"cursor={lo:02X} {hi:02X} "
                        f"species_ok={species_ok} "
                        f"party_ok={party_ok} hp_ok={hp_ok} "
                        f"raw_party={self.active_player_party_index()} "
                        f"resolved_party={self.resolved_active_player_party_index()} "
                        f"HP={self.active_player_hp()}"
                    )

                if stable_main >= self.decision_stable_frames:
                    if self._probe_main_menu_interactive((lo, hi)):
                        verified_pair = self.menu_cursor_pair()
                        self.phase = PHASE_READY
                        self._seen_main_this_battle = True

                        if self.debug:
                            print(
                                "[resolution] READY interactive "
                                f"cursor={verified_pair[0]:02X} "
                                f"{verified_pair[1]:02X} "
                                f"stable={stable_main}"
                            )
                        return True

                    # C1/C7/E9/EF was only a stale cursor under battle text
                    # (notably "is fast asleep").  Advance exactly one text
                    # page, then keep waiting for the real next-turn menu.
                    if self.debug:
                        print(
                            "[resolution] apparent MAIN is text overlay "
                            f"cursor={lo:02X} {hi:02X} -> continue"
                        )
                    stable_main = 0
                    stable_cursor = None
                    gap = 0
                    continue

                self._tick(1)
                gap += 1
                continue

            stable_main = 0
            stable_cursor = None

            # ----------------------------------------------------------
            # 상대 포켓몬을 쓰러뜨린 뒤의 SHIFT 질문.
            # 일반 텍스트용 A pulse에 맡기지 않고 명시적으로 NO를 고른다.
            # ----------------------------------------------------------
            if self._is_shift_prompt_cursor(lo, hi):
                self._decline_shift_prompt()
                gap = 0
                continue

            # ----------------------------------------------------------
            # A9 C4 충돌 처리.
            # phase가 TURN_RESOLUTION이면 기술 메뉴로 보지 않는다.
            # ----------------------------------------------------------
            same_as_move0 = (
                lo == MOVE_CURSORS[0]
                and hi == 0xC4
            )

            if same_as_move0:
                if self.debug and not last_same_a9_log:
                    print(
                        "[resolution] cursor=A9 C4 but "
                        "phase=TURN_RESOLUTION -> NOT MOVE_MENU"
                    )
                last_same_a9_log = True
            else:
                last_same_a9_log = False

            # 교체 직후 species가 아직 바뀌지 않은 경우도 계속 진행.
            # D05E/D05F 이벤트는 일반 pulse가 아니라 hold-until-clear로 처리.
            if any(flags):
                if self.debug:
                    print(
                        "[resolution] event RAM active "
                        f"D05E={flags[0]} D05F={flags[1]} -> A HOLD"
                    )

                self._resolution_event_hold()
                gap = 0
                continue

            # 일반 전투 텍스트도 있으므로 일정 간격으로 A pulse.
            if gap >= self.resolution_a_gap_frames:
                self._resolution_a_pulse()
                gap = 0
                continue

            self._tick(1)
            gap += 1

            if self.debug and frame % 60 == 0:
                action_result = self.action_result_or_took_turn()
                active_species = None
                if self.addr_battle_species is not None:
                    active_species = self.active_player_species()

                print(
                    "[resolution] waiting "
                    f"frame={frame:04d} "
                    f"phase={self.phase} "
                    f"cursor={lo:02X} {hi:02X} "
                    f"D05E={flags[0]} D05F={flags[1]} "
                    f"actionResult={action_result} "
                    f"species={active_species} "
                    f"required={required_species}"
                )

        self._release("a")
        raise BattleMacroTimeout(
            "next decision timeout: "
            f"phase={self.phase}, "
            f"cursor={self.menu_cursor_pair()}, "
            f"flags={self.resolution_flags()}, "
            f"required_species={required_species}, "
            f"required_party_index={required_party_index}, "
            f"require_positive_hp={require_positive_hp}"
        )

    # ------------------------------------------------------------------
    # Move action 0~3
    # ------------------------------------------------------------------

    def _open_fight_move_menu(self) -> int:
        """Open FIGHT with fresh A edges, clearing a stale C1 text overlay.

        Sleep/wake and other no-action messages can leave C1 C4 in RAM even
        though the four-choice menu is not interactive yet.  One held A only
        closes that text; it cannot also create the second edge needed to open
        FIGHT.  Retry after a full release until an A9/BD/D1/E5 C4 move cursor
        proves that the real move menu opened.
        """
        # Sleep/wake, Disable and multi-page attack-result text can reuse the
        # apparent C1 C4 cursor for several consecutive fresh-A edges.  Five
        # retries proved too short in live battles, so allow a generous bound;
        # the loop still exits immediately once a full move cursor is seen.
        max_attempts = 20
        frames_per_attempt = min(self.menu_timeout_frames, 60)

        for attempt in range(1, max_attempts + 1):
            self.phase = PHASE_OPEN_FIGHT
            self._move_main_cursor_to(MAIN_FIGHT)
            self.phase = PHASE_OPEN_FIGHT
            self._neutral_a_release()

            opened_pair = None
            self.pyboy.button_press("a")
            try:
                for frame in range(frames_per_attempt):
                    self._tick(1)
                    pair = self.menu_cursor_pair()
                    if pair in MOVE_CURSOR_PAIRS:
                        opened_pair = pair
                        if self.debug:
                            print(
                                "[move] FIGHT opened "
                                f"attempt={attempt} after={frame + 1}f "
                                f"cursor={pair[0]:02X} {pair[1]:02X}"
                            )
                        break
            finally:
                self._release("a")

            for _ in range(self.resolution_release_frames):
                self._tick(1)

            if opened_pair is not None:
                self.phase = PHASE_MOVE_MENU
                return self._wait_move_cursor_index(
                    stable_frames=1,
                    label="move menu",
                )

            if self.in_battle() is False:
                raise BattleMacroError(
                    "battle ended while clearing pre-FIGHT text"
                )

            pair = self.menu_cursor_pair()
            if self.debug:
                print(
                    "[move] FIGHT A consumed by text -> fresh-A retry "
                    f"attempt={attempt} cursor={pair[0]:02X} {pair[1]:02X}"
                )

            # If the message advanced into another resolution state, finish
            # that state first.  A stable MAIN result will be probed again on
            # the next attempt rather than being trusted by cursor alone.
            if self.menu_context() not in (MENU_MAIN, MENU_INITIAL):
                ready = self.wait_next_decision(
                    timeout_frames=self.turn_timeout_frames,
                )
                if not ready:
                    raise BattleMacroError(
                        "battle ended while advancing pre-FIGHT text"
                    )

        lo, hi = self.menu_cursor_pair()
        raise BattleMacroTimeout(
            "open FIGHT move menu failed after fresh-A retries: "
            f"cursor={lo:02X} {hi:02X}, attempts={max_attempts}"
        )

    def choose_move(self, move_index: int) -> bool:
        move_index = int(move_index)

        if not 0 <= move_index <= 3:
            raise ValueError(
                f"move_index must be 0..3, got {move_index}"
            )

        # --------------------------------------------------------------
        # OPEN_FIGHT
        # --------------------------------------------------------------
        current_index = self._open_fight_move_menu()

        if self.debug:
            lo, hi = self.menu_cursor_pair()
            print(
                "[move] MOVE_MENU "
                f"cursor={lo:02X} {hi:02X} "
                f"index={current_index} target={move_index}"
            )

        # --------------------------------------------------------------
        # 기술 커서 이동 (v6)
        #
        # MOVE 메뉴는 hold-until-cursor가 아니라 한 칸씩 독립 tap.
        # 특히 move2(D1), move3(E5)로 갈 때:
        #   A9 -> BD -> D1 -> E5
        # 각 단계마다 release + stable 검증을 한다.
        # --------------------------------------------------------------
        current_index = self._move_menu_cursor_to(
            move_index,
            current_index=current_index,
        )

        # --------------------------------------------------------------
        # 기술 확정.
        #
        # selected/used RAM equality는 stale할 수 있으므로 종료조건으로 쓰지 않는다.
        # Navigation의 A 방식처럼 충분히 hold 후 release하고,
        # 그 다음부터는 phase=TURN_RESOLUTION로 해석한다.
        # --------------------------------------------------------------
        self.phase = PHASE_CONFIRM_MOVE

        if self.debug:
            lo, hi = self.menu_cursor_pair()
            print(
                "[move] CONFIRM_MOVE "
                f"index={move_index} cursor={lo:02X} {hi:02X} "
                f"selected(old)=0x{self.player_selected_move():02X} "
                f"used(old)=0x{self.player_used_move():02X}"
            )

        # v6: 방향키 release와 확정 A 사이에 neutral 구간을 넣고
        # 최종 cursor를 다시 확인한 뒤 fresh A edge로 확정한다.
        self._confirm_move_fresh_a(move_index)

        # 여기부터 A9 C4가 다시 보여도 기술 메뉴가 아니다.
        return self.wait_next_decision()

    # ------------------------------------------------------------------
    # Switch action 4~9
    # ------------------------------------------------------------------

    def _forced_switch_pokemon(
        self,
        party_index: int,
        *,
        target_species: int,
    ) -> bool:
        """Execute a PPO replacement from an already-open forced party menu."""
        if self.active_player_hp() != 0:
            raise BattleMacroError(
                "forced switch requested while active HP is not zero: "
                f"HP={self.active_player_hp()}"
            )

        target_hp = self.party_hp(party_index)
        if target_hp <= 0:
            raise BattleMacroError(
                f"PPO forced switch target party {party_index} is fainted: "
                f"HP={target_hp}"
            )

        current_pair = self.menu_cursor_pair()
        if self._party_cursor_index(current_pair) is None:
            raise BattleMacroError(
                "forced switch party menu is not ready: "
                f"cursor={current_pair[0]:02X} {current_pair[1]:02X}"
            )

        self.phase = PHASE_FORCED_SWITCH_READY
        current_index = self._wait_party_cursor_index(
            stable_frames=self.decision_stable_frames,
            label="forced switch party menu",
        )

        while current_index < party_index:
            next_index = current_index + 1
            self._press_until_cursor(
                "down",
                PARTY_CURSORS[next_index],
                label=f"forced party {current_index}->{next_index}",
            )
            self._tick(2)
            current_index = next_index

        while current_index > party_index:
            next_index = current_index - 1
            self._press_until_cursor(
                "up",
                PARTY_CURSORS[next_index],
                label=f"forced party {current_index}->{next_index}",
            )
            self._tick(2)
            current_index = next_index

        actual_pair = self.menu_cursor_pair()
        if self._party_cursor_index(actual_pair) != party_index:
            raise BattleMacroError(
                "forced switch cursor verification failed: "
                f"target={party_index} "
                f"actual={actual_pair[0]:02X} {actual_pair[1]:02X}"
            )

        if self.debug:
            print(
                "[forced-switch] PPO selected "
                f"party={party_index} species=0x{target_species:02X} "
                f"HP={target_hp} cursor={actual_pair[0]:02X} {actual_pair[1]:02X}"
            )

        self.phase = PHASE_CONFIRM_FORCED_SWITCH
        self._note_player_party_index(party_index)
        self._neutral_a_release()
        self._hold_fixed(
            "a",
            self.confirm_hold_frames,
            label=f"confirm forced SWITCH party {party_index}",
        )

        # Do not return to PPO merely because a familiar cursor appeared.
        # The selected slot, species and positive BattleMon HP must all be live,
        # followed by the C1/C7/E9/EF C4 main battle menu.
        return self.wait_next_decision(
            required_species=target_species,
            required_party_index=party_index,
            require_positive_hp=True,
        )

    def switch_pokemon(self, party_index: int) -> bool:
        party_index = int(party_index)

        if not 0 <= party_index <= 5:
            raise ValueError(
                f"party_index must be 0..5, got {party_index}"
            )

        target_species = self.party_species(party_index)

        if target_species in (0x00, 0xFF):
            raise BattleMacroError(
                f"party {party_index} 슬롯에 유효한 species가 없습니다: "
                f"0x{target_species:02X}"
            )

        # A forced replacement is already on the party screen.  Opening PKMN
        # again would desynchronize and repeatedly select the fainted default
        # slot, so branch before the normal voluntary-switch flow.
        if (
            self.active_player_hp() == 0
            and self._party_cursor_index(self.menu_cursor_pair()) is not None
        ):
            return self._forced_switch_pokemon(
                party_index,
                target_species=target_species,
            )

        current_species = self.active_player_species()
        current_party_index = self.resolved_active_player_party_index()
        if current_party_index == party_index:
            raise BattleMacroError(
                f"party {party_index}가 현재 필드 포켓몬 슬롯입니다."
            )

        if self.party_hp(party_index) <= 0:
            raise BattleMacroError(
                f"party {party_index}는 기절 상태라 교체할 수 없습니다: "
                f"HP={self.party_hp(party_index)}"
            )

        if self.debug:
            print(
                "[switch] target "
                f"party={party_index} "
                f"species=0x{target_species:02X} "
                f"current=0x{current_species:02X}"
            )

        # --------------------------------------------------------------
        # OPEN_PKMN
        # --------------------------------------------------------------
        self.phase = PHASE_OPEN_PKMN
        self._move_main_cursor_to(MAIN_PKMN)
        # wait_main_menu()가 READY로 바꾸므로 PKMN 진입 직전에 복구.
        self.phase = PHASE_OPEN_PKMN

        self._press_until_cursor_in(
            "a",
            PARTY_CURSORS,
            timeout_frames=self.menu_timeout_frames,
            label="open PKMN menu",
        )
        self._release("a")
        self._tick(2)

        self.phase = PHASE_PARTY_MENU

        current_index = self._wait_party_cursor_index(
            stable_frames=1,
            label="party menu",
        )

        # --------------------------------------------------------------
        # party cursor 이동
        # --------------------------------------------------------------
        while current_index < party_index:
            next_index = current_index + 1
            self._press_until_cursor(
                "down",
                PARTY_CURSORS[next_index],
                label=f"party {next_index}",
            )
            self._tick(2)
            current_index = next_index

        while current_index > party_index:
            next_index = current_index - 1
            self._press_until_cursor(
                "up",
                PARTY_CURSORS[next_index],
                label=f"party {next_index}",
            )
            self._tick(2)
            current_index = next_index

        # --------------------------------------------------------------
        # 포켓몬 선택 -> SWITCH/STATS/CANCEL submenu
        # submenu의 정확한 cursor 값은 사용자가 아직 측정하지 않았으므로
        # PARTY cursor 집합에서 벗어났다는 것만 확인.
        # --------------------------------------------------------------
        selected_cursor = PARTY_CURSORS[party_index]

        self.pyboy.button_press("a")
        try:
            submenu_open = False

            for _ in range(self.menu_timeout_frames):
                self._tick(1)
                cursor = self.menu_cursor()

                if cursor not in PARTY_CURSORS:
                    submenu_open = True
                    break
        finally:
            self._release("a")

        if not submenu_open:
            raise BattleMacroTimeout(
                f"party {party_index} submenu open timeout: "
                f"cursor=0x{self.menu_cursor():02X}"
            )

        self._tick(3)

        if self.debug:
            lo, hi = self.menu_cursor_pair()
            print(
                "[switch] submenu opened "
                f"from=0x{selected_cursor:02X} now={lo:02X} {hi:02X}"
            )

        # --------------------------------------------------------------
        # SWITCH 확정.
        # wActionResultOrTookBattleTurn은 lazy하므로 성공 판정에 사용하지 않는다.
        # submenu 첫 항목 SWITCH를 Navigation식 A hold로 확정.
        # --------------------------------------------------------------
        self.phase = PHASE_CONFIRM_SWITCH
        self._note_player_party_index(party_index)

        self._hold_fixed(
            "a",
            self.confirm_hold_frames,
            label=f"confirm SWITCH party {party_index}",
        )

        # 선택한 species가 실제 필드에 올라온 뒤,
        # 메인 메뉴가 돌아왔을 때만 완료.
        return self.wait_next_decision(
            required_species=target_species,
            required_party_index=party_index,
            require_positive_hp=True,
        )

    # ------------------------------------------------------------------
    # RUN command (not part of the PPO 10-action space)
    # ------------------------------------------------------------------

    def run_away(self) -> bool:
        """메인 배틀 메뉴의 RUN을 선택하고 실제 탈출 여부를 반환한다.

        True: 야생 배틀에서 도망쳐 전투가 종료됨.
        False: 트레이너전/도주 실패 등으로 다음 배틀 메뉴가 돌아옴.
        """
        in_battle = self.in_battle()
        if in_battle is False:
            return False

        self.phase = PHASE_OPEN_RUN
        self._move_main_cursor_to(MAIN_RUN)
        # wait_main_menu()가 READY로 바꾸므로 확정 직전에 phase를 복구한다.
        self.phase = PHASE_OPEN_RUN

        self._hold_fixed(
            "a",
            self.confirm_hold_frames,
            label="confirm RUN",
        )

        # escape 성공 시 wait_next_decision()은 False(배틀 종료),
        # 실패/금지 시 True(다음 메인 메뉴)를 반환한다.
        next_decision = self.wait_next_decision()
        return not next_decision

    # ------------------------------------------------------------------
    # Unified action
    # ------------------------------------------------------------------

    def execute_action(self, action: int) -> bool:
        action = int(action)

        if 0 <= action <= 3:
            return self.choose_move(action)

        if 4 <= action <= 9:
            return self.switch_pokemon(action - 4)

        raise ValueError(
            f"battle action must be 0..9, got {action}"
        )


__all__ = [
    "BattleMacro",
    "BattleMacroError",
    "BattleMacroTimeout",
    "BattleMacroStopped",
    "BattleManualHandoff",
    "MacroDebugState",
    "PHASE_IDLE",
    "PHASE_FIRST_POKEMON",
    "PHASE_READY",
    "PHASE_OPEN_FIGHT",
    "PHASE_MOVE_MENU",
    "PHASE_CONFIRM_MOVE",
    "PHASE_OPEN_PKMN",
    "PHASE_PARTY_MENU",
    "PHASE_CONFIRM_SWITCH",
    "PHASE_FORCED_SWITCH_READY",
    "PHASE_CONFIRM_FORCED_SWITCH",
    "PHASE_OPEN_RUN",
    "PHASE_TURN_RESOLUTION",
    "PHASE_DECLINE_SHIFT",
    "PHASE_MOVE_LEARN_HANDOFF",
    "PHASE_BATTLE_END",
    "ADDR_FIRST_MON_YET",
    "ADDR_LAST_MENU_ITEM",
    "ADDR_MENU_CURSOR_LO",
    "ADDR_MENU_CURSOR_HI",
    "ADDR_RESOLUTION_FLAG_0",
    "ADDR_RESOLUTION_FLAG_1",
    "ADDR_PLAYER_SELECTED_MOVE",
    "ADDR_PLAYER_USED_MOVE",
    "MAIN_CURSORS",
    "MAIN_CURSOR_PAIRS",
    "MOVE_CURSORS",
    "MOVE_CURSOR_PAIRS",
    "MOVE_LEARN_CURSOR_PAIRS",
    "PARTY_CURSORS",
    "PARTY_CURSOR_PAIRS",
    "INITIAL_BATTLE_CURSOR_PAIR",
    "MENU_INITIAL",
    "MENU_MAIN",
    "MENU_MOVE",
    "MENU_PARTY",
    "MENU_UNKNOWN",
    "SHIFT_PROMPT_CURSOR_HI",
    "SHIFT_YES",
    "SHIFT_NO",
    "SHIFT_CURSORS",
]
