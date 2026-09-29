"""파티에 치트 포켓몬 주입 (Pokemon Red WRAM)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

# pret/pokered addresses
ADDR_PARTY_COUNT = 0xD163
ADDR_PARTY_SPECIES = 0xD164
ADDR_PARTY_MONS = 0xD16B
ADDR_PARTY_OT = 0xD273
ADDR_PARTY_NICKS = 0xD2B5
ADDR_POKEDEX_OWNED = 0xD2F7
ADDR_POKEDEX_SEEN = 0xD30A
ADDR_PLAYER_NAME = 0xD158
ADDR_PLAYER_ID = 0xD359

PARTY_LENGTH = 6
PARTYMON_STRUCT_LENGTH = 44  # $2C
NAME_LENGTH = 11
NAME_TERMINATOR = 0x50

# internal species id → (dex_no 1-based, type1, type2, base hp/atk/def/spd/spc) types: PSYCHIC=$18
_SPECIES: Dict[str, Dict[str, Any]] = {
    "mewtwo": {
        "id": 0x83,
        "dex": 150,
        "types": (0x18, 0x18),
        "base": (106, 110, 90, 130, 154),
        "catch_rate": 3,
        "default_moves": (0x5E, 0x55, 0x3A, 0x69),  # Psychic, Thunderbolt, Ice Beam, Recover
        "default_nick": "MEWTWO",
    },
    "mew": {
        "id": 0x15,
        "dex": 151,
        "types": (0x18, 0x18),
        "base": (100, 100, 100, 100, 100),
        "catch_rate": 45,
        "default_moves": (0x5E, 0x55, 0x3A, 0x69),  # Psychic, Thunderbolt, Ice Beam, Recover
        "default_nick": "MEW",
    },
    # 비전머신 운반용.
    "tentacool": {
        "id": 0x18,
        "dex": 72,
        "types": (0x15, 0x03),  # WATER, POISON
        "base": (40, 40, 35, 70, 100),
        "catch_rate": 190,
        "default_moves": (0x0F, 0x39, 0, 0),  # Cut, Surf
        "default_nick": "HMMON",
    },
}


def _encode_name(text: str, length: int = NAME_LENGTH) -> List[int]:
    """ASCII 대문자/숫자 → Gen1 문자 (간단 맵)."""
    out: List[int] = []
    for ch in (text or "").upper():
        if len(out) >= length - 1:
            break
        if "A" <= ch <= "Z":
            out.append(0x80 + (ord(ch) - ord("A")))
        elif "0" <= ch <= "9":
            out.append(0xF6 + (ord(ch) - ord("0")))
        elif ch in (" ", "-", "'"):
            out.append(0x7F if ch == " " else 0xE3 if ch == "-" else 0xE0)
        # 그 외 무시
    out.append(NAME_TERMINATOR)
    while len(out) < length:
        out.append(NAME_TERMINATOR)
    return out[:length]


def _u16(n: int) -> tuple[int, int]:
    n = max(0, min(0xFFFF, int(n)))
    return (n >> 8) & 0xFF, n & 0xFF


def _u24(n: int) -> tuple[int, int, int]:
    n = max(0, min(0xFFFFFF, int(n)))
    return (n >> 16) & 0xFF, (n >> 8) & 0xFF, n & 0xFF


def _calc_stat(base: int, iv: int, ev: int, level: int, *, is_hp: bool) -> int:
    """Gen1 스탯 (EV는 실제 저장값, sqrt(EV)/4 근사로 floor(sqrt(ev))/4)."""
    # max EV 65535 → floor(sqrt)=255
    import math

    ev_term = int(math.sqrt(max(0, ev))) // 4
    core = ((base + iv) * 2 + ev_term) * level // 100
    if is_hp:
        return core + level + 10
    return core + 5


def _resolve_species(spec: Any) -> Dict[str, Any]:
    if isinstance(spec, int):
        for meta in _SPECIES.values():
            if meta["id"] == spec:
                return dict(meta)
        raise ValueError(f"알 수 없는 species id: {spec}")
    key = str(spec).strip().lower()
    if key not in _SPECIES:
        raise ValueError(f"지원 species: {sorted(_SPECIES)} (받은 값: {spec!r})")
    return dict(_SPECIES[key])


def _set_dex_flag(pyboy, dex_no: int) -> None:
    """dex_no: 1..151 → owned/seen 비트 ON."""
    idx = int(dex_no) - 1
    if idx < 0:
        return
    byte_i, bit_i = divmod(idx, 8)
    for base in (ADDR_POKEDEX_OWNED, ADDR_POKEDEX_SEEN):
        addr = base + byte_i
        pyboy.memory[addr] = int(pyboy.memory[addr]) | (1 << bit_i)


def _party_has_species(pyboy, species_id: int) -> bool:
    count = int(pyboy.memory[ADDR_PARTY_COUNT])
    for i in range(min(count, PARTY_LENGTH)):
        if int(pyboy.memory[ADDR_PARTY_SPECIES + i]) == species_id:
            return True
    return False


def build_party_mon_bytes(
    *,
    species_id: int,
    types: Sequence[int],
    base: Sequence[int],
    catch_rate: int,
    moves: Sequence[int],
    level: int,
    ot_id: int,
    pps: Optional[Sequence[int]] = None,
) -> List[int]:
    """44바이트 party_struct."""
    level = max(1, min(100, int(level)))
    iv = 0xFFFF  # 올 15
    ev = 0xFFFF
    hp_iv = ((iv >> 12) & 1) * 8 + ((iv >> 8) & 1) * 4 + ((iv >> 4) & 1) * 2 + (iv & 1)
    atk_iv = (iv >> 12) & 0xF
    def_iv = (iv >> 8) & 0xF
    spd_iv = (iv >> 4) & 0xF
    spc_iv = iv & 0xF

    b_hp, b_atk, b_def, b_spd, b_spc = [int(x) for x in base]
    max_hp = _calc_stat(b_hp, hp_iv, ev, level, is_hp=True)
    atk = _calc_stat(b_atk, atk_iv, ev, level, is_hp=False)
    defense = _calc_stat(b_def, def_iv, ev, level, is_hp=False)
    speed = _calc_stat(b_spd, spd_iv, ev, level, is_hp=False)
    special = _calc_stat(b_spc, spc_iv, ev, level, is_hp=False)

    # Slow growth L100 exp = 1_000_000
    exp = level * level * level

    mv = list(moves)[:4]
    while len(mv) < 4:
        mv.append(0)
    # PP 를 안 주면 40 으로 채운다.
    if pps is None:
        pp_list = [0x28 if m else 0 for m in mv]
    else:
        pp_list = [int(p) & 0x3F for p in list(pps)[:4]]
        while len(pp_list) < 4:
            pp_list.append(0)
        pp_list = [p if mv[i] else 0 for i, p in enumerate(pp_list)]

    hp_hi, hp_lo = _u16(max_hp)
    data: List[int] = [
        species_id & 0xFF,
        hp_hi,
        hp_lo,
        level & 0xFF,  # BoxLevel
        0,  # Status
        int(types[0]) & 0xFF,
        int(types[1]) & 0xFF,
        catch_rate & 0xFF,
        *[m & 0xFF for m in mv],
        *_u16(ot_id),
        *_u24(exp),
        *_u16(ev),
        *_u16(ev),
        *_u16(ev),
        *_u16(ev),
        *_u16(ev),
        *_u16(iv),
        *pp_list,
        level & 0xFF,
        *_u16(max_hp),
        *_u16(atk),
        *_u16(defense),
        *_u16(speed),
        *_u16(special),
    ]
    assert len(data) == PARTYMON_STRUCT_LENGTH, len(data)
    return data


def give_party_mon(pyboy, cfg: Optional[Dict[str, Any]] = None) -> str:
    """파티에 치트 몬 추가."""
    cfg = dict(cfg or {})
    if not cfg.get("enabled", False):
        return "disabled"

    meta = _resolve_species(cfg.get("species", "mewtwo"))
    species_id = int(meta["id"])
    level = int(cfg.get("level", 100))
    nick = str(cfg.get("nickname") or meta["default_nick"])
    skip_if_have = bool(cfg.get("skip_if_have", True))

    count = int(pyboy.memory[ADDR_PARTY_COUNT])
    if count >= PARTY_LENGTH:
        return "party full (6)"
    if skip_if_have and _party_has_species(pyboy, species_id):
        return f"already have species 0x{species_id:02X}"

    moves = cfg.get("moves") or meta["default_moves"]
    ot_id = (int(pyboy.memory[ADDR_PLAYER_ID]) << 8) | int(
        pyboy.memory[ADDR_PLAYER_ID + 1]
    )

    mon = build_party_mon_bytes(
        species_id=species_id,
        types=meta["types"],
        base=meta["base"],
        catch_rate=int(meta["catch_rate"]),
        moves=moves,
        level=level,
        ot_id=ot_id,
        pps=cfg.get("pps"),
    )

    slot = count
    # species list
    pyboy.memory[ADDR_PARTY_SPECIES + slot] = species_id
    pyboy.memory[ADDR_PARTY_SPECIES + slot + 1] = 0xFF
    pyboy.memory[ADDR_PARTY_COUNT] = count + 1

    base = ADDR_PARTY_MONS + slot * PARTYMON_STRUCT_LENGTH
    for i, b in enumerate(mon):
        pyboy.memory[base + i] = b

    # OT = player name
    ot_bytes = [int(pyboy.memory[ADDR_PLAYER_NAME + i]) for i in range(NAME_LENGTH)]
    if ot_bytes[0] in (0, NAME_TERMINATOR):
        ot_bytes = _encode_name("RED")
    nick_bytes = _encode_name(nick)
    ot_base = ADDR_PARTY_OT + slot * NAME_LENGTH
    nick_base = ADDR_PARTY_NICKS + slot * NAME_LENGTH
    for i in range(NAME_LENGTH):
        pyboy.memory[ot_base + i] = ot_bytes[i]
        pyboy.memory[nick_base + i] = nick_bytes[i]

    _set_dex_flag(pyboy, int(meta["dex"]))
    return f"added {nick} (species=0x{species_id:02X} Lv{level}) slot={slot}"


def apply_party_mon_from_config(pyboy) -> Optional[str]:
    """config.yaml cheats.party_mon 적용."""
    from core.config import load_config

    party_cfg = (load_config().get("cheats") or {}).get("party_mon")
    if not isinstance(party_cfg, dict) or not party_cfg.get("enabled"):
        return None
    return give_party_mon(pyboy, party_cfg)
