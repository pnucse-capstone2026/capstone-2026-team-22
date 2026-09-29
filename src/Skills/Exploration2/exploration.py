"""play.py ExploreRunner 계약용 ExplorationSkill."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

Coord = Tuple[int, int]

_EXPLORATION_DIR = Path(__file__).resolve().parent
_PUFFER_MAIN = _EXPLORATION_DIR / "pokemonred_puffer-main"
_HOLD_RL = 24


def _ensure_puffer_main_on_path() -> Path:
    root = _PUFFER_MAIN.resolve()
    env = os.environ.get("POKEMONRED_PUFFER_MAIN")
    candidates = []
    if env:
        candidates.append(Path(env).expanduser().resolve())
    candidates.append(root)
    for cand in candidates:
        if (cand / "pokemonred_puffer" / "play_bridge.py").is_file():
            s = str(cand)
            if s not in sys.path:
                sys.path.insert(0, s)
            return cand
    raise ModuleNotFoundError(
        f"pokemonred_puffer-main 없음: {_PUFFER_MAIN} "
        "(또는 POKEMONRED_PUFFER_MAIN)"
    )


def _default_checkpoint() -> Path:
    env = os.environ.get("POKEMONRED_CHECKPOINT")
    if env:
        return Path(env).expanduser()
    _ensure_puffer_main_on_path()
    from pokemonred_puffer.checkpoints import latest_model_path
    from pokemonred_puffer.play_bridge import default_checkpoint

    local = latest_model_path(_EXPLORATION_DIR, nested=False)
    if local is not None:
        return local
    return default_checkpoint()


def default_model_path() -> Path:
    """실제로 쓰일 체크포인트 경로."""
    return _default_checkpoint()


class ExplorationSkill:
    """PlayExplorationPolicy(추론) + play.py pulse 러너 필드."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        *,
        device: str = "cpu",
        greedy: bool = False,
        cheat_profile: str = "full",
    ) -> None:
        self._model_path_arg = Path(model_path).expanduser() if model_path else None
        # 실제 경로는 load_policy() 에서 확정된다 (puffer import 를 미루기 위해 여기서 _default_checkpoint() 를 부르지 않는다).
        self.model_path: Optional[Path] = self._model_path_arg
        self.device = device
        self.greedy = greedy
        self.cheat_profile = cheat_profile

        self._rl = None  # PlayExplorationPolicy | None
        self.action_hold_frames = _HOLD_RL
        self.pulse_action = False
        self.active = False
        self.steps = 0
        self.max_steps = 1000
        self._last_error: Optional[str] = None

    def _clear_rl(self) -> None:
        if self._rl is not None and getattr(self._rl, "active", False):
            try:
                self._rl.stop("중단")
            except Exception:
                pass
        self._rl = None
        self.pulse_action = False
        self.action_hold_frames = _HOLD_RL

    def _ensure_policy_loaded(self) -> bool:
        if self._rl is not None:
            return True
        return self.load_policy()

    def load_policy(self) -> bool:
        try:
            _ensure_puffer_main_on_path()
            from pokemonred_puffer.play_bridge import PlayExplorationPolicy

            ckpt = self._model_path_arg if self._model_path_arg is not None else _default_checkpoint()
            if not Path(ckpt).is_file():
                self._last_error = f"모델 파일 없음: {ckpt}"
                print(
                    f"[explore] 모델 없음: {ckpt}\n"
                    "  Skills/Exploration2/ 에 model_*.pt 를 두거나 "
                    "POKEMONRED_CHECKPOINT 를 설정하세요."
                )
                self._clear_rl()
                return False

            self._rl = PlayExplorationPolicy(
                model_path=ckpt,
                device=self.device,
                greedy=self.greedy,
                cheat_profile=self.cheat_profile,
            )
            self.model_path = self._rl.model_path
            self.action_hold_frames = int(getattr(self._rl, "action_freq", _HOLD_RL))
            self.pulse_action = True
            self.max_steps = int(getattr(self._rl, "default_max_steps", self.max_steps))
            self._last_error = None
            return True
        except Exception as exc:
            self._last_error = str(exc)
            print(f"[explore] RL 로드 실패 — Exploration 중단: {exc}")
            self._clear_rl()
            return False

    def run_action_on_emulator(self, action_name: str) -> None:
        if self._rl is None:
            raise RuntimeError("RL policy not loaded")
        self._rl.execute_action(action_name)

    def bind_pyboy(self, pyboy, *, reduce_res: bool = True, two_bit: bool = True) -> bool:
        if not self._ensure_policy_loaded() or self._rl is None:
            return False
        self._rl.bind_pyboy(pyboy, reduce_res=reduce_res, two_bit=two_bit)
        return True

    def start(self, map_id: int, *, max_steps: int | None = None) -> bool:
        if not self._ensure_policy_loaded() or self._rl is None or self._rl.policy is None:
            print("[explore] RL 정책 없음 — Exploration 중단")
            self.active = False
            return False
        self.active = True
        self.steps = 0
        if max_steps is not None:
            self.max_steps = int(max_steps)
        self._rl.start(map_id, max_steps=self.max_steps)
        return True

    def stop(self, reason: str = "중단") -> None:
        if not self.active and self._rl is None:
            return
        self.active = False
        if self._rl is not None and getattr(self._rl, "active", False):
            self._rl.stop(reason)
            self.steps = int(getattr(self._rl, "steps", self.steps))
        else:
            print(f"[explore] 종료: {reason} (steps={self.steps})")

    def propose_action(
        self,
        gmap,
        map_id: int,
        current_pos: Coord,
    ) -> Optional[str]:
        if not self.active:
            return None
        if self._rl is None or self._rl.policy is None:
            print("[explore] RL 정책 없음 — Exploration 중단")
            self.active = False
            return None
        if not self._rl.active:
            self._rl.start(map_id, max_steps=self.max_steps)
        try:
            action = self._rl.propose_action(gmap, map_id, current_pos)
        except Exception as exc:
            print(f"[explore] RL 추론 실패 — Exploration 중단: {exc}")
            self.stop(f"추론 실패: {exc}")
            return None
        self.steps = int(getattr(self._rl, "steps", self.steps))
        if action is not None:
            return action
        if not self._rl.active:
            self.active = False
        return None
