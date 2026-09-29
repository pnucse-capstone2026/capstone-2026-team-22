"""play.py Exploration 스킬 ↔ 학습 체크포인트 브리지.

play.py 는 자체 PyBoy 를 돌리고, 스킬은 propose_action(...) 만 제공한다.
이 모듈은 pokemonred_puffer-main 의 observation/policy/action_exec/cheats 로
그 계약을 채운다 (별도 RedGymEnv 없음).

"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

from pokemonred_puffer.action_exec import DEFAULT_ACTION_FREQ, execute_rl_action
from pokemonred_puffer.actions import ACTION_NAMES
from pokemonred_puffer.emulator_runtime import get_game_coords, read_m, symbol_address
from pokemonred_puffer.exploration_cheats import resolve_cheat_profile
from pokemonred_puffer.play_env_parity import (
    TrainCheatBridge,
    apply_train_env_cheats,
    load_env_config,
)
from pokemonred_puffer.checkpoints import latest_model_path

# RedGymEnv VALID_ACTIONS 순서와 동일
RL_ACTIONS: tuple[str, ...] = ACTION_NAMES
Coord = Tuple[int, int]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def default_checkpoint() -> Path:
    """환경변수 → Exploration 최신 model_*.pt → runs/ 최신 model_*.pt."""
    env = os.environ.get("POKEMONRED_CHECKPOINT")
    if env:
        path = Path(env).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"POKEMONRED_CHECKPOINT not found: {path}")

    explore_dirs = [
        _PROJECT_ROOT.parent / "Skills" / "Exploration",
        _PROJECT_ROOT / "Skills" / "Exploration",
        # Skills/Exploration/pokemonred_puffer-main 중첩 복사본
        _PROJECT_ROOT.parent,
    ]
    for explore in explore_dirs:
        found = latest_model_path(explore, nested=False)
        if found is not None:
            return found.resolve()

    found = latest_model_path(_PROJECT_ROOT / "runs", nested=True)
    if found is not None:
        return found.resolve()

    raise FileNotFoundError(
        "No checkpoint. Place model_*.pt under Skills/Exploration/ "
        "or set POKEMONRED_CHECKPOINT."
    )


class ObsBridge:
    """play.py PyBoy → pokemonred_puffer.observation.get_obs 최소 env."""

    def __init__(
        self,
        pyboy: Any,
        *,
        reduce_res: bool = True,
        two_bit: bool = True,
    ) -> None:
        self.pyboy = pyboy
        self.screen = pyboy.screen
        self.reduce_res = bool(reduce_res)
        self.two_bit = bool(two_bit)
        self.visited_mask_tiles: Dict[int, Set[Coord]] = {}
        self._prev_coords: Optional[Tuple[int, int, int]] = None
        self._moved = 0
        self.target_map_id: Optional[int] = None
        self.target_map_ids: frozenset[int] = frozenset()
        self.connected_map_ids: frozenset[int] = frozenset()
        self._symbol_addresses: Dict[str, int] = {}

    def set_target(self, map_id: int) -> None:
        from pokemonred_puffer.data.target_scope import connected_map_ids, target_map_ids

        mid = int(map_id)
        self.target_map_id = mid
        self.target_map_ids = target_map_ids(mid)
        self.connected_map_ids = connected_map_ids(mid, self.target_map_ids)

    def _symbol_address(self, symbol: str) -> int:
        return symbol_address(self, symbol)

    def read_m(self, addr: str | int) -> int:
        return read_m(self, addr)

    def get_game_coords(self) -> Tuple[int, int, int]:
        return get_game_coords(self)

    def target_region(self, map_n: int):
        from pokemonred_puffer.data.target_scope import classify_target_region

        return classify_target_region(
            map_n, self.target_map_id, self.connected_map_ids, self.target_map_ids
        )

    def screen_obs(self):
        from pokemonred_puffer.observation import screen_obs

        return screen_obs(self)

    def reset_episode_obs(self, map_id: int | None = None) -> None:
        self.visited_mask_tiles.clear()
        self._prev_coords = None
        self._moved = 0
        if map_id is not None:
            self.set_target(int(map_id))


def load_agent(checkpoint: Path | str, device: str = "cpu"):
    """model_*.pt 로드 (CleanRL RecurrentPolicy 포함)."""
    import torch

    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # Unpickle용 정책 모듈 등록. pufferlib 없는 환경에서도 액션 수 검사/모킹이
    # 가능해야 하므로 여기서는 하드 실패하지 않는다.
    try:
        importlib.import_module("pokemonred_puffer.policies.multi_convolutional")
    except ModuleNotFoundError:
        pass
    agent = torch.load(path, map_location=device, weights_only=False)
    policy = agent
    while not hasattr(policy, "actor") and hasattr(policy, "policy"):
        policy = policy.policy
    action_count = getattr(getattr(policy, "actor", None), "out_features", None)
    if action_count != len(RL_ACTIONS):
        raise ValueError(
            f"Checkpoint has {action_count} actions, but this project uses "
            f"{len(RL_ACTIONS)} ({', '.join(RL_ACTIONS)}): {path}"
        )
    if hasattr(agent, "eval"):
        agent.eval()
    return agent


def policy_logits(agent, obs_t: dict, lstm_state):
    """Dict obs → action logits (+ 갱신된 LSTM state).

    저장 형태: CleanRL RecurrentPolicy → LSTMWrapper → MultiConvolutionalPolicy
    LSTMWrapper 는 flat 만 받으므로 안쪽 encode_observations(dict) 경로를 사용.
    """
    wrapper = agent
    if hasattr(agent, "policy") and not hasattr(agent, "recurrent"):
        if hasattr(agent.policy, "recurrent") and hasattr(agent.policy, "policy"):
            wrapper = agent.policy

    if hasattr(wrapper, "recurrent") and hasattr(wrapper, "policy"):
        inner = wrapper.policy
        hidden, lookup = inner.encode_observations(obs_t)
        batch = hidden.shape[0]
        seq = hidden.view(batch, 1, -1).transpose(0, 1)
        seq, lstm_state = wrapper.recurrent(seq, lstm_state)
        hidden = seq.transpose(0, 1).reshape(batch, -1)
        logits, _value = inner.decode_actions(hidden, lookup)
        return logits, lstm_state

    if hasattr(agent, "encode_observations") and hasattr(agent, "decode_actions"):
        hidden, lookup = agent.encode_observations(obs_t)
        logits, _value = agent.decode_actions(hidden, lookup)
        return logits, lstm_state

    raise TypeError(f"Unsupported policy type: {type(agent)!r}")


class PlayExplorationPolicy:
    """play.py ExplorationSkill 이 그대로 쓸 수 있는 RL 드라이버.

    계약:
      bind_pyboy(pyboy)
      start(map_id)
      stop(reason)
      propose_action(gmap, map_id, current_pos) -> "UP"|...|"LEFT"|None
      execute_action(action_name)  # 학습과 동일한 한 스텝
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        device: str = "cpu",
        greedy: bool = False,
        config_path: Path | str | None = None,
        # 학습 기본과 동일. 필요할 때만 "wild" 또는 "off"로 낮춘다.
        cheat_profile: str = "full",
    ) -> None:
        self.model_path = (
            Path(model_path).expanduser().resolve()
            if model_path
            else default_checkpoint()
        )
        self.device = device
        self.greedy = greedy
        self.cheat_profile = resolve_cheat_profile(cheat_profile)
        self.config_path = Path(config_path) if config_path else None

        self.env_config = load_env_config(self.config_path)
        self.action_freq = int(self.env_config.get("action_freq", DEFAULT_ACTION_FREQ))
        self.default_max_steps = int(self.env_config.get("max_steps", 2000))
        self.expect_reduce_res = bool(self.env_config.get("reduce_res", True))
        self.expect_two_bit = bool(self.env_config.get("two_bit", True))

        self.policy = None
        self._lstm_state = None
        self._obs_env: Optional[ObsBridge] = None
        self._cheat: Optional[TrainCheatBridge] = None
        self._pyboy: Any = None

        self.active = False
        self.steps = 0
        self.max_steps = self.default_max_steps

        self.load_policy()

    def load_policy(self) -> None:
        self.policy = load_agent(self.model_path, self.device)
        self._lstm_state = None
        print(
            f"[explore] RL 정책 로드: {self.model_path.name} ({self.device}) "
            f"action_freq={self.action_freq} cheats={self.cheat_profile}"
        )

    def bind_pyboy(
        self,
        pyboy: Any,
        *,
        reduce_res: bool | None = None,
        two_bit: bool | None = None,
    ) -> None:
        rr = self.expect_reduce_res if reduce_res is None else bool(reduce_res)
        tb = self.expect_two_bit if two_bit is None else bool(two_bit)
        if rr != self.expect_reduce_res or tb != self.expect_two_bit:
            raise ValueError(
                f"Obs flags must match training config: "
                f"reduce_res={self.expect_reduce_res}, two_bit={self.expect_two_bit} "
                f"(got reduce_res={rr}, two_bit={tb})"
            )
        if self._pyboy is pyboy and self._obs_env is not None:
            return
        self._pyboy = pyboy
        self._obs_env = ObsBridge(pyboy, reduce_res=rr, two_bit=tb)
        if self.cheat_profile != "off":
            self._cheat = apply_train_env_cheats(
                pyboy, self.env_config, profile=self.cheat_profile
            )
        else:
            self._cheat = None

    def start(self, map_id: int, *, max_steps: int | None = None) -> None:
        self.active = True
        self.steps = 0
        self.max_steps = int(max_steps) if max_steps is not None else self.default_max_steps
        self._lstm_state = None
        if self._obs_env is not None:
            self._obs_env.reset_episode_obs(int(map_id))
        if self._cheat is not None:
            self._cheat.apply_episode_start()
        if self.policy is None:
            self.load_policy()
        print(
            f"[explore] 시작 (map_id={map_id}, "
            f"max_steps={self.max_steps}, action_freq={self.action_freq}, "
            f"cheats={self.cheat_profile})"
        )

    def stop(self, reason: str = "중단") -> None:
        if not self.active:
            return
        self.active = False
        self._lstm_state = None
        print(f"[explore] 종료: {reason} (steps={self.steps})")

    def execute_action(self, action_name: str) -> None:
        """학습 환경과 동일한 입력 타이밍으로 한 스텝 실행한다."""
        if self._pyboy is None:
            raise RuntimeError("bind_pyboy() first")
        if self._cheat is not None:
            self._cheat.before_action()
        execute_rl_action(
            self._pyboy,
            action_name,
            action_freq=self.action_freq,
            disable_ai_actions=bool(
                self._cheat.disable_ai_actions if self._cheat is not None else False
            ),
            realtime=True,
        )
        if self._cheat is not None:
            self._cheat.after_action()

    def propose_action(
        self,
        gmap,
        map_id: int,
        current_pos: Coord,
    ) -> Optional[str]:
        del gmap, current_pos
        if not self.active:
            return None

        self.steps += 1
        if self.steps > self.max_steps:
            self.stop(f"상한 {self.max_steps}스텝")
            return None

        if self.policy is None or self._obs_env is None:
            return None

        try:
            import torch
            from pokemonred_puffer.observation import get_obs

            if self._obs_env.target_map_id is None:
                self._obs_env.set_target(int(map_id))

            obs = get_obs(self._obs_env)
            # Dict obs: LSTMWrapper.forward 는 flat 만 받음 → encode 경로 사용
            # (train evaluate 는 vecenv flat obs 로 agent(...) 호출)
            obs_t = {
                key: torch.as_tensor(value).unsqueeze(0).to(self.device)
                for key, value in obs.items()
            }
            with torch.no_grad():
                logits, self._lstm_state = policy_logits(
                    self.policy, obs_t, self._lstm_state
                )
                if self.greedy:
                    idx = int(torch.argmax(logits, dim=-1).reshape(-1)[0].item())
                else:
                    dist = torch.distributions.Categorical(logits=logits)
                    idx = int(dist.sample().reshape(-1)[0].item())
            if 0 <= idx < len(RL_ACTIONS):
                return RL_ACTIONS[idx]
            return None
        except Exception as exc:
            raise RuntimeError("exploration policy inference failed") from exc
