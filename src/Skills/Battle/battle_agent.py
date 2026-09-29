"""
battle_agent_5.py

Pokemon Red(PyBoy) 실게임 Battle RL 연결기.

연결 구조
---------
BattleRAMReader
    -> BattleObs.model_input()
       {
           "observation": (276,) float32,
           "action_mask": (10,) int64,
       }
    -> Masked PPO policy
    -> RL action 0..9
    -> rl_action_to_macro()
    -> BattleMacro.execute_action()

학습 모델의 action 순서
----------------------
0..5 : switch party 0..5
6..9 : move 0..3

BattleMacro 순서
----------------
0..3 : move 0..3
4..9 : switch party 0..5

따라서 model output을 BattleMacro에 직접 넣지 않는다.
항상 rl_action_to_macro()를 거친다.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

# SB3 2.9는 Gymnasium을 사용하지만, 환경에 구형 `gym`이 설치돼 있으면
# 호환 검사 목적으로 함께 import한다. PyBoy의 SDL2 창이 실행 중인 runtime에
# 이 선택적 legacy package를 늦게 import할 이유가 없고 실제 환경에서 그
# 직후 SDL 루프가 종료됐다. ModuleNotFound와 같은 상태로 만들어 SB3가
# Gymnasium 경로만 사용하게 한다. 학습/모델 형식에는 영향이 없다.
if "gym" not in sys.modules:
    sys.modules["gym"] = None

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .battle_obs import (
    BattleObs,
    BattleObsError,
    N_FEATURES,
    GLOBAL_DIM,
    MOVE_SLOT_DIM,
    OPP_MOVE_SLOT_DIM,
    OWN_TEAM_SLOT_DIM,
    OPP_TEAM_SLOT_DIM,
    OBSERVATION_LAYOUT_VERSION,
    MOVE_CATEGORY_INDEX,
    EFFECTIVENESS_INDEX,
    PERSISTENT_EFFECT_INDEX,
    PERSISTENT_EFFECT_ACTIVE_INDEX,
    OPPONENT_DEBUFF_INDEX,
    DEBUFF_REMAINING_CAPACITY_INDEX,
    rl_action_to_macro,
)
from .hierarchical_policy import (
    MaskedActorCriticPolicy as _RuntimeHierarchicalPolicy,
)


RL_ACTION_NAMES = (
    "switch0",
    "switch1",
    "switch2",
    "switch3",
    "switch4",
    "switch5",
    "move0",
    "move1",
    "move2",
    "move3",
)

MACRO_ACTION_NAMES = (
    "move0",
    "move1",
    "move2",
    "move3",
    "switch0",
    "switch1",
    "switch2",
    "switch3",
    "switch4",
    "switch5",
)


# ===========================================================================
# Legacy embedded policy (final model.zip에서는 사용하지 않음)
# ===========================================================================
#
# 예전 shared-candidate checkpoint의 state_dict에는:
#   move_scorer.*
#   switch_scorer.*
#   action_type_scorer.*
# 가 저장되어 있다.
#
# 따라서 일반 SB3 action_net actor가 아니라, 학습 당시 사용한
# slot-independent shared candidate scorer 구조를 정확히 복원해야 한다.
#
class FeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(
            observation_space,
            features_dim=N_FEATURES,
        )

    def forward(self, obs):
        x = obs["observation"].float()
        return torch.nan_to_num(
            x,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )


class _LegacyMaskedActorCriticPolicy(ActorCriticPolicy):
    """
    Masked PPO actor with slot-independent shared candidate scorers.

    Action:
      0..5 = switch candidates
      6..9 = move candidates

    Actor:
      - one shared switch_scorer for six party candidates
      - one shared move_scorer for four move candidates

    Critic:
      ordinary vf=[128, 128]
    """

    NUM_SWITCHES = 6
    NUM_MOVES = 4
    INITIAL_SWITCH_LOGIT_PRIOR = -0.250

    def __init__(self, *args, **kwargs):
        # PPO.load()가 저장된 policy kwargs를 전달할 수 있으므로
        # architecture 관련 중복 값은 제거하고 학습 당시 구조를 강제한다.
        kwargs.pop("features_extractor_class", None)
        kwargs.pop("net_arch", None)

        super().__init__(
            *args,
            **kwargs,
            # Actor는 SB3의 일반 latent_pi -> action_net을 사용하지 않는다.
            # Critic만 표준 128x128 value network를 사용.
            net_arch=dict(
                pi=[],
                vf=[128, 128],
            ),
            features_extractor_class=FeaturesExtractor,
        )

        # evaluator/debug compatibility
        self._mask = None
        self._last_effective_mask = None

    def _build(self, lr_schedule) -> None:
        """Build critic plus shared switch/move candidate scorers."""
        self._build_mlp_extractor()

        # Actor logits are NOT produced by ordinary SB3 action_net.
        # Identity has no trainable state_dict parameters.
        self.action_net = nn.Identity()

        # Standard critic head.
        self.value_net = nn.Linear(
            self.mlp_extractor.latent_dim_vf,
            1,
        )

        # permutation-invariant context:
        # global
        # + mean(own moves)
        # + mean(revealed opponent moves)
        # + mean(own party)
        # + mean(revealed opponent party)
        self.context_dim = (
            GLOBAL_DIM
            + MOVE_SLOT_DIM
            + OPP_MOVE_SLOT_DIM
            + OWN_TEAM_SLOT_DIM
            + OPP_TEAM_SLOT_DIM
        )

        # SAME scorer reused for all four move slots.
        self.move_scorer = nn.Sequential(
            nn.Linear(
                self.context_dim + MOVE_SLOT_DIM,
                128,
            ),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # SAME scorer reused for all six switch slots.
        self.switch_scorer = nn.Sequential(
            nn.Linear(
                self.context_dim + OWN_TEAM_SLOT_DIM,
                128,
            ),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Battle_RL_r2: switch 후보군과 move 후보군의 logit 기준을
        # 공통 context에서 직접 보정한다.
        self.action_type_scorer = nn.Sequential(
            nn.Linear(self.context_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 2),
        )

        if self.ortho_init:
            sqrt2 = 2.0 ** 0.5

            self.mlp_extractor.apply(
                lambda module: self.init_weights(
                    module,
                    gain=sqrt2,
                )
            )
            self.value_net.apply(
                lambda module: self.init_weights(
                    module,
                    gain=1.0,
                )
            )
            self.move_scorer.apply(
                lambda module: self.init_weights(
                    module,
                    gain=sqrt2,
                )
            )
            self.switch_scorer.apply(
                lambda module: self.init_weights(
                    module,
                    gain=sqrt2,
                )
            )
            self.action_type_scorer.apply(
                lambda module: self.init_weights(
                    module,
                    gain=sqrt2,
                )
            )

            # Near-uniform initial action preferences.
            self.init_weights(
                self.move_scorer[-1],
                gain=0.01,
            )
            self.init_weights(
                self.switch_scorer[-1],
                gain=0.01,
            )
            self.init_weights(
                self.action_type_scorer[-1],
                gain=0.01,
            )

        with torch.no_grad():
            self.action_type_scorer[-1].bias[0] = (
                self.INITIAL_SWITCH_LOGIT_PRIOR
            )
            self.action_type_scorer[-1].bias[1] = 0.0

        # scorer parameters must exist before optimizer creation.
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _split_features(
        self,
        features: torch.Tensor,
    ):
        """Split compact features back into candidate groups."""
        if features.ndim == 1:
            features = features.unsqueeze(0)

        batch_size = features.shape[0]
        offset = 0

        global_state = features[
            :,
            offset:offset + GLOBAL_DIM,
        ]
        offset += GLOBAL_DIM

        size = self.NUM_MOVES * MOVE_SLOT_DIM
        own_moves = features[
            :,
            offset:offset + size,
        ].reshape(
            batch_size,
            self.NUM_MOVES,
            MOVE_SLOT_DIM,
        )
        offset += size

        size = self.NUM_MOVES * OPP_MOVE_SLOT_DIM
        opponent_moves = features[
            :,
            offset:offset + size,
        ].reshape(
            batch_size,
            self.NUM_MOVES,
            OPP_MOVE_SLOT_DIM,
        )
        offset += size

        size = self.NUM_SWITCHES * OWN_TEAM_SLOT_DIM
        own_team = features[
            :,
            offset:offset + size,
        ].reshape(
            batch_size,
            self.NUM_SWITCHES,
            OWN_TEAM_SLOT_DIM,
        )
        offset += size

        size = self.NUM_SWITCHES * OPP_TEAM_SLOT_DIM
        opponent_team = features[
            :,
            offset:offset + size,
        ].reshape(
            batch_size,
            self.NUM_SWITCHES,
            OPP_TEAM_SLOT_DIM,
        )
        offset += size

        if offset != N_FEATURES:
            raise RuntimeError(
                "Observation layout mismatch: "
                f"parsed={offset}, expected={N_FEATURES}"
            )

        return (
            global_state,
            own_moves,
            opponent_moves,
            own_team,
            opponent_team,
        )

    def _candidate_logits(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Score every candidate with shared slot-independent networks."""
        (
            global_state,
            own_moves,
            opponent_moves,
            own_team,
            opponent_team,
        ) = self._split_features(features)

        batch_size = features.shape[0]

        context = torch.cat(
            [
                global_state,
                own_moves.mean(dim=1),
                opponent_moves.mean(dim=1),
                own_team.mean(dim=1),
                opponent_team.mean(dim=1),
            ],
            dim=-1,
        )
        action_type_logits = self.action_type_scorer(context)

        # Switch action logits 0..5.
        switch_context = context.unsqueeze(1).expand(
            -1,
            self.NUM_SWITCHES,
            -1,
        )
        switch_inputs = torch.cat(
            [
                switch_context,
                own_team,
            ],
            dim=-1,
        )

        switch_logits = self.switch_scorer(
            switch_inputs.reshape(
                batch_size * self.NUM_SWITCHES,
                -1,
            )
        ).reshape(
            batch_size,
            self.NUM_SWITCHES,
        )
        switch_logits = switch_logits + action_type_logits[:, 0:1]

        # Move action logits 6..9.
        move_context = context.unsqueeze(1).expand(
            -1,
            self.NUM_MOVES,
            -1,
        )
        move_inputs = torch.cat(
            [
                move_context,
                own_moves,
            ],
            dim=-1,
        )

        move_logits = self.move_scorer(
            move_inputs.reshape(
                batch_size * self.NUM_MOVES,
                -1,
            )
        ).reshape(
            batch_size,
            self.NUM_MOVES,
        )
        move_logits = move_logits + action_type_logits[:, 1:2]

        return torch.cat(
            [
                switch_logits,
                move_logits,
            ],
            dim=-1,
        )

    @staticmethod
    def _validated_mask(
        action_logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)

        valid_mask = mask > 0.5
        valid_counts = valid_mask.sum(dim=-1)

        if torch.any(valid_counts == 0):
            bad_rows = torch.nonzero(
                valid_counts == 0,
                as_tuple=False,
            ).flatten().tolist()

            raise RuntimeError(
                "ALL-ZERO ACTION MASK detected. "
                f"bad batch rows={bad_rows}, "
                f"action_mask={mask.detach().cpu().tolist()}"
            )

        if valid_mask.shape != action_logits.shape:
            raise RuntimeError(
                "Action-mask shape mismatch: "
                f"mask={tuple(valid_mask.shape)}, "
                f"logits={tuple(action_logits.shape)}"
            )

        return valid_mask

    def _effective_action_mask(
        self,
        features: torch.Tensor,
        action_logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Battle_RL_r2 정책과 같은 persistent/debuff 중복 필터."""
        base_valid = self._validated_mask(action_logits, mask)
        _, own_moves, _, _, _ = self._split_features(features)

        persistent = own_moves[:, :, PERSISTENT_EFFECT_INDEX] > 0.5
        active = own_moves[:, :, PERSISTENT_EFFECT_ACTIVE_INDEX] > 0.5
        debuff = own_moves[:, :, OPPONENT_DEBUFF_INDEX] > 0.0
        debuff_saturated = (
            own_moves[:, :, DEBUFF_REMAINING_CAPACITY_INDEX] <= 1.0e-6
        )
        damaging = own_moves[:, :, MOVE_CATEGORY_INDEX] > 0.5
        type_immune = (
            damaging
            & (own_moves[:, :, EFFECTIVENESS_INDEX] <= 1.0e-6)
        )
        redundant = (
            (persistent & active)
            | (debuff & debuff_saturated)
            | type_immune
        )

        effective = base_valid.clone()
        move_slice = slice(
            self.NUM_SWITCHES,
            self.NUM_SWITCHES + self.NUM_MOVES,
        )
        effective[:, move_slice] &= ~redundant

        empty_rows = effective.sum(dim=-1) == 0
        if torch.any(empty_rows):
            effective[empty_rows] = base_valid[empty_rows]

        return effective

    def _distribution_from_features(
        self,
        obs,
        features: torch.Tensor,
    ):
        action_logits = self._candidate_logits(features)

        if not torch.isfinite(action_logits).all():
            raise FloatingPointError(
                "Policy produced non-finite logits before masking: "
                f"{action_logits.detach().cpu().tolist()}"
            )

        mask = obs["action_mask"]
        self._mask = mask

        valid_mask = self._effective_action_mask(
            features,
            action_logits,
            mask,
        )
        self._last_effective_mask = valid_mask

        masked_logits = action_logits.masked_fill(
            ~valid_mask,
            -1.0e9,
        )

        if not torch.isfinite(masked_logits).all():
            raise FloatingPointError(
                "Policy produced non-finite logits after masking."
            )

        return self.action_dist.proba_distribution(
            action_logits=masked_logits
        )

    def forward(
        self,
        obs,
        deterministic=False,
    ):
        features = self.extract_features(obs)

        if isinstance(features, tuple):
            raise RuntimeError(
                "Shared-candidate policy requires "
                "share_features_extractor=True"
            )

        latent_vf = self.mlp_extractor.forward_critic(
            features
        )
        values = self.value_net(latent_vf)

        distribution = self._distribution_from_features(
            obs,
            features,
        )
        actions = distribution.get_actions(
            deterministic=deterministic
        )
        log_prob = distribution.log_prob(actions)

        actions = actions.reshape(
            (-1, *self.action_space.shape)
        )

        return actions, values, log_prob


    def evaluate_actions(
        self,
        obs,
        actions,
    ):
        features = self.extract_features(obs)

        if isinstance(features, tuple):
            raise RuntimeError(
                "Shared-candidate policy requires "
                "share_features_extractor=True"
            )

        latent_vf = self.mlp_extractor.forward_critic(
            features
        )
        values = self.value_net(latent_vf)

        distribution = self._distribution_from_features(
            obs,
            features,
        )

        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        return values, log_prob, entropy

    def get_distribution(self, obs):
        features = self.extract_features(obs)

        if isinstance(features, tuple):
            raise RuntimeError(
                "Shared-candidate policy requires "
                "share_features_extractor=True"
            )

        return self._distribution_from_features(
            obs,
            features,
        )


# 최종 model.zip은 Python 3.14에서 v15 policy class를 cloudpickle로 저장했다.
# Python 3.11 runtime에서는 그 code object를 직접 실행하지 않고, source가
# 정확히 같은 hierarchical_policy.py 클래스를 custom_objects로 주입한다.
# 따라서 RL/train.py와 poke-env/Showdown 학습 의존성도 필요하지 않다.


# ===========================================================================
# NumPy 2.x pickle -> NumPy 1.x compatibility
# ===========================================================================
#
# NumPy 2.x는 내부 module 경로가 numpy._core.* 로 저장될 수 있다.
# NumPy 1.23.x에는 동일 구현이 numpy.core.* 아래 있으므로,
# PPO.load() 전에 pickle import 경로만 alias한다.
#
# 이 패치는 모델 weight를 바꾸지 않고, pickle 역직렬화 시 module 이름만
# 호환시킨다. NumPy 자체를 monkey-patch하지 않는다.
#
def _install_numpy_pickle_compat() -> list[tuple[str, str]]:
    """
    NumPy 2.x에서 저장된 pickle을 NumPy 1.x에서 읽을 수 있도록
    numpy._core.* -> numpy.core.* alias를 sys.modules에 등록한다.

    반환:
        실제 등록된 (alias, target) 목록.
    """
    import importlib
    import numpy as np

    installed: list[tuple[str, str]] = []

    # NumPy 2.x에서는 numpy._core가 실제 package이므로 아무것도 할 필요 없음.
    try:
        importlib.import_module("numpy._core")
        return installed
    except ModuleNotFoundError:
        pass

    # NumPy 1.x에서만 실행.
    aliases = {
        "numpy._core": "numpy.core",
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core.umath": "numpy.core.umath",
        "numpy._core.fromnumeric": "numpy.core.fromnumeric",
        "numpy._core.shape_base": "numpy.core.shape_base",
        "numpy._core.numerictypes": "numpy.core.numerictypes",
        "numpy._core._dtype": "numpy.core._dtype",
        "numpy._core._type_aliases": "numpy.core._type_aliases",
        "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
    }

    for alias_name, target_name in aliases.items():
        if alias_name in sys.modules:
            continue

        try:
            target_module = importlib.import_module(target_name)
        except ModuleNotFoundError:
            # NumPy minor version에 없는 내부 module은 그냥 건너뛴다.
            continue

        sys.modules[alias_name] = target_module
        installed.append((alias_name, target_name))

    # pickle이 getattr(np, "_core") 형태로 접근하는 경우도 대비.
    if not hasattr(np, "_core"):
        try:
            setattr(np, "_core", importlib.import_module("numpy.core"))
        except Exception:
            pass

    return installed



class BattleAgentError(RuntimeError):
    pass


@dataclass(frozen=True)
class BattleDecision:
    rl_action: int
    macro_action: int
    rl_action_name: str
    macro_action_name: str
    action_mask: tuple[int, ...]


@dataclass(frozen=True)
class PolicyDiagnostics:
    """
    한 PPO decision에서 실제로 계산된 진단값.

    raw_logits:
        shared candidate scorer가 mask 적용 전에 낸 10개 점수.

    masked_logits:
        invalid action에 -1e9를 적용한 뒤의 점수.

    probabilities:
        masked_logits에 softmax를 적용한 최종 행동 확률.

    state_value:
        PPO critic의 V(s).

    observation:
        이번 decision에서 모델이 실제로 받은 276개 float feature.
        test_agent_5.py가 move feature를 사람이 읽기 좋게 출력할 때 사용한다.
    """
    state_value: float
    raw_logits: tuple[float, ...]
    masked_logits: tuple[float, ...]
    probabilities: tuple[float, ...]
    observation: tuple[float, ...]


@dataclass(frozen=True)
class BattleStepResult:
    decision: BattleDecision
    next_decision_ready: bool
    battle_still_active: bool


class BattleAgent:
    """
    학습된 Masked PPO와 emulator BattleObs/BattleMacro를 연결한다.

    Parameters
    ----------
    battle_ram:
        gb_memory.BattleRAMReader

    battle_macro:
        battle_macro.BattleMacro 또는 같은 execute_action() 인터페이스

    model_path:
        학습된 .zip 모델 경로, 예: models/all_5m.zip

    deterministic:
        True면 evaluation 때와 동일하게 가장 높은 확률의 legal action 선택.
    """

    def __init__(
        self,
        *,
        battle_ram,
        battle_macro,
        model_path: str | Path,
        battle_obs: BattleObs | None = None,
        species_path: str | Path | None = None,
        moves_path: str | Path | None = None,
        device: str = "auto",
        deterministic: bool = True,
        policy_module: str | None = None,
        debug: bool = False,
    ):
        self.battle_ram = battle_ram
        self.battle_macro = battle_macro
        self.model_path = Path(model_path)
        self.device = str(device)
        self.deterministic = bool(deterministic)
        self.debug = bool(debug)

        if battle_obs is None:
            self.battle_obs = BattleObs(
                battle_ram,
                species_path=species_path,
                moves_path=moves_path,
                debug=debug,
            )
        else:
            self.battle_obs = battle_obs

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"RL model이 없습니다: {self.model_path}"
            )

        # Python 3.11에서 안전하게 정의된 source-identical v15 class 사용.
        # 학습용 RL/train.py와 poke-env는 runtime에 필요하지 않다.
        self.policy_module = "Skills.Battle.hierarchical_policy.MaskedActorCriticPolicy"

        # NumPy 2.x에서 저장된 SB3 pickle을 NumPy 1.23.x에서 읽을 때
        # "No module named numpy._core"가 발생하는 경우를 위한 호환 alias.
        numpy_compat = _install_numpy_pickle_compat()

        if self.debug and numpy_compat:
            print(
                "[BattleAgent] NumPy pickle compatibility aliases:"
            )
            for alias_name, target_name in numpy_compat:
                print(
                    f"  {alias_name} -> {target_name}"
                )

        try:
            self.model = PPO.load(
                str(self.model_path),
                device=self.device,
                custom_objects={
                    # model.zip was saved by Python 3.14.  Never execute that
                    # cloudpickled code object in the Python 3.11 macro env;
                    # use the source-identical runtime class instead.
                    "policy_class": _RuntimeHierarchicalPolicy,
                },
            )
        except Exception as exc:
            raise BattleAgentError(
                "PPO model load 실패. "
                "NumPy 호환 + runtime hierarchical policy 복원 뒤에도 "
                "load가 실패했습니다. "
                "모델 재학습부터 하지 말고 아래 cause를 확인하세요. "
                f"model={self.model_path} | "
                f"numpy={np.__version__} | "
                f"cause={type(exc).__name__}: {exc}"
            ) from exc

        self._validate_loaded_model()
        self.policy_module = (
            f"{type(self.model.policy).__module__}."
            f"{type(self.model.policy).__qualname__}"
        )

        if self.debug:
            print(
                "[BattleAgent] model ready "
                f"path={self.model_path} "
                f"device={getattr(self.model, 'device', self.device)} "
                f"policy_module={self.policy_module}"
            )

    # ------------------------------------------------------------------
    # Model loading / validation
    # ------------------------------------------------------------------

    @staticmethod
    def _import_custom_policy(
        explicit_module: str | None = None,
    ) -> str:
        """
        backward-compatible helper.

        하위 호환용 helper. 최종 model.zip의 Python 3.14 code object 대신
        source-identical Python 3.11 runtime policy를 사용한다.
        """
        return "Skills.Battle.hierarchical_policy.MaskedActorCriticPolicy"


    def _validate_loaded_model(self) -> None:
        """
        구형 observation checkpoint를 실수로 연결하는 것을 막는다.
        """
        space = getattr(self.model, "observation_space", None)

        if space is None:
            raise BattleAgentError(
                "loaded PPO에 observation_space가 없습니다."
            )

        spaces = getattr(space, "spaces", None)

        if spaces is None:
            raise BattleAgentError(
                "이 PPO model은 Dict observation 모델이 아닙니다. "
                "현재 Battle RL policy는 observation + action_mask Dict를 "
                "사용해야 합니다. "
                f"loaded_space={space!r}"
            )

        required = {"observation", "action_mask"}
        missing = required - set(spaces.keys())

        if missing:
            raise BattleAgentError(
                "model observation Dict key 불일치: "
                f"missing={sorted(missing)}, "
                f"keys={sorted(spaces.keys())}"
            )

        obs_shape = tuple(
            getattr(spaces["observation"], "shape", ())
        )
        mask_shape = tuple(
            getattr(spaces["action_mask"], "shape", ())
        )

        if obs_shape != (N_FEATURES,):
            raise BattleAgentError(
                "PPO observation feature 수가 현재 battle_obs와 다릅니다: "
                f"model={obs_shape}, battle_obs={(N_FEATURES,)}. "
                f"현재 layout={OBSERVATION_LAYOUT_VERSION}. "
                "구형 194/198/202 feature 모델은 호환되지 않습니다."
            )

        if mask_shape != (10,):
            raise BattleAgentError(
                "PPO action_mask shape가 10이 아닙니다: "
                f"model={mask_shape}"
            )

        obs_dtype = np.dtype(getattr(spaces["observation"], "dtype", None))
        mask_dtype = np.dtype(getattr(spaces["action_mask"], "dtype", None))

        if obs_dtype != np.dtype(np.float32):
            raise BattleAgentError(
                "PPO observation dtype가 float32가 아닙니다: "
                f"model={obs_dtype}"
            )

        if mask_dtype != np.dtype(np.int64):
            raise BattleAgentError(
                "PPO action_mask dtype가 int64가 아닙니다: "
                f"model={mask_dtype}"
            )

        action_space = getattr(self.model, "action_space", None)
        n_actions = getattr(action_space, "n", None)

        if n_actions is not None and int(n_actions) != 10:
            raise BattleAgentError(
                "PPO action space가 10개가 아닙니다: "
                f"n={n_actions}"
            )

        required_policy_methods = (
            "_hierarchical_components",
            "_deterministic_hierarchical_actions",
        )
        missing_methods = [
            name
            for name in required_policy_methods
            if not callable(getattr(self.model.policy, name, None))
        ]
        if missing_methods:
            raise BattleAgentError(
                "checkpoint policy가 최종 hierarchical v15 구조가 아닙니다: "
                f"missing={missing_methods}, "
                f"policy={type(self.model.policy).__name__}"
            )

    # ------------------------------------------------------------------
    # Battle lifecycle
    # ------------------------------------------------------------------

    def reset_battle(self) -> None:
        """새 배틀이 시작될 때 한 번 호출."""
        self.battle_obs.reset_battle()

        if self.debug:
            print("[BattleAgent] battle tracker reset")

    # alias
    start_battle = reset_battle

    # ------------------------------------------------------------------
    # Observation -> PPO
    # ------------------------------------------------------------------

    def get_model_input(self, state=None) -> dict[str, np.ndarray]:
        """
        RAM -> 학습 PPO가 받은 정확한 Dict 형식.
        """
        model_input = self.battle_obs.model_input(state)
        self.battle_obs.validate_model_input(model_input)
        return model_input

    def predict_with_diagnostics(
        self,
        state=None,
        *,
        deterministic: bool | None = None,
    ) -> tuple[BattleDecision, PolicyDiagnostics]:
        """
        현재 state에서 PPO 행동 + 내부 scorer 값을 함께 반환한다.

        반환되는 probability는 action_mask 적용 뒤 최종 확률이다.
        raw_logits는 mask 적용 전 shared candidate scorer 출력이다.
        """
        if deterministic is None:
            deterministic = self.deterministic

        model_input = self.get_model_input(state)

        observation = model_input["observation"]
        mask = model_input["action_mask"]

        if observation.shape != (N_FEATURES,):
            raise BattleAgentError(
                f"obs shape error: {observation.shape}"
            )

        if mask.shape != (10,):
            raise BattleAgentError(
                f"mask shape error: {mask.shape}"
            )

        valid_actions = np.flatnonzero(mask > 0.5)

        if len(valid_actions) == 0:
            raise BattleAgentError(
                "ALL-ZERO ACTION MASK: PPO가 선택할 행동이 없습니다."
            )

        obs_tensor, _ = self.model.policy.obs_to_tensor(
            model_input
        )

        self.model.policy._mask = obs_tensor["action_mask"]

        with torch.no_grad():
            features = self.model.policy.extract_features(
                obs_tensor
            )

            if isinstance(features, tuple):
                raise BattleAgentError(
                    "shared-candidate policy expected a shared feature tensor"
                )

            components = self.model.policy._hierarchical_components(
                obs_tensor,
                features,
            )
            raw_logits = components["raw_candidate_logits"]
            valid_mask = components["effective_mask"]
            masked_logits = components["joint_logits"]

            probabilities = torch.softmax(
                masked_logits,
                dim=-1,
            )

            latent_vf = self.model.policy.mlp_extractor.forward_critic(
                features
            )
            values = self.model.policy.value_net(
                latent_vf
            )

            if deterministic:
                action_tensor = (
                    self.model.policy._deterministic_hierarchical_actions(
                        components
                    )
                )
            else:
                distribution = (
                    self.model.policy.action_dist.proba_distribution(
                        action_logits=masked_logits
                    )
                )
                action_tensor = distribution.get_actions(deterministic=False)

        rl_action = int(action_tensor.item())

        if not 0 <= rl_action <= 9:
            raise BattleAgentError(
                f"RL action 범위 오류: {rl_action}"
            )

        if mask[rl_action] <= 0.5:
            raise BattleAgentError(
                "MASK ERROR: PPO가 invalid action을 선택했습니다. "
                f"selected={rl_action}, "
                f"valid={valid_actions.tolist()}, "
                f"mask={mask.astype(int).tolist()}"
            )

        macro_action = rl_action_to_macro(rl_action)

        decision = BattleDecision(
            rl_action=rl_action,
            macro_action=macro_action,
            rl_action_name=RL_ACTION_NAMES[rl_action],
            macro_action_name=MACRO_ACTION_NAMES[macro_action],
            action_mask=tuple(
                int(v)
                for v in (
                    valid_mask.detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                    .astype(bool)
                    .tolist()
                )
            ),
        )

        raw_np = (
            raw_logits.detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
        masked_np = (
            masked_logits.detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
        prob_np = (
            probabilities.detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
        value_np = (
            values.detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        diagnostics = PolicyDiagnostics(
            state_value=float(value_np[0]),
            raw_logits=tuple(float(x) for x in raw_np.tolist()),
            masked_logits=tuple(float(x) for x in masked_np.tolist()),
            probabilities=tuple(float(x) for x in prob_np.tolist()),
            observation=tuple(
                float(x)
                for x in observation.reshape(-1).tolist()
            ),
        )

        if len(diagnostics.raw_logits) != 10:
            raise BattleAgentError(
                "PPO raw logits shape mismatch: "
                f"{len(diagnostics.raw_logits)}"
            )

        if len(diagnostics.probabilities) != 10:
            raise BattleAgentError(
                "PPO probability shape mismatch: "
                f"{len(diagnostics.probabilities)}"
            )

        if self.debug:
            print(
                "[BattleAgent] decision "
                f"RL={decision.rl_action}:{decision.rl_action_name} "
                f"-> macro={decision.macro_action}:{decision.macro_action_name} "
                f"valid={valid_actions.tolist()} "
                f"value={diagnostics.state_value:.6f}"
            )

        return decision, diagnostics

    def predict(
        self,
        state=None,
        *,
        deterministic: bool | None = None,
    ) -> BattleDecision:
        """
        기존 API 유지.
        진단값이 필요하면 predict_with_diagnostics()를 사용한다.
        """
        decision, _ = self.predict_with_diagnostics(
            state,
            deterministic=deterministic,
        )
        return decision


    # ------------------------------------------------------------------
    # PPO -> BattleMacro
    # ------------------------------------------------------------------

    def act(
        self,
        state=None,
        *,
        deterministic: bool | None = None,
    ) -> BattleStepResult:
        """
        한 번의 RL decision을 실제 게임에서 끝까지 실행한다.

        호출 전제:
            BattleMacro가 판단하는 플레이어 메인 배틀 메뉴 READY 상태.

        실행:
            RAM -> model input
            -> PPO action
            -> macro action 변환
            -> BattleMacro.execute_action()
            -> 다음 decision menu까지 진행
            -> 상대 공개정보 / turn tracker 갱신
        """
        decision = self.predict(
            state,
            deterministic=deterministic,
        )

        if self.debug:
            print(
                "[BattleAgent] execute "
                f"{decision.macro_action_name}"
            )

        try:
            ready = bool(
                self.battle_macro.execute_action(
                    decision.macro_action
                )
            )
        except Exception as exc:
            raise BattleAgentError(
                "BattleMacro action 실행 실패: "
                f"RL={decision.rl_action_name}, "
                f"macro={decision.macro_action_name}"
            ) from exc

        # BattleMacro v5는 다음 decision menu가 실제 돌아오면 True,
        # battle end면 False를 반환한다.
        battle_still_active = False

        try:
            after_state = self.battle_ram.read()
            battle_still_active = bool(
                getattr(after_state, "in_battle", False)
            )
        except Exception:
            after_state = None

        if ready and battle_still_active:
            # 반드시 macro가 다음 메뉴까지 처리한 뒤 호출.
            self.battle_obs.finish_turn(after_state)

        if self.debug:
            print(
                "[BattleAgent] action complete "
                f"next_ready={ready} "
                f"in_battle={battle_still_active}"
            )

        return BattleStepResult(
            decision=decision,
            next_decision_ready=ready,
            battle_still_active=battle_still_active,
        )

    # 더 읽기 쉬운 alias.
    step = act

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def describe_model(self) -> str:
        space = getattr(self.model, "observation_space", None)
        action_space = getattr(self.model, "action_space", None)
        policy = getattr(self.model, "policy", None)

        return "\n".join([
            "[BattleAgent model]",
            f"path={self.model_path}",
            f"policy={type(policy).__name__}",
            f"observation_space={space}",
            f"action_space={action_space}",
            f"deterministic={self.deterministic}",
        ])


__all__ = [
    "BattleAgent",
    "BattleAgentError",
    "BattleDecision",
    "PolicyDiagnostics",
    "BattleStepResult",
    "RL_ACTION_NAMES",
    "MACRO_ACTION_NAMES",
    "FeaturesExtractor",
    "_LegacyMaskedActorCriticPolicy",
    "_install_numpy_pickle_compat",
]
