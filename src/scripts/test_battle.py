"""Battle PPO 체크포인트의 런타임 계약을 빠르게 검사한다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import battle_opt  # noqa: E402
from Skills.Battle.battle_obs import N_FEATURES  # noqa: E402
from Skills.Battle.hierarchical_policy import (  # noqa: E402
    MaskedActorCriticPolicy,
)


def _model_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=str(
            battle_opt("model", "Skills/Battle/models/model.zip")
        ),
    )
    args = parser.parse_args()
    model_path = _model_path(args.model)

    # model.zip 은 NumPy 2.x 에서 pickle 됐다.
    from Skills.Battle.battle_agent import _install_numpy_pickle_compat

    _install_numpy_pickle_compat()

    from stable_baselines3 import PPO

    model = PPO.load(
        str(model_path),
        device="cpu",
        custom_objects={"policy_class": MaskedActorCriticPolicy},
    )
    spaces = model.observation_space.spaces
    obs_shape = tuple(spaces["observation"].shape)
    mask_shape = tuple(spaces["action_mask"].shape)
    action_count = int(model.action_space.n)

    if obs_shape != (N_FEATURES,):
        raise RuntimeError(f"observation shape 불일치: {obs_shape}")
    if mask_shape != (10,):
        raise RuntimeError(f"action_mask shape 불일치: {mask_shape}")
    if action_count != 10:
        raise RuntimeError(f"action 개수 불일치: {action_count}")

    print(
        "[battle smoke] OK "
        f"model={model_path} policy={type(model.policy).__name__} "
        f"observation={obs_shape} mask={mask_shape} actions={action_count}"
    )


if __name__ == "__main__":
    main()
