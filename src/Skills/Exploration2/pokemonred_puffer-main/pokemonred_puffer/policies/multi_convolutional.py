"""탐험 전용 멀티입력 CNN 정책 + LSTM 래퍼."""

from __future__ import annotations

import pufferlib.emulation
import pufferlib.models
import pufferlib.pytorch
import torch
from torch import nn

from pokemonred_puffer.observation import PIXEL_VALUES

_NUM_DIRECTIONS = 4
_NUM_LAST_ACTIONS = 5
_NUM_MOVEMENT_RESULTS = 4
_NUM_MAP_SCOPES = 3
_CNN_CHANNELS = 64
_MAP_FEATURES = 256
_SCOPE_EMBED_DIM = 8


# torch 2.2.0 기준 F.one_hot은 trace 불가 → 수동 one-hot
def one_hot(tensor, num_classes):
    index = torch.arange(0, num_classes, device=tensor.device)
    return (tensor.view([*tensor.shape, 1]) == index.view([1] * tensor.ndim + [num_classes])).to(
        torch.int64
    )


def apply_scope_film(
    feat: torch.Tensor,
    scope_idx: torch.Tensor,
    scope_embed: nn.Embedding,
    scope_film: nn.Linear,
) -> torch.Tensor:
    """Modulate map-memory features by map_scope; zero init is identity."""
    channels = feat.shape[1]
    gb = scope_film(scope_embed(scope_idx.long().reshape(-1)))
    gamma = 1 + torch.tanh(gb[:, :channels])
    beta = torch.tanh(gb[:, channels:])
    return feat * gamma + beta


def run_lstm_with_resets(
    recurrent: nn.LSTM,
    sequence: torch.Tensor,
    state,
    reset_before: torch.Tensor,
):
    """Run ``sequence`` while clearing state before reset observations."""
    timesteps, batch_size, _ = sequence.shape
    if tuple(reset_before.shape) != (batch_size, timesteps):
        raise ValueError(
            f"reset mask must be {(batch_size, timesteps)}, got {tuple(reset_before.shape)}"
        )

    outputs = []
    for timestep in range(timesteps):
        if state is not None:
            keep = (~reset_before[:, timestep].bool()).to(sequence.dtype).view(1, batch_size, 1)
            state = (state[0] * keep, state[1] * keep)
        output, state = recurrent(sequence[timestep : timestep + 1], state)
        outputs.append(output)
    return torch.cat(outputs, dim=0), state


class MultiConvolutionalRNN(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)

    def forward_with_resets(self, x, state, reset_before):
        """Training forward that resets recurrent state at episode boundaries."""
        x_shape, space_shape = x.shape, self.obs_shape
        x_n, space_n = len(x_shape), len(space_shape)
        if x_shape[-space_n:] != space_shape:
            raise ValueError("Invalid input tensor shape", x.shape)
        if x_n == space_n + 1:
            batch_size, timesteps = x_shape[0], 1
        elif x_n == space_n + 2:
            batch_size, timesteps = x_shape[:2]
        else:
            raise ValueError("Invalid input tensor shape", x.shape)

        flat = x.reshape(batch_size * timesteps, *space_shape)
        hidden, lookup = self.policy.encode_observations(flat)
        hidden = hidden.reshape(batch_size, timesteps, self.input_size).transpose(0, 1)
        hidden, state = run_lstm_with_resets(
            self.recurrent, hidden, state, reset_before.reshape(batch_size, timesteps)
        )
        hidden = hidden.transpose(0, 1).reshape(batch_size * timesteps, self.hidden_size)
        logits, value = self.policy.decode_actions(hidden, lookup)
        return logits, value, state


class MultiConvolutionalPolicy(nn.Module):
    """Screen, local tracking-mask, and movement encoders.

    RedGymEnv 탐험 obs Dict 키:
    screen, tracking_mask, map_scope, direction, last_action, movement_result.
    map_scope modulates tracking-derived map features and is also concatenated
    as a one-hot scalar.
    """

    def __init__(
        self,
        env: pufferlib.emulation.GymnasiumPufferEnv,
        hidden_size: int = 512,
        channels_last: bool = True,
        downsample: int = 1,
    ):
        super().__init__()
        self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
        self.num_actions = env.single_action_space.n
        self.channels_last = channels_last
        self.downsample = downsample
        self.hidden_size = hidden_size
        # GymnasiumPufferEnv.env 는 래퍼 스택일 수 있음 → .unwrapped 가 RedGymEnv
        self.two_bit = env.env.unwrapped.two_bit

        self.screen_convs = nn.Sequential(
            nn.LazyConv2d(32, 8, stride=2),
            nn.ReLU(),
            nn.LazyConv2d(64, 4, stride=2),
            nn.ReLU(),
            nn.LazyConv2d(_CNN_CHANNELS, 3, stride=2),
            nn.ReLU(),
        )
        self.tracking_convs = nn.Sequential(
            nn.LazyConv2d(16, 3),
            nn.ReLU(),
            nn.LazyConv2d(32, 3, stride=2),
            nn.ReLU(),
        )
        self.map_projection = nn.Sequential(
            nn.LazyLinear(_MAP_FEATURES),
            nn.ReLU(),
        )
        self.scope_embed = nn.Embedding(_NUM_MAP_SCOPES, _SCOPE_EMBED_DIM)
        self.scope_film = nn.Linear(_SCOPE_EMBED_DIM, _MAP_FEATURES * 2)
        nn.init.zeros_(self.scope_film.weight)
        nn.init.zeros_(self.scope_film.bias)

        self.encode_linear = nn.Sequential(
            nn.LazyLinear(hidden_size),
            nn.ReLU(),
        )

        self.actor = nn.LazyLinear(self.num_actions)
        self.value_fn = nn.LazyLinear(1)

        self.register_buffer(
            "screen_buckets", torch.tensor(PIXEL_VALUES, dtype=torch.uint8), persistent=False
        )
        self.register_buffer(
            "unpack_mask",
            torch.tensor([0xC0, 0x30, 0x0C, 0x03], dtype=torch.uint8),
            persistent=False,
        )
        self.register_buffer(
            "unpack_shift",
            torch.tensor([6, 4, 2, 0], dtype=torch.uint8),
            persistent=False,
        )

    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, observations):
        # GymnasiumPufferEnv는 Dict obs를 flat uint8 버퍼로 패킹
        if isinstance(observations, dict):
            obs = observations
        else:
            # 패킹 버퍼가 float로 승격된 경우만 되돌림
            if observations.dtype != torch.uint8:
                observations = observations.to(dtype=torch.uint8)
            obs = pufferlib.pytorch.nativize_tensor(observations, self.dtype)

        screen = obs["screen"].to(dtype=torch.uint8)
        restored_shape = (screen.shape[0], screen.shape[1], screen.shape[2] * 4, screen.shape[3])

        if self.two_bit:
            screen = torch.index_select(
                self.screen_buckets,
                0,
                ((screen.reshape((-1, 1)) & self.unpack_mask) >> self.unpack_shift).flatten().int(),
            ).reshape(restored_shape)

        if self.channels_last:
            screen = screen.permute(0, 3, 1, 2)
        if self.downsample > 1:
            screen = screen[:, :, :: self.downsample, :: self.downsample]

        screen_feat = self.screen_convs(screen.float() / 255.0)

        tracking_mask = obs["tracking_mask"].float() / 255.0
        if self.channels_last:
            tracking_mask = tracking_mask.permute(0, 3, 1, 2)

        tracking_feat = torch.flatten(self.tracking_convs(tracking_mask), 1)
        map_feat = self.map_projection(tracking_feat)
        scope_idx = obs["map_scope"].long().reshape(-1)
        map_feat = apply_scope_film(map_feat, scope_idx, self.scope_embed, self.scope_film)

        scalar_parts = [
            torch.flatten(screen_feat, 1),
            map_feat,
            one_hot(obs["map_scope"].int(), _NUM_MAP_SCOPES).float().squeeze(1),
            one_hot(obs["direction"].int(), _NUM_DIRECTIONS).float().squeeze(1),
            one_hot(obs["last_action"].int(), _NUM_LAST_ACTIONS).float().squeeze(1),
            one_hot(obs["movement_result"].int(), _NUM_MOVEMENT_RESULTS).float().squeeze(1),
        ]
        cat_obs = torch.cat(scalar_parts, dim=-1)
        return self.encode_linear(cat_obs), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value
