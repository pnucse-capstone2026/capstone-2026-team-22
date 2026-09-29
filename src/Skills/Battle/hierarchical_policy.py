"""Standalone copy of the final v15 hierarchical inference policy.

This module intentionally imports only Torch, Stable-Baselines3, and the
runtime observation constants.  It must not import RL.train/RL.env/poke-env.
The class body is kept identical to Battle_RL_r2/RL/train.py so a checkpoint
saved under Python 3.14 can be loaded safely by the Python 3.11 runtime using
custom_objects, without executing cross-version cloudpickled code objects.
"""

from __future__ import annotations

import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .battle_obs import (
    N_FEATURES,
    GLOBAL_DIM,
    SWITCH_CONTEXT_START_INDEX,
    SWITCH_NEED_GLOBAL_INDEX,
    MOVE_SLOT_DIM,
    MOVE_CATEGORY_INDEX,
    ATTACK_QUALITY_RATIO_INDEX,
    BEST_ATTACK_INDEX,
    ATTACK_SCORE_NORM_INDEX,
    EFFECTIVENESS_INDEX,
    EFFECTIVE_ACCURACY_INDEX,
    MIN_DAMAGE_RATIO_INDEX,
    KO_ON_HIT_INDEX,
    SAFE_IMMEDIATE_KO_INDEX,
    STATUS_UTILITY_INDEX,
    PERSISTENT_EFFECT_INDEX,
    PERSISTENT_EFFECT_ACTIVE_INDEX,
    OPPONENT_DEBUFF_INDEX,
    DEBUFF_REMAINING_CAPACITY_INDEX,
    OPP_MOVE_SLOT_DIM,
    OWN_TEAM_SLOT_DIM,
    OWN_TEAM_MATCHUP_GAIN_INDEX,
    OWN_TEAM_SWITCH_QUALITY_INDEX,
    OWN_TEAM_BEST_SWITCH_INDEX,
    OPP_TEAM_SLOT_DIM,
)

class FeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=N_FEATURES)

    def forward(self, obs):
        x = obs["observation"].float()
        return torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)


class MaskedActorCriticPolicy(ActorCriticPolicy):
    """Hierarchical masked PPO actor with three action families.

    External action contract is unchanged:
      0..5 = switch candidates
      6..9 = move candidates

    Internally the policy factorizes the action as:
      1) family: SWITCH / ATTACK / STATUS
      2) candidate inside that family

    The stochastic PPO distribution is the exact joint distribution
    P(family) * P(candidate | family), flattened back to the original 10 actions.
    Deterministic inference is family-first, then candidate-first.  Therefore a
    family with three good moves no longer loses to one switch merely because its
    probability mass is split across three individual move actions.
    """

    NUM_SWITCHES = 6
    NUM_MOVES = 4
    NUM_FAMILIES = 3

    FAMILY_SWITCH = 0
    FAMILY_ATTACK = 1
    FAMILY_STATUS = 2
    FAMILY_NAMES = ("switch", "attack", "status")

    # Attack remains the ordinary default.  The explicit observed switch_need
    # feature below can locally override this prior in tactical states.
    INITIAL_SWITCH_LOGIT_PRIOR = -0.350
    INITIAL_ATTACK_LOGIT_PRIOR = 0.550
    INITIAL_STATUS_LOGIT_PRIOR = -0.200
    # Fixed routing residuals encode the intended high-level decision while
    # leaving PPO free to learn the remaining context.  `switch_need` is already
    # conservatively gated by env.py, so a value of one means a real matchup
    # flip or safe severe-debuff escape, not merely a less-bad bench option.
    TACTICAL_SWITCH_ROUTE_BIAS = 2.750
    TACTICAL_MOVE_SUPPRESSION = 0.750
    TACTICAL_STATUS_SUPPRESSION = 0.250
    BAD_SWITCH_FALLBACK_PENALTY = 1.500
    BAD_SWITCH_MOVE_BONUS = 0.750
    FINISHER_SWITCH_SUPPRESSION = 4.000
    FINISHER_ATTACK_ROUTE_BIAS = 4.000
    FINISHER_STATUS_SUPPRESSION = 2.000
    NEG_INF = -1.0e9

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            net_arch=dict(pi=[], vf=[128, 128]),
            features_extractor_class=FeaturesExtractor,
        )
        self._mask = None
        self._last_effective_mask = None
        self._last_family_logits = None
        self._last_family_probs = None
        self._last_family_valid_mask = None
        self._last_family_quality = None
        self._last_raw_candidate_logits = None
        self._last_joint_logits = None

    def _build(self, lr_schedule) -> None:
        """Build critic, candidate scorers, and the three-family gate."""
        self._build_mlp_extractor()

        # Actor logits are produced by the custom hierarchy below.
        self.action_net = nn.Identity()
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        # Candidate scorers may use all observed candidates as context.  The
        # FAMILY gate, however, gets a different LEGAL-ONLY context below so
        # locked/irrelevant move slots cannot dilute the best legal action signal.
        self.candidate_context_dim = (
            GLOBAL_DIM
            + 2 * MOVE_SLOT_DIM
            + OPP_MOVE_SLOT_DIM
            + 2 * OWN_TEAM_SLOT_DIM
            + OPP_TEAM_SLOT_DIM
        )
        self.family_context_dim = (
            GLOBAL_DIM
            + 4 * MOVE_SLOT_DIM      # legal attack mean/max + legal status mean/max
            + OPP_MOVE_SLOT_DIM
            + 2 * OWN_TEAM_SLOT_DIM # legal switch mean/max
            + OPP_TEAM_SLOT_DIM
            + self.NUM_FAMILIES      # normalized legal candidate counts
        )

        self.move_scorer = nn.Sequential(
            nn.Linear(self.candidate_context_dim + MOVE_SLOT_DIM, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.switch_scorer = nn.Sequential(
            nn.Linear(self.candidate_context_dim + OWN_TEAM_SLOT_DIM, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Small learnable semantic residuals give candidate learning a sane
        # bootstrap instead of asking near-zero random MLP weights to discover
        # HornAttack>Tackle and useful-status>Growl from scratch.
        self.move_quality_residual = nn.Linear(MOVE_SLOT_DIM, 1, bias=False)
        self.switch_quality_residual = nn.Linear(OWN_TEAM_SLOT_DIM, 1, bias=False)

        # Family 0=switch, 1=damaging attack, 2=status/setup.
        self.family_scorer = nn.Sequential(
            nn.Linear(self.family_context_dim, 96),
            nn.Tanh(),
            nn.Linear(96, 64),
            nn.Tanh(),
            nn.Linear(64, self.NUM_FAMILIES),
        )

        # Learnable family-quality anchors.  They multiply the best LEGAL
        # absolute switch quality / attack score / status utility respectively.
        self.family_quality_scales = nn.Parameter(
            torch.tensor([0.40, 0.50, 0.30], dtype=torch.float32)
        )
        self.switch_need_family_scale = nn.Parameter(
            torch.tensor(0.75, dtype=torch.float32)
        )

        if self.ortho_init:
            sqrt2 = 2.0 ** 0.5
            self.mlp_extractor.apply(
                lambda module: self.init_weights(module, gain=sqrt2)
            )
            self.value_net.apply(
                lambda module: self.init_weights(module, gain=1.0)
            )
            self.move_scorer.apply(
                lambda module: self.init_weights(module, gain=sqrt2)
            )
            self.switch_scorer.apply(
                lambda module: self.init_weights(module, gain=sqrt2)
            )
            self.family_scorer.apply(
                lambda module: self.init_weights(module, gain=sqrt2)
            )

            self.init_weights(self.move_scorer[-1], gain=0.01)
            self.init_weights(self.switch_scorer[-1], gain=0.01)
            self.init_weights(self.family_scorer[-1], gain=0.01)

        # Explicit, LEARNABLE candidate priors.  PPO may change every one of
        # these weights; they only fix the poor v9 bootstrap at ~1M steps.
        with torch.no_grad():
            self.move_quality_residual.weight.zero_()
            self.move_quality_residual.weight[0, ATTACK_QUALITY_RATIO_INDEX] = 0.55
            self.move_quality_residual.weight[0, BEST_ATTACK_INDEX] = 0.25
            self.move_quality_residual.weight[0, ATTACK_SCORE_NORM_INDEX] = 0.20
            self.move_quality_residual.weight[0, EFFECTIVE_ACCURACY_INDEX] = 0.20
            self.move_quality_residual.weight[0, MIN_DAMAGE_RATIO_INDEX] = 0.40
            self.move_quality_residual.weight[0, KO_ON_HIT_INDEX] = 0.40
            self.move_quality_residual.weight[0, SAFE_IMMEDIATE_KO_INDEX] = 1.00
            self.move_quality_residual.weight[0, STATUS_UTILITY_INDEX] = 0.35

            self.switch_quality_residual.weight.zero_()
            self.switch_quality_residual.weight[0, OWN_TEAM_SWITCH_QUALITY_INDEX] = 0.55
            self.switch_quality_residual.weight[0, OWN_TEAM_BEST_SWITCH_INDEX] = 0.25
            self.switch_quality_residual.weight[0, OWN_TEAM_MATCHUP_GAIN_INDEX] = 0.20

        # Attack-first initial family prior; learned return can override it.
        with torch.no_grad():
            self.family_scorer[-1].bias[self.FAMILY_SWITCH] = (
                self.INITIAL_SWITCH_LOGIT_PRIOR
            )
            self.family_scorer[-1].bias[self.FAMILY_ATTACK] = (
                self.INITIAL_ATTACK_LOGIT_PRIOR
            )
            self.family_scorer[-1].bias[self.FAMILY_STATUS] = (
                self.INITIAL_STATUS_LOGIT_PRIOR
            )

        # Custom heads must exist before optimizer construction.
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _split_features(self, features: torch.Tensor):
        """Split env.py's compact vector back into candidate groups."""
        if features.ndim == 1:
            features = features.unsqueeze(0)

        batch_size = features.shape[0]
        offset = 0

        global_state = features[:, offset:offset + GLOBAL_DIM]
        offset += GLOBAL_DIM

        size = self.NUM_MOVES * MOVE_SLOT_DIM
        own_moves = features[:, offset:offset + size].reshape(
            batch_size, self.NUM_MOVES, MOVE_SLOT_DIM
        )
        offset += size

        size = self.NUM_MOVES * OPP_MOVE_SLOT_DIM
        opponent_moves = features[:, offset:offset + size].reshape(
            batch_size, self.NUM_MOVES, OPP_MOVE_SLOT_DIM
        )
        offset += size

        size = self.NUM_SWITCHES * OWN_TEAM_SLOT_DIM
        own_team = features[:, offset:offset + size].reshape(
            batch_size, self.NUM_SWITCHES, OWN_TEAM_SLOT_DIM
        )
        offset += size

        size = self.NUM_SWITCHES * OPP_TEAM_SLOT_DIM
        opponent_team = features[:, offset:offset + size].reshape(
            batch_size, self.NUM_SWITCHES, OPP_TEAM_SLOT_DIM
        )
        offset += size

        if offset != N_FEATURES:
            raise RuntimeError(
                f"Observation layout mismatch: parsed={offset}, expected={N_FEATURES}"
            )

        return global_state, own_moves, opponent_moves, own_team, opponent_team

    def _candidate_context_from_parts(
        self,
        global_state: torch.Tensor,
        own_moves: torch.Tensor,
        opponent_moves: torch.Tensor,
        own_team: torch.Tensor,
        opponent_team: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                global_state,
                own_moves.mean(dim=1),
                own_moves.max(dim=1).values,
                opponent_moves.mean(dim=1),
                own_team.mean(dim=1),
                own_team.max(dim=1).values,
                opponent_team.mean(dim=1),
            ],
            dim=-1,
        )

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(values.dtype).unsqueeze(-1)
        count = weights.sum(dim=1).clamp_min(1.0)
        return (values * weights).sum(dim=1) / count

    @staticmethod
    def _masked_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        expanded = mask.unsqueeze(-1)
        masked = values.masked_fill(~expanded, -1.0e9)
        maxima = masked.max(dim=1).values
        any_valid = mask.any(dim=1, keepdim=True)
        return torch.where(any_valid, maxima, torch.zeros_like(maxima))

    def _family_context_from_parts(
        self,
        global_state: torch.Tensor,
        own_moves: torch.Tensor,
        opponent_moves: torch.Tensor,
        own_team: torch.Tensor,
        opponent_team: torch.Tensor,
        action_family_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Only actions that can ACTUALLY be selected contribute to the family
        # summary.  This is the main v10 change: locked slots no longer dilute
        # the attack/status/switch evidence seen by the family gate.
        switch_mask = action_family_mask[:, self.FAMILY_SWITCH, :self.NUM_SWITCHES]
        attack_mask = action_family_mask[:, self.FAMILY_ATTACK, self.NUM_SWITCHES:]
        status_mask = action_family_mask[:, self.FAMILY_STATUS, self.NUM_SWITCHES:]

        counts = torch.stack(
            [
                switch_mask.sum(dim=1).to(own_team.dtype) / float(self.NUM_SWITCHES),
                attack_mask.sum(dim=1).to(own_moves.dtype) / float(self.NUM_MOVES),
                status_mask.sum(dim=1).to(own_moves.dtype) / float(self.NUM_MOVES),
            ],
            dim=-1,
        )

        return torch.cat(
            [
                global_state,
                self._masked_mean(own_moves, attack_mask),
                self._masked_max(own_moves, attack_mask),
                self._masked_mean(own_moves, status_mask),
                self._masked_max(own_moves, status_mask),
                opponent_moves.mean(dim=1),
                self._masked_mean(own_team, switch_mask),
                self._masked_max(own_team, switch_mask),
                opponent_team.mean(dim=1),
                counts,
            ],
            dim=-1,
        )

    def _candidate_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Return slot-independent raw candidate scores (no family bias added)."""
        (
            global_state,
            own_moves,
            opponent_moves,
            own_team,
            opponent_team,
        ) = self._split_features(features)

        batch_size = features.shape[0]
        context = self._candidate_context_from_parts(
            global_state,
            own_moves,
            opponent_moves,
            own_team,
            opponent_team,
        )

        switch_context = context.unsqueeze(1).expand(
            -1, self.NUM_SWITCHES, -1
        )
        switch_inputs = torch.cat([switch_context, own_team], dim=-1)
        switch_logits = self.switch_scorer(
            switch_inputs.reshape(batch_size * self.NUM_SWITCHES, -1)
        ).reshape(batch_size, self.NUM_SWITCHES)
        switch_logits = switch_logits + self.switch_quality_residual(
            own_team.reshape(batch_size * self.NUM_SWITCHES, -1)
        ).reshape(batch_size, self.NUM_SWITCHES)

        move_context = context.unsqueeze(1).expand(-1, self.NUM_MOVES, -1)
        move_inputs = torch.cat([move_context, own_moves], dim=-1)
        move_logits = self.move_scorer(
            move_inputs.reshape(batch_size * self.NUM_MOVES, -1)
        ).reshape(batch_size, self.NUM_MOVES)
        move_logits = move_logits + self.move_quality_residual(
            own_moves.reshape(batch_size * self.NUM_MOVES, -1)
        ).reshape(batch_size, self.NUM_MOVES)

        return torch.cat([switch_logits, move_logits], dim=-1)

    def _family_logits(
        self,
        features: torch.Tensor,
        action_family_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (
            global_state,
            own_moves,
            opponent_moves,
            own_team,
            opponent_team,
        ) = self._split_features(features)

        context = self._family_context_from_parts(
            global_state,
            own_moves,
            opponent_moves,
            own_team,
            opponent_team,
            action_family_mask,
        )

        switch_mask = action_family_mask[:, self.FAMILY_SWITCH, :self.NUM_SWITCHES]
        attack_mask = action_family_mask[:, self.FAMILY_ATTACK, self.NUM_SWITCHES:]
        status_mask = action_family_mask[:, self.FAMILY_STATUS, self.NUM_SWITCHES:]

        switch_quality = self._masked_max(
            own_team[:, :, OWN_TEAM_SWITCH_QUALITY_INDEX:OWN_TEAM_SWITCH_QUALITY_INDEX + 1],
            switch_mask,
        ).squeeze(-1)
        attack_quality = self._masked_max(
            own_moves[:, :, ATTACK_SCORE_NORM_INDEX:ATTACK_SCORE_NORM_INDEX + 1],
            attack_mask,
        ).squeeze(-1)
        status_quality = self._masked_max(
            own_moves[:, :, STATUS_UTILITY_INDEX:STATUS_UTILITY_INDEX + 1],
            status_mask,
        ).squeeze(-1)
        finisher_available = self._masked_max(
            own_moves[:, :, SAFE_IMMEDIATE_KO_INDEX:SAFE_IMMEDIATE_KO_INDEX + 1],
            attack_mask,
        ).squeeze(-1)
        family_quality = torch.stack(
            [switch_quality, attack_quality, status_quality], dim=-1
        )

        logits = self.family_scorer(context)
        logits = logits + family_quality * self.family_quality_scales.unsqueeze(0)

        # A transparent learnable bootstrap for the cross-family decision.  At
        # switch_need=1, SWITCH gets +.75 and ATTACK gets -.1875 initially;
        # ordinary states (need=0) keep the attack-first default unchanged.
        switch_need = global_state[:, SWITCH_NEED_GLOBAL_INDEX]
        effective_switch_need = switch_need * (1.0 - finisher_available)
        switch_need_direction = torch.stack(
            [
                effective_switch_need,
                -0.25 * effective_switch_need,
                torch.zeros_like(effective_switch_need),
            ],
            dim=-1,
        )
        logits = logits + (
            switch_need_direction * self.switch_need_family_scale
        )

        # Family-first tactical routing:
        #   * a verified switch need strongly favors SWITCH;
        #   * an unfavorable active matchup with no viable switch sends the
        #     decision back to the two move families (ATTACK / STATUS).
        # The latter is the water/water/water-vs-Grass safeguard: choosing a
        # different but still unfavorable Water Pokemon is not rewarded merely
        # because it is the least-bad bench member.
        current_matchup_score = global_state[:, SWITCH_CONTEXT_START_INDEX]
        unfavorable = torch.clamp(
            -current_matchup_score / 0.100,
            min=0.0,
            max=1.0,
        )
        # A minimum-roll KO is the top-level exception: do not route away from
        # it merely because switching would otherwise improve the matchup.
        # This is a fixed, observable priority; PPO still learns which finisher
        # move to use inside the ATTACK family.
        move_fallback = (
            unfavorable
            * (1.0 - effective_switch_need)
            * (1.0 - finisher_available)
        )
        routing_residual = torch.stack(
            [
                (
                    self.TACTICAL_SWITCH_ROUTE_BIAS * effective_switch_need
                    - self.BAD_SWITCH_FALLBACK_PENALTY * move_fallback
                    - self.FINISHER_SWITCH_SUPPRESSION * finisher_available
                ),
                (
                    -self.TACTICAL_MOVE_SUPPRESSION * effective_switch_need
                    + self.BAD_SWITCH_MOVE_BONUS * move_fallback
                    + self.FINISHER_ATTACK_ROUTE_BIAS * finisher_available
                ),
                (
                    -self.TACTICAL_STATUS_SUPPRESSION * effective_switch_need
                    + self.BAD_SWITCH_MOVE_BONUS * move_fallback
                    - self.FINISHER_STATUS_SUPPRESSION * finisher_available
                ),
            ],
            dim=-1,
        )
        logits = logits + routing_residual
        return logits, family_quality

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
        """Apply state-change semantics on top of poke-env legality."""
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
        move_slice = slice(self.NUM_SWITCHES, self.NUM_SWITCHES + self.NUM_MOVES)
        effective[:, move_slice] &= ~redundant

        empty_rows = effective.sum(dim=-1) == 0
        if torch.any(empty_rows):
            effective[empty_rows] = base_valid[empty_rows]

        return effective

    def _family_action_masks(
        self,
        features: torch.Tensor,
        effective_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (family_valid[B,3], action_family_mask[B,3,10])."""
        _, own_moves, _, _, _ = self._split_features(features)
        batch_size = features.shape[0]
        device = features.device

        action_family_mask = torch.zeros(
            (batch_size, self.NUM_FAMILIES, self.NUM_SWITCHES + self.NUM_MOVES),
            dtype=torch.bool,
            device=device,
        )

        switch_valid = effective_mask[:, :self.NUM_SWITCHES]
        move_valid = effective_mask[:, self.NUM_SWITCHES:]
        damaging = own_moves[:, :, MOVE_CATEGORY_INDEX] > 0.5

        action_family_mask[:, self.FAMILY_SWITCH, :self.NUM_SWITCHES] = switch_valid
        action_family_mask[:, self.FAMILY_ATTACK, self.NUM_SWITCHES:] = (
            move_valid & damaging
        )
        action_family_mask[:, self.FAMILY_STATUS, self.NUM_SWITCHES:] = (
            move_valid & ~damaging
        )

        family_valid = action_family_mask.any(dim=-1)
        return family_valid, action_family_mask

    def _hierarchical_components(
        self,
        obs,
        features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build exact flattened P(family) * P(candidate|family) distribution."""
        raw_candidate_logits = self._candidate_logits(features)

        if not torch.isfinite(raw_candidate_logits).all():
            raise FloatingPointError("Policy produced non-finite candidate logits.")

        mask = obs["action_mask"]
        self._mask = mask
        effective_mask = self._effective_action_mask(
            features,
            raw_candidate_logits,
            mask,
        )
        family_valid, action_family_mask = self._family_action_masks(
            features,
            effective_mask,
        )

        # Family scoring is intentionally AFTER legality/redundancy masking so
        # its pooled summaries contain legal candidates only.
        family_logits, family_quality = self._family_logits(
            features,
            action_family_mask,
        )
        if not torch.isfinite(family_logits).all():
            raise FloatingPointError("Policy produced non-finite family logits.")

        # At least one action exists, therefore at least one family must exist.
        if torch.any(family_valid.sum(dim=-1) == 0):
            raise RuntimeError("Hierarchical policy produced an all-zero family mask.")

        masked_family_logits = family_logits.masked_fill(
            ~family_valid,
            self.NEG_INF,
        )
        family_log_probs = torch.log_softmax(masked_family_logits, dim=-1)
        family_probs = torch.softmax(masked_family_logits, dim=-1)

        joint_logits = torch.full_like(raw_candidate_logits, self.NEG_INF)
        conditional_log_probs = torch.full(
            (
                features.shape[0],
                self.NUM_FAMILIES,
                self.NUM_SWITCHES + self.NUM_MOVES,
            ),
            self.NEG_INF,
            dtype=features.dtype,
            device=features.device,
        )

        for family in range(self.NUM_FAMILIES):
            fam_mask = action_family_mask[:, family, :]
            masked_candidates = raw_candidate_logits.masked_fill(
                ~fam_mask,
                self.NEG_INF,
            )
            fam_cond_log_probs = torch.log_softmax(masked_candidates, dim=-1)
            conditional_log_probs[:, family, :] = fam_cond_log_probs
            fam_joint = family_log_probs[:, family:family + 1] + fam_cond_log_probs
            joint_logits = torch.where(fam_mask, fam_joint, joint_logits)

        if not torch.isfinite(joint_logits).all():
            raise FloatingPointError("Policy produced non-finite hierarchical logits.")

        self._last_effective_mask = effective_mask
        self._last_family_logits = masked_family_logits
        self._last_family_probs = family_probs
        self._last_family_valid_mask = family_valid
        self._last_family_quality = family_quality
        self._last_raw_candidate_logits = raw_candidate_logits
        self._last_joint_logits = joint_logits

        return {
            "raw_candidate_logits": raw_candidate_logits,
            "family_logits": masked_family_logits,
            "family_probs": family_probs,
            "family_valid": family_valid,
            "family_quality": family_quality,
            "action_family_mask": action_family_mask,
            "conditional_log_probs": conditional_log_probs,
            "joint_logits": joint_logits,
            "effective_mask": effective_mask,
        }

    def hierarchical_debug(self, obs) -> dict[str, torch.Tensor]:
        """Expose hierarchy components for evaluate_7.py diagnostics."""
        features = self.extract_features(obs)
        if isinstance(features, tuple):
            raise RuntimeError(
                "Hierarchical policy requires share_features_extractor=True"
            )
        return self._hierarchical_components(obs, features)

    def _distribution_from_features(self, obs, features: torch.Tensor):
        components = self._hierarchical_components(obs, features)
        return self.action_dist.proba_distribution(
            action_logits=components["joint_logits"]
        )

    def _deterministic_hierarchical_actions(
        self,
        components: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Argmax family first, then argmax candidate inside that family."""
        family_choice = torch.argmax(components["family_logits"], dim=-1)
        raw_candidates = components["raw_candidate_logits"]
        family_masks = components["action_family_mask"]

        actions: list[torch.Tensor] = []
        for row in range(raw_candidates.shape[0]):
            family = int(family_choice[row].item())
            candidate_mask = family_masks[row, family]
            masked = raw_candidates[row].masked_fill(
                ~candidate_mask,
                self.NEG_INF,
            )
            actions.append(torch.argmax(masked, dim=-1))
        return torch.stack(actions, dim=0)

    def select_hierarchical_action(self, obs, deterministic: bool = True) -> torch.Tensor:
        """Public helper used by evaluation and runtime deterministic inference."""
        features = self.extract_features(obs)
        if isinstance(features, tuple):
            raise RuntimeError(
                "Hierarchical policy requires share_features_extractor=True"
            )
        components = self._hierarchical_components(obs, features)
        if deterministic:
            return self._deterministic_hierarchical_actions(components)
        distribution = self.action_dist.proba_distribution(
            action_logits=components["joint_logits"]
        )
        return distribution.get_actions(deterministic=False)

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        if isinstance(features, tuple):
            raise RuntimeError(
                "Hierarchical policy requires share_features_extractor=True"
            )

        latent_vf = self.mlp_extractor.forward_critic(features)
        values = self.value_net(latent_vf)
        components = self._hierarchical_components(obs, features)
        distribution = self.action_dist.proba_distribution(
            action_logits=components["joint_logits"]
        )

        if deterministic:
            actions = self._deterministic_hierarchical_actions(components)
        else:
            actions = distribution.get_actions(deterministic=False)

        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def _predict(self, observation, deterministic: bool = False) -> torch.Tensor:
        # SB3 model.predict() reaches this method. Keep runtime behavior identical
        # to evaluate_7.py: family-first when deterministic=True.
        return self.select_hierarchical_action(
            observation,
            deterministic=deterministic,
        )

    def evaluate_actions(self, obs, actions):
        features = self.extract_features(obs)
        if isinstance(features, tuple):
            raise RuntimeError(
                "Hierarchical policy requires share_features_extractor=True"
            )

        latent_vf = self.mlp_extractor.forward_critic(features)
        values = self.value_net(latent_vf)
        distribution = self._distribution_from_features(obs, features)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs):
        features = self.extract_features(obs)
        if isinstance(features, tuple):
            raise RuntimeError(
                "Hierarchical policy requires share_features_extractor=True"
            )
        return self._distribution_from_features(obs, features)

__all__ = ["FeaturesExtractor", "MaskedActorCriticPolicy"]

