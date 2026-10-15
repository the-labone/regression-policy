"""State or visually conditioned action-chunk model, independent of its objective.

Based on real-stanford/diffusion_policy (MIT), revision
5ba07ac6661db573af695b419a7947ecb704690f. See ../../LICENSE.diffusion_policy.
"""

import copy
from collections.abc import Mapping

import torch
from torch import Tensor, nn

from ...base import BasePolicy, PolicyTarget, SampleSpec, Timestep


class ActionChunkPolicy(BasePolicy):
    """Backbone adapter with observation encoding and action-chunk alignment.

    Compose with diffusion, flow, or another compatible objective. Both backbones
    accept fractional times and return unconstrained action-shaped predictions;
    the objective determines their meaning and owns the loss and sampling loop.
    ``forward`` is only a backbone prediction on encoded inputs. Normalized
    clean targets and decoded actions remain 7D/10D per arm, as configured;
    conversion to native controller commands belongs to the environment adapter.
    ``validate_values=False`` skips per-call finite checks for trusted inputs;
    shape/type/device checks remain. The trainer checks the resulting loss.
    An optional visual encoder maps camera histories and normalized proprioception
    to backbone features; the default state path retains its checkpoint layout.
    """

    def __init__(
        self,
        model: nn.Module,
        normalizer: nn.Module,
        *,
        n_action_steps: int = 8,
        action_start_index: int | None = None,
        validate_values: bool = True,
        obs_encoder: nn.Module | None = None,
    ):
        super().__init__()
        self.model = model
        self.obs_encoder = obs_encoder
        if obs_encoder is not None and obs_encoder.output_dim != model.obs_dim:
            raise ValueError("Observation encoder output must match backbone obs_dim.")
        self.validate_values = validate_values
        self.horizon = model.horizon
        self.obs_dim = model.obs_dim
        self.action_dim = model.action_dim
        self.n_obs_steps = model.n_obs_steps
        self.normalizer = self._prepare_normalizer(normalizer)
        self._configure_actions(
            n_action_steps,
            self.n_obs_steps - 1 if action_start_index is None else action_start_index,
        )

    def _prepare_normalizer(self, normalizer: nn.Module) -> nn.Module:
        """Own a frozen copy on the model's device and dtype."""
        if not isinstance(normalizer, nn.Module) or not all(
            callable(getattr(normalizer, name, None))
            for name in (
                "normalize_obs",
                "unnormalize_obs",
                "normalize_action",
                "unnormalize_action",
            )
        ):
            raise TypeError(
                "normalizer must be an nn.Module with obs/action affine methods."
            )
        return (
            copy.deepcopy(normalizer)
            .requires_grad_(False)
            .to(device=self.device, dtype=self.dtype)
        )

    def _configure_actions(self, n_action_steps, start):
        if type(n_action_steps) is not int or n_action_steps < 1:
            raise ValueError("n_action_steps must be a positive integer.")
        if type(start) is not int or start < 0 or start + n_action_steps > self.horizon:
            raise ValueError(
                "Execution action slice must fit inside the prediction horizon."
            )
        self.n_action_steps = n_action_steps
        self.action_start_index = start

    def get_extra_state(self):
        return {
            "version": 1,
            "n_action_steps": self.n_action_steps,
            "action_start_index": self.action_start_index,
        }

    def set_extra_state(self, state):
        if state.get("version") != 1:
            raise ValueError("Unsupported action-chunk policy state version.")
        self._configure_actions(state["n_action_steps"], state["action_start_index"])

    def _check_tensor(self, value: Tensor, name: str):
        if not value.is_floating_point() or (
            self.validate_values and not torch.isfinite(value).all()
        ):
            raise ValueError(f"{name} must be a finite floating-point tensor.")
        if value.device != self.device:
            raise ValueError(f"{name} must be on the policy device ({self.device}).")

    def encode_condition(self, observation: Tensor | Mapping[str, Tensor]) -> Tensor:
        if isinstance(observation, Mapping):
            if "past_action" in observation:
                raise ValueError("Conditioning on past_action is not supported.")
            observation = observation["obs"]
        if self.obs_encoder is not None:
            condition = self.obs_encoder(observation, self.normalizer, self.n_obs_steps)
            self._check_tensor(condition, "encoded observations")
            return condition.to(dtype=self.dtype)
        if (
            observation.ndim != 3
            or observation.shape[0] < 1
            or observation.shape[1] < self.n_obs_steps
            or observation.shape[2] != self.obs_dim
        ):
            raise ValueError("obs must have shape (B, T >= n_obs_steps, obs_dim).")
        self._check_tensor(observation, "obs")
        return self.normalizer.normalize_obs(observation[:, : self.n_obs_steps])

    def encode_target(self, batch: Mapping[str, Tensor]) -> PolicyTarget:
        if "valid_mask" in batch:
            raise ValueError("Masked DP training is not enabled; DP includes padding.")
        action = batch["action"]
        obs = batch["obs"]
        batch_size = (
            obs["proprio"].shape[0] if self.obs_encoder is not None else obs.shape[0]
        )
        if action.shape != (batch_size, self.horizon, self.action_dim):
            raise ValueError("action must have shape (B, horizon, action_dim).")
        self._check_tensor(action, "action")
        # action_valid_mask is informational: DP trains on repeated edge actions.
        return PolicyTarget(self.normalizer.normalize_action(action))

    def sample_spec(self, condition: Tensor) -> SampleSpec:
        return SampleSpec(
            (condition.shape[0], self.horizon, self.action_dim),
            condition.device,
            condition.dtype,
        )

    def forward(self, sample: Tensor, time: Timestep, condition: Tensor) -> Tensor:
        return self.model(sample, time, condition)

    def decode_prediction(self, prediction: Tensor) -> dict[str, Tensor]:
        if prediction.ndim != 3 or prediction.shape[1:] != (
            self.horizon,
            self.action_dim,
        ):
            raise ValueError("prediction must have shape (B, horizon, action_dim).")
        self._check_tensor(prediction, "prediction")
        action_pred = self.normalizer.unnormalize_action(prediction)
        start = self.action_start_index
        return {
            "action": action_pred[:, start : start + self.n_action_steps],
            "action_pred": action_pred,
        }

    def get_optimizer_groups(self, weight_decay: float):
        groups = self.model.get_optimizer_groups(weight_decay=weight_decay)
        if self.obs_encoder is not None:
            groups.append(
                {
                    "params": self.obs_encoder.parameters(),
                    "weight_decay": self.obs_encoder.weight_decay,
                }
            )
        return groups
