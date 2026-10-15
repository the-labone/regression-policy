"""GR00T N1.7 model adapter for the shared policy/objective interface."""

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from ..base import BasePolicy, PolicyTarget, SampleSpec, Timestep
from .action_head import GR00TActionHead
from .processing import GR00TCondition, model_inputs


class GR00TPolicy(BasePolicy):
    """Compose a GR00T action head with an optional vision-language backbone.

    Accepts a flat, processor-prepared batch: normalized/padded ``state`` (B,T,S),
    ``embodiment_id`` (B,), normalized ``action`` (B,H,D) and ``action_mask``
    (B,H,D). With a backbone, the remaining fields are its prepared inputs. With
    backbone=None, supply backbone_features, backbone_attention_mask and (for
    AlternateVLDiT) image_mask; cached features must come from a frozen backbone.

    ``decode_prediction`` returns normalized padded actions, ready for GR00T's
    external processor's unapply step. Relative-action reconstruction needs the
    original observation; it must not use mutable last-observation policy state.
    This layer does not yet provide a raw-observation or simulator-ready API.
    """

    def __init__(
        self,
        action_head: GR00TActionHead,
        *,
        backbone: nn.Module | None = None,
        freeze_backbone: bool = True,
        n_action_steps: int | None = None,
        action_start_index: int = 0,
    ):
        super().__init__()
        if not isinstance(action_head, GR00TActionHead):
            raise TypeError("action_head must be a GR00TActionHead.")
        self.action_head = action_head
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.horizon = action_head.horizon
        self.action_dim = action_head.action_dim
        self._configure_actions(
            self.horizon if n_action_steps is None else n_action_steps,
            action_start_index,
        )
        if backbone is not None and freeze_backbone:
            backbone.requires_grad_(False)
        self.train(action_head.training)

    @property
    def device(self):
        return self.action_head.device

    @property
    def dtype(self):
        return self.action_head.dtype

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone is not None and self.freeze_backbone:
            self.backbone.eval()
        return self

    def _configure_actions(self, count, start):
        if (
            type(count) is not int
            or count < 1
            or type(start) is not int
            or start < 0
            or start + count > self.horizon
        ):
            raise ValueError("Execution action slice must fit the prediction horizon.")
        self.n_action_steps = count
        self.action_start_index = start

    def encode_condition(self, observation: Mapping[str, Tensor]) -> GR00TCondition:
        """One backbone pass per pipeline call, independently of solver steps."""
        # Do not pass supervision into the VLM or store it in the condition.
        raw_inputs = {
            key: value
            for key, value in observation.items()
            if key not in ("action", "action_mask")
        }
        state = model_inputs(raw_inputs["state"], device=self.device, dtype=self.dtype)
        embodiment = model_inputs(
            raw_inputs["embodiment_id"], device=self.device, dtype=self.dtype
        )
        expected = (self.action_head.state_history_length, self.action_head.state_dim)
        if (
            state.ndim != 3
            or state.shape[1:] != expected
            or not state.is_floating_point()
        ):
            raise ValueError(
                f"state must be floating point with shape (B, {expected})."
            )
        if embodiment.shape != (state.shape[0],) or embodiment.dtype != torch.long:
            raise ValueError("embodiment_id must be int64 with shape (B,).")
        if self.action_head.validate_values and (
            (embodiment < 0).any()
            or (embodiment >= self.action_head.num_embodiments).any()
        ):
            raise ValueError("embodiment_id is outside the configured category range.")
        if self.backbone is None:
            output = model_inputs(
                {
                    key: raw_inputs[key]
                    for key in (
                        "backbone_features",
                        "backbone_attention_mask",
                        "image_mask",
                    )
                    if key in raw_inputs
                },
                device=self.device,
                dtype=self.dtype,
            )
        else:
            backbone_dtype = next(self.backbone.parameters()).dtype
            prepared = self.backbone.prepare_input(
                model_inputs(raw_inputs, device=self.device, dtype=backbone_dtype)
            )
            if self.freeze_backbone:
                with torch.no_grad():
                    output = self.backbone(prepared)
            else:
                output = self.backbone(prepared)
        features = output["backbone_features"]
        mask = output["backbone_attention_mask"]
        if (
            features.ndim != 3
            or features.shape[0] != state.shape[0]
            or mask.shape != features.shape[:2]
        ):
            raise ValueError("Backbone features/mask must have shapes (B,L,C)/(B,L).")
        if self.action_head.use_alternate_vl_dit and (
            "image_mask" not in output
            or output["image_mask"].shape != mask.shape
            or output["image_mask"].dtype != torch.bool
            or mask.dtype != torch.bool
        ):
            raise ValueError(
                "AlternateVLDiT needs boolean image/attention masks with shape (B,L)."
            )
        # Keep gradients when fine-tuning a backbone with a different compute dtype.
        output = dict(output)
        output["backbone_features"] = features.to(dtype=self.dtype)
        return self.action_head.encode_condition(output, state, embodiment)

    def encode_target(self, batch: Mapping[str, Tensor]) -> PolicyTarget:
        action = batch["action"]
        mask = batch["action_mask"]
        expected = (batch["state"].shape[0], self.horizon, self.action_dim)
        if action.shape != expected or not action.is_floating_point():
            raise ValueError(
                f"Prepared action must be floating point with shape {expected}."
            )
        if mask.shape != action.shape:
            raise ValueError("action_mask must match the padded action shape exactly.")
        if self.action_head.validate_values and (
            not torch.isfinite(action).all()
            or not torch.isfinite(mask).all()
            or (mask < 0).any()
        ):
            raise ValueError(
                "Actions and masks must be finite; mask weights must be nonnegative."
            )
        return PolicyTarget(
            action.to(device=self.device, dtype=self.dtype), mask.to(device=self.device)
        )

    def sample_spec(self, condition: GR00TCondition) -> SampleSpec:
        return SampleSpec(
            (condition.state_features.shape[0], self.horizon, self.action_dim),
            self.device,
            self.dtype,
        )

    def forward(
        self, sample: Tensor, time: Timestep, condition: GR00TCondition
    ) -> Tensor:
        return self.action_head(sample, time, condition)

    def decode_prediction(self, prediction: Tensor) -> dict[str, Tensor]:
        if prediction.ndim != 3 or prediction.shape[1:] != (
            self.horizon,
            self.action_dim,
        ):
            raise ValueError("prediction must have shape (B, horizon, max_action_dim).")
        start = self.action_start_index
        return {
            "action": prediction[:, start : start + self.n_action_steps],
            "action_pred": prediction,
        }

    def get_extra_state(self):
        return {
            "version": 1,
            "time_mode": self.action_head.time_mode,
            "num_timestep_buckets": self.action_head.num_timestep_buckets,
            "horizon": self.horizon,
            "action_dim": self.action_dim,
            "n_action_steps": self.n_action_steps,
            "action_start_index": self.action_start_index,
            "freeze_backbone": self.freeze_backbone,
            "state_dropout_prob": self.action_head.state_dropout_prob,
        }

    def set_extra_state(self, state):
        # Reconstruct the same configured model before loading native weights.
        # Do not silently change the interpretation of times or normalized actions.
        if state != self.get_extra_state():
            raise ValueError("GR00T policy settings differ from the saved checkpoint.")
