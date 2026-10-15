"""Linear flow matching and Euler sampling through the BasePolicy interface."""

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from ....policies.base import BasePolicy, SampleSpec
from ...base import BaseObjective
from ...common import (
    gaussian_noise,
    loss_precision,
    loss_weights,
    validate_prediction,
    validate_target,
)


class FlowObjective(BaseObjective):
    """Predict clean sample minus Gaussian noise along a linear interpolant.

    For encoded target x and noise e, z(t) = (1-t)e + tx, default t ~ U[0,1].
    The policy predicts dz/dt = x-e, with continuous, unscaled time in [0,1].
    This is generative velocity in the policy's representation, not robot velocity.
    Normalization, action representation and decoding belong to the policy.

    Defaults follow much-ado-about-noising's flow objective: squared L2 summed
    over the last (feature) axis, averaged over all other axes, scaled by 0.1.
    Inference uses nine uniform Euler steps and zero initialization; stochastic
    mode starts from Gaussian noise. Training always starts from Gaussian noise.
    No clipping, rotation projection or denormalization happens in this objective.
    Optional Beta sampling, affine time scaling and reversal support native VLA
    recipes. Explicit timesteps always use the canonical noise-to-data coordinate.

    Masks extend the upstream loss: weighted element mean times feature count,
    so an all-ones mask equals the unmasked loss. Conditions remain opaque and
    targets may have any rank >= 2, including action chunks and video latents.
    Set validate_values=False for fullgraph training on trusted data; metadata
    checks remain, and the caller must check loss/gradient finiteness.
    """

    def __init__(
        self,
        *,
        num_inference_steps: int = 9,
        sample_mode: str = "zero",
        norm_type: str = "l2",
        loss_scale: float = 0.1,
        validate_values: bool = True,
        time_beta: tuple[float, float] | None = None,
        time_scale: float = 1.0,
        time_offset: float = 0.0,
        time_flip: bool = False,
        inference_time_dtype: str = "float32",
    ):
        super().__init__()
        self.validate_values = validate_values
        self._configure(
            num_inference_steps=num_inference_steps,
            sample_mode=sample_mode,
            norm_type=norm_type,
            loss_scale=loss_scale,
            time_beta=time_beta,
            time_scale=time_scale,
            time_offset=time_offset,
            time_flip=time_flip,
            inference_time_dtype=inference_time_dtype,
        )

    def compute_loss(
        self,
        policy: BasePolicy,
        target: Tensor,
        condition: Any,
        *,
        loss_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
        timesteps: Tensor | None = None,
    ) -> Tensor:
        """Train one velocity prediction on an encoded clean target.

        Optional noise/timesteps override random draws for controlled comparisons.
        A loss mask changes only reduction, not interpolation or the velocity target.
        """
        timesteps, noise = self._prepare_training_inputs(
            target, timesteps=timesteps, noise=noise, generator=generator
        )

        # One time per example, broadcast across its sequence/spatial dimensions.
        t = timesteps.to(target.dtype).reshape((-1,) + (1,) * (target.ndim - 1))
        interpolated_sample = (1 - t) * noise + t * target
        target_velocity = target - noise
        predicted_velocity = self._predict_velocity(
            policy, interpolated_sample, timesteps, condition
        )

        error = self._velocity_error(predicted_velocity, target_velocity)
        return self._reduce_loss(error, loss_mask)

    @torch.no_grad()
    def sample(
        self,
        policy: BasePolicy,
        condition: Any,
        *,
        sample_spec: SampleSpec,
        generator: torch.Generator | None = None,
        initial_noise: Tensor | None = None,
    ) -> Tensor:
        """Integrate the velocity field from t=0 to t=1 with uniform Euler steps."""
        if policy.training:
            raise RuntimeError("Call policy.eval() before sampling to disable dropout.")
        trajectory = self._initial_sample(sample_spec, initial_noise, generator)

        # Query at each interval's left endpoint; the last update lands at t=1.
        dt = 1.0 / self.num_inference_steps
        for step in range(self.num_inference_steps):
            timesteps = torch.full(
                (sample_spec.shape[0],),
                step * dt,
                device=sample_spec.device,
                dtype=getattr(torch, self.inference_time_dtype),
            )
            velocity = self._predict_velocity(policy, trajectory, timesteps, condition)
            trajectory = trajectory + dt * velocity.to(trajectory.dtype)
        return trajectory

    def _prepare_training_inputs(
        self,
        target: Tensor,
        *,
        timesteps: Tensor | None,
        noise: Tensor | None,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, Tensor]:
        """Validate the target and draw configured times and Gaussian noise.

        Time is drawn before noise to keep seeded runs reproducible. Inference's
        sample_mode does not affect training noise. Value checks are optional
        because they synchronize CUDA tensors and prevent fullgraph compilation.
        """
        validate_target(target, check_values=self.validate_values)
        if timesteps is None:
            if self.time_beta is None:
                timesteps = torch.rand(
                    (target.shape[0],), device=target.device, generator=generator
                )
            else:
                # LeRobot/OpenPI samples Beta on CPU, including for CUDA models.
                # Dirichlet is the primitive used by torch.distributions.Beta;
                # the explicit generator avoids changing unrelated RNG streams.
                concentration = torch.tensor(
                    self.time_beta, dtype=torch.float32
                ).expand(target.shape[0], 2)
                if generator is not None and generator.device.type != "cpu":
                    raise ValueError(
                        "Beta time sampling requires a CPU generator or explicit timesteps."
                    )
                timesteps = torch._sample_dirichlet(concentration, generator=generator)[
                    :, 0
                ]
                timesteps = timesteps.to(target.device)
            timesteps = timesteps * self.time_scale + self.time_offset
            if self.time_flip:
                timesteps = 1 - timesteps
        if timesteps.shape != (target.shape[0],) or not timesteps.is_floating_point():
            raise ValueError("timesteps must be floating-point (B,).")
        if self.validate_values and (
            not torch.isfinite(timesteps).all()
            or (timesteps < 0).any()
            or (timesteps > 1).any()
        ):
            raise ValueError("timesteps must be floating-point (B,) within [0, 1].")
        timesteps = timesteps.to(device=target.device)
        noise = gaussian_noise(
            target, noise, generator=generator, check_values=self.validate_values
        )
        return timesteps, noise

    @staticmethod
    def _predict_velocity(
        policy: BasePolicy, sample: Tensor, timesteps: Tensor, condition: Any
    ) -> Tensor:
        prediction = policy(sample, timesteps, condition)
        validate_prediction(prediction, sample)
        return prediction

    def _velocity_error(self, prediction: Tensor, expected: Tensor) -> Tensor:
        """Elementwise velocity error, before feature reduction or loss masking."""
        # Accumulate low-precision errors in fp32, as autocast MSE does.
        prediction = loss_precision(prediction)
        expected = loss_precision(expected)
        if self.norm_type == "l2":
            return (prediction - expected).square()
        if self.norm_type == "l1":
            return (prediction - expected).abs()
        return F.smooth_l1_loss(prediction, expected, reduction="none")

    def _reduce_loss(self, error: Tensor, loss_mask: Tensor | None) -> Tensor:
        """Sum features, average other axes, then apply the configured scale.

        For (B,H,D), the unmasked loss is scale * sum(error) / (B*H).
        Weighted reduction uses scale * D * sum(error*weights) / sum(weights),
        retaining the same scale when all weights are one.
        """
        if loss_mask is None:
            return self.loss_scale * error.sum(dim=-1).mean()

        weights = loss_weights(loss_mask, error, check_values=self.validate_values)
        denominator = weights.sum()
        return self.loss_scale * error.shape[-1] * (error * weights).sum() / denominator

    def _initial_sample(
        self,
        sample_spec: SampleSpec,
        initial_noise: Tensor | None,
        generator: torch.Generator | None,
    ) -> Tensor:
        """Explicit initial_noise overrides the mode and is never modified in place."""
        if initial_noise is not None:
            sample_spec.validate(
                initial_noise, "initial_noise", check_values=self.validate_values
            )
            return initial_noise.clone()
        if self.sample_mode == "stochastic":
            return torch.randn(
                sample_spec.shape,
                device=sample_spec.device,
                dtype=sample_spec.dtype,
                generator=generator,
            )
        return torch.zeros(
            sample_spec.shape, device=sample_spec.device, dtype=sample_spec.dtype
        )

    def _configure(
        self,
        *,
        num_inference_steps: int,
        sample_mode: str,
        norm_type: str,
        loss_scale: float,
        time_beta: tuple[float, float] | None = None,
        time_scale: float = 1.0,
        time_offset: float = 0.0,
        time_flip: bool = False,
        inference_time_dtype: str = "float32",
    ) -> None:
        """Use the same configuration validation on construction and restoration."""
        if type(num_inference_steps) is not int or num_inference_steps < 1:
            raise ValueError("num_inference_steps must be a positive integer.")
        if sample_mode not in ("zero", "stochastic"):
            raise ValueError("sample_mode must be 'zero' or 'stochastic'.")
        if norm_type not in ("l2", "l1", "smooth_l1"):
            raise ValueError("norm_type must be 'l2', 'l1' or 'smooth_l1'.")
        if not math.isfinite(loss_scale) or loss_scale <= 0:
            raise ValueError("loss_scale must be finite and positive.")
        self.num_inference_steps = num_inference_steps
        self.sample_mode = sample_mode
        self.norm_type = norm_type
        self.loss_scale = float(loss_scale)
        if time_beta is not None and (
            len(time_beta) != 2
            or any(not math.isfinite(x) or x <= 0 for x in time_beta)
        ):
            raise ValueError(
                "time_beta must contain two positive finite concentrations."
            )
        if not (
            math.isfinite(time_scale)
            and math.isfinite(time_offset)
            and 0 < time_scale <= 1
            and 0 <= time_offset <= 1 - time_scale + 1e-12
        ):
            raise ValueError("Scaled training times must remain in [0, 1].")
        if type(time_flip) is not bool or inference_time_dtype not in (
            "float32",
            "float64",
        ):
            raise ValueError("Invalid time_flip or inference_time_dtype.")
        self.time_beta = tuple(time_beta) if time_beta is not None else None
        self.time_scale, self.time_offset, self.time_flip = (
            time_scale,
            time_offset,
            time_flip,
        )
        self.inference_time_dtype = inference_time_dtype

    def get_extra_state(self):
        return {
            "version": 1,
            "num_inference_steps": self.num_inference_steps,
            "sample_mode": self.sample_mode,
            "norm_type": self.norm_type,
            "loss_scale": self.loss_scale,
            "time_beta": self.time_beta,
            "time_scale": self.time_scale,
            "time_offset": self.time_offset,
            "time_flip": self.time_flip,
            "inference_time_dtype": self.inference_time_dtype,
        }

    def set_extra_state(self, state):
        if state.get("version") != 1:
            raise ValueError("Unsupported flow objective state version.")
        self._configure(
            num_inference_steps=state["num_inference_steps"],
            sample_mode=state["sample_mode"],
            norm_type=state["norm_type"],
            loss_scale=state["loss_scale"],
            time_beta=state.get("time_beta"),
            time_scale=state.get("time_scale", 1.0),
            time_offset=state.get("time_offset", 0.0),
            time_flip=state.get("time_flip", False),
            inference_time_dtype=state.get("inference_time_dtype", "float32"),
        )
