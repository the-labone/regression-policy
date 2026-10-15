"""DDPM/DDIM learning and sampling through the BasePolicy interface.

Follows real-stanford/diffusion_policy (MIT), revision
5ba07ac6661db573af695b419a7947ecb704690f. See ../../../policies/LICENSE.diffusion_policy.
"""

import copy
import inspect
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler
from torch import Tensor

from ....policies.base import BasePolicy, SampleSpec
from ...base import BaseObjective
from ...common import gaussian_noise, loss_weights


def _scheduler_from_config(name: str, config: Mapping):
    classes = {"DDPMScheduler": DDPMScheduler, "DDIMScheduler": DDIMScheduler}
    if name not in classes:
        raise ValueError(f"Unsupported diffusion scheduler: {name!r}.")
    cls = classes[name]
    kwargs = {key: value for key, value in config.items() if not key.startswith("_")}
    unknown = kwargs.keys() - inspect.signature(cls.__init__).parameters.keys()
    if unknown:
        raise ValueError(f"Unsupported {name} configuration fields: {sorted(unknown)}")
    return cls(**kwargs)


class DiffusionObjective(BaseObjective):
    """Diffusion over floating-point samples of any shape, with opaque conditions.

    The policy supplies one prediction. No observation preprocessing, action
    normalization, architecture selection or execution slicing happens here.
    Masks weight the loss only; conditional inpainting is not implemented.
    ``validate_values=False`` trusts input values while retaining metadata checks,
    so loss computation can be compiled without Tensor-to-Python checks. The caller
    then owns validation of supplied timesteps, noise and masks.
    """

    def __init__(
        self,
        *,
        noise_scheduler: DDPMScheduler | DDIMScheduler | None = None,
        num_inference_steps: int | None = None,
        scheduler_step_kwargs: Mapping | None = None,
        validate_values: bool = True,
    ):
        super().__init__()
        self.validate_values = validate_values
        if noise_scheduler is None:
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=100,
                beta_schedule="squaredcos_cap_v2",
                variance_type="fixed_small",
                clip_sample=True,
                prediction_type="epsilon",
            )
        self._configure_sampling(
            noise_scheduler, num_inference_steps, scheduler_step_kwargs or {}
        )

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # Diffusers' scheduler is not an nn.Module. Move its training coefficients
        # before compilation so add_noise does not capture a CPU -> GPU copy and
        # prevent CUDA graph capture. Keep their dtype: add_noise/get_velocity own
        # the original DP casts to the target dtype, including mixed precision.
        alphas = self.noise_scheduler.alphas_cumprod
        device = fn(torch.empty(0, device=alphas.device)).device
        self.noise_scheduler.alphas_cumprod = alphas.to(device=device)
        return self

    def _configure_sampling(self, scheduler, steps, step_kwargs):
        if type(scheduler) not in (DDPMScheduler, DDIMScheduler):
            raise TypeError("Only DDPM and DDIM schedulers are supported.")
        if scheduler.config.prediction_type not in (
            "epsilon",
            "sample",
            "v_prediction",
        ):
            raise ValueError("Unsupported diffusion prediction_type.")
        if getattr(scheduler.config, "variance_type", "fixed_small") in (
            "learned",
            "learned_range",
        ):
            raise ValueError(
                "Learned variance requires a different backbone output head."
            )
        train_steps = scheduler.config.num_train_timesteps
        steps = train_steps if steps is None else steps
        if type(steps) is not int or steps < 1:
            raise ValueError("num_inference_steps must be a positive integer.")
        if steps > train_steps:
            raise ValueError("num_inference_steps cannot exceed num_train_timesteps.")
        # Pinned diffusers 0.11.1 DDPM advances t -> t-1, even for strided schedules.
        if isinstance(scheduler, DDPMScheduler) and steps != train_steps:
            raise ValueError(
                "DP-compatible DDPM requires all training timesteps; use DDIM for reduced-step sampling."
            )
        allowed = (
            {"eta", "use_clipped_model_output"}
            if isinstance(scheduler, DDIMScheduler)
            else set()
        )
        if set(step_kwargs) - allowed:
            raise ValueError(
                f"Unsupported scheduler step arguments: {sorted(set(step_kwargs) - allowed)}"
            )
        if "eta" in step_kwargs and (
            not torch.isfinite(torch.tensor(step_kwargs["eta"]))
            or step_kwargs["eta"] < 0
        ):
            raise ValueError("DDIM eta must be finite and nonnegative.")
        self.noise_scheduler = copy.deepcopy(scheduler)
        self.num_inference_steps = steps
        self.scheduler_step_kwargs = dict(step_kwargs)

    def get_extra_state(self):
        return {
            "version": 1,
            "scheduler_class": type(self.noise_scheduler).__name__,
            "scheduler_config": copy.deepcopy(dict(self.noise_scheduler.config)),
            "num_inference_steps": self.num_inference_steps,
            "scheduler_step_kwargs": self.scheduler_step_kwargs.copy(),
        }

    def set_extra_state(self, state):
        if state.get("version") != 1:
            raise ValueError("Unsupported diffusion objective state version.")
        scheduler = _scheduler_from_config(
            state["scheduler_class"], state["scheduler_config"]
        )
        self._configure_sampling(
            scheduler, state["num_inference_steps"], state["scheduler_step_kwargs"]
        )

    @staticmethod
    def _prediction(policy, sample, timestep, condition):
        prediction = policy(sample, timestep, condition)
        if not isinstance(prediction, Tensor) or prediction.shape != sample.shape:
            raise ValueError("policy prediction must match the sample shape.")
        if prediction.device != sample.device:
            raise ValueError("policy prediction must match the sample device.")
        return prediction

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
        # SampleSpec is for the sampling boundary. Avoid constructing that frozen
        # dataclass inside a compiled loss (unsupported by older Torch runtimes).
        if not target.ndim or not target.numel() or not target.is_floating_point():
            raise ValueError("target must be a nonempty floating-point sample batch.")
        if self.validate_values and not torch.isfinite(target).all():
            raise ValueError("target must be finite.")
        noise = gaussian_noise(
            target, noise, generator=generator, check_values=self.validate_values
        )
        if timesteps is None:
            timesteps = torch.randint(
                self.noise_scheduler.config.num_train_timesteps,
                (target.shape[0],),
                device=target.device,
                generator=generator,
            )
        if (
            timesteps.shape != (target.shape[0],)
            or timesteps.dtype != torch.long
            or (
                self.validate_values
                and (
                    (timesteps < 0).any()
                    or (
                        timesteps >= self.noise_scheduler.config.num_train_timesteps
                    ).any()
                )
            )
        ):
            raise ValueError(
                "timesteps must be int64 (B,) within the training schedule."
            )
        timesteps = timesteps.to(target.device)
        noisy_sample = self.noise_scheduler.add_noise(target, noise, timesteps)
        prediction = self._prediction(policy, noisy_sample, timesteps, condition)
        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            expected = noise
        elif prediction_type == "sample":
            expected = target
        else:
            expected = self.noise_scheduler.get_velocity(target, noise, timesteps)
        if loss_mask is None:
            return F.mse_loss(prediction, expected)
        error = F.mse_loss(prediction, expected, reduction="none")
        weights = loss_weights(loss_mask, error, check_values=self.validate_values)
        denominator = weights.sum()
        return (error * weights).sum() / denominator

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
        if policy.training:
            raise RuntimeError("Call policy.eval() before sampling to disable dropout.")
        if initial_noise is None:
            trajectory = torch.randn(
                sample_spec.shape,
                device=sample_spec.device,
                dtype=sample_spec.dtype,
                generator=generator,
            )
        else:
            sample_spec.validate(
                initial_noise, "initial_noise", check_values=self.validate_values
            )
            trajectory = initial_noise.clone()
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for timestep in self.noise_scheduler.timesteps:
            prediction = self._prediction(policy, trajectory, timestep, condition)
            trajectory = self.noise_scheduler.step(
                prediction,
                timestep,
                trajectory,
                generator=generator,
                **self.scheduler_step_kwargs,
            ).prev_sample
        return trajectory
