"""Weighted independent MSEs during training; native routed MSE at validation."""
import torch

from ht_regression.objectives.regression.MSE import MSEObjective
from ht_regression.objectives.common import (
    loss_precision, loss_weights, validate_prediction, validate_target,
)


class OverlapMSEObjective(MSEObjective):
    def compute_loss(self, policy, target, condition, *, loss_mask=None, generator=None):
        # Validation and residual reports measure the policy actually deployed.
        if not policy.training:
            return super().compute_loss(policy, target, condition,
                                        loss_mask=loss_mask, generator=generator)
        validate_target(target, check_values=self.validate_values)
        weights = condition.get('expert_weights')
        if weights is None or weights.shape != (target.shape[0], len(policy.experts)):
            raise ValueError('Training requires one normalized expert-weight vector per window.')
        if self.validate_values and (
            not torch.isfinite(weights).all() or (weights < 0).any()
            or not torch.allclose(weights.sum(-1), torch.ones_like(weights[:, 0]))
        ):
            raise ValueError('Expert weights must be nonnegative, finite and sum to one.')
        sample = torch.zeros_like(target)
        time = torch.zeros(target.shape[0], device=target.device, dtype=torch.float32)
        precise_target = loss_precision(target)
        error = torch.zeros_like(precise_target)
        for index, expert in enumerate(policy.experts):
            prediction = expert(sample, time, condition['state'])
            validate_prediction(prediction, sample)
            squared = (loss_precision(prediction)-precise_target).square()
            error = error + weights[:, index, None, None].to(squared.dtype) * squared
        if loss_mask is None:
            return self.loss_scale * error.mean()
        mask = loss_weights(loss_mask, error, check_values=self.validate_values)
        return self.loss_scale * (error*mask).sum() / mask.sum()
