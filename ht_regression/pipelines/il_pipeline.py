"""Imitation-learning composition of a policy with an objective."""

import torch
from torch import nn

from ..objectives.base import BaseObjective
from ..policies.base import BasePolicy


class PolicyPipeline(nn.Module):
    """Bind model and objective for training, checkpointing and evaluation.

    The policy owns representations and action alignment. The objective owns
    the learning and sampling algorithm. Neither stores a reference to the other.
    """

    def __init__(self, policy: BasePolicy, objective: BaseObjective):
        super().__init__()
        if not isinstance(policy, BasePolicy) or not isinstance(
            objective, BaseObjective
        ):
            raise TypeError("Expected a BasePolicy and a BaseObjective.")
        self.policy = policy
        self.objective = objective

    @property
    def device(self):
        return self.policy.device

    @property
    def dtype(self):
        return self.policy.dtype

    def forward(self, sample, time, condition):
        return self.policy(sample, time, condition)

    def compute_loss(self, batch, **kwargs):
        condition = self.policy.encode_condition(batch)
        target = self.policy.encode_target(batch)
        return self.objective.compute_loss(
            self.policy, target.sample, condition, loss_mask=target.loss_mask, **kwargs
        )

    @torch.no_grad()
    def predict_action(self, observation, **kwargs):
        if self.training or self.policy.training:
            raise RuntimeError("Call pipeline.eval() before predict_action().")
        condition = self.policy.encode_condition(observation)
        prediction = self.objective.sample(
            self.policy,
            condition,
            sample_spec=self.policy.sample_spec(condition),
            **kwargs,
        )
        return self.policy.decode_prediction(prediction)

    def reset(self):
        self.policy.reset()

    def get_optimizer_groups(self, weight_decay: float):
        return self.policy.get_optimizer_groups(weight_decay)
