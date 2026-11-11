"""Carry training memberships outside the unchanged observation-only policy."""
from collections.abc import Mapping

from experiments.toolhang_multistage.policy import StageMSEPolicy


def state_condition(condition):
    return condition['state'] if isinstance(condition, Mapping) else condition


class OverlapStagePolicy(StageMSEPolicy):
    def encode_condition(self, observation):
        state = super().encode_condition(observation)
        weights = observation.get('expert_weights') if isinstance(observation, Mapping) else None
        return {'state': state, 'expert_weights': weights}

    def sample_spec(self, condition):
        return super().sample_spec(state_condition(condition))

    def stages_from_condition(self, condition):
        return super().stages_from_condition(state_condition(condition))

    def forward(self, sample, time, condition):
        # Metadata never enters the router or a backbone, including at eval.
        return super().forward(sample, time, state_condition(condition))
