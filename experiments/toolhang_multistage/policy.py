"""Six independent native MSE experts with an observation-only Tool-Hang gate."""
import copy

import torch
from torch import nn

from ht_regression.policies.from_scratch.action_chunk_policy import ActionChunkPolicy

STAGE_NAMES = ('frame_pickup', 'frame_transport', 'frame_align_insert',
               'tool_pickup', 'tool_transport', 'tool_align_hang')
DEFAULT_THRESHOLDS = dict(lift_height=.09, frame_base_xy=.10, tool_frame_distance=.20)


def stage_ids(observation, thresholds=None):
    """Native 1.2 global-first layout: object44 + EEF3/4 + gripper2.

    Uses the latest supplied observation only. No action, future state, elapsed
    time, hidden simulator state or irreversible phase latch is consulted.
    """
    if observation.ndim != 3 or observation.shape[-1] != 53:
        raise ValueError('Expected native Tool-Hang observations (B,T,53).')
    thresholds = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    current = observation[:, -1]
    base, frame, tool = current[:, :3], current[:, 14:17], current[:, 28:31]
    assembled, hung = current[:, 42] > .5, current[:, 43] > .5
    frame_low = frame[:, 2] - base[:, 2] < thresholds['lift_height']
    tool_low = tool[:, 2] - base[:, 2] < thresholds['lift_height']
    frame_far = (frame[:, :2] - base[:, :2]).square().sum(-1) > thresholds['frame_base_xy']**2
    tool_far = (tool - frame).square().sum(-1) > thresholds['tool_frame_distance']**2
    frame_phase = torch.where(frame_low, 0, torch.where(frame_far, 1, 2))
    tool_phase = torch.where(tool_low & ~hung, 3, torch.where(tool_far & ~hung, 4, 5))
    return torch.where(assembled, tool_phase, frame_phase)


class StageMSEPolicy(ActionChunkPolicy):
    """Preserve native targets, normalization, action alignment and MSE objective.

    Dense masked training retains static shapes for upstream fullgraph compile.
    Each sample contributes a gradient only to its selected expert. Eager eval
    dispatches subsets to avoid computing unused experts. All experts start at
    the same native initialization, but own separate storage and optimizer state.
    """
    def __init__(self, model, normalizer, *, thresholds=None, **kwargs):
        super().__init__(model, normalizer, **kwargs)
        if self.obs_dim != 53 or self.obs_encoder is not None:
            raise ValueError('StageMSEPolicy requires native 53D Tool-Hang state.')
        self.other_experts = nn.ModuleList(copy.deepcopy(model) for _ in range(5))
        self.thresholds = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
        if set(self.thresholds) != set(DEFAULT_THRESHOLDS) or any(
            not 0 < float(value) < 1 for value in self.thresholds.values()
        ):
            raise ValueError('Expected positive geometric thresholds below one metre.')

    @property
    def experts(self):
        return (self.model, *self.other_experts)

    def stages_from_condition(self, condition):
        raw = self.normalizer.unnormalize_obs(condition)
        return stage_ids(raw, self.thresholds)

    def forward(self, sample, time, condition):
        stages = self.stages_from_condition(condition)
        if self.training:
            # No stage balancing: native MSE still averages over the original B.
            result = torch.zeros_like(sample)
            for index, expert in enumerate(self.experts):
                prediction = expert(sample, time, condition)
                result = result + torch.where((stages == index)[:, None, None], prediction, 0.)
            return result
        result = torch.empty_like(sample)
        for index, expert in enumerate(self.experts):
            rows = (stages == index).nonzero(as_tuple=True)[0]
            if rows.numel():
                subtime = time[rows] if torch.is_tensor(time) and time.ndim and time.numel() > 1 else time
                result[rows] = expert(sample[rows], subtime, condition[rows])
        return result

    def get_optimizer_groups(self, weight_decay):
        return [group for expert in self.experts
                for group in expert.get_optimizer_groups(weight_decay=weight_decay)]

    def get_extra_state(self):
        return {**super().get_extra_state(), 'stage_router_version': 1,
                'thresholds': dict(self.thresholds)}

    def set_extra_state(self, state):
        super().set_extra_state(state)
        if state.get('stage_router_version') != 1 or state['thresholds'] != self.thresholds:
            raise ValueError('Checkpoint stage router differs from configured router.')
