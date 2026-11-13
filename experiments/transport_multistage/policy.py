"""Six native Transformer experts for the original opposed-arm Transport layout."""
from collections.abc import Mapping
import copy
import math

import torch
from torch import nn
from ht_regression.policies.from_scratch.action_chunk_policy import ActionChunkPolicy

STAGE_NAMES = ('lid_removal', 'payload_pickup', 'donor_transport',
               'handover', 'receiver_transport', 'target_placement')
DEFAULT_THRESHOLDS = dict(source_y=-.25, donor_y=-.12, receiver_y=.18,
    lift_height=.12, lid_xy=.22, lid_height=.08, donor_release=.14,
    receiver_near=.16, target_xy=.18)


def checked_thresholds(thresholds):
    values = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    if set(values) != set(DEFAULT_THRESHOLDS) or any(
        not math.isfinite(float(v)) for v in values.values()
    ):
        raise ValueError('Expected all finite Transport thresholds.')
    if not (-1 < values['source_y'] < values['donor_y'] < 0 < values['receiver_y'] < 1) or any(
        not 0 < values[k] < 1 for k in values if k not in ('source_y','donor_y','receiver_y')
    ):
        raise ValueError('Expected ordered Y planes and positive sub-metre distances.')
    return values


def stage_ids(observation, thresholds=None):
    """Route using only the latest raw object41 + two arm pose/gripper9 states.

    Fixed Y planes refer to robosuite 1.2's default tables centred at +/-0.45 m.
    Geometric stages allow reversals; they do not assert actual grasp contacts.
    Trash clearing proceeds concurrently and both arms' actions remain joint.
    """
    if observation.ndim != 3 or observation.shape[-1] != 59:
        raise ValueError('Expected native Transport observations (B,T,59).')
    t = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    current = observation[:, -1]
    payload, lid, target = current[:, :3], current[:, 14:17], current[:, 21:24]
    near_target = ((payload[:, :2]-target[:, :2]).square().sum(-1) < t['target_xy']**2
                   ) | (current[:, 39] > .5)
    receiver = ((current[:, 27:30].square().sum(-1) > t['donor_release']**2)
                & (current[:, 33:36].square().sum(-1) < t['receiver_near']**2))
    stage = torch.where(near_target, 5, torch.where(receiver | (payload[:, 1] > t['receiver_y']),4,3))
    stage = torch.where(payload[:, 1] < t['donor_y'], 2, stage)
    source = payload[:, 1] < t['source_y']
    stage = torch.where(source & (payload[:, 2]-target[:, 2] < t['lift_height']),1,stage)
    covered = (source & ((lid[:, :2]-payload[:, :2]).square().sum(-1) < t['lid_xy']**2)
               & (lid[:, 2]-payload[:, 2] > t['lid_height']))
    return torch.where(covered,0,stage)


def state_condition(condition):
    return condition['state'] if isinstance(condition, Mapping) else condition


class TransportStagePolicy(ActionChunkPolicy):
    def __init__(self, model, normalizer, *, thresholds=None, **kwargs):
        super().__init__(model, normalizer, **kwargs)
        if self.obs_dim != 59 or self.action_dim != 20 or self.obs_encoder is not None:
            raise ValueError('Transport requires native 59D observations and 20D ABS model actions.')
        self.thresholds = checked_thresholds(thresholds)
        self.other_experts = nn.ModuleList(copy.deepcopy(model) for _ in range(5))

    @property
    def experts(self):
        return (self.model, *self.other_experts)

    def encode_condition(self, observation):
        state = super().encode_condition(observation)
        weights = observation.get('expert_weights') if isinstance(observation, Mapping) else None
        return {'state':state, 'expert_weights':weights}

    def sample_spec(self, condition):
        return super().sample_spec(state_condition(condition))

    def stages_from_condition(self, condition):
        return stage_ids(self.normalizer.unnormalize_obs(state_condition(condition)),self.thresholds)

    def forward(self, sample, time, condition):
        state = state_condition(condition)
        stages = self.stages_from_condition(state)
        if self.training:
            result = torch.zeros_like(sample)
            for index,expert in enumerate(self.experts):
                result = result + torch.where((stages==index)[:,None,None],expert(sample,time,state),0.)
            return result
        result = torch.empty_like(sample)
        for index,expert in enumerate(self.experts):
            rows = (stages==index).nonzero(as_tuple=True)[0]
            if rows.numel():
                subtime = time[rows] if torch.is_tensor(time) and time.ndim and time.numel()>1 else time
                result[rows] = expert(sample[rows],subtime,state[rows])
        return result

    def get_optimizer_groups(self, weight_decay):
        return [group for expert in self.experts
                for group in expert.get_optimizer_groups(weight_decay=weight_decay)]

    def get_extra_state(self):
        return {**super().get_extra_state(),'transport_router_version':1,
                'thresholds':dict(self.thresholds)}

    def set_extra_state(self, state):
        super().set_extra_state(state)
        if state.get('transport_router_version') != 1 or state.get('thresholds') != self.thresholds:
            raise ValueError('Checkpoint Transport router differs from configured router.')
