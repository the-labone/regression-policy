"""Native windows plus episode-local expert memberships near stage boundaries."""
import numpy as np
import torch

from ht_regression.data.robomimic.dataset import RobomimicStateDataset
from experiments.toolhang_multistage.policy import stage_ids


def boundary_weights(phases, overlap_steps=8):
    """Share [b-radius,b+radius) between both stages incident to boundary b.

    b is the first state in the new stage. Intersecting bands take a set union,
    so repeated switches cannot increase a window's total weight above one.
    The caller invokes this independently for every demonstration.
    """
    phases = np.asarray(phases)
    if (phases.ndim != 1 or not np.issubdtype(phases.dtype, np.integer)
            or np.any((phases < 0) | (phases >= 6))):
        raise ValueError('Expected a vector of stage indices in [0,5].')
    if type(overlap_steps) is not int or overlap_steps < 0:
        raise ValueError('overlap_steps must be a nonnegative integer.')
    membership = np.eye(6, dtype=bool)[phases]
    for boundary in np.flatnonzero(phases[1:] != phases[:-1]) + 1:
        lo, hi = max(0, boundary-overlap_steps), min(len(phases), boundary+overlap_steps)
        membership[lo:hi, phases[boundary-1]] = True
        membership[lo:hi, phases[boundary]] = True
    weights = membership.astype(np.float32)
    return weights / weights.sum(axis=1, keepdims=True)


class OverlapStateDataset(RobomimicStateDataset):
    """Keep native observations/actions/splits/normalization; add loss metadata."""
    def __init__(self, *args, overlap_steps=8, **kwargs):
        super().__init__(*args, **kwargs)
        if self.obs_keys != ('object','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos'):
            raise ValueError('Overlap labels require native Tool-Hang state ordering.')
        self.overlap_steps = overlap_steps
        self._expert_weights = []
        for observations, _ in self._episodes:
            phases = stage_ids(torch.from_numpy(observations[:, None])).numpy()
            self._expert_weights.append(boundary_weights(phases, overlap_steps))

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        episode, start = self._windows[index]
        weights = self._expert_weights[episode]
        decision = int(np.clip(start+self.n_obs_steps-1, 0, len(weights)-1))
        sample['expert_weights'] = torch.from_numpy(weights[decision].copy())
        return sample
