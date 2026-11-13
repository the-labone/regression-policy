"""Original Transport windows with episode-local, observation-derived memberships."""
import numpy as np
import torch
from ht_regression.data.robomimic.dataset import RobomimicStateDataset
from experiments.toolhang_overlap.dataset import boundary_weights
from experiments.transport_multistage.policy import checked_thresholds, stage_ids

OBS_KEYS = ('object','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos',
            'robot1_eef_pos','robot1_eef_quat','robot1_gripper_qpos')


def validate_metadata(metadata):
    kwargs = metadata['env_kwargs']
    if (metadata['env_name'] != 'TwoArmTransport'
            or kwargs.get('env_configuration') != 'single-arm-opposed'
            or list(kwargs.get('robots',[])) != ['Panda','Panda']
            or not np.allclose(kwargs.get('tables_boundary',[.8,1.2,.05]),[.8,1.2,.05])
            or not np.allclose(kwargs.get('bin_size',[.3,.3,.15]),[.3,.3,.15])):
        raise ValueError('Transport router requires the original opposed-Panda table/bin layout.')


class TransportOverlapDataset(RobomimicStateDataset):
    def __init__(self,*args,overlap_steps=8,thresholds=None,**kwargs):
        super().__init__(*args,**kwargs)
        validate_metadata(self._env_meta)
        if self.obs_keys != OBS_KEYS or self.obs_dim != 59 or self.action_dim != 20:
            raise ValueError('Transport overlap requires native 59D state and two-arm ABS actions.')
        self.overlap_steps = overlap_steps
        self.thresholds = checked_thresholds(thresholds)
        self._expert_weights = []
        for observations,_ in self._episodes:
            phases = stage_ids(torch.from_numpy(observations[:,None]),self.thresholds).numpy()
            self._expert_weights.append(boundary_weights(phases,overlap_steps))

    def __getitem__(self,index):
        sample = super().__getitem__(index)
        episode,start = self._windows[index]
        weights = self._expert_weights[episode]
        decision = int(np.clip(start+self.n_obs_steps-1,0,len(weights)-1))
        sample['expert_weights'] = torch.from_numpy(weights[decision].copy())
        return sample
