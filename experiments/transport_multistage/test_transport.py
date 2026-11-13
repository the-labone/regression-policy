"""Behavior tests for the original dual-arm Transport observation/action contract."""
import copy
import importlib.util
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from ht_regression.data.robomimic.dataset import RobomimicStateDataset
from ht_regression.data.robomimic.normalizer import RobomimicNormalizer
from ht_regression.pipelines import PolicyPipeline
from ht_regression.policies.from_scratch.backbones import TransformerBackbone
from experiments.toolhang_overlap.objective import OverlapMSEObjective


def states():
    """Six hand-checked scenes, with intentionally misleading previous frames."""
    obs = torch.zeros(6, 2, 59)
    current = obs[:, -1]
    current[:, :3] = torch.tensor([[.2,-.45,.826], [.2,-.45,.826],
        [.2,-.3,1.1], [.1,.05,1.05], [.0,.22,1.1], [-.19,.43,1.0]])
    current[:, 14:17] = torch.tensor([-.18,-.45,.825])
    current[0, 14:17] = torch.tensor([.2,-.45,.976])
    current[:, 21:24] = torch.tensor([-.2,.45,.805])
    current[:, 24:27] = torch.tensor([.2,.45,.805])
    current[:, 27:30] = torch.tensor([.0,.0,-.07])
    current[4:, 27:30] = torch.tensor([.0,.3,-.1])
    current[:, 33:36] = torch.tensor([.0,-.5,-.1])
    current[3:, 33:36] = torch.tensor([.0,.0,-.05])
    current[:, 41:44] = current[:, :3]-current[:, 27:30]
    current[:, 50:53] = current[:, :3]-current[:, 33:36]
    obs[:, 0] = current.flip(0)
    return obs


class TransportTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.assertIsNotNone(importlib.util.find_spec('experiments.transport_multistage.policy'),
                             'Transport implementation is missing')
        from experiments.transport_multistage.policy import stage_ids, TransportStagePolicy
        self.route, self.Policy = stage_ids, TransportStagePolicy

    def pipeline(self, **kwargs):
        torch.manual_seed(41)
        model = TransformerBackbone(action_dim=20, obs_dim=59, horizon=10,
            n_obs_steps=2, n_layer=1, n_emb=16, n_head=2, p_drop_attn=0.)
        norm = RobomimicNormalizer(59,20,'abs',normalization='minmax')
        norm.obs_scale.fill_(2.3); norm.obs_offset.fill_(.7)
        norm.action_scale.fill_(1); norm.action_offset.zero_()
        return PolicyPipeline(self.Policy(model,norm,validate_values=False,**kwargs),
                              OverlapMSEObjective(validate_values=False))

    def test_scene_routing_latest_frame_and_recovery(self):
        obs = states()
        torch.testing.assert_close(self.route(obs), torch.arange(6))
        # Receiver can acquire the payload before it crosses y=0.18.
        obs[4,-1,1] = .05
        torch.testing.assert_close(self.route(obs), torch.arange(6))
        # A return to the initial state must undo a later phase.
        sequence = obs[[5,0,4,1,3,2]]
        torch.testing.assert_close(self.route(sequence), torch.tensor([5,0,4,1,3,2]))
        pipeline = self.pipeline()
        torch.testing.assert_close(pipeline.policy.stages_from_condition(
            pipeline.policy.encode_condition({'obs':obs})), torch.arange(6))
        with self.assertRaises(ValueError): self.route(torch.zeros(2,2,53))

    def test_independent_weighted_losses_gradients_and_no_unused_expert_updates(self):
        pipeline = self.pipeline().train()
        with torch.no_grad():
            for index, expert in enumerate(pipeline.policy.experts):
                for p in expert.parameters(): p.zero_()
                expert.head.bias.fill_([-1,1,2,3,4,5][index])
        batch = dict(obs=states()[[0,2]],action=torch.zeros(2,10,20),
            expert_weights=torch.tensor([[.5,.5,0,0,0,0],[0,0,1,0,0,0]]))
        loss = pipeline.compute_loss(batch)
        torch.testing.assert_close(loss,torch.tensor(2.5))
        loss.backward()
        for index,expected in enumerate([-.025,.025,.1,0,0,0]):
            torch.testing.assert_close(pipeline.policy.experts[index].head.bias.grad,
                                      torch.full((20,),float(expected)))

    def test_inference_isolates_metadata_and_preserves_dual_arm_execution(self):
        pipeline = self.pipeline().eval()
        with torch.no_grad():
            for index,expert in enumerate(pipeline.policy.experts):
                for p in expert.parameters(): p.zero_()
                expert.head.bias.fill_(index)
            obs = states()
            batch = dict(obs=obs, action=torch.zeros(6,10,20),
                         expert_weights=torch.full((6,6),float('nan')))
            pred = pipeline.predict_action(batch)
            expected = torch.arange(6).float()[:,None,None].expand(6,10,20)
            torch.testing.assert_close(pred['action_pred'],expected)
            torch.testing.assert_close(pred['action'],expected[:,1:9])
            torch.testing.assert_close(pipeline.predict_action({'obs':obs})['action_pred'],expected)
            torch.testing.assert_close(pipeline.compute_loss(batch),expected.square().mean())

    def test_initialization_independence_compile_and_checkpoint_router(self):
        from experiments.transport_multistage.policy import DEFAULT_THRESHOLDS
        pipeline = self.pipeline().train()
        params = [next(expert.parameters()) for expert in pipeline.policy.experts]
        self.assertEqual(len({p.data_ptr() for p in params}),6)
        for param in params[1:]: torch.testing.assert_close(param,params[0],rtol=0,atol=0)
        batch = dict(obs=states(),action=torch.randn(6,10,20),expert_weights=torch.eye(6))
        eager = pipeline.compute_loss(batch); eager.backward()
        gradients = [p.grad.clone() for p in pipeline.parameters() if p.requires_grad]
        pipeline.zero_grad(set_to_none=True)
        compiled = torch.compile(pipeline.compute_loss,backend='eager',fullgraph=True)
        result = compiled(batch); result.backward()
        torch.testing.assert_close(result,eager)
        for param,grad in zip((p for p in pipeline.parameters() if p.requires_grad),gradients):
            torch.testing.assert_close(param.grad,grad)
        restored = self.pipeline(); restored.load_state_dict(pipeline.state_dict())
        changed = dict(DEFAULT_THRESHOLDS,lift_height=.13)
        with self.assertRaises(ValueError): self.pipeline(thresholds=changed).load_state_dict(pipeline.state_dict())
        with self.assertRaises(ValueError): self.pipeline(thresholds=dict(DEFAULT_THRESHOLDS,lift_height=float('nan')))

    def test_resolved_recipe_preserves_transport_defaults(self):
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from ht_regression.training.robomimic.config import register_experiment_configs
        root = Path(__file__).resolve().parents[2]
        register_experiment_configs()
        with initialize_config_dir(version_base='1.3',config_dir=str(root/'configs')):
            base = compose(config_name='train',overrides=[
                'experiment=robomimic/state/transport/mh/ht_transformer_nu200',
                'objective=mse',
                'policy._target_=ht_regression.policies.from_scratch.action_chunk_policy.ActionChunkPolicy',
                'dataset.path=/data/transport.hdf5','training.output_dir=/tmp/transport-config-test'])
            cfg = compose(config_name='train',overrides=[
                'experiment=robomimic/state/transport/mh/mse_multistage6_overlap8',
                'dataset.path=/data/transport.hdf5','training.output_dir=/tmp/transport-config-test'])
        a,b = [OmegaConf.to_container(c,resolve=True) for c in (base,cfg)]
        expected = copy.deepcopy(a)
        expected['name']=b['name']
        for key in ('dataset','policy','objective'): expected[key]['_target_']=b[key]['_target_']
        expected['dataset']['overlap_steps']=8
        expected['dataset']['thresholds']=None
        expected['policy']['thresholds']=None
        self.assertEqual(expected,b)
        from experiments.transport_multistage.policy import DEFAULT_THRESHOLDS
        cfg.policy.thresholds=dict(DEFAULT_THRESHOLDS,lift_height=.13)
        self.assertEqual(OmegaConf.to_container(cfg.dataset.thresholds),dict(DEFAULT_THRESHOLDS,lift_height=.13))

    @unittest.skipUnless(os.environ.get('TRANSPORT_DATASET'),'Set TRANSPORT_DATASET for real-data checks')
    def test_native_data_parity_overlap_and_geometry(self):
        from experiments.transport_multistage.dataset import TransportOverlapDataset, validate_metadata
        kwargs = dict(path=os.environ['TRANSPORT_DATASET'],action_space='abs',source_action_space='abs',
            horizon=10,n_obs_steps=2,n_action_steps=8,val_ratio=.02,seed=42,normalization='minmax')
        native = RobomimicStateDataset(**kwargs); actual = TransportOverlapDataset(**kwargs)
        from experiments.transport_multistage.policy import DEFAULT_THRESHOLDS
        alternate = TransportOverlapDataset(**kwargs,thresholds=dict(DEFAULT_THRESHOLDS,source_y=-.5))
        torch.testing.assert_close(alternate[0]['expert_weights'],torch.tensor([0.,0.,1.,0.,0.,0.]))
        torch.testing.assert_close(actual[0]['expert_weights'],torch.tensor([1.,0.,0.,0.,0.,0.]))
        validate_metadata(native._env_meta)
        bad = copy.deepcopy(native._env_meta); bad['env_kwargs']['env_configuration']='single-arm-parallel'
        with self.assertRaises(ValueError): validate_metadata(bad)
        bad = copy.deepcopy(native._env_meta); bad['env_kwargs']['tables_boundary']=[.8,2.,.05]
        with self.assertRaises(ValueError): validate_metadata(bad)
        for key,value in native.normalizer.state_dict().items():
            other = actual.normalizer.state_dict()[key]
            if torch.is_tensor(value): torch.testing.assert_close(value,other,rtol=0,atol=0)
            else: self.assertEqual(value,other)
        for left,right in [(native,actual),(native.get_validation_dataset(),actual.get_validation_dataset())]:
            self.assertEqual(left.demo_names,right.demo_names); self.assertEqual(len(left),len(right))
            for i in np.linspace(0,len(left)-1,40,dtype=int):
                a,b = left[i],right[i]
                for key in a: torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
                torch.testing.assert_close(b['obs'][:,27:30],b['obs'][:,:3]-b['obs'][:,41:44],atol=1e-6,rtol=1e-6)
                torch.testing.assert_close(b['obs'][:,33:36],b['obs'][:,:3]-b['obs'][:,50:53],atol=1e-6,rtol=1e-6)
                ep,start = right._windows[i]; obs = right._episodes[ep][0]
                phases = self.route(torch.from_numpy(obs[:,None])).numpy();t=start+1
                allowed = {int(phases[t])}
                for boundary in np.flatnonzero(np.diff(phases))+1:
                    if boundary-8<=t<boundary+8: allowed.update([int(phases[boundary-1]),int(phases[boundary])])
                expected = torch.zeros(6)
                for stage in allowed: expected[stage]=1/len(allowed)
                torch.testing.assert_close(b['expert_weights'],expected)


if __name__=='__main__': unittest.main()
