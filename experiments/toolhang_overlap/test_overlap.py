"""Behavior tests for shared boundary states and separately weighted MSEs."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from ht_regression.data.robomimic.dataset import RobomimicStateDataset
from ht_regression.data.robomimic.normalizer import RobomimicNormalizer
from ht_regression.policies.from_scratch.backbones import TransformerBackbone
from ht_regression.pipelines import PolicyPipeline
from experiments.toolhang_multistage.test_policy import states


class OverlapTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.assertIsNotNone(importlib.util.find_spec('experiments.toolhang_overlap.dataset'),
                             'overlap implementation is missing')
        from experiments.toolhang_overlap.dataset import boundary_weights
        from experiments.toolhang_overlap.policy import OverlapStagePolicy
        from experiments.toolhang_overlap.objective import OverlapMSEObjective
        self.weights, self.Policy, self.Objective = boundary_weights, OverlapStagePolicy, OverlapMSEObjective

    def make_pipeline(self):
        torch.manual_seed(41)
        model = TransformerBackbone(action_dim=10, obs_dim=53, horizon=10,
            n_obs_steps=2, n_layer=1, n_emb=16, n_head=2, p_drop_attn=0.)
        norm = RobomimicNormalizer(53, 10, 'abs', normalization='minmax')
        norm.obs_scale.fill_(2.3); norm.obs_offset.fill_(.7)
        norm.action_scale.fill_(1); norm.action_offset.zero_()
        policy = self.Policy(model, norm, validate_values=False)
        return PolicyPipeline(policy, self.Objective(validate_values=False))

    def test_exact_eight_frames_each_side_and_intersecting_bands(self):
        w = self.weights(np.repeat([0,1,2],12),8)
        np.testing.assert_allclose(w[3], [1,0,0,0,0,0])
        for t in (4,11,12,15):
            np.testing.assert_allclose(w[t], [.5,.5,0,0,0,0])
        for t in (16,19):
            np.testing.assert_allclose(w[t], [1/3,1/3,1/3,0,0,0])
        for t in (20,23,24,31):
            np.testing.assert_allclose(w[t], [0,.5,.5,0,0,0])
        np.testing.assert_allclose(w[32], [0,0,1,0,0,0])
        np.testing.assert_allclose(w.sum(1),1)

    def test_reversal_deduplication_skips_and_episode_edges(self):
        np.testing.assert_allclose(self.weights(np.array([0,1,0]),8),
            [[.5,.5,0,0,0,0]]*3)
        np.testing.assert_allclose(self.weights(np.array([0,2]),8),
            [[.5,0,.5,0,0,0]]*2)
        np.testing.assert_allclose(self.weights(np.array([5,5]),8),
            [[0,0,0,0,0,1]]*2)
        np.testing.assert_allclose(self.weights(np.array([0,1]),0),
            [[1,0,0,0,0,0],[0,1,0,0,0,0]])
        with self.assertRaises(ValueError): self.weights(np.array([0,6]),8)
        with self.assertRaises(ValueError): self.weights(np.array([0,1]),-1)

    def test_sum_of_losses_not_loss_of_average_and_gradient_weights(self):
        pipeline = self.make_pipeline().train()
        with torch.no_grad():
            for i, expert in enumerate(pipeline.policy.experts):
                for p in expert.parameters(): p.zero_()
                expert.head.bias.fill_([-1,1,2,3,4,5][i])
        batch = dict(obs=states()[[0,2]],action=torch.zeros(2,10,10),
            expert_weights=torch.tensor([[.5,.5,0,0,0,0],[0,0,1,0,0,0]]))
        loss = pipeline.compute_loss(batch)
        torch.testing.assert_close(loss,torch.tensor(2.5))
        loss.backward()
        for i, expected in enumerate([-.05,.05,.2,0,0,0]):
            torch.testing.assert_close(pipeline.policy.experts[i].head.bias.grad,
                                      torch.full((10,),float(expected)))

    def test_inference_and_validation_ignore_training_metadata(self):
        pipeline = self.make_pipeline().eval()
        with torch.no_grad():
            for i, expert in enumerate(pipeline.policy.experts): expert.head.bias.fill_(i)
        batch = dict(obs=states(),action=torch.randn(6,10,10),
                     expert_weights=torch.full((6,6),float('nan')))
        with torch.no_grad():
            ordinary = pipeline.predict_action({'obs':batch['obs']})
            with_metadata = pipeline.predict_action(batch)
            torch.testing.assert_close(with_metadata['action_pred'],ordinary['action_pred'])
            target = pipeline.policy.encode_target(batch).sample
            pred = pipeline.policy.normalizer.normalize_action(ordinary['action_pred'])
            torch.testing.assert_close(pipeline.compute_loss(batch),(pred-target).square().mean())
            torch.testing.assert_close(ordinary['action'],ordinary['action_pred'][:,1:9])

    def test_zero_overlap_matches_original_loss_and_fullgraph_gradients(self):
        pipeline = self.make_pipeline().train()
        batch = dict(obs=states(),action=torch.randn(6,10,10),expert_weights=torch.eye(6))
        cond = pipeline.policy.encode_condition(batch)
        target = pipeline.policy.encode_target(batch).sample
        original = (pipeline.policy(torch.zeros_like(target),torch.zeros(6),cond)-target).square().mean()
        torch.testing.assert_close(pipeline.compute_loss(batch),original)
        batch['expert_weights'] = torch.tensor([[.5,.5,0,0,0,0]]*6)
        eager = pipeline.compute_loss(batch)
        eager.backward()
        grads = [p.grad.clone() for p in pipeline.parameters() if p.requires_grad]
        pipeline.zero_grad(set_to_none=True)
        compiled = torch.compile(pipeline.compute_loss,backend='eager',fullgraph=True)
        result = compiled(batch); result.backward()
        torch.testing.assert_close(result,eager)
        for actual,expected in zip((p.grad for p in pipeline.parameters() if p.requires_grad),grads):
            torch.testing.assert_close(actual,expected)
        with self.assertRaises(ValueError):
            pipeline.compute_loss(dict(obs=states(),action=torch.zeros(6,10,10)))

    @unittest.skipUnless(os.environ.get('TOOLHANG_DATASET'), 'Set TOOLHANG_DATASET for the real-data parity check')
    def test_native_windows_split_and_normalizer_unchanged(self):
        from experiments.toolhang_overlap.dataset import OverlapStateDataset
        kwargs = dict(path=os.environ['TOOLHANG_DATASET'],
            action_space='abs',source_action_space='abs',horizon=10,n_obs_steps=2,
            n_action_steps=8,val_ratio=.02,seed=42,normalization='minmax')
        native = RobomimicStateDataset(**kwargs)
        overlap = OverlapStateDataset(**kwargs,overlap_steps=8)
        self.assertEqual(len(native),len(overlap))
        self.assertEqual(native.demo_names,overlap.demo_names)
        self.assertEqual(native.get_validation_dataset().demo_names,overlap.get_validation_dataset().demo_names)
        self.assertEqual(set(overlap.demo_names)&set(overlap.get_validation_dataset().demo_names),set())
        for key,value in native.normalizer.state_dict().items():
            other=overlap.normalizer.state_dict()[key]
            if torch.is_tensor(value): torch.testing.assert_close(value,other,rtol=0,atol=0)
            else: self.assertEqual(value,other)
        for left,right in [(native,overlap),(native.get_validation_dataset(),overlap.get_validation_dataset())]:
            for i in np.linspace(0,len(left)-1,40,dtype=int):
                a,b=left[i],right[i]
                for key in a: torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
                ep,start=right._windows[i];obs=right._episodes[ep][0]
                from experiments.toolhang_multistage.policy import stage_ids
                phase=stage_ids(torch.from_numpy(obs[:,None])).numpy()
                t=max(0,min(start+1,len(obs)-1))
                # Independent reference: incident stages of boundaries within the band.
                allowed={int(phase[t])}
                for boundary in np.where(phase[1:]!=phase[:-1])[0]+1:
                    if boundary-8<=t<boundary+8: allowed.update([int(phase[boundary-1]),int(phase[boundary])])
                expected=torch.zeros(6)
                for stage in allowed: expected[stage]=1/len(allowed)
                torch.testing.assert_close(b['expert_weights'],expected)



if __name__=='__main__': unittest.main()
