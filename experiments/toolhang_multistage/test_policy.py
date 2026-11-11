"""Behavior tests: current observation routing and independent expert learning."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from ht_regression.data.robomimic.normalizer import RobomimicNormalizer
from ht_regression.policies.from_scratch.backbones import TransformerBackbone
from ht_regression.pipelines import PolicyPipeline
from ht_regression.objectives.regression.MSE import MSEObjective


def states():
    """Hand-picked native-layout states, expected stages0..5."""
    obs = torch.zeros(6, 2, 53)
    obs[..., 2] = .8
    obs[..., 14:17] = torch.tensor([-.08, .04, 1.03])
    obs[..., 28:31] = torch.tensor([.05, -.22, .84])
    obs[0, -1, 16] = .84
    obs[1, -1, 14:17] = torch.tensor([-.2, -.2, 1.1])
    obs[3:, -1, 42] = 1
    obs[4, -1, 28:31] = torch.tensor([.2, -.3, 1.1])
    obs[5, -1, 28:31] = torch.tensor([-.05, -.09, .95])
    # Earlier frame deliberately belongs to another stage.
    obs[:, 0] = obs[0, -1].clone()
    return obs


class PolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        spec = importlib.util.find_spec('experiments.toolhang_multistage.policy')
        self.assertIsNotNone(spec, 'multi-stage policy implementation is missing')
        from experiments.toolhang_multistage.policy import StageMSEPolicy, stage_ids
        self.StageMSEPolicy, self.stage_ids = StageMSEPolicy, stage_ids

    def make_policy(self):
        torch.manual_seed(41)
        model = TransformerBackbone(action_dim=10, obs_dim=53, horizon=10,
            n_obs_steps=2, n_layer=1, n_emb=16, n_head=2, p_drop_attn=0.)
        normalizer = RobomimicNormalizer(53, 10, 'abs', normalization='minmax')
        normalizer.obs_scale.fill_(2.3)
        normalizer.obs_offset.fill_(.7)
        return self.StageMSEPolicy(model, normalizer, validate_values=False)

    def test_current_observation_and_recovery(self):
        obs = states()
        self.assertEqual(self.stage_ids(obs).tolist(), [0,1,2,3,4,5])
        obs[5, -1, 30] = .84  # Dropped tool must return to pickup.
        self.assertEqual(self.stage_ids(obs)[5].item(), 3)
        obs[5, -1, 43] = 1  # An actually hanging low tool remains in final phase.
        self.assertEqual(self.stage_ids(obs)[5].item(), 5)

    def test_dense_sparse_native_loss_and_chunk_alignment(self):
        p = self.make_policy()
        # Deliberately make expert outputs distinguishable.
        with torch.no_grad():
            for i, model in enumerate(p.experts):
                model.head.bias.fill_(i * .1)
        batch = {'obs': states(), 'action': torch.randn(6,10,10)}
        cond = p.encode_condition(batch)
        x = torch.zeros(6,10,10); t = torch.zeros(6)
        dense = p.train()(x,t,cond)
        sparse = p.eval()(x,t,cond)
        manual = torch.cat([expert(x[i:i+1],t[i:i+1],cond[i:i+1])
                            for i,expert in enumerate(p.experts)])
        torch.testing.assert_close(dense, manual)
        torch.testing.assert_close(sparse, manual)
        pipeline = PolicyPipeline(p, MSEObjective(validate_values=False))
        torch.testing.assert_close(pipeline.compute_loss(batch),
            (manual-p.encode_target(batch).sample).square().mean())
        result = pipeline.eval().predict_action(batch)
        torch.testing.assert_close(result['action'],result['action_pred'][:,1:9])

    def test_only_selected_expert_receives_gradient_and_all_are_optimized(self):
        p = self.make_policy().train()
        params = [x for group in p.get_optimizer_groups(.001) for x in group['params']]
        self.assertEqual(len(params),len({id(x) for x in params}))
        self.assertEqual({id(x) for x in params},{id(x) for x in p.parameters() if x.requires_grad})
        batch = {'obs':states()[2:3], 'action':torch.randn(1,10,10)}
        PolicyPipeline(p,MSEObjective(validate_values=False)).compute_loss(batch).backward()
        for i,expert in enumerate(p.experts):
            norm = sum(float(x.grad.square().sum()) for x in expert.parameters() if x.grad is not None)
            self.assertGreater(norm,0) if i==2 else self.assertEqual(norm,0)
        addresses = [{x.data_ptr() for x in e.parameters()} for e in p.experts]
        for i in range(6):
            for j in range(i):self.assertTrue(addresses[i].isdisjoint(addresses[j]))

    def test_checkpoint_state_and_fullgraph(self):
        p=self.make_policy().train()
        batch={'obs':states(),'action':torch.randn(6,10,10)}
        pipeline=PolicyPipeline(p,MSEObjective(validate_values=False))
        loss=pipeline.compute_loss(batch)
        compiled=torch.compile(pipeline.compute_loss,backend='eager',fullgraph=True)
        torch.testing.assert_close(compiled(batch),loss)
        p.eval(); q=self.make_policy().eval();q.load_state_dict(p.state_dict())
        x=torch.zeros(6,10,10);t=torch.zeros(6)
        torch.testing.assert_close(p(x,t,p.encode_condition(batch)),q(x,t,q.encode_condition(batch)))



if __name__ == '__main__':unittest.main()
