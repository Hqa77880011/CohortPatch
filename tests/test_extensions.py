import unittest

import numpy as np
import torch

from cohortpatch.activity import make_model
from cohortpatch.attribute import public_features, support_gate
from cohortpatch.baselines import batches, batch_select, exact_rosters
from cohortpatch.profiles import split_blocks
from cohortpatch.trajectory import reconstruct
from cohortpatch.disaggregation import infer_participation


class ExtensionChecks(unittest.TestCase):
    def test_mr_fixed_k_and_variable_union(self):
        rng = np.random.default_rng(17)
        groups = batches(10, 3)
        self.assertEqual(sorted(map(len, groups)), [3, 3, 4])
        choice = batch_select(groups, set(range(10)), [0]*10, 7, 3, rng)
        self.assertEqual(len(choice), 7)
        with self.assertRaises(ValueError):
            batches(10, 3, strict=True)
        legal = list(exact_rosters([set(range(6))], set(range(6)), 3, 3, 3))
        self.assertEqual(len(legal),20)  # all twenty nonempty three-subsets

    def test_cnn_every_parameter_receives_update(self):
        torch.set_num_threads(1); torch.manual_seed(17)
        model = make_model(3, 2, "cnn")
        original = [p.detach().clone() for p in model.parameters()]
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        loss = torch.nn.functional.cross_entropy(model(torch.randn(8, 3, 128)), torch.tensor([0,1]*4))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0 for p in model.parameters()))
        optimizer.step()
        self.assertTrue(all(not torch.equal(a, b) for a, b in zip(original, model.parameters())))

    def test_profile_time_gap_and_unsupported_har(self):
        sensor = dict(client=np.zeros(20, int), split=np.zeros(20, int), start=np.arange(20)*100,
                      end=np.arange(20)*100+80, x=np.zeros((20, 3, 128)), y=np.zeros(20, int))
        victim, prior, manifest = split_blocks(sensor)
        self.assertTrue(manifest["sample_disjoint"])
        self.assertGreaterEqual(prior["start"].min()-victim["end"].max(), 80)
        sensor.pop("start")
        self.assertEqual(split_blocks(sensor)[2]["status"], "unsupported_profile_prior")

    def test_dynamic_reconstruction_and_public_feature_gate(self):
        a = np.eye(3)
        true = np.array([[.1, .2], [.3, .1], [.2, -.1]])
        reconstructed, _ = reconstruct(a, true, dynamic=True, steps=10)
        np.testing.assert_allclose(np.einsum("rn,rnd->rd", a, reconstructed), true, atol=1e-8)
        trace = dict(roster=np.ones((2,3)), availability=np.ones((2,3)), updates=np.ones((2,2)),
                     public_models=np.ones((2,2)), layer_sizes=[2], initial_class=[0,0,0])
        before = public_features(trace, 0)
        trace["private_target_update"] = np.ones(1000)*999
        np.testing.assert_array_equal(before, public_features(trace, 0))
        self.assertFalse(support_gate(np.array([0,1]*3), np.arange(6), np.repeat([0,1,2],2))[0])

    def test_lam_nullspace_and_unique_summary_solution(self):
        roster=np.array([[1,0],[0,1],[1,1],[1,0]])
        updates=np.random.default_rng(1).normal(size=(2,6))
        summary=np.ones((1,4))
        estimate,report=infer_participation(roster@updates,summary,summary@roster,2,seconds=10)
        np.testing.assert_array_equal(estimate,roster)
        self.assertTrue(all(row['unique_optimal_participation'] for row in report['clients']))


if __name__ == "__main__":
    unittest.main()
