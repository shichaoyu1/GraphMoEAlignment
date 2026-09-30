import copy
import unittest

import torch

from glioma.data.idea4_5 import make_availability_splits, partition_training
from glioma.models.idea4_5 import Idea45AvailabilityModel
from glioma.training.idea4_5_audit import audit_pair, brier_loss, masked_view


class AvailabilityAuditTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = Idea45AvailabilityModel(
            token_dim=4, num_modalities=3, num_classes=2,
            shared_dim=8, spd_dim=3, local_topology="uniform", moment_mode="mean_augmented",
            readout_mode="nodewise",
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        splits = make_availability_splits(num_samples=36, token_count=5, token_dim=4, seed=3)
        self.update = torch.utils.data.default_collate([splits["train"][i] for i in range(4)])
        self.measure = torch.utils.data.default_collate([splits["train"][i] for i in range(4, 8)])

    def test_gate_keeps_forward_and_changes_backward(self):
        inputs = self.update
        self.model.eval()
        normal = self.model(inputs["tokens"], inputs["modality_mask"], inputs["token_mask"])["logits"]
        blocked = self.model(
            inputs["tokens"], inputs["modality_mask"], inputs["token_mask"],
            gradient_edge=(0, 1, 0.0),
        )["logits"]
        torch.testing.assert_close(normal, blocked, rtol=0, atol=1e-7)
        self.model.zero_grad(set_to_none=True)
        brier_loss(normal, inputs["label"]).backward()
        normal_gradient = self.model.fusion.token_adapters[1].weight.grad.clone()
        self.model.zero_grad(set_to_none=True)
        brier_loss(blocked, inputs["label"]).backward()
        blocked_gradient = self.model.fusion.token_adapters[1].weight.grad
        self.assertGreater(float((normal_gradient - blocked_gradient).abs().max()), 1e-8)

    def test_nodewise_readout_makes_uniform_communication_functional(self):
        inputs = self.update
        self.model.eval()
        with torch.no_grad():
            uniform = self.model(inputs["tokens"], inputs["modality_mask"], inputs["token_mask"])["logits"]
            identity = self.model(
                inputs["tokens"], inputs["modality_mask"], inputs["token_mask"],
                topology_override="identity",
            )["logits"]
        self.assertGreater(float((uniform - identity).abs().max()), 1e-6)

    def test_v3_conditions_keep_objects_labels_and_splits_paired(self):
        exact = make_availability_splits(
            num_samples=60, token_count=5, token_dim=4, extra_information="exact_copy",
            data_seed=17, split_seed=29,
        )
        novel = make_availability_splits(
            num_samples=60, token_count=5, token_dim=4, extra_information="novel_signal",
            data_seed=17, split_seed=29,
        )
        for split in ("train", "val", "test"):
            self.assertEqual(exact[split].sample_ids.tolist(), novel[split].sample_ids.tolist())
            torch.testing.assert_close(exact[split].labels, novel[split].labels)
        torch.testing.assert_close(exact["train"].tokens[:, :, 0], exact["train"].tokens[:, :, 2])

    def test_roles_are_group_disjoint(self):
        splits = make_availability_splits(num_samples=80, token_count=5, token_dim=4, seed=13)
        _, indices = partition_training(splits["train"], split_seed=7, synthetic=True)
        role_groups = {
            role: set(splits["train"].group_ids[values]) for role, values in indices.items()
        }
        self.assertFalse(role_groups["fit"] & role_groups["select"])
        self.assertFalse(role_groups["fit"] & role_groups["measure"])
        self.assertFalse(role_groups["select"] & role_groups["measure"])

    def test_fixed_public_edge_weight_is_availability_invariant(self):
        model = Idea45AvailabilityModel(
            token_dim=4, num_modalities=3, num_classes=2, shared_dim=8, spd_dim=3,
            local_topology="uniform", edge_normalization="fixed", readout_mode="nodewise",
        ).eval()
        rich = masked_view(self.update, (0, 1, 2), torch.device("cpu"))
        sparse = masked_view(self.update, (0, 1), torch.device("cpu"))
        with torch.no_grad():
            rich_edge = model(rich["tokens"], rich["modality_mask"], rich["token_mask"])[
                "fusion"
            ]["local_adjacency"][..., 0, 1]
            sparse_edge = model(sparse["tokens"], sparse["modality_mask"], sparse["token_mask"])[
                "fusion"
            ]["local_adjacency"][..., 0, 1]
        torch.testing.assert_close(rich_edge, sparse_edge)

    def test_four_branches_share_state_and_report_interaction(self):
        self.model.train()
        output = self.model(
            self.update["tokens"], self.update["modality_mask"], self.update["token_mask"]
        )
        brier_loss(output["logits"], self.update["label"]).backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        original = copy.deepcopy(self.model.state_dict())
        optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        audit = audit_pair(
            self.model, self.optimizer, self.update, self.measure,
            sparse=(0, 1), rich=(0, 1, 2), edge=(0, 1),
        )
        self.assertTrue(self.model.training)
        for key, tensor in original.items():
            torch.testing.assert_close(tensor, self.model.state_dict()[key])
        current_optimizer = self.optimizer.state_dict()
        self.assertEqual(optimizer_state["param_groups"], current_optimizer["param_groups"])
        for index, state in optimizer_state["state"].items():
            for key, value in state.items():
                torch.testing.assert_close(value, current_optimizer["state"][index][key])
        risks = audit["branches"]
        expected = (
            risks["rich_on"]["risk"]["sparse"] - risks["rich_off"]["risk"]["sparse"]
            - risks["sparse_on"]["risk"]["sparse"] + risks["sparse_off"]["risk"]["sparse"]
        )
        self.assertAlmostEqual(audit["J"], expected)
        self.assertAlmostEqual(
            audit["delta_abs_rich_to_sparse"],
            risks["rich_on"]["risk"]["sparse"] - audit["baseline"]["sparse"],
        )

    def test_mechanism_audit_has_no_clipping_or_optimizer_history(self):
        audit = audit_pair(
            self.model, self.optimizer, self.update, self.measure,
            sparse=(0, 1), rich=(0, 1, 2), edge=(0, 1),
            audit_optimizer="mechanism_sgd", mechanism_lr=1e-3, clip_norm=None,
        )
        self.assertNotIn("optimizer_background", audit["controls"])
        for branch in audit["branches"].values():
            self.assertEqual(branch["clip_scale"], 1.0)
            self.assertIn("all", branch["gradient_norms"])
            self.assertIn("all", branch["displacement_norms"])


if __name__ == "__main__":
    unittest.main()
