import unittest
from argparse import Namespace

import torch

from glioma.config.paper_profiles import apply_paper_profile
from glioma.eval.semantic_alignment_eval import (
    _intervention_outputs,
    _uses_topomoe_artifact_protocol,
    _wrong_family_routing,
    intervention_metrics,
)
from glioma.models.glioma_topomoe_net import GliomaTopoMoENet
from glioma.modules.topo_moe import TopoMoE
from glioma.semantic.losses import topomoe_family_balanced_losses


class TopoMoEV2Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.prior = torch.tensor(
            [
                [0.7, 0.3, 0.0, 0.0],
                [0.3, 0.7, 0.0, 0.0],
                [0.0, 0.0, 0.6, 0.4],
                [0.0, 0.0, 0.4, 0.6],
            ],
            dtype=torch.float32,
        )
        self.family_ids = [0, 0, 1, 1]

    def build_module(self, **kwargs):
        return TopoMoE(
            shared_dim=8,
            topo_prior=self.prior,
            anchor_family_ids=self.family_ids,
            num_families=3,
            version="v2",
            **kwargs,
        )

    def test_topology_modes_are_row_normalized(self):
        for mode in ("prior_only", "learned_only", "prior_plus_learned"):
            module = self.build_module(topo_mode=mode)
            effective = module.effective_topology()
            self.assertTrue(torch.allclose(effective.sum(dim=-1), torch.ones(4), atol=1e-6))
            if mode == "prior_only":
                self.assertTrue(torch.allclose(effective, self.prior))

    def test_prior_zero_edge_has_gradient_and_opens(self):
        module = self.build_module(topo_mode="prior_plus_learned")
        optimizer = torch.optim.SGD(module.parameters(), lr=1.0)
        initial = float(module.effective_topology()[0, 2].detach())
        optimizer.zero_grad()
        loss = -module.effective_topology()[0, 2]
        loss.backward()
        self.assertGreater(abs(float(module.A_raw.grad[0, 2])), 0.0)
        optimizer.step()
        self.assertGreater(float(module.effective_topology()[0, 2].detach()), initial)

    def test_beta_initialization_and_probability_mass(self):
        module = self.build_module(topo_beta_init=0.1)
        output = module(torch.randn(2, 3, 8), torch.randn(4, 8))
        self.assertNotIn("routed_logits", output)
        self.assertAlmostEqual(float(torch.sigmoid(module.topo_beta_logit).detach()), 0.1, places=5)
        for family in range(module.num_anchor_families):
            anchor_ids = torch.where(module.family_ids == family)[0]
            conditional_mass = torch.exp(output["family_log_probs"][..., anchor_ids]).sum(dim=-1)
            self.assertTrue(torch.allclose(conditional_mass, torch.ones_like(conditional_mass), atol=1e-6))
        anchor_mass = torch.exp(output["routed_log_probs"]).sum(dim=-1)
        residual_mass = output["routing_weights"][..., module.residual_index]
        self.assertTrue(torch.allclose(anchor_mass + residual_mass, torch.ones_like(anchor_mass), atol=1e-5))

    def test_family_balanced_targets_ignore_positive_count(self):
        module = self.build_module()
        output = module(torch.randn(1, 1, 8), torch.randn(4, 8))
        family_loss, within_loss = topomoe_family_balanced_losses(
            output["routing_weights"],
            output["family_log_probs"],
            [[0, 1, 2]],
            self.family_ids,
            module.residual_index,
        )
        expected = -0.5 * (
            torch.log(output["routing_weights"][0, 0, 0])
            + torch.log(output["routing_weights"][0, 0, 1])
        )
        self.assertTrue(torch.allclose(family_loss, expected, atol=1e-6))
        self.assertTrue(torch.isfinite(within_loss))

    def test_v1_compatibility_output(self):
        module = TopoMoE(
            shared_dim=8,
            topo_prior=self.prior,
            anchor_family_ids=self.family_ids,
            num_families=3,
            version="v1",
            route_mixture="product",
        )
        output = module(torch.randn(1, 3, 8), torch.randn(4, 8))
        self.assertIsNotNone(output["routed_logits"])
        self.assertIsNone(output["routed_log_probs"])

    def test_js_specialization_margin(self):
        module = self.build_module(specialize_margin=0.05)
        identical = torch.tensor([[[0.45, 0.45, 0.1], [0.45, 0.45, 0.1]]])
        separated = torch.tensor([[[0.89, 0.01, 0.1], [0.01, 0.89, 0.1]]])
        self.assertGreater(float(module._specialize_v2(identical)), float(module._specialize_v2(separated)))

    def test_remove_expert_and_wrong_family_routing(self):
        module = self.build_module()
        output = module(
            torch.randn(1, 3, 8),
            torch.randn(4, 8),
            disabled_family_ids=[0],
        )
        self.assertTrue(torch.all(output["routing_weights"][..., 0] < 1e-7))
        self.assertTrue(torch.allclose(output["routing_weights"].sum(-1), torch.ones(1, 3), atol=1e-6))
        forced = _wrong_family_routing(
            [[0, 2], [0], [2]],
            self.family_ids,
            2,
            2,
            (1, 3, 3),
            torch.device("cpu"),
        )
        self.assertEqual(int(forced[0, 0].argmax()), 2)
        self.assertEqual(int(forced[0, 1].argmax()), 1)
        self.assertEqual(int(forced[0, 2].argmax()), 0)

    def test_paper2_model_has_no_legacy_modules_by_default(self):
        model = GliomaTopoMoENet(
            z_slices=3,
            feat_dim=16,
            shared_dim=8,
            topo_prior=self.prior,
            anchor_family_ids=self.family_ids,
            num_families=3,
        )
        self.assertIsNone(model.graph_builder)
        self.assertFalse(hasattr(model, "diffusion"))
        self.assertFalse(hasattr(model, "private_heads"))
        output = model(
            torch.randn(1, 4, 3, 32, 32),
            region_masks=torch.ones(1, 3, 3, 32, 32),
            return_extras=True,
            anchor_prototypes=torch.randn(4, 8),
        )
        for loss in output["losses"].values():
            self.assertTrue(torch.isfinite(loss))
        objective = sum(output["losses"].values()) - output["extras"]["routed_log_probs"][..., 0].mean()
        objective.backward()
        self.assertIsNotNone(model.topo_moe.A_raw.grad)
        self.assertTrue(torch.isfinite(model.topo_moe.A_raw.grad).all())
        self.assertGreater(float(model.topo_moe.A_raw.grad.abs().sum()), 0.0)

    def test_paper2_profile_preserves_v1_and_exposes_graph_ablation(self):
        args = Namespace(
            paper_config="paper2",
            variant="full",
            graph_type="learnable",
            no_private=False,
            no_diffusion=False,
            moe_module="none",
            topomoe_version="v1",
        )
        result = apply_paper_profile(args)
        self.assertEqual(result.topomoe_version, "v1")
        self.assertEqual(result.graph_type, "learnable")
        self.assertFalse(result.no_private)
        self.assertTrue(result.no_diffusion)
        args.topomoe_version = "v2"
        args.variant = "graph_shared_only"
        self.assertEqual(apply_paper_profile(args).graph_type, "learnable")

    def test_paper2_v1_uses_new_artifact_protocol(self):
        model = type("Paper2V1", (), {"use_topo_moe": True, "topomoe_version": "v1"})()
        self.assertTrue(_uses_topomoe_artifact_protocol(model, Namespace(paper_config="paper2")))
        self.assertFalse(_uses_topomoe_artifact_protocol(model, Namespace(paper_config="paper1")))
        model.topomoe_version = "v2"
        self.assertTrue(_uses_topomoe_artifact_protocol(model, Namespace(paper_config="none")))

    def test_intervention_schema_contains_subgroup_deltas(self):
        records = {
            "routed_scores": torch.tensor(
                [[5.0, 0.0, 0.0, 0.0], [0.0, 0.0, 5.0, 0.0], [5.0, 0.0, 0.0, 0.0]]
            ).numpy(),
            "intervention_scores": {
                "force_wrong_family": torch.tensor(
                    [[0.0, 0.0, 5.0, 0.0], [5.0, 0.0, 0.0, 0.0], [0.0, 0.0, 5.0, 0.0]]
                ).numpy()
            },
            "query_targets": [[0], [2], [0]],
            "query_subject_ids": ["a", "b", "c"],
            "family_ids": [0, 0, 1, 1],
            "family_names": ["pathology", "molecular", "residual"],
            "query_records": [
                {"node_name": "Core"},
                {"node_name": "Edema"},
                {"node_name": "Core"},
            ],
        }
        scenario = intervention_metrics(records)["scenarios"]["force_wrong_family"]
        self.assertIn("delta_map", scenario["by_target_family"]["pathology"])
        self.assertIn("delta_mrr", scenario["by_node"]["Core"])

    def test_interventions_are_deterministic_and_complete(self):
        model = GliomaTopoMoENet(
            z_slices=3,
            feat_dim=16,
            shared_dim=8,
            topo_prior=self.prior,
            anchor_family_ids=self.family_ids,
            num_families=3,
        )
        shared = torch.randn(1, 3, 8)
        prototypes = torch.randn(4, 8)
        targets = [[0], [2], [0, 2]]
        first = _intervention_outputs(model, shared, prototypes, targets, 42, ["Core", "Edema", "Enhancing"])
        second = _intervention_outputs(model, shared, prototypes, targets, 42, ["Core", "Edema", "Enhancing"])
        expected = {
            "shuffle_topology",
            "force_wrong_family",
            "anchor_family_permutation",
            "mask_node_core",
            "remove_residual_branch",
        }
        self.assertTrue(expected.issubset(first))
        for name in first:
            self.assertTrue((first[name] == second[name]).all())


if __name__ == "__main__":
    unittest.main()
