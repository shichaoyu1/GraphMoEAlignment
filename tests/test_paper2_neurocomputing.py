import unittest
from argparse import Namespace
import json
import os
import tempfile

import numpy as np
import torch

from glioma.config.paper_profiles import apply_paper2_experiment_profile
from glioma.cli.audit_paper2_neurocomputing import main as audit_main
from glioma.data.utsw_dataset import patient_cross_validation_split
from glioma.models.glioma_topomoe_net import GliomaTopoMoENet
from glioma.modules.topo_moe import TopoMoE
from glioma.semantic.losses import direct_to_routed_distillation_loss
from glioma.semantic.topology import controlled_topology_prior
from glioma.semantic.paper2_statistics import paired_bootstrap, patient_metrics


class Paper2NeurocomputingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.prior = torch.tensor(
            [
                [0.8, 0.2, 0.0, 0.0],
                [0.2, 0.8, 0.0, 0.0],
                [0.0, 0.0, 0.7, 0.3],
                [0.0, 0.0, 0.3, 0.7],
            ],
            dtype=torch.float32,
        )
        self.family_ids = [0, 0, 1, 1]

    def test_profiles_resolve_to_explicit_controlled_models(self):
        base = dict(
            experiment_profile="prior_guided_router",
            paper_config="none",
            topomoe_version="v1",
            node_mode="modalities",
            graph_type="learnable",
            no_private=False,
            no_diffusion=False,
            moe_module="none",
            topo_mode="prior_plus_learned",
            disable_topology_refinement=False,
            epochs=30,
            direct_stage_epochs=None,
            router_stage_epochs=None,
            joint_stage_epochs=None,
            stage_patience=None,
        )
        args = apply_paper2_experiment_profile(Namespace(**base))
        self.assertTrue(args.routing_enabled)
        self.assertFalse(args.use_residual_expert)
        self.assertEqual(args.topo_mode, "prior_only")
        self.assertEqual(args.context_mode, "topology")
        base["experiment_profile"] = "direct_only"
        direct = apply_paper2_experiment_profile(Namespace(**base))
        self.assertFalse(direct.routing_enabled)
        self.assertEqual(direct.router_stage_epochs, 0)

    def test_no_residual_router_has_complete_anchor_probability_mass(self):
        module = TopoMoE(
            shared_dim=8,
            topo_prior=self.prior,
            anchor_family_ids=self.family_ids,
            num_families=2,
            version="v2",
            use_residual_expert=False,
        )
        output = module(torch.randn(2, 3, 8), torch.randn(4, 8))
        self.assertIsNone(module.residual_index)
        self.assertIsNone(module.residual_expert)
        self.assertTrue(
            torch.allclose(
                torch.exp(output["routed_log_probs"]).sum(dim=-1),
                torch.ones(2, 3),
                atol=1e-6,
            )
        )
        self.assertEqual(float(output["diagnostics"]["residual_usage"].detach()), 0.0)

    def test_unstructured_context_ignores_topology_override(self):
        module = TopoMoE(
            shared_dim=8,
            topo_prior=self.prior,
            anchor_family_ids=self.family_ids,
            num_families=2,
            version="v2",
            use_residual_expert=False,
            context_mode="unstructured",
            refine_prototypes=False,
        )
        shared = torch.randn(2, 3, 8)
        prototypes = torch.randn(4, 8)
        first = module(shared, prototypes, topology_override=torch.eye(4))["routed_scores"]
        second = module(shared, prototypes, topology_override=torch.full((4, 4), 0.25))["routed_scores"]
        self.assertTrue(torch.allclose(first, second, atol=1e-6))

    def test_direct_only_model_reuses_encoder_without_router(self):
        model = GliomaTopoMoENet(
            z_slices=3,
            feat_dim=16,
            shared_dim=8,
            routing_enabled=False,
        )
        output = model(
            torch.randn(1, 4, 3, 32, 32),
            region_masks=torch.ones(1, 3, 3, 32, 32),
            return_extras=True,
        )
        self.assertFalse(model.use_topo_moe)
        self.assertIsNone(output["extras"]["routed_scores"])
        self.assertEqual(tuple(output["extras"]["shared"].shape), (1, 3, 8))

    def test_distillation_and_control_priors_are_deterministic(self):
        queries = torch.randn(6, 8, requires_grad=True)
        prototypes = torch.randn(4, 8)
        routed = torch.randn(6, 4, requires_grad=True)
        loss = direct_to_routed_distillation_loss(queries, prototypes, routed)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(routed.grad)
        first = controlled_topology_prior(5, "random", seed=13)
        second = controlled_topology_prior(5, "random", seed=13)
        self.assertTrue(torch.allclose(first, second))
        self.assertTrue(torch.allclose(first.sum(dim=-1), torch.ones(5), atol=1e-6))

    def test_patient_cv_is_deterministic_and_leak_free(self):
        cases = [
            {
                "subject_id": f"p{idx:02d}",
                "label": idx % 3,
                "metadata": {
                    "IDH": "mutated",
                    "MGMT": "methylated" if idx % 2 else "",
                    "1p19Q CODEL": "co-deleted" if idx % 4 else "",
                },
            }
            for idx in range(30)
        ]
        first = patient_cross_validation_split(cases, num_folds=5, fold=2, seed=17)
        second = patient_cross_validation_split(cases, num_folds=5, fold=2, seed=17)
        self.assertEqual(
            {name: [case["subject_id"] for case in values] for name, values in first.items()},
            {name: [case["subject_id"] for case in values] for name, values in second.items()},
        )
        sets = [{case["subject_id"] for case in first[name]} for name in ("train", "val", "test")]
        self.assertFalse(sets[0] & sets[1])
        self.assertFalse(sets[0] & sets[2])
        self.assertFalse(sets[1] & sets[2])
        self.assertEqual(sum(map(len, sets)), len(cases))

    def test_patient_metrics_cluster_queries_before_bootstrap(self):
        payload = {
            "subject_ids": ["a", "a", "b", "b"],
            "query_targets": [[0], [1], [0], [1]],
            "baseline": [[3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1]],
            "candidate": [[3, 2, 1], [1, 3, 2], [3, 2, 1], [1, 3, 2]],
        }
        baseline = patient_metrics(payload, "baseline")
        candidate = patient_metrics(payload, "candidate")
        result = paired_bootstrap(baseline, candidate, n_bootstrap=200, seed=3)
        self.assertEqual(result["n_patients"], 2)
        self.assertGreater(result["metrics"]["map"]["mean_delta"], 0)
        self.assertTrue(np.isfinite(result["metrics"]["mrr"]["ci95"]).all())

    def test_audit_cli_writes_reproducible_paper_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            direct_root = os.path.join(directory, "direct")
            prior_root = os.path.join(directory, "prior")
            out_dir = os.path.join(directory, "audit")
            os.makedirs(direct_root)
            os.makedirs(prior_root)
            payload = {
                "subject_ids": ["a", "a", "b", "b"],
                "query_targets": [[0], [1], [0], [1]],
                "direct_scores": [[3, 1, 0], [1, 3, 0], [3, 1, 0], [1, 3, 0]],
                "routed_scores": [[3, 1, 0], [1, 3, 0], [3, 1, 0], [1, 3, 0]],
                "intervention_scores": {
                    "uniform_routing": [[1, 3, 0], [3, 1, 0], [1, 3, 0], [3, 1, 0]]
                },
            }
            for root in (direct_root, prior_root):
                with open(os.path.join(root, "patient_level_records.json"), "w", encoding="utf-8") as file:
                    json.dump(payload, file)
            result = audit_main(
                [
                    "--method",
                    f"direct_only=direct_scores|{direct_root}",
                    "--method",
                    f"prior_guided_router=routed_scores|{prior_root}",
                    "--out_dir",
                    out_dir,
                    "--bootstrap",
                    "50",
                ]
            )
            self.assertEqual(result["analysis_unit"], "patient")
            for filename in (
                "paper2_statistical_audit.json",
                "table2_patient_metrics.csv",
                "SUBMISSION_GATE_REPORT.md",
                "figure2_model_performance.png",
                "figure4_routing_interventions.png",
            ):
                self.assertTrue(os.path.isfile(os.path.join(out_dir, filename)), filename)


if __name__ == "__main__":
    unittest.main()
