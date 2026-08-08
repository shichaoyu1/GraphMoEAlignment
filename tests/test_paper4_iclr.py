import json
import os
import tempfile
import unittest

import numpy as np
import torch

from glioma.cli.aggregate_paper4_diagnostics import aggregate
from glioma.cli.train_paper4_benchmark import main as train_benchmark
from glioma.cli.train_semantic_alignment import _has_routed_training, build_parser, resolve_seeds
from glioma.data.paper4_benchmarks import load_npz_splits, make_synthetic_splits
from glioma.eval.paper4_diagnostics import (
    coadaptation_diagnostics,
    diagnostic_gains,
    edge_recovery,
    edge_stability,
)
from glioma.modules.hierarchical_spd_fusion import (
    HierarchicalSPDGraphFusion,
    SPDMomentGraphFusion,
)


class Paper4SeedAndTopologyTests(unittest.TestCase):
    def test_paper4_published_baseline_does_not_enable_topomoe_routing(self):
        paper4 = build_parser().parse_args(
            [
                "--data_root", "unused",
                "--paper_config", "paper4",
                "--paper4_fusion_backend", "published_baseline",
                "--paper4_baseline", "hemis",
            ]
        )
        self.assertFalse(_has_routed_training(paper4))

        topomoe = build_parser().parse_args(
            ["--data_root", "unused", "--moe_module", "topo_moe", "--topomoe_version", "v2"]
        )
        self.assertTrue(_has_routed_training(topomoe))
        topomoe.routing_enabled = False
        self.assertFalse(_has_routed_training(topomoe))

    def test_legacy_and_explicit_seeds_resolve_independently(self):
        legacy = resolve_seeds(build_parser().parse_args(["--data_root", "unused", "--seed", "7"]))
        self.assertEqual((legacy.split_seed, legacy.model_seed), (7, 7))
        explicit = resolve_seeds(
            build_parser().parse_args(
                ["--data_root", "unused", "--seed", "7", "--split_seed", "11", "--model_seed", "13"]
            )
        )
        self.assertEqual((explicit.seed, explicit.split_seed, explicit.model_seed), (7, 11, 13))

    def test_uniform_local_pooling_equals_identity(self):
        torch.manual_seed(3)
        module = SPDMomentGraphFusion(
            token_dim=6,
            shared_dim=8,
            num_modalities=4,
            spd_dim=4,
            local_topology="identity",
        ).eval()
        tokens = torch.randn(2, 1, 4, 7, 6)
        identity = module(tokens)
        uniform = module(tokens, local_topology_override="uniform")
        torch.testing.assert_close(
            identity["group_representation"], uniform["group_representation"], atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(identity["fused_nodes"], uniform["fused_nodes"], atol=1e-5, rtol=1e-5)

    def test_token_mask_excludes_padded_values(self):
        torch.manual_seed(5)
        module = SPDMomentGraphFusion(
            token_dim=5,
            shared_dim=8,
            num_modalities=2,
            spd_dim=3,
            local_topology="identity",
        ).eval()
        tokens = torch.randn(2, 1, 2, 6, 5)
        token_mask = torch.ones(2, 1, 2, 6, dtype=torch.bool)
        token_mask[..., -2:] = False
        corrupted = tokens.clone()
        corrupted[..., -2:, :] = 1e4
        clean = module(tokens, token_mask=token_mask)
        padded = module(corrupted, token_mask=token_mask)
        torch.testing.assert_close(clean["spd_matrices"], padded["spd_matrices"])

    def test_local_and_upper_topology_are_independent(self):
        torch.manual_seed(7)
        module = HierarchicalSPDGraphFusion(
            token_dim=6,
            shared_dim=8,
            family_ids=[0, 0, 1, 1],
            family_names=["pathology", "molecular", "residual"],
            spd_dim=4,
            local_topology="identity",
            upper_topology="learned",
        ).eval()
        result = module(torch.randn(2, 3, 4, 7, 6), torch.randn(4, 8))
        self.assertEqual(result["local_topology"], "identity")
        self.assertEqual(result["upper_topology"], "learned")
        local_offdiag = result["local_adjacency"] * (1 - torch.eye(4))
        self.assertEqual(float(local_offdiag.abs().sum()), 0.0)
        self.assertGreater(
            float((result["upper_adjacency"] * (1 - torch.eye(6))).detach().sum()), 0.0
        )

    def test_constructor_intervention_is_eval_only(self):
        module = HierarchicalSPDGraphFusion(
            token_dim=6,
            shared_dim=8,
            family_ids=[0, 1],
            family_names=["pathology", "molecular", "residual"],
            spd_dim=4,
            graph_intervention="uniform",
        )
        tokens = torch.randn(1, 3, 4, 6, 6)
        prototypes = torch.randn(2, 8)
        training = module(tokens, prototypes)
        module.eval()
        evaluation = module(tokens, prototypes)
        self.assertEqual(training["graph_intervention"], "none")
        self.assertEqual(evaluation["graph_intervention"], "uniform")


class Paper4BenchmarkAndDiagnosticTests(unittest.TestCase):
    def test_synthetic_and_npz_adapters(self):
        splits, graph = make_synthetic_splits(
            num_samples=40, num_modalities=4, token_count=6, token_dim=5, seed=9
        )
        self.assertEqual(graph.shape, (4, 4))
        self.assertEqual(tuple(splits["train"].tokens.shape[1:]), (1, 4, 6, 5))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "fixture.npz")
            payload = {}
            for split in ("train", "val", "test"):
                count = 8 if split == "train" else 4
                payload[f"{split}_tokens"] = np.random.randn(count, 1, 3, 5, 4).astype(np.float32)
                payload[f"{split}_labels"] = np.random.randn(count).astype(np.float32)
            np.savez(path, **payload)
            loaded = load_npz_splits(path)
            self.assertEqual(tuple(loaded["train"].tokens.shape), (8, 1, 3, 5, 4))

    def test_diagnostic_formulas_and_edge_metrics(self):
        gains = diagnostic_gains(
            {
                "spd_cross_graph": 0.70,
                "spd_uniform_graph": 0.73,
                "spd_identity_graph": 0.69,
                "euclidean_cross_graph": 0.65,
            }
        )
        self.assertAlmostEqual(gains["geometry_gain"], 0.05)
        self.assertAlmostEqual(gains["communication_gain"], 0.04)
        self.assertAlmostEqual(gains["allocation_gain"], -0.03)
        coadaptation = coadaptation_diagnostics(0.70, 0.68, 0.73)
        self.assertAlmostEqual(coadaptation["intervention_cost"], 0.02)
        self.assertAlmostEqual(coadaptation["coadaptation_gap"], 0.05)
        planted = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        recovered = edge_recovery(planted + 0.1, planted)
        self.assertAlmostEqual(recovered["edge_auroc"], 1.0)
        stable = edge_stability([planted, planted * 0.8], top_k=2)
        self.assertEqual(stable["topk_jaccard_mean"], 1.0)

    def test_synthetic_training_smoke(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = train_benchmark(
                [
                    "--dataset", "synthetic",
                    "--out_dir", out_dir,
                    "--num_samples", "48",
                    "--token_count", "5",
                    "--token_dim", "4",
                    "--spd_dim", "3",
                    "--shared_dim", "8",
                    "--epochs", "1",
                    "--batch_size", "8",
                    "--eval_missing_rates", "0",
                    "--eval_noise_stds", "0",
                    "--cpu",
                ]
            )
            self.assertIn("accuracy", result["metrics"])
            for name in (
                "config.json",
                "test_metrics.json",
                "intervention_metrics.json",
                "robustness_metrics.json",
                "adjacency.json",
                "planted_adjacency.json",
            ):
                self.assertTrue(os.path.exists(os.path.join(out_dir, name)), name)

    def test_avmnist_and_mosei_training_fixtures(self):
        rng = np.random.default_rng(12)
        with tempfile.TemporaryDirectory() as root:
            for dataset, modalities in (("avmnist", 2), ("mosei", 3)):
                path = os.path.join(root, f"{dataset}.npz")
                payload = {}
                for split, count in (("train", 16), ("val", 8), ("test", 8)):
                    payload[f"{split}_tokens"] = rng.normal(
                        size=(count, 1, modalities, 5, 4)
                    ).astype(np.float32)
                    if dataset == "mosei":
                        payload[f"{split}_labels"] = rng.normal(size=count).astype(np.float32)
                    else:
                        payload[f"{split}_labels"] = rng.integers(0, 2, size=count, dtype=np.int64)
                np.savez(path, **payload)
                out_dir = os.path.join(root, f"{dataset}_run")
                result = train_benchmark(
                    [
                        "--dataset", dataset,
                        "--data_path", path,
                        "--out_dir", out_dir,
                        "--num_classes", "2",
                        "--spd_dim", "3",
                        "--shared_dim", "8",
                        "--epochs", "1",
                        "--batch_size", "8",
                        "--eval_missing_rates", "0",
                        "--eval_noise_stds", "0",
                        "--cpu",
                    ]
                )
                self.assertIn("accuracy", result["metrics"])
                if dataset == "mosei":
                    self.assertIn("binary_f1", result["metrics"])

    def test_factorial_aggregate_and_split_guard(self):
        with tempfile.TemporaryDirectory() as root:
            variants = {
                "spd_cross_graph": 0.70,
                "spd_uniform_graph": 0.73,
                "spd_identity_graph": 0.69,
                "euclidean_cross_graph": 0.65,
            }
            for variant, score in variants.items():
                directory = os.path.join(root, variant, "paper4", "split_42_model_101")
                os.makedirs(directory)
                with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as file:
                    json.dump({"split_seed": 42, "model_seed": 101}, file)
                with open(os.path.join(directory, "splits.json"), "w", encoding="utf-8") as file:
                    json.dump({"train": ["a"], "val": ["b"], "test": ["c"]}, file)
                with open(os.path.join(directory, "test_metrics.json"), "w", encoding="utf-8") as file:
                    json.dump({"map": score}, file)
                with open(os.path.join(directory, "adjacency.json"), "w", encoding="utf-8") as file:
                    json.dump(np.eye(4).tolist(), file)
                if variant == "spd_cross_graph":
                    with open(os.path.join(directory, "intervention_metrics.json"), "w", encoding="utf-8") as file:
                        json.dump({"uniform": {"map": 0.68}}, file)
            payload = aggregate(
                root,
                preferred_metric="map",
                expected_variants=list(variants),
                expected_split_seeds=[42],
                expected_model_seeds=[101],
            )
            self.assertEqual(payload["status"], "factorial_complete")
            self.assertAlmostEqual(payload["paired_diagnostics"]["coadaptation_gap"]["mean"], 0.05)


if __name__ == "__main__":
    unittest.main()
