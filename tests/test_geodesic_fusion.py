import json
import os
import tempfile
import unittest
from argparse import Namespace

import numpy as np
import torch

from glioma.cli.aggregate_geodesic_runs import aggregate
from glioma.cli.train_semantic_alignment import _checkpoint_selection, build_parser
from glioma.anchors import semantic_anchors
from glioma.config.paper_profiles import apply_paper_profile
from glioma.models.glioma_geodesic_fusion_net import GliomaGeodesicFusionNet
from glioma.modules.geodesic_fusion import GeodesicModalityGraphFusion
from glioma.semantic.losses import geodesic_path_semantic_loss
from glioma.objectives import SemanticPrototypeBank
from glioma.training.engine import run_epoch
from glioma.eval.semantic_alignment_eval import evaluate_and_save
from glioma.visualization.geodesic_fusion import (
    build_geodesic_payloads,
    save_geodesic_figures,
)


class GeodesicFusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.module = GeodesicModalityGraphFusion(shared_dim=8, path_steps=5)
        self.nodes = torch.randn(2, 3, 4, 8)
        self.prototypes = torch.randn(6, 8)

    def test_six_pairs_endpoints_and_reversal(self):
        self.assertEqual(self.module.num_pairs, 6)
        start = torch.randn(4, 8)
        end = torch.randn(4, 8)
        context = torch.randn(4, 8)
        forward, _ = self.module.pair_path(start, end, context)
        reverse, _ = self.module.pair_path(end, start, context)
        torch.testing.assert_close(forward[:, 0], start, atol=1e-6, rtol=0)
        torch.testing.assert_close(forward[:, -1], end, atol=1e-6, rtol=0)
        torch.testing.assert_close(forward, reverse.flip(dims=(1,)), atol=1e-6, rtol=1e-6)

    def test_euclidean_path_has_zero_deviation(self):
        module = GeodesicModalityGraphFusion(shared_dim=8, fusion_mode="euclidean")
        output = module(self.nodes, self.prototypes)
        self.assertEqual(float(output["path_deviation"].abs().max()), 0.0)
        torch.testing.assert_close(output["geodesic_energy"], output["linear_energy"])

    def test_land_metric_is_positive_and_support_is_detached(self):
        points = torch.randn(2, 3, 6, 4, 8, requires_grad=True)
        supports = torch.randn(2, 10, 8, requires_grad=True)
        metric = self.module.land_metric_diagonal(points, supports)
        self.assertTrue(torch.isfinite(metric).all())
        self.assertTrue((metric > 0).all())
        metric.mean().backward()
        self.assertIsNotNone(points.grad)
        self.assertIsNone(supports.grad)

    def test_one_optimizer_step_reduces_fixed_endpoint_energy(self):
        module = GeodesicModalityGraphFusion(shared_dim=8, path_steps=5)
        optimizer = torch.optim.SGD(
            list(module.geopath_net.parameters()) + [module.bend_logit], lr=1e-4
        )
        before = module(self.nodes.detach(), self.prototypes.detach())["geo_energy_loss"]
        optimizer.zero_grad()
        before.backward()
        optimizer.step()
        after = module(self.nodes.detach(), self.prototypes.detach())["geo_energy_loss"]
        self.assertLess(float(after.detach()), float(before.detach()))

    def test_missing_modality_masks_edges_and_normalizes_rows(self):
        mask = torch.tensor([[1, 0, 1, 1], [1, 1, 1, 1]], dtype=torch.bool)
        output = self.module(self.nodes, self.prototypes, modality_mask=mask)
        adjacency = output["adjacency"]
        self.assertEqual(float(adjacency[0, :, 1].abs().sum().detach()), 0.0)
        self.assertEqual(float(adjacency[0, :, :, 1].abs().sum().detach()), 0.0)
        row_sum = adjacency.sum(dim=-1)
        torch.testing.assert_close(row_sum[0, :, [0, 2, 3]], torch.ones_like(row_sum[0, :, [0, 2, 3]]))
        self.assertEqual(float(row_sum[0, :, 1].abs().sum().detach()), 0.0)

    def test_single_available_modality_is_finite(self):
        mask = torch.tensor([[1, 0, 0, 0], [0, 0, 0, 1]], dtype=torch.bool)
        output = self.module(self.nodes, self.prototypes, modality_mask=mask)
        self.assertTrue(torch.isfinite(output["fused_nodes"]).all())
        self.assertEqual(int(output["pair_valid"].sum()), 0)

    def test_path_semantic_targets_and_gradients(self):
        output = self.module(self.nodes, self.prototypes)
        prototypes = self.prototypes.clone().requires_grad_(True)
        targets = [[0], [1], [2], [3], [4], [5]]
        loss = geodesic_path_semantic_loss(
            output["interior_paths"], output["pair_valid"], targets, prototypes
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(prototypes.grad)

    def test_paper4_model_forward_backward_and_boundaries(self):
        model = GliomaGeodesicFusionNet(z_slices=3, feat_dim=16, shared_dim=8)
        self.assertFalse(hasattr(model, "graph_builder"))
        self.assertFalse(hasattr(model, "diffusion"))
        self.assertFalse(hasattr(model, "private_heads"))
        self.assertFalse(hasattr(model, "topo_moe"))
        output = model(
            torch.randn(1, 4, 3, 32, 32),
            region_masks=torch.ones(1, 3, 3, 32, 32),
            anchor_prototypes=self.prototypes,
            return_extras=True,
        )
        self.assertEqual(tuple(output["extras"]["shared"].shape), (1, 3, 8))
        objective = output["extras"]["shared"].mean() + output["losses"]["geo_energy"]
        objective.backward()
        self.assertIsNotNone(model.modality_encoders[0].net[0].weight.grad)
        self.assertIsNotNone(model.fusion.geopath_net[0].weight.grad)
        self.assertIsNotNone(model.fusion.message_net[0].weight.grad)

    def test_paper4_profile_and_cli_defaults(self):
        args = Namespace(
            paper_config="paper4",
            node_mode="modalities",
            graph_type="learnable",
            no_private=False,
            no_diffusion=False,
            moe_module="topo_moe",
        )
        result = apply_paper_profile(args)
        self.assertEqual(result.node_mode, "regions")
        self.assertEqual(result.graph_type, "no_graph")
        self.assertTrue(result.no_private)
        self.assertTrue(result.no_diffusion)
        self.assertEqual(result.moe_module, "none")
        parsed = build_parser().parse_args(["--data_root", "unused", "--paper_config", "paper4"])
        self.assertEqual(parsed.fusion_mode, "geodesic")
        self.assertEqual(parsed.geo_path_steps, 5)

    def test_checkpoint_selection_fallback_schema(self):
        score, selection = _checkpoint_selection({}, -1.25, 3)
        self.assertEqual(score, -1.25)
        self.assertTrue(selection["fallback"])
        self.assertEqual(selection["metric"], "loss_fallback")

    def test_run_epoch_paper4_losses_and_diagnostics(self):
        args = build_parser().parse_args(["--data_root", "unused", "--paper_config", "paper4"])
        args.node_mode = "regions"
        args.alignment_objective = "clip"
        metadata = {
            "Tumor Grade": "II",
            "Tumor Type": "Glioma",
            "IDH": "Mutant",
            "MGMT": "Methylated",
            "1p19Q CODEL": "Intact",
        }
        anchors = semantic_anchors(metadata)
        key_to_id = {anchor["key"]: idx for idx, anchor in enumerate(anchors)}
        case_lookup = {
            "case_a": {"metadata": metadata},
            "case_b": {"metadata": metadata},
        }
        loader = [
            {
                "images": torch.randn(2, 4, 3, 32, 32),
                "region_masks": torch.ones(2, 3, 3, 32, 32),
                "subject_id": ["case_a", "case_b"],
            }
        ]
        model = GliomaGeodesicFusionNet(z_slices=3, feat_dim=16, shared_dim=8)
        bank = SemanticPrototypeBank(len(anchors), 8)
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(bank.parameters()), lr=1e-4)
        result = run_epoch(
            model,
            bank,
            loader,
            optimizer,
            torch.device("cpu"),
            args,
            case_lookup,
            key_to_id,
            1,
            {
                "medclip_ignore_ids": {},
                "family_ids": [],
                "family_names": [],
                "residual_index": 0,
            },
        )
        self.assertTrue(np.isfinite(result["losses"]["total"]))
        self.assertIn("geo_energy", result["losses"])
        self.assertIn("path_semantic", result["losses"])
        self.assertIn("energy_ratio", result["fusion"])
        self.assertGreater(result["fusion"]["geopath_gradient_norm"], 0.0)

    def test_five_ablation_configurations_change_only_requested_controls(self):
        configs = {
            "full": ("geodesic", "case_and_anchors", True),
            "euclidean": ("euclidean", "case_and_anchors", True),
            "case_only": ("geodesic", "case_only", True),
            "no_graph": ("geodesic", "case_and_anchors", False),
            "concat": ("concat", "case_and_anchors", False),
        }
        for mode, support, graph in configs.values():
            module = GeodesicModalityGraphFusion(
                8, fusion_mode=mode, metric_support=support, use_graph=graph
            )
            self.assertEqual(module.fusion_mode, mode)
            self.assertEqual(module.metric_support, support)
            self.assertEqual(module.use_graph, graph)

    def test_server_launcher_declares_full_protocol(self):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        with open(
            os.path.join(project_root, "run_server_paper4_geodesic_full.sh"),
            "r",
            encoding="utf-8",
        ) as file:
            launcher = file.read()
        for variant in (
            "full_geodesic_graph",
            "euclidean_graph",
            "case_only_metric",
            "geodesic_no_graph",
            "latent_concat",
        ):
            self.assertIn(f"run_variant {variant}", launcher)
        self.assertIn('SEEDS="${SEEDS:-42 43 44}"', launcher)
        self.assertIn('PAPER4_EPOCHS="${PAPER4_EPOCHS:-50}"', launcher)
        self.assertIn("PAPER4_ALIGN_MAX_CASES=", launcher)
        self.assertIn("aggregate_geodesic_runs", launcher)

    def test_payloads_and_figures(self):
        output = self.module(self.nodes, self.prototypes)
        records = {
            "case_count": 2,
            "prototypes": self.prototypes.numpy(),
            "geodesic_fusion": {
                "pair_indices": output["pair_indices"].tolist(),
                "adjacency": output["adjacency"].detach().numpy(),
                "geodesic_energy": output["geodesic_energy"].detach().numpy(),
                "linear_energy": output["linear_energy"].detach().numpy(),
                "energy_ratio": output["energy_ratio"].detach().numpy(),
                "path_deviation": output["path_deviation"].detach().numpy(),
                "representative_paths": [
                    {"subject_id": "x", "paths": output["paths"][0].detach().tolist()}
                ],
            },
        }
        diagnostics, graph = build_geodesic_payloads(
            records, {"fusion_mode": "geodesic", "metric_support": "case_and_anchors"}
        )
        self.assertIn("energy_ratio", diagnostics)
        self.assertEqual(len(graph["edges"]), 18)
        with tempfile.TemporaryDirectory() as out_dir:
            figures = save_geodesic_figures(records, graph, out_dir)
            self.assertEqual(len(figures), 3)
            self.assertTrue(all(os.path.exists(os.path.join(out_dir, name)) for name in figures))

    def test_paper4_evaluation_writes_only_geodesic_artifacts(self):
        args = build_parser().parse_args(["--data_root", "unused", "--paper_config", "paper4"])
        args.node_mode = "regions"
        metadata = {
            "Tumor Grade": "II",
            "Tumor Type": "Glioma",
            "IDH": "Mutant",
            "MGMT": "Methylated",
            "1p19Q CODEL": "Intact",
        }
        anchor_vocab = semantic_anchors(metadata)
        key_to_id = {anchor["key"]: idx for idx, anchor in enumerate(anchor_vocab)}
        case_lookup = {"case_a": {"metadata": metadata}, "case_b": {"metadata": metadata}}
        loader = [
            {
                "images": torch.randn(2, 4, 3, 32, 32),
                "region_masks": torch.ones(2, 3, 3, 32, 32),
                "subject_id": ["case_a", "case_b"],
            }
        ]
        model = GliomaGeodesicFusionNet(z_slices=3, feat_dim=16, shared_dim=8)
        bank = SemanticPrototypeBank(len(anchor_vocab), 8)
        with tempfile.TemporaryDirectory() as out_dir:
            evaluate_and_save(
                model,
                bank,
                loader,
                torch.device("cpu"),
                args,
                case_lookup,
                key_to_id,
                anchor_vocab,
                out_dir,
                checkpoint_type="paper4_direct_best",
                run_interventions=False,
                figure_context={
                    "seed": 42,
                    "fusion_mode": "geodesic",
                    "metric_support": "case_and_anchors",
                    "fusion_graph": True,
                },
            )
            required = {
                "test_metrics.json",
                "geodesic_diagnostics.json",
                "fusion_graph.json",
                "fusion_figure_manifest.json",
                "geodesic_path_projection.png",
                "modality_geodesic_graph.png",
                "geodesic_energy_comparison.png",
            }
            self.assertTrue(required.issubset(set(os.listdir(out_dir))))
            self.assertFalse(any(name.startswith("semantic_unit_") for name in os.listdir(out_dir)))


class GeodesicAggregationTests(unittest.TestCase):
    def test_multiseed_ablation_aggregation(self):
        variants = [
            "full_geodesic_graph",
            "euclidean_graph",
            "case_only_metric",
            "geodesic_no_graph",
            "latent_concat",
        ]
        with tempfile.TemporaryDirectory() as root:
            for variant in variants:
                for seed in (42, 43):
                    seed_dir = os.path.join(root, variant, "paper4", f"seed_{seed}")
                    os.makedirs(seed_dir)
                    payloads = {
                        "anchor_vocab.json": [{"label": "A"}, {"label": "B"}],
                        "splits.json": {"train": ["a"], "val": ["b"], "test": ["c"]},
                        "test_metrics.json": {"map": 0.5 + seed * 1e-4, "mrr": 0.6},
                        "geodesic_diagnostics.json": {
                            "energy_ratio": {"mean": 0.9},
                            "path_deviation": {"mean": 0.1},
                        },
                        "fusion_graph.json": {
                            "modality_names": ["T1", "T1ce", "T2", "FLAIR"],
                            "region_names": ["Core", "Edema", "Enhancing"],
                            "adjacency_mean": np.tile(np.eye(4)[None], (3, 1, 1)).tolist(),
                        },
                    }
                    for name, payload in payloads.items():
                        with open(os.path.join(seed_dir, name), "w", encoding="utf-8") as file:
                            json.dump(payload, file)
            manifest = aggregate(root)
            self.assertEqual(set(manifest["variants"]), set(variants))
            self.assertTrue(os.path.exists(os.path.join(root, "aggregate", "aggregate_geodesic.json")))
            self.assertEqual(len(manifest["figures"]), 2)


if __name__ == "__main__":
    unittest.main()
