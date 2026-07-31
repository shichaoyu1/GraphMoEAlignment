import json
import os
import tempfile
import unittest

import numpy as np
import torch

from glioma.anchors import semantic_anchors
from glioma.cli.aggregate_manifold_runs import CONFIRM_CORE_VARIANTS, SCREEN_VARIANTS, aggregate
from glioma.cli.train_semantic_alignment import build_parser
from glioma.eval.semantic_alignment_eval import evaluate_and_save
from glioma.models.glioma_geodesic_fusion_net import GliomaGeodesicFusionNet
from glioma.modules.hierarchical_spd_fusion import (
    HierarchicalSPDGraphFusion,
    spd_expm,
    spd_geodesic,
    spd_logm,
    trace_normalize,
)
from glioma.modules.published_fusion_baselines import PublishedModalityFusion
from glioma.visualization.manifold_fusion import save_manifold_figures
from glioma.objectives import SemanticPrototypeBank
from glioma.training.engine import run_epoch


class ManifoldOperatorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.module = HierarchicalSPDGraphFusion(
            token_dim=12,
            shared_dim=8,
            family_ids=[0, 0, 1, 1],
            family_names=["pathology", "molecular", "residual"],
            family_prior=torch.eye(3),
            spd_dim=4,
        )
        self.tokens = torch.randn(2, 3, 4, 9, 12, requires_grad=True)
        self.prototypes = torch.randn(4, 8, requires_grad=True)

    def test_log_exp_roundtrip_and_geodesic_endpoints(self):
        raw = torch.randn(3, 4, 4)
        matrix = raw @ raw.transpose(-1, -2) + 0.2 * torch.eye(4)
        torch.testing.assert_close(spd_expm(spd_logm(matrix)), matrix, atol=2e-4, rtol=2e-4)
        path = spd_geodesic(matrix[0], matrix[1], torch.linspace(0, 1, 5))
        torch.testing.assert_close(path[0], matrix[0], atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(path[-1], matrix[1], atol=2e-4, rtol=2e-4)

    def test_trace_normalization_removes_global_scale(self):
        raw = torch.randn(2, 4, 4)
        matrix = raw @ raw.transpose(-1, -2) + 0.1 * torch.eye(4)
        normalized, _ = trace_normalize(matrix)
        scaled, _ = trace_normalize(17.0 * matrix)
        torch.testing.assert_close(normalized, scaled, atol=1e-5, rtol=1e-5)

    def test_masked_graph_and_finite_backward(self):
        mask = torch.tensor([[1, 0, 1, 1], [1, 1, 1, 1]], dtype=torch.bool)
        output = self.module(self.tokens, self.prototypes, mask)
        self.assertEqual(tuple(output["fused_nodes"].shape), (2, 3, 8))
        self.assertEqual(float(output["local_adjacency"][0, :, 1].detach().abs().sum()), 0.0)
        loss = output["fused_nodes"].mean() + output["condition_loss"] + output["topology_loss"]
        loss.backward()
        self.assertTrue(torch.isfinite(self.tokens.grad).all())
        self.assertTrue(torch.isfinite(self.prototypes.grad).all())

    def test_cross_budget_and_graph_interventions(self):
        output = self.module(self.tokens.detach(), self.prototypes.detach())
        local = output["local_adjacency"]
        upper = output["upper_adjacency"]
        local_offdiag = local.sum(dim=-1) - torch.diagonal(local, dim1=-2, dim2=-1)
        upper_offdiag = upper.sum(dim=-1) - torch.diagonal(upper, dim1=-2, dim2=-1)
        torch.testing.assert_close(local_offdiag, torch.full_like(local_offdiag, 0.35), atol=1e-5, rtol=0)
        torch.testing.assert_close(upper_offdiag, torch.full_like(upper_offdiag, 0.40), atol=1e-5, rtol=0)
        torch.testing.assert_close(
            upper[:, :3, 3:].sum(dim=-1),
            torch.full_like(upper[:, :3, 3:].sum(dim=-1), 0.25),
            atol=1e-5,
            rtol=0,
        )
        identity = self.module(
            self.tokens.detach(), self.prototypes.detach(), graph_intervention="identity"
        )
        self.assertGreater(float((output["fused_nodes"] - identity["fused_nodes"]).detach().abs().sum()), 0.0)
        first = self.module(self.tokens.detach(), self.prototypes.detach(), graph_intervention="shuffle")
        second = self.module(self.tokens.detach(), self.prototypes.detach(), graph_intervention="shuffle")
        torch.testing.assert_close(first["local_adjacency"], second["local_adjacency"])
        no_cross_type = self.module(
            self.tokens.detach(), self.prototypes.detach(), graph_intervention="no_region_family"
        )
        self.assertEqual(
            float(no_cross_type["upper_adjacency"][:, :3, 3:].detach().abs().sum()), 0.0
        )

    def test_single_modality_falls_back_to_identity(self):
        mask = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.bool)
        output = self.module(self.tokens.detach(), self.prototypes.detach(), mask)
        diagonal = torch.diagonal(output["local_adjacency"], dim1=-2, dim2=-1).detach()
        self.assertEqual(float(diagonal[0, :, 0].min()), 1.0)
        self.assertEqual(float(diagonal[1, :, 1].min()), 1.0)

    def test_published_baselines_mask_and_backward(self):
        nodes = torch.randn(2, 3, 4, 8, requires_grad=True)
        mask = torch.tensor([[1, 1, 0, 1], [1, 1, 1, 1]], dtype=torch.bool)
        outputs = []
        for mode in ("latent_concat", "hemis", "gmu", "mbt_style"):
            result = PublishedModalityFusion(8, mode=mode)(nodes, modality_mask=mask)
            self.assertEqual(tuple(result["fused_nodes"].shape), (2, 3, 8))
            outputs.append(result["fused_nodes"])
        sum(value.sum() for value in outputs).backward()
        self.assertTrue(torch.isfinite(nodes.grad).all())

    def test_published_baselines_complete_training_epoch(self):
        metadata = {
            "Tumor Grade": "II",
            "Tumor Type": "Glioma",
            "IDH": "Mutant",
            "MGMT": "Methylated",
            "1p19Q CODEL": "Intact",
        }
        anchors = semantic_anchors(metadata)
        key_to_id = {anchor["key"]: index for index, anchor in enumerate(anchors)}
        case_lookup = {
            "case_a": {"metadata": metadata},
            "case_b": {"metadata": metadata},
        }
        loader = [{
            "images": torch.randn(2, 4, 3, 32, 32),
            "region_masks": torch.ones(2, 3, 3, 32, 32),
            "subject_id": ["case_a", "case_b"],
        }]
        loss_context = {
            "medclip_ignore_ids": {},
            "family_ids": [],
            "family_names": [],
            "residual_index": 0,
        }
        for mode in ("latent_concat", "hemis", "gmu", "mbt_style"):
            args = build_parser().parse_args([
                "--data_root", "unused",
                "--paper_config", "paper4",
                "--paper4_fusion_backend", "published_baseline",
                "--paper4_baseline", mode,
            ])
            model = GliomaGeodesicFusionNet(
                z_slices=3,
                feat_dim=16,
                shared_dim=8,
                paper4_fusion_backend="published_baseline",
                paper4_baseline=mode,
            )
            bank = SemanticPrototypeBank(len(anchors), 8)
            optimizer = torch.optim.AdamW(
                list(model.parameters()) + list(bank.parameters()), lr=1e-4
            )
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
                loss_context,
            )
            self.assertTrue(np.isfinite(result["losses"]["total"]), mode)
            self.assertEqual(result["losses"]["nonfinite_batches"], 0.0, mode)
            self.assertGreater(result["fusion"]["geopath_gradient_norm"], 0.0, mode)

    def test_model_does_not_use_labels(self):
        model = GliomaGeodesicFusionNet(
            z_slices=3,
            feat_dim=16,
            shared_dim=8,
            paper4_fusion_backend="spd_hierarchical",
            spd_dim=4,
            anchor_family_ids=[0, 0, 1, 1],
            anchor_family_names=["pathology", "molecular", "residual"],
            family_prior=torch.eye(3),
        ).eval()
        images = torch.randn(1, 4, 3, 32, 32)
        masks = torch.ones(1, 3, 3, 32, 32)
        prototypes = torch.randn(4, 8)
        with torch.no_grad():
            left = model(images, labels=torch.tensor([0]), region_masks=masks, anchor_prototypes=prototypes, return_extras=True)
            right = model(images, labels=torch.tensor([99]), region_masks=masks, anchor_prototypes=prototypes, return_extras=True)
        torch.testing.assert_close(left["extras"]["shared"], right["extras"]["shared"])


class ManifoldArtifactTests(unittest.TestCase):
    def test_evaluation_writes_manifold_artifacts(self):
        args = build_parser().parse_args(
            ["--data_root", "unused", "--paper_config", "paper4", "--paper4_fusion_backend", "spd_hierarchical"]
        )
        metadata = {
            "Tumor Grade": "2",
            "Tumor Type": "Glioma",
            "IDH": "mutated",
            "MGMT": "methylated",
            "1p19Q CODEL": "co-deleted",
        }
        from glioma.anchors import semantic_anchors

        anchor_vocab = semantic_anchors(metadata)
        key_to_id = {anchor["key"]: index for index, anchor in enumerate(anchor_vocab)}
        family_ids = [0 if anchor.get("source") == "Pathology" else 1 for anchor in anchor_vocab]
        model = GliomaGeodesicFusionNet(
            z_slices=3,
            feat_dim=16,
            shared_dim=8,
            paper4_fusion_backend="spd_hierarchical",
            spd_dim=4,
            anchor_family_ids=family_ids,
            anchor_family_names=["pathology", "molecular", "residual"],
            family_prior=torch.eye(3),
        )
        bank = SemanticPrototypeBank(len(anchor_vocab), 8)
        loader = [{
            "images": torch.randn(2, 4, 3, 32, 32),
            "region_masks": torch.ones(2, 3, 3, 32, 32),
            "subject_id": ["a", "b"],
        }]
        case_lookup = {subject: {"metadata": metadata} for subject in ("a", "b")}
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
                run_interventions=True,
                figure_context={"seed": 42, "fusion_backend": "spd_hierarchical", "spd_geometry": "spd"},
            )
            required = {
                "manifold_feature_stats.json",
                "manifold_case_records.json",
                "manifold_graph_records.npz",
                "manifold_topology.json",
                "graph_role_metrics.json",
                "paper4_manifold_figure_manifest.json",
                "paper4_manifold_overview.png",
                "paper4_scale_to_manifold.png",
                "paper4_hierarchical_topology.png",
                "paper4_graph_participation.png",
                "paper4_graph_interventions.png",
                "paper4_manifold_ablation.png",
            }
            self.assertTrue(required.issubset(set(os.listdir(out_dir))))
            with open(os.path.join(out_dir, "graph_role_metrics.json"), encoding="utf-8") as file:
                role = json.load(file)
            self.assertEqual(
                set(role["interventions"]),
                {"identity", "uniform", "shuffle", "no_local", "no_upper", "no_region_family"},
            )
            with open(os.path.join(out_dir, "test_metrics.json"), encoding="utf-8") as file:
                metrics = json.load(file)
            self.assertIn("target_family", metrics["subgroups"]["macro"])

    def test_five_figures_and_full_protocol_launcher(self):
        output = self.module_records()
        with tempfile.TemporaryDirectory() as out_dir:
            figures = save_manifold_figures(output, {"recall@1": 0.7, "map": 0.6, "mrr": 0.8}, self.anchor_vocab(), out_dir)
            self.assertEqual(len(figures), 6)
            self.assertTrue(all(os.path.getsize(os.path.join(out_dir, name)) > 0 for name in figures))
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        with open(os.path.join(root, "run_server_paper4_graph_evidence.sh"), encoding="utf-8") as file:
            launcher = file.read()
        for variant in SCREEN_VARIANTS:
            self.assertIn(variant, launcher)
        self.assertIn('STAGE="${STAGE:-screen}"', launcher)

    def test_multiseed_aggregate_status(self):
        with tempfile.TemporaryDirectory() as root:
            for variant in SCREEN_VARIANTS:
                for seed in (42, 43, 44):
                    directory = os.path.join(root, variant, "paper4", f"seed_{seed}")
                    os.makedirs(directory)
                    payloads = {
                        "anchor_vocab.json": self.anchor_vocab(),
                        "splits.json": {"train": ["a"], "val": ["b"], "test": ["c"]},
                        "test_metrics.json": {"recall@1": 0.7, "map": 0.6, "mrr": 0.8},
                    }
                    if variant == "spd_cross_graph":
                        payloads["manifold_topology.json"] = {
                            "local_adjacency_mean": np.tile(np.eye(4)[None], (3, 1, 1)).tolist(),
                            "upper_adjacency_mean": np.eye(6).tolist(),
                            "upper_node_names": ["Core", "Edema", "Enhancing", "pathology", "molecular", "residual"],
                        }
                    for name, payload in payloads.items():
                        with open(os.path.join(directory, name), "w", encoding="utf-8") as file:
                            json.dump(payload, file)
            manifest = aggregate(root)
            self.assertEqual(manifest["status"], "final_multiseed")
            self.assertTrue(os.path.exists(os.path.join(root, "aggregate", "paper4_manifold_ablation.png")))

    def test_confirmation_reuses_screening_selection_and_requires_five_seeds(self):
        with tempfile.TemporaryDirectory() as root:
            def write_seed(variant, seed, validation_score):
                directory = os.path.join(root, variant, "paper4", f"seed_{seed}")
                os.makedirs(directory, exist_ok=True)
                payloads = {
                    "anchor_vocab.json": self.anchor_vocab(),
                    "splits.json": {"train": [f"train-{seed}"], "val": [f"val-{seed}"], "test": [f"test-{seed}"]},
                    "test_metrics.json": {"recall@1": 0.7, "map": 0.6 + seed * 1e-5, "mrr": 0.8},
                    "checkpoint_manifest.json": {"direct": {"metric": "map", "score": validation_score}},
                }
                if variant == "spd_cross_graph":
                    payloads["manifold_topology.json"] = {
                        "local_adjacency_mean": np.tile(np.eye(4)[None], (3, 1, 1)).tolist(),
                        "upper_adjacency_mean": np.eye(6).tolist(),
                        "upper_node_names": ["Core", "Edema", "Enhancing", "pathology", "molecular", "residual"],
                    }
                for name, payload in payloads.items():
                    with open(os.path.join(directory, name), "w", encoding="utf-8") as file:
                        json.dump(payload, file)

            baseline_scores = {"latent_concat": 0.60, "hemis": 0.66, "gmu": 0.63, "mbt_style": 0.64}
            for variant in SCREEN_VARIANTS:
                for seed in (42, 43, 44):
                    write_seed(variant, seed, baseline_scores.get(variant, 0.62))
            aggregate(root, stage="screen", expected_seeds=[42, 43, 44])
            for variant in CONFIRM_CORE_VARIANTS + ["hemis"]:
                for seed in (45, 46):
                    write_seed(variant, seed, baseline_scores.get(variant, 0.62))
            manifest = aggregate(root, stage="confirm", expected_seeds=[42, 43, 44, 45, 46])
            self.assertEqual(manifest["status"], "final_multiseed")
            self.assertEqual(manifest["best_published_baseline"], "hemis")

    @staticmethod
    def anchor_vocab():
        return [
            {"label": "Pathology A", "source": "Pathology"},
            {"label": "Molecular B", "source": "Gene"},
        ]

    @staticmethod
    def module_records():
        rng = np.random.default_rng(4)
        subjects = ["a", "b"]
        query_records = []
        targets = []
        for subject, grade in zip(subjects, ["2", "3"]):
            for region in ("Necrotic/Core", "Edema", "Enhancing"):
                query_records.append({"subject_id": subject, "node_name": region, "grade": grade, "target_labels": ["Pathology A"]})
                targets.append([0])
        return {
            "query_records": query_records,
            "query_targets": targets,
            "direct_scores": rng.normal(size=(6, 2)),
            "manifold_fusion": {
                "subject_ids": subjects,
                "local_adjacency": np.tile(np.eye(4)[None, None], (2, 3, 1, 1)),
                "upper_adjacency": np.tile(np.eye(6)[None], (2, 1, 1)),
                "upper_node_names": ["Necrotic/Core", "Edema", "Enhancing", "pathology", "molecular", "residual"],
                "raw_scales": np.abs(rng.normal(size=(2, 3, 4))),
                "raw_spd_traces": np.abs(rng.normal(size=(2, 3, 4))),
                "condition_numbers": 1.0 + np.abs(rng.normal(size=(2, 3, 4))),
                "spd_eigenvalues": np.abs(rng.normal(size=(2, 3, 4, 4))),
            },
        }


if __name__ == "__main__":
    unittest.main()
