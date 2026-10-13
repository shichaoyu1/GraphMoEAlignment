"""Scientific protocol, isolation, and end-to-end tests for opt-in revision runs."""

from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from glioma.anchors import semantic_anchors, target_anchor_keys
from glioma.cli.aggregate_paper2_revision import paired_summary
from glioma.cli.run_paper2_revision import build_plan, experiments
from glioma.cli.train_paper2_revision import build_parser, main
from glioma.cli.evaluate_paper2_transfer import main as transfer_main
from glioma.cli.train_semantic_alignment import build_parser as legacy_parser
from glioma.config.paper_profiles import apply_paper2_experiment_profile, apply_paper_profile
from glioma.paper2_revision.evaluation import gate_interventions, summarize
from glioma.paper2_revision.losses import (family_supervision_loss, observed_field_classification_loss,
                                        proxy_anchor_loss, retrieval_loss, target_masks)
from glioma.paper2_revision.model import FactorizedHead
from glioma.paper2_revision.protocol import (NODES, QueryTarget, fit_family_prior, fit_frequency,
                                          query_target, validate_splits)
from glioma.semantic.vocab import build_anchor_vocab
from glioma.data.case_discovery import discover_semantic_cases
from glioma.data.utsw_dataset import patient_cross_validation_split
from glioma.models.glioma_topomoe_net import GliomaTopoMoENet
from glioma.objectives import SemanticPrototypeBank


class RevisionProtocolTests(unittest.TestCase):
    def setUp(self):
        self.cases = [
            {"subject_id": "a", "metadata": {"Tumor Grade": "4", "Tumor Type": "GBM", "IDH": "wild type", "MGMT": "unknown", "1p19Q CODEL": "non-codeleted"}},
            {"subject_id": "b", "metadata": {"Tumor Grade": "2", "Tumor Type": "astro", "IDH": "mutated", "MGMT": "methylated"}},
        ]
        self.vocab, self.keys = build_anchor_vocab(self.cases)
        self.family_ids = [0 if a["source"] == "Pathology" else 1 for a in self.vocab]

    def test_targets_match_legacy_policy_without_mutating_metadata(self):
        original = deepcopy(self.cases)
        for case in self.cases:
            for node in NODES:
                for policy in ("region_rules", "all_patient_anchors"):
                    target = query_target(case["metadata"], node, policy, self.vocab)
                    expected = [self.keys[key] for key in target_anchor_keys(case["metadata"], node, policy)]
                    self.assertEqual(list(target.positives), expected)
                    self.assertTrue(set(target.positives) <= set(target.observed_candidates))
        self.assertEqual(self.cases, original)
        edema = query_target(self.cases[1]["metadata"], "Edema", "region_rules", self.vocab)
        mgmt = next(a["key"] for a in semantic_anchors(self.cases[1]["metadata"]) if a["field"] == "MGMT")
        self.assertIn(mgmt, edema.rule_excluded_known)

    def test_unknown_is_not_positive_and_observed_protocol_masks_candidates(self):
        target = query_target(self.cases[0]["metadata"], "Enhancing", "region_rules", self.vocab)
        positives, full = target_masks([target], len(self.vocab), "cpu")
        _, observed = target_masks([target], len(self.vocab), "cpu", "observed_fields")
        mgmt = next(i for i, a in enumerate(self.vocab) if a["field"] == "MGMT")
        self.assertFalse(bool(positives[0, mgmt]))
        self.assertTrue(bool(full[0, mgmt]))
        self.assertFalse(bool(observed[0, mgmt]))

    def test_oov_and_fallback_are_visible_and_empty_splits_rejected(self):
        target = query_target({"IDH": "unseen"}, "Enhancing", "region_rules", self.vocab)
        self.assertTrue(target.fallback)
        self.assertEqual(len(target.oov_keys), 1)
        self.assertFalse(target.positives)
        with self.assertRaises(ValueError):
            validate_splits({"train": self.cases, "val": [], "test": self.cases})
        with self.assertRaises(ValueError):
            validate_splits({"train": self.cases[:1], "val": self.cases[1:], "test": self.cases[:1]})

    def test_frequency_and_family_priors_use_only_explicit_train_rows(self):
        probabilities = fit_frequency(self.cases[:1], "region_rules", self.vocab, by_node=True)
        changed_test = deepcopy(self.cases[1])
        changed_test["metadata"]["IDH"] = "wild type"
        np.testing.assert_array_equal(probabilities, fit_frequency(self.cases[:1], "region_rules", self.vocab, True))
        self.assertFalse(np.array_equal(probabilities, fit_frequency(self.cases + [changed_test], "region_rules", self.vocab, True)))
        priors = fit_family_prior(self.cases[:1], "region_rules", self.vocab, self.family_ids, True)
        self.assertGreater(priors[0, 1], priors[2, 1])
        np.testing.assert_allclose(priors.sum(-1), 1)

    def test_multilabel_unknown_field_has_zero_gradient(self):
        target = query_target(self.cases[0]["metadata"], "Enhancing", "region_rules", self.vocab)
        logits = torch.randn(1, len(self.vocab), requires_grad=True)
        observed_field_classification_loss(logits, [target], self.vocab).backward()
        mgmt = [i for i, a in enumerate(self.vocab) if a["field"] == "MGMT"]
        self.assertTrue(torch.equal(logits.grad[:, mgmt], torch.zeros_like(logits.grad[:, mgmt])))

    def test_published_loss_formulas_and_masked_gradients(self):
        logits = torch.tensor([[2.0, 0.0, -1.0]], requires_grad=True)
        positive = torch.tensor([[True, True, False]])
        candidates = torch.ones_like(positive)
        expected = -torch.log_softmax(logits, -1)[0, :2].mean()
        torch.testing.assert_close(retrieval_loss(logits, positive, candidates, True), expected)
        cosine = torch.tensor([[0.8, -0.4], [-0.2, 0.7]], requires_grad=True)
        mask = torch.eye(2, dtype=torch.bool)
        expected_pos = torch.log1p((torch.exp(-32 * (cosine - 0.1)) * mask).sum(0)).mean()
        expected_neg = torch.log1p((torch.exp(32 * (cosine + 0.1)) * ~mask).sum(0)).mean()
        torch.testing.assert_close(proxy_anchor_loss(cosine, mask, torch.ones_like(mask)), expected_pos + expected_neg)
        masked_candidates = torch.tensor([[True, False], [True, True]])
        loss = proxy_anchor_loss(cosine, mask, masked_candidates)
        loss.backward()
        self.assertEqual(float(cosine.grad[0, 1]), 0)

    def test_equal_family_supervision_and_explicit_off_scope(self):
        gates = torch.tensor([[0.2, 0.8]], requires_grad=True)
        loss = family_supervision_loss(gates, torch.tensor([[True, True, True]]), torch.tensor([0, 0, 1]))
        torch.testing.assert_close(loss, -0.5 * gates.log().sum())
        args = build_parser().parse_args(["--data_root", "unused", "--out_dir", "unused", "--family_supervision", "off"])
        self.assertEqual(args.lambda_within, 0.3)


class RevisionModelTests(unittest.TestCase):
    def test_static_gates_ignore_patient_features_and_gate_changes_preserve_family_rank(self):
        torch.manual_seed(3)
        head = FactorizedHead(8, [0, 0, 1, 1], gate="node")
        features, prototypes = torch.randn(2, 3, 8), torch.randn(4, 8)
        first = head(features, prototypes, 0.07)
        uniform = head(features, prototypes, 0.07, torch.full((2, 3, 2), 0.5))
        torch.testing.assert_close(first["gates"][0], first["gates"][1])
        torch.testing.assert_close(first["scores"][..., 0] - first["scores"][..., 1], uniform["scores"][..., 0] - uniform["scores"][..., 1])
        torch.testing.assert_close(first["scores"].exp().sum(-1), torch.ones(2, 3))

    def test_all_head_factors_backward_and_probability_mass(self):
        for projection in ("shared", "separate"):
            for gate in ("node", "dynamic"):
                head = FactorizedHead(8, [0, 0, 1, 1], projection, gate)
                features = torch.randn(2, 3, 8, requires_grad=True)
                prototypes = torch.randn(4, 8, requires_grad=True)
                output = head(features, prototypes, 0.1)
                (-output["scores"][..., 0].mean()).backward()
                self.assertTrue(torch.isfinite(features.grad).all())
                self.assertTrue(torch.isfinite(prototypes.grad).all())
                self.assertTrue(all(p.grad is not None for p in head.parameters()))

    def test_node_interventions_exhaust_nonidentity_and_patient_shuffle_has_no_fixed_points(self):
        subjects = [subject for subject in ("a", "b", "c") for _ in NODES]
        nodes = list(NODES) * 3
        gates = np.asarray([[0.2 + 0.01 * i, 0.8 - 0.01 * i] for i in range(9)])
        conditional = np.log(np.full((9, 4), 0.5))
        outputs, audit = gate_interventions(gates, conditional, subjects, nodes, [0, 0, 1, 1], 42)
        self.assertEqual(audit["node_permutation_count"], 5)
        self.assertEqual(len([key for key in outputs if key.startswith("node_permutation")]), 5)
        for key, mapping in audit["mappings"].items():
            if key.startswith("node_permutation"):
                self.assertTrue(any(source != dest for source, dest in mapping.items()))
            else:
                self.assertTrue(all(source != dest for source, dest in mapping.items()))
        second = gate_interventions(gates, conditional, subjects, nodes, [0, 0, 1, 1], 42)[0]
        for name in outputs:
            np.testing.assert_array_equal(outputs[name], second[name])


class RevisionIsolationTests(unittest.TestCase):
    def test_legacy_profiles_and_parser_defaults_are_unchanged(self):
        for paper in ("paper1", "paper2", "paper3", "paper4"):
            args = legacy_parser().parse_args(["--data_root", "unused", "--paper_config", paper])
            args = apply_paper2_experiment_profile(apply_paper_profile(args))
            self.assertEqual(args.experiment_profile, "legacy")
            self.assertFalse(hasattr(args, "candidate_policy"))
        args = legacy_parser().parse_args(["--data_root", "unused", "--experiment_profile", "prior_guided_router"])
        args = apply_paper2_experiment_profile(args)
        self.assertEqual((args.context_mode, args.topo_mode, args.score_mode), ("topology", "prior_only", "conditional"))

    def test_plan_is_supplemental_and_nonexecuting(self):
        args = Namespace(data_root="unused", out_root="unused", metadata_tsv=None, group="mechanism", reference_root=None,
                         cv_folds=5, folds=None, cv_seed=2026, seeds=[42], epochs=40, router_epochs=15, joint_epochs=10,
                         batch_size=4, num_workers=0, segmentation_source="not_documented", cpu=True)
        plan = build_plan(args)
        self.assertEqual(len(plan), 20)
        self.assertTrue(all(item["command"][2] == "glioma.cli.train_paper2_revision" for item in plan))
        self.assertEqual(len(experiments("supervision")), 3)
        args.seeds = [42, 42]
        with self.assertRaises(ValueError):
            build_plan(args)

    def test_policy_change_does_not_reuse_incompatible_reference_weights(self):
        args = Namespace(data_root="unused", out_root="unused", metadata_tsv=None, group="supervision", reference_root="reference",
                         cv_folds=5, folds=[0], cv_seed=2026, seeds=[42], epochs=40, router_epochs=15, joint_epochs=10,
                         batch_size=4, num_workers=0, segmentation_source="not_documented", cpu=True)
        for item in build_plan(args):
            if "all_patient_anchors" in item["command"]:
                self.assertNotIn("--reference_run", item["command"])
                self.assertIn("--splits_file", item["command"])

    def test_pairing_rejects_target_changes_and_missing_runs(self):
        target = QueryTarget((0,), (0, 1), (), (), False)
        metrics = summarize([[2, 1]], [target], ["a"], ["Edema"], {"a": "4"}, [0, 1])
        records = {"target_policy": "region_rules", "candidate_policy": "full_gallery", "anchor_keys": ["x", "y"],
                   "family_ids": [0, 1], "subject_ids": ["a"], "node_names": ["Edema"], "query_targets": [[0]],
                   "observed_candidates": [[0, 1]], "metrics": metrics}
        left = {(2026, 0, 42): records}
        result = paired_summary(left, deepcopy(left), bootstrap=20)
        self.assertEqual(result["metrics"]["map"]["mean_delta"], 0)
        modified = deepcopy(left)
        modified[(2026, 0, 42)]["query_targets"] = [[1]]
        with self.assertRaises(ValueError):
            paired_summary(left, modified)
        with self.assertRaises(ValueError):
            paired_summary(left, {})


def write_fixture(root, n=15, prefix="synthetic_"):
    import nibabel as nib
    root = Path(root)
    data = root / "data"
    data.mkdir()
    metadata = root / "metadata.tsv"
    lines = ["Subject ID\tTumor Grade\tTumor Type\tIDH\tMGMT\t1p19Q CODEL"]
    rng = np.random.default_rng(17)
    for i in range(n):
        subject = prefix + str(i)
        folder = data / subject
        folder.mkdir()
        for modality in ("t1", "t1ce", "t2", "flair"):
            image = rng.normal(size=(16, 16, 3)).astype(np.float32)
            nib.save(nib.Nifti1Image(image, np.eye(4)), folder / ("brain_" + modality + ".nii.gz"))
        seg = np.zeros((16, 16, 3), dtype=np.float32)
        seg[2:6, 2:6] = 1
        seg[6:10, 6:10] = 2
        seg[10:14, 10:14] = 4
        nib.save(nib.Nifti1Image(seg, np.eye(4)), folder / "tumorseg_manual_correction.nii.gz")
        lines.append(f"{subject}\t{2 + i % 3}\t{'GBM' if i % 2 else 'astro'}\t{'mutated' if i % 2 else 'wild type'}\t{'methylated' if i % 2 else 'unknown'}\tnon-codeleted")
    metadata.write_text("\n".join(lines), encoding="utf-8")
    return data, metadata


class RevisionEndToEndTests(unittest.TestCase):
    def test_external_frequency_transfer_uses_frozen_vocabulary_and_rejects_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_data, source_metadata = write_fixture(root)
            run = root / "source_run"
            main(["--data_root", str(source_data), "--metadata_tsv", str(source_metadata), "--out_dir", str(run),
                  "--method", "frequency", "--cv_folds", "3"])
            original = (run / "frequency_fit.json").read_bytes()
            with self.assertRaises(ValueError):
                transfer_main(["--source_run", str(run), "--data_root", str(source_data), "--metadata_tsv", str(source_metadata),
                               "--out_dir", str(root / "bad_transfer"), "--cpu"])
            self.assertFalse((root / "bad_transfer").exists())
            external = root / "external"
            external.mkdir()
            data, metadata = write_fixture(external, n=6, prefix="independent_")
            out = root / "transfer"
            transfer_main(["--source_run", str(run), "--data_root", str(data), "--metadata_tsv", str(metadata), "--out_dir", str(out), "--cpu"])
            self.assertEqual((run / "frequency_fit.json").read_bytes(), original)
            records = json.loads((out / "external_records.json").read_text(encoding="utf-8"))
            self.assertEqual(records["metrics"]["full_gallery"]["n_patients"], 6)
            self.assertEqual(records["split"], "external")

    def test_calibration_reference_freezes_encoder_and_rejects_changed_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, metadata = write_fixture(root)
            reference = root / "reference"
            reference.mkdir()
            cases = discover_semantic_cases(str(data), str(metadata))
            splits = patient_cross_validation_split(cases, 3, 0, 2026)
            vocab, _ = build_anchor_vocab(splits["train"])
            membership = {name: [case["subject_id"] for case in values] for name, values in splits.items()}
            config = {"z_slices": 3, "roi_size": 16, "feat_dim": 8, "shared_dim": 8,
                      "temperature": 0.07, "target_policy": "region_rules", "experiment_profile": "direct_only"}
            for name, value in (("splits", membership), ("config", config), ("anchor_vocab", vocab)):
                (reference / (name + ".json")).write_text(json.dumps(value), encoding="utf-8")
            encoder = GliomaTopoMoENet(z_slices=3, feat_dim=8, shared_dim=8, routing_enabled=False)
            bank = SemanticPrototypeBank(len(vocab), 8)
            torch.save({"model": encoder.state_dict(), "bank": bank.state_dict(), "anchor_vocab": vocab}, reference / "best_semantic_alignment.pt")
            out = root / "calibration"
            common = ["--data_root", str(data), "--metadata_tsv", str(metadata), "--cv_folds", "3",
                      "--roi_size", "16", "--z_slices", "3", "--feat_dim", "8", "--shared_dim", "8",
                      "--batch_size", "2", "--epochs", "1", "--router_epochs", "1", "--joint_epochs", "1",
                      "--method", "calibration", "--gate", "node", "--reference_run", str(reference), "--cpu"]
            main(common + ["--out_dir", str(out)])
            state = torch.load(out / "best_router.pt", weights_only=True)
            for key, value in encoder.state_dict().items():
                self.assertTrue(torch.equal(state["encoder." + key], value))
            self.assertTrue(torch.equal(state["bank.prototypes"], bank.prototypes))
            with self.assertRaises(ValueError):
                main(common + ["--out_dir", str(root / "bad"), "--target_policy", "all_patient_anchors"])
            self.assertFalse((root / "bad").exists())
            with self.assertRaisesRegex(ValueError, "prefer_registered"):
                main(common + ["--out_dir", str(root / "bad_registration"), "--prefer_registered"])
            self.assertFalse((root / "bad_registration").exists())

    def test_real_nifti_discovery_training_and_artifact_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, metadata = write_fixture(root)
            common = ["--data_root", str(data), "--metadata_tsv", str(metadata), "--cv_folds", "3",
                      "--roi_size", "16", "--z_slices", "3", "--feat_dim", "8", "--shared_dim", "8",
                      "--batch_size", "2", "--epochs", "1", "--router_epochs", "1", "--joint_epochs", "1",
                      "--segmentation_source", "synthetic fixture masks", "--cpu"]
            for method in ("node_frequency", "direct", "global", "multilabel", "mean_positive_proxy", "proxy_anchor_multilabel", "factorized"):
                out = root / method
                main(common + ["--out_dir", str(out), "--method", method, "--projection", "separate", "--gate", "node"])
                self.assertTrue((out / "completed.json").is_file())
                records = json.loads((out / "test_records.json").read_text(encoding="utf-8"))
                self.assertEqual(len(records["scores"]), 15)
                self.assertEqual(records["metrics"]["full_gallery"]["n_patients"], 5)
                self.assertEqual(records["split"], "test")
                self.assertTrue((out / "val_records.json").is_file())
                if method == "factorized":
                    self.assertEqual(records["intervention_audit"]["node_permutation_count"], 5)
                    self.assertIn("direct_metrics", records)
                    external = root / "independent"
                    external.mkdir()
                    external_data, external_metadata = write_fixture(external, n=3, prefix="external_")
                    transfer_main(["--source_run", str(out), "--data_root", str(external_data), "--metadata_tsv", str(external_metadata),
                                   "--out_dir", str(root / "model_transfer"), "--cpu"])
                    transfer_records = json.loads((root / "model_transfer" / "external_records.json").read_text(encoding="utf-8"))
                    self.assertEqual(transfer_records["metrics"]["full_gallery"]["n_patients"], 3)
                with self.assertRaises(FileExistsError):
                    main(common + ["--out_dir", str(out), "--method", method])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
