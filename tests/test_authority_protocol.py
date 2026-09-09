"""Acceptance tests for resume, data coupling, status accounting and server manifests."""

import dataclasses
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from glioma.cli.aggregate_authority import cluster_rows, paired_comparisons
from glioma.cli.run_authority_protocol import development_jobs, formal_jobs, freeze_development, load_frozen
from glioma.cli.train_authority import atomic_json, canonical_hash, run_training, source_hash
from glioma.data.authority_benchmarks import AuthoritySynthetic, intervention_grid
from glioma.eval.authority_diagnostics import evaluate_suite, metrics, predict
from glioma.models.authority_fusion import AuthorityFusion, authority_loss, mean_augmented_spd
from glioma.modules.authority_projection import GeneralAuthorityProjection, project_disjoint_orders


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(8)

    def test_formal_counts_and_no_implicit_learning_rate(self):
        development, main, attribution = development_jobs(), formal_jobs("main"), formal_jobs("attribution")
        self.assertEqual((len(development), len(main), len(attribution)), (60, 300, 210))
        self.assertEqual({j["config"]["lr"] for j in main}, {None})
        self.assertEqual({j["config"]["data_seed"] for j in development}, {41})
        self.assertEqual({j["config"]["data_seed"] for j in main}, set(range(42,47)))
        self.assertEqual({j["config"]["model_seed"] for j in main}, {101,102,103})
        self.assertEqual(sum(j["config"]["attribution"] == "role_shift" for j in attribution), 30)
        self.assertEqual(sum(j["config"]["train_n"] == 2048 for j in attribution), 60)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                load_frozen(directory)
            with self.assertRaises(ValueError):
                freeze_development(directory)

    def test_fixed_label_interventions_and_only_target_switch_changes_labels(self):
        for task in ("s1", "s2"):
            generator = AuthoritySynthetic(task, count=32)
            clean = generator.materialize()
            for name, parameters in intervention_grid(task):
                changed = generator.materialize(**parameters)
                self.assertEqual(clean.sample_ids, changed.sample_ids)
                if name != "target_switch":
                    self.assertTrue(torch.equal(clean.labels, changed.labels), name)

    def test_freeze_uses_only_complete_equal_budget_validation_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for job in development_jobs():
                directory = root / "development" / job["id"]
                directory.mkdir(parents=True)
                config = dict(job["config"],source_hash=source_hash(),rule_version="authority-v1",evaluation_mode="none")
                config["config_hash"] = canonical_hash(config)
                atomic_json(directory/"config.json",config)
                atomic_json(directory/"DONE.json",dict(source_hash=source_hash(),config_hash=config["config_hash"],
                                                       best_validation_brier=abs(config["lr"]-.0003)))
            frozen = freeze_development(root)
            self.assertTrue(all(value["lr"] == .0003 for value in frozen["selected"].values()))
            self.assertEqual(frozen, load_frozen(root))
            self.assertEqual(len(json.loads((root/"main_manifest.json").read_text())),300)
            frozen["selected"]["s1/acf"]["lr"] = .9
            atomic_json(root/"frozen_protocol.json",frozen)
            with self.assertRaises(ValueError):
                load_frozen(root)

    def test_mean_information_survives_and_box_projection_pools_before_clipping(self):
        token = torch.randn(8,16,7)
        self.assertFalse(torch.allclose(mean_augmented_spd(token), mean_augmented_spd(token+2)))
        projected = project_disjoint_orders(torch.tensor([[-3., .5, .4]]), torch.tensor([[1]]), torch.tensor([[True]]))
        torch.testing.assert_close(projected, torch.tensor([[0.,0.,.4]]))

    def test_reliability_only_changes_auxiliary_evidence_in_both_tasks(self):
        for task in ("s1", "s2"):
            generator = AuthoritySynthetic(task,count=12)
            a, b = generator.materialize(rb=.6), generator.materialize(rb=.95)
            row = torch.arange(12)[:,None]
            owner = a.evidence.authority_sources
            torch.testing.assert_close(a.evidence.tokens[row,0,owner],b.evidence.tokens[row,0,owner],atol=0,rtol=0)
            torch.testing.assert_close(a.evidence.source_quality[row,owner],b.evidence.source_quality[row,owner],atol=0,rtol=0)
            model = AuthorityFusion(task).eval()
            torch.testing.assert_close(model(a.evidence).protected_states,model(b.evidence).protected_states,atol=1e-6,rtol=0)

    def test_invalid_sources_exit_fusion_and_unknown_has_no_hard_rule(self):
        for task in ("s1", "s2"):
            generator = AuthoritySynthetic(task, count=16)
            invalid = generator.materialize(event="invalid")
            e = invalid.evidence
            tokens, statements, quality = e.tokens.clone(), e.statements.clone(), e.source_quality.clone()
            for source in range(4):
                unavailable = e.verified_validity[:,source] == 0
                tokens[unavailable,0,source] += 8
                statements[unavailable,source] ^= 1
                quality[unavailable,source] = .99
            model = AuthorityFusion(task).eval()
            a = model(e)
            b = model(dataclasses.replace(e,tokens=tokens,statements=statements,source_quality=quality))
            torch.testing.assert_close(a.probabilities, b.probabilities, atol=1e-6, rtol=0)
            unknown = generator.materialize(event="unknown")
            result = predict(model, unknown)
            report = metrics(result, unknown)
            self.assertIsNone(report["avr"])
            self.assertEqual(report["avr_denominator"], 0)
            self.assertEqual(report["prediction_coverage"], 1.)
            all_missing = generator.materialize(event="all_missing")
            report = metrics(predict(model, all_missing), all_missing)
            self.assertIsNone(report["brier"])
            self.assertIsNone(report["risk_at_coverage_0.5"])
            self.assertEqual(report["prediction_coverage"], 0.)

    def test_solver_failure_is_distinct_and_reaches_model_output(self):
        def fail(*args, **kwargs):
            raise RuntimeError("Injected numerical failure")
        with patch("glioma.modules.authority_projection._convex_layer", return_value=fail):
            batch = AuthoritySynthetic("s1",count=3).materialize()
            result = AuthorityFusion("s1",projection_backend="general")(batch.evidence)
            self.assertEqual(result.status.tolist(), [4,4,4])
            torch.testing.assert_close(result.p_star, result.p0)

    @unittest.skipUnless(importlib.util.find_spec("cvxpylayers"), "Optional general projection dependencies unavailable")
    def test_general_model_matches_analytic_and_backpropagates_in_both_tasks(self):
        for task in ("s1", "s2"):
            batch = AuthoritySynthetic(task,count=4).materialize()
            model = AuthorityFusion(task)
            analytic = model(batch.evidence)
            model.projection_backend = "general"
            general = model(batch.evidence)
            self.assertEqual(general.status.tolist(),[0]*4)
            torch.testing.assert_close(analytic.p_star,general.p_star,atol=1e-6,rtol=0)
            authority_loss(general,batch.labels,task,"acf").backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_epoch_boundary_resume_matches_uninterrupted_training(self):
        config = dict(task="s1", variant="acf", train_n=40, val_n=16, test_n=16, epochs=2, batch_size=16)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_training(config, root/"continuous", evaluate=False)
            run_training(config, root/"resumed", evaluate=False, stop_after=1)
            result = run_training(config, root/"resumed", evaluate=False)
            a = torch.load(root/"continuous/last.pt",weights_only=False)
            b = torch.load(root/"resumed/last.pt",weights_only=False)
            self.assertEqual(a["best_brier"], b["best_brier"])
            for key in a["model"]:
                torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)
            self.assertEqual(result, run_training(config, root/"resumed", evaluate=False))
            with self.assertRaises(ValueError):
                run_training(dict(config, lr=.0003), root/"resumed", evaluate=False)

    def test_paired_event_artifacts_and_bypass_positive_control(self):
        with tempfile.TemporaryDirectory() as temp:
            summary = evaluate_suite(AuthorityFusion("s1"), AuthoritySynthetic("s1",split="test",count=6), temp,
                                     {"data_seed":41,"model_seed":100}, batch_size=3)
            self.assertEqual(len(summary), 42)
            self.assertEqual(summary["conflict_strength_8"]["protected_shift_max"], 0.)
            self.assertGreater(summary["bypass"]["protected_shift_max"], 1e-5)
            self.assertIsNone(summary["invalid"]["avr"])
            with np.load(Path(temp)/"target_switch.npz",allow_pickle=False) as saved:
                self.assertEqual(len(saved["event_ids"]),6)
                self.assertFalse(saved["shift_eligible"].any())
                self.assertFalse(np.array_equal(saved["labels_before"],saved["labels_after"]))

    def test_cluster_unit_is_data_seed_not_events_or_model_runs(self):
        rows = []
        for seed in range(42,47):
            for model in (101,102,103):
                for variant, value in (("acf",seed/100), ("learned_joint",seed/100+.1)):
                    rows.append(dict(phase="main",task="s1",variant=variant,attribution="base",event="clean",
                                     data_seed=seed,model_seed=model,brier=value))
        clusters = cluster_rows(rows)
        self.assertEqual(len(clusters),10)
        pairs, data_pairs, intervals = paired_comparisons(rows)
        self.assertEqual((len(pairs),len(data_pairs)), (15,5))
        brier = next(row for row in intervals if row["metric"] == "brier")
        self.assertEqual(brier["independent_data_seeds"],5)
        self.assertAlmostEqual(brier["mean_difference"],-.1)


if __name__ == "__main__":
    unittest.main()
