import json
import os
import tempfile
import unittest

from glioma.cli.aggregate_topomoe_runs import aggregate
from glioma.cli.train_semantic_alignment import _checkpoint_selection, _metric_score


class TopoMoEAggregationTests(unittest.TestCase):
    def _write(self, path, payload):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file)

    def _seed_payloads(self, seed_dir, offset):
        os.makedirs(seed_dir)
        self._write(
            os.path.join(seed_dir, "anchor_vocab.json"),
            [
                {"label": "Pathology A", "source": "Pathology"},
                {"label": "IDH mutated", "source": "Gene"},
            ],
        )
        self._write(
            os.path.join(seed_dir, "test_metrics.json"),
            {
                "checkpoint_type": "routed_best",
                "case_count": 10,
                "direct": {"map": 0.6 + offset},
                "routed": {"map": 0.5 + offset},
            },
        )
        self._write(
            os.path.join(seed_dir, "routing_spectrum.json"),
            {
                "family_names": ["pathology", "molecular", "residual"],
                "by_node": {
                    "Edema": {
                        "mean": {"pathology": 0.5 + offset, "molecular": 0.4 - offset, "residual": 0.1},
                    }
                },
                "by_grade": {
                    "4": {
                        "mean": {"pathology": 0.5 + offset, "molecular": 0.4 - offset, "residual": 0.1},
                    }
                },
            },
        )
        self._write(
            os.path.join(seed_dir, "topomoe_topology.json"),
            {
                "topology_mode": "prior_plus_learned",
                "A_prior": [[1.0, 0.0], [0.0, 1.0]],
                "A_initial": [[0.99, 0.01], [0.01, 0.99]],
                "A_effective": [[0.98 - offset, 0.02 + offset], [0.02 + offset, 0.98 - offset]],
                "diagnostics": {"new_edge_mass": 0.02 + offset},
            },
        )
        self._write(
            os.path.join(seed_dir, "intervention_metrics.json"),
            {
                "baseline": {"map": 0.5 + offset},
                "scenarios": {
                    "remove_pathology_expert": {"delta_map": -0.1 - offset, "delta_mrr": -0.05},
                },
            },
        )

    def test_multiseed_aggregation_and_figures(self):
        with tempfile.TemporaryDirectory() as root:
            paper_root = os.path.join(root, "paper2")
            os.makedirs(paper_root)
            self._seed_payloads(os.path.join(paper_root, "seed_42"), 0.0)
            self._seed_payloads(os.path.join(paper_root, "seed_43"), 0.02)
            manifest = aggregate(root)
            aggregate_dir = os.path.join(root, "aggregate")
            self.assertEqual(manifest["seed_count"], 2)
            self.assertEqual(manifest["checkpoint_types"], ["routed_best"])
            self.assertEqual(manifest["topology_modes"], ["prior_plus_learned"])
            self.assertTrue(os.path.exists(os.path.join(aggregate_dir, "aggregate_metrics.json")))
            self.assertTrue(os.path.exists(os.path.join(aggregate_dir, "topomoe_topology_stability.png")))

    def test_checkpoint_metric_selection(self):
        score, metric = _metric_score({"map": 0.7, "mrr": 0.8}, -1.0)
        self.assertEqual((score, metric), (0.7, "map"))
        score, metric = _metric_score({}, -2.0)
        self.assertEqual((score, metric), (-2.0, "loss_fallback"))
        direct_score, direct = _checkpoint_selection({"map": 0.7}, -2.0, 3)
        routed_score, routed = _checkpoint_selection({"map": 0.4}, direct_score, 3)
        self.assertEqual((direct_score, routed_score), (0.7, 0.4))
        self.assertFalse(direct["fallback"])
        _, fallback = _checkpoint_selection({}, direct_score, 4)
        self.assertTrue(fallback["fallback"])


if __name__ == "__main__":
    unittest.main()
