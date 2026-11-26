from copy import deepcopy

import numpy as np
import pytest
import torch

from glioma.cli import idea3_role_readout as readout
from glioma.data.utsw_authority import features, rule_active
from glioma.eval.idea3_role_leakage import TARGET, audit_leakage


def _row(i, label, positive):
    record = {"IDH": "mutated", TARGET: "co-deleted" if positive else "non co-deleted",
              "MGMT": "NA", "IDH_Testing Method": "IHC", "scope_eligible": positive,
              "curated_record_accepted": True, "assay_validity": "unreported"}
    active = rule_active(record)
    return {"subject_id": str(i), "label": label, "features": features(record),
            "record": record, "active": active, "molecular_candidate": positive,
            "reference_conflict": active and label == 2}


def test_leakage_audits_derived_flags_and_cached_features():
    rows = [_row(i, i % 3, i % 2 == 0) for i in range(10)]
    manifest = {"rows": rows, "folds": {r["subject_id"]: i % 5 for i, r in enumerate(rows)}}
    audit = audit_leakage(manifest)
    assert audit["proxy_by_measured_target"]["molecular_candidate"]["true"]["co-deleted"] == 5
    assert audit["fold_input_support"]["1"]["absent"]["without_exact_train_match"] == 2
    rows[0]["features"][3:6] = [0, 0, 1]
    with pytest.raises(ValueError, match="source record"):
        audit_leakage(manifest)


def test_readout_pairs_differ_only_by_target_and_ignores_test_labels(monkeypatch):
    class Encoder(torch.nn.Module):
        def encode_mri(self, images):
            return images

    def loader(rows, source, batch_size):
        return [{"images": torch.ones(len(rows), 4, 7),
                 "molecular": torch.tensor([r["features"] for r in rows])}]

    monkeypatch.setattr(readout, "loader", loader)
    encoder = Encoder().eval()
    train = [_row(i, i % 3, i % 3 == 0) for i in range(18)]
    test = [_row(18, 0, True), _row(19, 1, False)]
    changed = deepcopy(test)
    for row in changed:
        row["label"] = 2
        row["features"][12] = 1 - row["features"][12]
        row["active"] = not row["active"]
    result = readout.matched_readout_fold(encoder, train, test, None, 0, "cpu", 4)
    other = readout.matched_readout_fold(encoder, train, changed, None, 0, "cpu", 4)
    columns = ["p2", "p3", "p4"]
    assert result["state"].tolist() == ["O", "C", "O", "C"]
    np.testing.assert_allclose(result[columns], other[columns])
    np.testing.assert_allclose(result.loc[0, columns].to_numpy(float), result.loc[2, columns].to_numpy(float))
    assert result.loc[1, "p2"] > result.loc[3, "p2"]
    np.testing.assert_allclose(result["brier"], ((result[columns] - np.eye(3)[result.label]) ** 2).mean(axis=1))
    with pytest.raises(ValueError, match="overlap"):
        readout.matched_readout_fold(encoder, train, train[:1], None, 0, "cpu", 4)
