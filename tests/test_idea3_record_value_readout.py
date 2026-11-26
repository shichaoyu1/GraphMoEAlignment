from copy import deepcopy

import numpy as np
import pandas as pd
import torch

from glioma.cli import idea3_record_value_readout as readout
from glioma.data.utsw_authority import features
from glioma.eval.idea3_role_leakage import TARGET


def patient(index, value, label, mri):
    record = {"IDH": "mutated", TARGET: value, "MGMT": "NA", "IDH_Testing Method": "IHC",
              "scope_eligible": value == "co-deleted"}
    return {"subject_id": str(index), "record": record, "label": label,
            "features": features(record), "active": record["scope_eligible"], "mri": mri}


def batches(rows, source, batch_size):
    for start in range(0, len(rows), batch_size):
        group = rows[start:start + batch_size]
        yield {"molecular": torch.tensor([r["features"] for r in group]),
               "images": torch.tensor([r["mri"] for r in group]).float()[:, None]}


class FrozenEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_encoded = 0

    def encode_mri(self, images):
        self.n_encoded += len(images)
        return images[:, None].expand(-1, 4, 7)


def test_nested_readouts_share_mri_and_add_only_recording_then_value(monkeypatch):
    monkeypatch.setattr(readout, "loader", batches)
    categories = ("NA", "non co-deleted", "co-deleted")
    train = [patient(i, categories[i % 3], i % 3, i // 3) for i in range(18)]
    test = [patient(20 + i, value, i, 1.) for i, value in enumerate(categories)]
    model = FrozenEncoder().eval()
    result = readout.record_value_fold(model, train, test, None, 0, "cpu", 4)
    assert model.n_encoded == len(train) + len(test)
    rows = result[result.readout.eq("X")]
    assert rows.codeletion_measured.tolist() == [False, True, True]
    assert pd.isna(rows.codeletion_positive.iloc[0])
    assert rows.codeletion_positive.iloc[1:].tolist() == [False, True]
    probs = ["p2", "p3", "p4"]
    x, xm, xmv = [result[result.readout.eq(name)][probs].to_numpy() for name in readout.READOUTS]
    np.testing.assert_allclose(x[0], x[1])
    np.testing.assert_allclose(x[1], x[2])
    np.testing.assert_allclose(xm[1], xm[2])
    assert xm[0, 0] > xm[1, 0]
    assert xmv[2, 2] > xmv[1, 2]
    expected = ((result[probs].to_numpy() - np.eye(3)[result.label]) ** 2).mean(axis=1)
    np.testing.assert_allclose(result.brier, expected)


def test_unseen_rows_labels_and_archived_flags_do_not_affect_predictions(monkeypatch):
    monkeypatch.setattr(readout, "loader", batches)
    train = [patient(i, "co-deleted" if i % 2 else "NA", i % 3, i % 3 + i / 100.) for i in range(18)]
    test = [patient(30, "co-deleted", 0, 1.5)]
    changed = deepcopy(test)
    changed[0]["label"] = 2
    changed[0]["active"] = False
    # The diagnosis scope and duplicate target channels cannot enter through the archived report.
    changed[0]["features"][12] = 0.
    changed[0]["features"][3:6] = [0., 0., 1.]
    changed.append(patient(31, "non co-deleted", 1, 1000.))
    model = FrozenEncoder().eval()
    before = readout.record_value_fold(model, train, test, None, 0, "cpu", 4)
    after = readout.record_value_fold(model, train, changed, None, 0, "cpu", 4)
    other = after[after.subject_id.eq("30")]
    np.testing.assert_allclose(before[["p2", "p3", "p4"]], other[["p2", "p3", "p4"]])
