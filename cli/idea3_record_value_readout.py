"""Fold-local linear grading readouts for the availability and value of 1p/19q."""

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from glioma.cli.utsw_authority import loader
from glioma.data.utsw_authority import FIELDS
from glioma.eval.idea3_record_value import READOUTS
from glioma.eval.idea3_role_leakage import TARGET


@torch.inference_mode()
def _design(model, rows, source, device, batch_size):
    """Build one row per patient, ordered as X, recorded indicator, positive indicator."""
    blocks = []
    for batch in loader(rows, source, batch_size=batch_size):
        mri = model.encode_mri(batch["images"].to(device)).flatten(1).cpu().numpy()
        report = batch["molecular"].cpu().numpy()
        # IDH, MGMT and testing method only. Exclude the target triplet and scope bit.
        blocks.append(np.concatenate([mri, report[:, :3], report[:, 6:12]], axis=1))

    results = [row["record"][TARGET] for row in rows]
    measured = np.isin(results, FIELDS[TARGET])
    positive = np.equal(results, "co-deleted")
    return np.column_stack([np.concatenate(blocks), measured, positive]).astype(np.float64)


def record_value_fold(residual_model, train_rows, test_rows, source, fold, device, batch_size):
    """Fit the three nested readouts on identical train rows and return held-out predictions."""
    if residual_model.training:
        raise ValueError("Use the archived MRI encoder in evaluation mode")
    train_labels = np.array([r["label"] for r in train_rows])
    if set(train_labels) != {0, 1, 2}:
        raise ValueError("The training fold needs all three reference grades")
    train = _design(residual_model, train_rows, source, device, batch_size)
    test = _design(residual_model, test_rows, source, device, batch_size)
    test_labels = np.array([r["label"] for r in test_rows])
    measured = test[:, -2].astype(bool)
    positive = pd.array(test[:, -1].astype(bool), dtype="boolean")
    positive[~measured] = pd.NA  # Unknown and measured-negative remain distinct in the export.
    patients = pd.DataFrame({
        "fold": fold, "subject_id": [r["subject_id"] for r in test_rows], "label": test_labels,
        "codeletion_measured": measured, "codeletion_positive": positive,
    })

    frames = []
    base_width = train.shape[1] - 2
    for added, readout in enumerate(READOUTS):
        width = base_width + added
        scaler = StandardScaler().fit(train[:, :width])
        predictor = LogisticRegression(C=1., solver="lbfgs", max_iter=2000, random_state=101)
        predictor.fit(scaler.transform(train[:, :width]), train_labels)
        p = predictor.predict_proba(scaler.transform(test[:, :width]))
        frames.append(patients.assign(
            readout=readout, p2=p[:, 0], p3=p[:, 1], p4=p[:, 2],
            brier=np.square(p - np.eye(3)[test_labels]).mean(axis=1),
        ))
    return pd.concat(frames, ignore_index=True)
