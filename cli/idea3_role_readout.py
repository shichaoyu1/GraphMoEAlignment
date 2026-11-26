"""Training-state-matched soft readout on the archived fold-specific MRI encoder."""

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from glioma.cli.utsw_authority import loader
from glioma.data.utsw_authority import FIELDS
from glioma.eval.idea3_role_leakage import TARGET


@torch.inference_mode()
def _paired_inputs(model, rows, source, device, batch_size):
    design = []
    for batch in loader(rows, source, batch_size=batch_size):
        # The encoder sees only the recipient's MRI. Both views share that encoding.
        mri = model.encode_mri(batch["images"].to(device)).flatten(1).cpu().numpy()
        visible = batch["molecular"][:, :12].cpu().numpy()  # excludes diagnosis-derived scope
        complementary = np.concatenate([visible, mri], axis=1)
        absent = complementary.copy()
        absent[:, 3:6] = [0., 0., 1.]
        design.append(np.stack([absent, complementary], axis=1).reshape(2 * len(mri), -1))
    return np.concatenate(design).astype(np.float64)


def matched_readout_fold(residual_model, train_rows, test_rows, source, fold, device, batch_size):
    """Fit on measured train patients only, predict O/C for each measured test patient."""
    train = [r for r in train_rows if r["record"][TARGET] in FIELDS[TARGET]]
    test = [r for r in test_rows if r["record"][TARGET] in FIELDS[TARGET]]
    ids = [r["subject_id"] for r in train + test]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate patient or train/test overlap in the role-matched readout")
    train_labels = np.repeat([r["label"] for r in train], 2)
    if set(train_labels) != {0, 1, 2} or not test:
        raise ValueError("Role-matched readout needs three measured training grades and measured test patients")
    if residual_model.training:
        raise ValueError("Use the archived MRI encoder in evaluation mode")
    x_train = _paired_inputs(residual_model, train, source, device, batch_size)
    scaler = StandardScaler().fit(x_train)
    predictor = LogisticRegression(C=1., solver="lbfgs", max_iter=2000, random_state=101)
    predictor.fit(scaler.transform(x_train), train_labels, sample_weight=np.full(len(train_labels), .5))

    x_test = _paired_inputs(residual_model, test, source, device, batch_size)
    p = predictor.predict_proba(scaler.transform(x_test))
    labels = np.repeat([r["label"] for r in test], 2)
    return pd.DataFrame({
        "fold": fold,
        "subject_id": np.repeat([r["subject_id"] for r in test], 2),
        "label": labels,
        "state": np.tile(["O", "C"], len(test)),
        "p2": p[:, 0], "p3": p[:, 1], "p4": p[:, 2],
        "brier": np.square(p - np.eye(3)[labels]).mean(axis=1),
    })
