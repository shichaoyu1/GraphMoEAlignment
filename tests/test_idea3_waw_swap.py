from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from glioma.data.waw_grade import MASKS, SCANS, prepare_images
from glioma.eval.idea3_rule_mri_swap import MODELS
from glioma.eval.idea3_waw_swap import external_results


def test_unavailable_scan_is_a_failed_examination(tmp_path):
    for archive in (MASKS, SCANS):
        with ZipFile(tmp_path / archive, "w"):
            pass
    audit = prepare_images([{"subject_id": "26"}], tmp_path, tmp_path)
    assert not audit["complete"] and audit["accepted"] == []
    assert "26_0_brain_t1n" in audit["errors"]["26"]


def test_ensemble_scores_probabilities_before_patient_bootstrap():
    ids, labels = ["a", "b", "c"], [0, 1, 0]
    rows = []
    for fold in range(5):
        for model in MODELS:
            for i, sid in enumerate(ids):
                for j, donor in enumerate(ids):
                    p = 0.1 * fold + (0 if model == "molecular_projected" else 0.2 * j)
                    rows.append(dict(fold=fold, model=model, subject_id=sid, image_subject_id=donor,
                                     label=labels[i], reference_grade=labels[i]+2, allowed_grades="2|3",
                                     raw_p2=(1-p)*0.8, raw_p3=p*0.8, raw_p4=0.2, p2=1-p, p3=p, p4=0.))
    frame = pd.DataFrame(rows)
    _, _, patients, report = external_results(frame, ids)
    primary = report["models"]["read_only_residual"]
    assert primary["own_advantage"]["n"] == 3
    assert primary["n_pairs"] == 6
    per = patients[patients.model == "read_only_residual"].set_index("subject_id")
    assert per.loc["a", "own_brier"] == pytest.approx(0.2**2)
    assert per.loc["a", "donor_brier"] == pytest.approx((0.4**2 + 0.6**2)/2)
    assert report["models"]["molecular_projected"]["own_advantage"]["estimate"] == pytest.approx(0)
    assert len(primary["own_advantage_by_checkpoint"]) == 5
    # Recompute leave-one-out contrasts from the remaining two patients and their actual MRIs.
    errors = (np.array([0.2, 0.4, 0.6])[None, :] - np.array(labels)[:, None])**2
    leave = [(errors[i, j]+errors[j, i]-errors[i, i]-errors[j, j])/2 for i, j in [(0, 1), (1, 2), (0, 2)]]
    assert primary["leave_one_patient_out_range"] == pytest.approx([min(leave), max(leave)])
    with pytest.raises(ValueError, match="Incomplete or duplicate"):
        external_results(frame[frame.fold != 4], ids)
