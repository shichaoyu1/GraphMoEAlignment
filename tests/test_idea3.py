import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import average_precision_score

from glioma.cli.idea3_anchor import preference_summary
from glioma.cli.idea3_authority import build_derangement, withdraw_report
from glioma.data.idea3 import fixed_view, grid, phenotype_target
from glioma.data.utsw_authority import GradeEvidence, features, molecular_record
from glioma.eval.idea3 import check_pairs, hypervolume, patient_draws, weighted_ap
from glioma.models.idea3 import ReadOnlyGradeResidual, task_losses
from glioma.training.idea3 import INTERMEDIATE, WEIGHTS


def test_mask_only_changes_phenotype_target(tmp_path):
    affine = np.eye(4)
    intensity = np.arange(512, dtype=np.float32).reshape(8, 8, 8)
    image = nib.Nifti1Image(intensity, affine)
    mask = np.zeros_like(intensity)
    mask.flat[:3] = [1, 2, 4]
    file = tmp_path / "tumorseg_manual_correction.nii.gz"
    nib.save(nib.Nifti1Image(mask, affine), file)
    view = fixed_view([image] * 4, size=4)
    assert phenotype_target([file], grid(image))["phenotype"] == 1 / 3
    mask.flat[0] = 2
    nib.save(nib.Nifti1Image(mask, affine), file)
    assert phenotype_target([file], grid(image))["phenotype"] == 2 / 3
    np.testing.assert_array_equal(view, fixed_view([image] * 4, size=4))
    mask.flat[0] = 3
    nib.save(nib.Nifti1Image(mask, affine), file)
    target = phenotype_target([file], grid(image))
    assert target["phenotype"] is None and target["unmapped_3_voxels"] == 1
    mask.flat[0] = 5
    nib.save(nib.Nifti1Image(mask, affine), file)
    assert phenotype_target([file], grid(image))["phenotype_status"] == "unmapped_labels"
    alternate = tmp_path / "rtumorseg_manual_correction.nii.gz"
    mask.flat[0] = 1
    nib.save(nib.Nifti1Image(mask, affine), alternate)
    assert phenotype_target([file, alternate], grid(image))["phenotype"] == 1 / 3


def test_weighted_ap_handles_ties_missing_labels_and_repeated_patients():
    labels = np.array([0, 1, 1, 0, -1])
    scores = np.array([.7, .7, .8, .5, .99])
    weights = np.array([[1, 1, 1, 1, 1], [0, 2, 1, 2, 0], [2, 0, 0, 2, 1]])
    obtained = weighted_ap(labels, scores, weights)
    for i in range(2):
        valid = labels >= 0
        assert obtained[i] == pytest.approx(average_precision_score(labels[valid], scores[valid],
                                                                    sample_weight=weights[i, valid]))
    assert np.isnan(obtained[2])
    assert hypervolume([(.2, .9), (.5, .6), (.8, .4), (.3, .3)]) == pytest.approx(.48)


def test_loss_masks_missing_targets_without_gradients():
    logits = torch.ones((2, 3), requires_grad=True)
    prediction = torch.tensor([.7, .3], requires_grad=True)
    anchors = torch.tensor([[-1., 1., 0.], [1., 0., -1.]])
    target = torch.tensor([float("nan"), .2])
    anchor, phenotype = task_losses(logits, prediction, anchors, target)
    (anchor + phenotype).backward()
    assert phenotype.item() == pytest.approx(.1)
    assert not logits.grad[anchors < 0].any() and prediction.grad[0] == 0


def test_read_only_residual_preserves_rule_and_report_state():
    torch.manual_seed(101)
    record = molecular_record({"IDH": "mutated", "1p19Q CODEL": "co-deleted", "IDH_Testing Method": "NGS",
                               "Tumor Type": "OLIGODENDROGLIOMA, NOS"})
    evidence = GradeEvidence(torch.empty(2, 0), torch.tensor([features(record)] * 2),
                             torch.ones((2, 5), dtype=torch.bool), torch.tensor([True, False]))
    model = ReadOnlyGradeResidual().eval()
    feats = torch.rand(2, 4, 7)
    a = model.predict_from_features(evidence, feats)
    b = model.predict_from_features(evidence, feats + 1)
    assert torch.equal(a["designated_state"], b["designated_state"])
    assert a["p0"][0, 2] == b["p0"][0, 2] and a["p_star"][0, 2] == b["p_star"][0, 2] == 0
    assert a["p_star"][0, 0] != b["p_star"][0, 0]
    assert torch.allclose(a["p_star"].sum(1), torch.ones(2))
    withdrawn = model.predict_from_features(withdraw_report(evidence), feats)
    assert withdrawn["allowed"].all() and not withdrawn["active"].any()
    permutation = build_derangement(20, 1)
    assert sorted(permutation) == list(range(20)) and not np.any(permutation == np.arange(20))


def test_invalid_pairing_rejected():
    rows = pd.DataFrame({"subject_id": ["a", "b"], "fold": [0, 1], "label": [0, 1]})
    shifted = rows.copy()
    shifted.loc[0, "fold"] = 2
    with pytest.raises(ValueError, match="same patients, folds and targets"):
        check_pairs(rows, shifted, ["label"])


def test_gated_stage_compares_identical_patients_at_held_out_weights():
    targets = np.tile([0., 1.], 4)
    rows = []
    for method, grid in (("fixed", WEIGHTS), ("conditioned", (*WEIGHTS, *INTERMEDIATE))):
        for w in grid:
            rows.append(pd.DataFrame({"subject_id": list("abcdefgh"), "fold": [0] * 8, "seed": [42] * 8,
                                      "weight": w, "method": method, "idh": targets, "1p19q": targets,
                                      "mgmt": targets, "phenotype": targets, "p_idh": .5 + w * (targets - .5),
                                      "p_1p19q": .5 + w * (targets - .5), "p_mgmt": .5 + w * (targets - .5),
                                      "pred_phenotype": .5 + w * (targets - .5)}))
    report, intermediate = preference_summary(pd.concat(rows), patient_draws(8, 15))
    assert len(report["fixed"]["weights"]) == len(WEIGHTS)
    assert len(report["conditioned"]["weights"]) == len(WEIGHTS) + len(INTERMEDIATE)
    assert report["conditioned"]["hypervolume"] == pytest.approx(report["interpolation"]["hypervolume"])
    for result, w in zip(intermediate, INTERMEDIATE):
        assert result.weight.eq(w).all()
        np.testing.assert_allclose(result.pred_phenotype, .5 + w * (targets - .5))
