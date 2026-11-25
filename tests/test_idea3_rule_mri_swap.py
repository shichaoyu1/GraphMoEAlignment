from dataclasses import replace

import pandas as pd
import pytest
import torch

from glioma.cli.idea3_rule_mri_swap import replay_fold, swap_fold, verify_invariance, verify_replay
from glioma.data.utsw_authority import compile_grade
from glioma.eval.idea3_rule_mri_swap import MODELS, patient_results, summarize, validate_pairs


def toy_batch(ids, grades):
    labels = torch.tensor(grades)
    report = torch.zeros(len(ids), 13)
    report[:, [1, 4, 9, 12]] = 1
    return {"subject_id": ids, "images": labels.float().reshape(-1, 1), "molecular": report,
            "available": torch.ones(len(ids), 5, dtype=torch.bool), "active": torch.ones(len(ids), dtype=torch.bool),
            "label": labels}


def toy_prediction(evidence, use_mri):
    compiled = compile_grade(evidence)
    grade3 = .1 + .8 * evidence.images.flatten() if use_mri else torch.full_like(evidence.images.flatten(), .5)
    p = torch.stack([1 - grade3, grade3, torch.zeros_like(grade3)], dim=1)
    raw = .75 * p
    raw[:, 2] = .25
    return {**compiled, "p0": raw, "p_star": p}


def test_swap_reuses_recipient_reference_and_weights_patients_equally():
    models = {name: lambda e, name=name: toy_prediction(e, name != "molecular_projected") for name in MODELS}
    clean, swapped, patients = [], [], {}
    for fold, batch in enumerate((toy_batch(["a", "b"], [0, 1]), toy_batch(["c", "d", "e"], [0, 0, 0]))):
        ids = batch["subject_id"]
        patients[fold] = ids
        clean.append(replay_fold(models, [batch], "cpu", fold))
        swapped.append(swap_fold(models, [batch], ids, "cpu", fold, 2))
    frame = pd.concat(clean + swapped, ignore_index=True)
    validate_pairs(frame, patients)
    with pytest.raises(ValueError, match="Incomplete or duplicate"):
        validate_pairs(frame.iloc[:-1], patients)
    pairs, per_patient = patient_results(frame)
    verify_invariance(pairs)
    report = summarize(per_patient, repetitions=100)
    primary = report["models"]["read_only_residual"]["own_advantage"]
    assert primary["n"] == 5
    assert primary["estimate"] == pytest.approx(2 * .8 / 5)
    assert primary["leave_one_fold_out_range"] == pytest.approx([0, .8])
    assert report["models"]["molecular_projected"]["own_advantage"]["ci95"] == [0., 0.]
    assert report["own_brier_contrasts"]["molecular_projected"]["estimate"] == pytest.approx(.24)
    assert report["models"]["read_only_residual"]["n_pairs"] == 8
    with pytest.raises(ValueError, match="protected probabilities"):
        verify_invariance(pairs.assign(raw_p4=pairs.raw_p4 + .01 * pairs.condition.eq("donor")))


def test_clean_replay_checks_projection_metadata_and_raw_probabilities():
    batch = toy_batch(["a", "b"], [0, 1])
    models = {"read_only_residual": lambda e: toy_prediction(e, True)}
    clean = replay_fold(models, [batch], "cpu", 0)
    assert verify_replay(clean, clean.iloc[::-1]) == {"read_only_residual": 0.}
    with pytest.raises(ValueError, match="Clean replay exceeds"):
        verify_replay(clean, clean.assign(raw_p4=clean.raw_p4 + .00002))
    with pytest.raises(ValueError, match="archived allowed_grades"):
        verify_replay(clean, clean.assign(allowed_grades="2|3|4"))


def test_predictor_cannot_change_rule():
    batch = toy_batch(["a", "b"], [0, 1])
    models = {"acf": lambda e: toy_prediction(replace(e, active=~e.active), True)}
    with pytest.raises(ValueError, match="Compiled rule changed"):
        replay_fold(models, [batch], "cpu", 0)
