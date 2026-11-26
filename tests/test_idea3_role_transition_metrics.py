import pandas as pd
import pytest

from glioma.eval.idea3_role_transition import analyze_states, compare_swap_states, paired_effect, report_markdown
from glioma.eval.idea3_rule_mri_swap import MODELS


def _states():
    rows = []
    for fold, subject, label, measured, eligible in (
        (0, "a", 0, True, True), (0, "b", 1, True, True),
        (1, "c", 2, True, False), (1, "d", 2, False, False),
    ):
        for model in MODELS:
            for state in ("O", "C", "A"):
                active = state == "A" and eligible
                # In fold 0 the visible field identifies the grade. In fold 1 it adds nothing.
                p = (0., 0., 1.)
                if fold == 0 and state != "O":
                    p = (1., 0., 0.) if label == 0 else (0., 1., 0.)
                rows.append(dict(fold=fold, subject_id=subject, model=model, state=state, label=label,
                                 codeletion_measured=measured, archive_active=eligible, active=active,
                                 allowed_grades="2|3" if active else "2|3|4", status=0 if active else 1,
                                 raw_p2=p[0], raw_p3=p[1], raw_p4=p[2], p2=p[0], p3=p[1], p4=p[2]))
    return pd.DataFrame(rows)


def test_soft_effect_matches_three_class_mean_and_patient_denominator():
    frame = _states().sample(frac=1, random_state=17)
    patients, summary = analyze_states(frame, repetitions=120)
    assert len(patients) == 16
    assert summary["measured_reference_grades"] == {"2": 1, "3": 1, "4": 1}
    assert summary["structure"]["n_active"] == 2
    effect = summary["models"]["read_only_residual"]["soft_advantage"]
    assert effect["n"] == 3
    assert effect["estimate"] == pytest.approx(4 / 9)
    assert effect["ci95"] == pytest.approx([4 / 9, 4 / 9])
    assert effect["leave_one_fold_out"] == pytest.approx({"0": 0, "1": 2 / 3})
    assert "O minus C" in report_markdown(summary)
    measured = patients.query("model == 'read_only_residual' and codeletion_measured")
    assert paired_effect(measured.set_index(["fold", "subject_id"]).soft_advantage, repetitions=120) == effect
    single = paired_effect(measured.iloc[:1].set_index(["fold", "subject_id"]).soft_advantage, repetitions=5)
    assert single["ci95"] == [single["estimate"]] * 2
    assert single["leave_one_fold_out_range"] is None
    summary.update(study_mode="smoke", paired_state_linear_readout=single,
                   alternate_minus_own_field_brier3={"read_only_residual": single})
    table = report_markdown(summary)
    assert "Short technical run" in table
    assert "Fold-trained linear readout" in table
    assert "Alternate measured-field" in table


def test_states_reject_incomplete_pairs_and_stale_activation():
    frame = _states()
    with pytest.raises(ValueError, match="Missing paired"):
        analyze_states(frame.iloc[1:])
    keep = frame.state.eq("C") & frame.archive_active
    with pytest.raises(ValueError, match="allowed grades"):
        analyze_states(frame.assign(active=frame.active | keep))
    with pytest.raises(ValueError, match="Invalid probabilities"):
        analyze_states(frame.assign(p3=float("nan")))


def test_swap_interaction_matches_three_class_loss_in_both_states():
    frames = {}
    for state in ("A", "C"):
        patients = _states().query("state == @state and archive_active").copy()
        patients["image_subject_id"] = patients.subject_id
        donors = patients.copy()
        donors["image_subject_id"] = donors.subject_id.map({"a": "b", "b": "a"})
        if state == "A":
            donors[["p2", "p3"]] = donors[["p3", "p2"]].to_numpy()
        frames[state] = pd.concat([patients, donors], ignore_index=True)
    paired, result = compare_swap_states(frames["A"], frames["C"], 30)
    assert result["n_patients"] == 2
    assert paired.n_donors.eq(1).all()
    assert result["models"]["read_only_residual"]["estimate"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="MRI pairs"):
        compare_swap_states(frames["A"], frames["C"].iloc[:-1], 30)
