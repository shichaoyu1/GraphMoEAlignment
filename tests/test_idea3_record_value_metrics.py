import pandas as pd
import pytest

from glioma.eval.idea3_record_value import READOUTS, report_markdown, summarize


def _predictions():
    rows = []
    for sid, fold, positive in (("a", 0, True), ("b", 0, False), ("c", 1, True), ("d", 1, None)):
        for readout in READOUTS:
            # X fails for everyone. M corrects the unreported case. V corrects the two positives
            # and worsens the unreported case, which is deliberately outside the primary estimand.
            correct = ((readout == "X+M" and positive is None)
                       or (readout == "X+M+V" and positive is True))
            rows.append(dict(subject_id=sid, fold=fold, label=0, codeletion_measured=positive is not None,
                             codeletion_positive=positive, readout=readout, p2=float(correct),
                             p3=float(not correct), p4=0.))
    return pd.DataFrame(rows).sample(frac=1, random_state=8)


def test_nesting_effect_uses_measured_patients_and_complete_availability_contrast():
    results = summarize(_predictions(), repetitions=120)
    assert (results["n_positive"], results["n_negative"], results["n_unreported"]) == (2, 1, 1)
    value = results["value_given_record"]
    assert value["n"] == 3 and value["estimate"] == pytest.approx(4 / 9)
    assert value["leave_one_fold_out"] == pytest.approx({"0": 2 / 3, "1": 1 / 3})
    availability = results["record_availability"]
    assert availability["n"] == 4 and availability["estimate"] == pytest.approx(1 / 6)
    assert results["brier3"]["measured"] == pytest.approx({"X": 2 / 3, "X+M": 2 / 3, "X+M+V": 2 / 9})
    assert results["brier3"]["all"] == pytest.approx({"X": 2 / 3, "X+M": 1 / 2, "X+M+V": 1 / 3})
    assert "进一步降低" in report_markdown(results)
    results["study_mode"] = "smoke"
    assert "短测" in report_markdown(results)


def test_incomplete_readout_for_unreported_patient_is_rejected():
    rows = _predictions()
    missing = rows.subject_id.eq("d") & rows.readout.eq("X+M+V")
    with pytest.raises(ValueError, match="all three readouts"):
        summarize(rows[~missing], repetitions=10)
