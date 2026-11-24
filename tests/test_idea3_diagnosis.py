import numpy as np
import pandas as pd
import pytest

from glioma.cli.diagnose_idea3_anchor import diagnose
from glioma.cli.replay_idea3_anchor import check_prepared


def test_fold_offsets_move_pooled_ap_with_identical_within_fold_rankings():
    y = np.array([0, 0, 1, 0, 1, 1])
    low = np.array([.25, .27, .30, .70, .74, .81])
    high = low + np.array([.5, .5, .5, -.5, -.5, -.5])
    rows = []
    for method in ("fixed", "probe"):
        for weight in (.05, .95):
            score = high if (method, weight) == ("fixed", .95) else low
            rows.append(pd.DataFrame({"subject_id": list("abcdef"), "fold": [0] * 3 + [1] * 3,
                                      "seed": 42, "weight": weight, "method": method,
                                      "idh": y, "1p19q": y, "mgmt": y, "phenotype": .5,
                                      "p_idh": score, "p_1p19q": score, "p_mgmt": score}))
    report, _ = diagnose(pd.concat(rows), 42, 12)
    delta = {(r["method"], r["scope"]): r["estimate"] for r in report["endpoint_contrasts"]
             if r["field"] == "macro" and r["metric"] == "ap_balanced"}
    assert delta["fixed", "pooled"] < 0
    assert delta["fixed", "fold_mean"] == delta["fixed", "rank_pooled"] == 0
    assert delta["probe", "pooled"] == 0


def test_replay_checks_stable_manifest_data_across_different_paths():
    inputs = {"matrix_size": 64, "metadata_sha256": "same"}
    row = {"subject_id": "A", "anchors": [None, 1, 0], "phenotype": .5,
           "manual_mask": "manual.nii.gz", "phenotype_status": "eligible", "images": ["test.nii.gz"]}
    pilot = {"grid": {}, "splits": [], "fields": ["idh"],
             "inputs": {**inputs, "data_root": "local"}, "rows": [{**row, "source_signature": "local"}]}
    prepared = {**pilot, "inputs": {**inputs, "data_root": "server"}, "rows": [{**row, "source_signature": "server"}]}
    check_prepared(pilot, prepared)
    prepared["rows"][0]["phenotype"] = .6
    with pytest.raises(ValueError, match="do not match"):
        check_prepared(pilot, prepared)
