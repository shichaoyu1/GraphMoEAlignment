from argparse import Namespace

import numpy as np
import pandas as pd

from glioma.cli.train_idea3_matched import full_gate, run_point


def test_matched_point_exports_and_resumes(tmp_path):
    prepared = tmp_path / "prepared"
    cache = prepared / "cache"
    cache.mkdir(parents=True)
    rng = np.random.default_rng(42)
    rows = []
    for i in range(6):
        sid = f"synthetic{i}"
        np.savez(cache / f"{sid}.npz", images=rng.random((4, 64, 64, 64), dtype=np.float32).astype(np.float16))
        rows.append({"subject_id": sid, "anchors": [i % 2] * 3, "phenotype": i / 6})
    args = Namespace(epochs=1, batch_size=2, device="cpu", matrix="verify")
    params = (prepared, tmp_path / "verify", rows[:2], rows[2:4], rows[4:], 0, 42, .05, .05, args, {}, {})
    result = run_point(*params)
    assert len(result) == 12
    assert {"selected", "final", "selected_probe", "final_probe", "selected_raw_probe", "final_raw_probe"} == set(result.method)
    pd.testing.assert_frame_equal(run_point(*params), result)


def test_full_gate_resamples_same_patients_across_seeds():
    rows = []
    for seed in (42, 43, 44):
        for weight in (.05, .95):
            for i in range(8):
                y = i % 2
                p = .5 if weight == .05 else .2 + .6 * y
                rows.append(dict(seed=seed, weight=weight, method="final", fold=i // 4, subject_id=str(i),
                                 idh=y, mgmt=y, phenotype=.4, p_idh=p, p_mgmt=p, pred_phenotype=.3 if weight == .05 else .1,
                                 **{"1p19q": y, "p_1p19q": p}))
    result = full_gate(pd.DataFrame(rows), (42, 43, 44), 30)
    assert result["passed"] and result["n_patients"] == 8
