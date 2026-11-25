from argparse import Namespace
import sys

import numpy as np
import pandas as pd

from glioma.cli.train_idea3_matched import full_gate, run_point
from glioma.cli import screen_idea3_matched, train_idea3_matched


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


def test_screen_uses_the_same_run_configuration_as_full(monkeypatch, tmp_path):
    prepared = tmp_path / "prepared"
    split = {role: [role] for role in ("train", "val", "test")}
    manifest = {"rows": [{"subject_id": sid} for sid in split], "splits": [split] * 5}
    calls = []

    def record(_prepared, out, _train, _val, _test, fold, seed, weight, preference, settings, _images, provenance):
        calls.append((out, fold, seed, weight, preference, settings.matrix, settings.epochs,
                      settings.batch_size, settings.device, provenance))
        return pd.DataFrame({"phenotype": [.5]})

    for module in (screen_idea3_matched, train_idea3_matched):
        monkeypatch.setattr(module, "manifest_data", lambda _path: (manifest, "digest"))
        monkeypatch.setattr(module, "run_point", record)
        monkeypatch.setattr(module, "full_gate", lambda *_: {"passed": False})
        monkeypatch.setattr(module, "atomic_csv", lambda *_: None)
        monkeypatch.setattr(module, "save_json", lambda *_: None)
    monkeypatch.setattr(train_idea3_matched, "diagnose", lambda *_: ({}, None))
    monkeypatch.setattr(sys, "argv", ["screen", "--prepared", str(prepared), "--output", str(tmp_path)])
    screen_idea3_matched.main()
    screen = calls[:]
    calls.clear()
    monkeypatch.setattr(sys, "argv", ["full", "--prepared", str(prepared), "--output", str(tmp_path / "full"),
                                     "--matrix", "full"])
    train_idea3_matched.main()
    assert len(screen) == 10 and len(calls) == 45
    assert screen == [r for r in calls if r[2] == 42 and r[3] in (.05, .95)]
