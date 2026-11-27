import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from glioma.eval.idea3_mri_mix import frozen_probabilities, score_outer, summarize
from glioma.eval.idea3_rule_mri_swap import MODELS
from glioma.objectives.idea3_mri_mix import BUDGETS, READ_ONLY, fit_development, fit_mixture


def example(fold=0):
    rng = np.random.default_rng(31 + fold)
    ids = [f"{fold}-{i}" for i in range(8)]
    labels = np.array([0, 1] * 4)
    report = rng.uniform(.15, .85, len(ids))
    pairs = []
    for i, sid in enumerate(ids):
        for j, image in enumerate(ids):
            q = np.clip(.3 * report[i] + .7 * (.1 + .8 * labels[j]) + rng.normal(0, .07), 0, 1)
            for model in MODELS:
                val = report[i] if model == "molecular_projected" else q
                pairs.append(dict(fold=fold, subject_id=sid, image_subject_id=image, label=labels[i],
                                  reference_grade=labels[i]+2, allowed_grades="2|3", model=model,
                                  p2=1-val, p3=val, p4=0., raw_p2=.8*(1-val), raw_p3=.8*val, raw_p4=.2))
    rows = [dict(subject_id=sid, label=label) for sid, label in zip(ids, labels)]
    return pd.DataFrame(pairs), rows


def test_continuous_zero_budget_solution_and_unlimited_endpoint():
    # On the swap observations excess risk is a^2 - 0.6a. On own MRI a=1 is optimal.
    x = np.ones((1, 1))
    assert fit_mixture((x, np.array([1.])), (x, np.array([.3])), 0.)[0] == pytest.approx(.6)
    assert fit_mixture((x, np.array([1.])), (x, np.array([.3])), None)[0] == pytest.approx(1.)
    assert fit_mixture((x, np.array([1.])), (x, np.array([0.])), 0.)[0] == 0


def test_degenerate_minima_use_donor_risk_then_equal_weights():
    own = (np.array([[.5, .5], [.5, .5]]), np.array([.5, .5]))
    donor = (np.array([[1., 0.], [1., 0.]]), np.full(2, .2))
    np.testing.assert_allclose(fit_mixture(own, donor, 0.), [.2, .8])
    np.testing.assert_allclose(fit_mixture(own, own, None), [.5, .5])
    flat = (np.zeros((2, 2)), np.ones(2))
    np.testing.assert_array_equal(fit_mixture(flat, flat, 0.), [0., 0.])
    np.testing.assert_allclose(fit_mixture(flat, donor, 0.), [.2, .2])


def test_adaptive_beats_dense_feasible_grid_and_constant_development_fits():
    pairs, ids = example()
    dev, _ = frozen_probabilities(pairs, ids, 0)
    solutions = pd.DataFrame(fit_development(dev)).set_index(["method", "budget"])
    grid = np.linspace(0, 1, 81)
    a, b = np.meshgrid(grid, grid)
    report = dev.molecular_projected.to_numpy()[:, None]
    delta = dev[READ_ONLY].to_numpy()[:, None] - report
    q = report + (a.ravel() + (b-a).ravel() * abs(delta)) * delta
    loss = (q - dev.label.to_numpy()[:, None])**2
    own = dev.condition.eq("own").to_numpy()
    for name, budget in BUDGETS.items():
        adaptive = solutions.loc[("adaptive", name)]
        valid = loss[~own].mean(axis=0) <= adaptive.dev_report_brier + (np.inf if budget is None else budget)
        assert adaptive.dev_own_brier <= loss[own].mean(axis=0)[valid].min() + 1e-10
        assert adaptive.dev_own_brier <= solutions.loc[("constant", name)].dev_own_brier + 1e-10
        if budget is not None:
            assert adaptive.dev_swap_brier <= adaptive.dev_report_brier + budget + 1e-12
    # Replicating a single recipient's whole donor pool cannot change its weight in the fit.
    copied = dev[dev.condition.eq("donor") & dev.subject_id.eq(ids[0]["subject_id"])]
    extra = pd.DataFrame(fit_development(pd.concat([dev, copied]))).set_index(["method", "budget"])
    np.testing.assert_allclose(solutions[["a", "b"]], extra[["a", "b"]], atol=1e-8)


def test_frozen_parameters_transfer_to_outer_patients_and_score_paired_intervals():
    data = [frozen_probabilities(*example(fold), fold)[0] for fold in range(2)]
    predictions = []
    for fold, outer in enumerate(data):
        fits = fit_development(data[1-fold])
        scored = score_outer(outer, fits, .4)
        changed = score_outer(outer.assign(label=1-outer.label), fits, .4)
        np.testing.assert_array_equal(scored[["mri_weight", "p3"]], changed[["mri_weight", "p3"]])
        predictions.append(scored)
    all_predictions = pd.concat(predictions, ignore_index=True)
    patients, report = summarize(all_predictions, 200)
    assert report["n_patients"] == 16
    group = patients[patients.budget.eq("0")]
    own = group.pivot(index=["fold", "subject_id"], columns="method", values="own_brier")
    p = report["primary"]
    assert p["constant_minus_adaptive_own"]["estimate"] == pytest.approx((own.constant - own.adaptive).mean())
    expected = group[group.method.eq("adaptive")].swap_brier.mean() - own.report_only.mean()
    assert p["adaptive_swap_minus_report"]["estimate"] == pytest.approx(expected)
    assert report["risks"]["0"]["report_only"]["swap_minus_report"]["ci95"] == [0., 0.]
    assert set(p["constant_minus_adaptive_own"]["leave_one_fold_out"]) == {"0", "1"}
    assert p["supported"] == (p["adaptive_swap_minus_report"]["ci95"][1] <= 0
                              and p["constant_minus_adaptive_own"]["ci95"][0] > 0)
    assert np.all(all_predictions.p4 == 0)


def test_frozen_inputs_reject_wrong_donor_and_changed_protected_branch():
    pairs, ids = example()
    broken = pairs.copy()
    broken.loc[0, "image_subject_id"] = "another-fold"
    with pytest.raises(ValueError, match="pairs"):
        frozen_probabilities(broken, ids, 0)
    bad = pairs.index[pairs.model.eq(READ_ONLY) & pairs.subject_id.ne(pairs.image_subject_id)][0]
    pairs.loc[bad, ["raw_p2", "raw_p4"]] += [-.001, .001]
    with pytest.raises(ValueError, match="protected"):
        frozen_probabilities(pairs, ids, 0)


def test_all_five_checkpoints_fit_only_on_their_own_validation_patients(tmp_path, monkeypatch):
    from glioma.cli import idea3_mri_mix as cli

    source, residual = tmp_path / "authority", tmp_path / "residual"
    (source / "splits").mkdir(parents=True)
    cases = [[f"{k}-{i}" for i in range(n)] for k, n in enumerate(cli.EXPECTED_FOLD_SIZES)]
    assignment = {sid: k for k, ids in enumerate(cases) for sid in ids}
    rows = [dict(subject_id=sid, active=True, label=i % 2) for ids in cases for i, sid in enumerate(ids)]
    (source / "manifest.json").write_text(json.dumps(dict(rule_version=cli.RULE_VERSION, rows=rows, folds=assignment)))
    (source / "image_audit.json").write_text('{"complete":true,"errors":[]}')
    for k, test in enumerate(cases):
        dev = cases[(k+1) % 5]
        train = sorted(set(assignment) - set(dev + test))
        (source / "splits" / f"fold{k}.json").write_text(json.dumps(dict(validation=dev, test=test, train=train)))
        checkpoint = residual / f"fold{k}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "best.pt").write_bytes(b"fixture")
    monkeypatch.setattr(cli, "RESIDUAL_SHA256", [hashlib.sha256(b"fixture").hexdigest()] * 5)
    monkeypatch.setattr(cli, "verify_fold_archive", lambda *args: None)
    monkeypatch.setattr(cli, "load_models", lambda source, residual, manifest, train, fold, device: fold)
    monkeypatch.setattr(cli, "scopeless_models", lambda model: model)
    events = []

    def infer(model, source, subset, fold, device, batch_size, archive=None):
        assert model == fold
        events.append(("infer", fold, {r["subject_id"] for r in subset}))
        pairs = pd.DataFrame([dict(fold=fold, subject_id=r["subject_id"], image_subject_id=d["subject_id"],
                                   label=r["label"], condition="own" if r is d else "donor",
                                   **{m: .4 for m in MODELS}) for r in subset for d in subset])
        return pairs, dict(n_swaps=len(subset)*(len(subset)-1), prediction_source="fixture")

    def fit(dev):
        events.append(("fit", int(dev.fold.iloc[0]), set(dev.subject_id)))
        return fit_development(dev)

    monkeypatch.setattr(cli, "predict_split", infer)
    monkeypatch.setattr(cli, "fit_development", fit)
    out = tmp_path / "output"
    report = cli.run(source, residual, out, cli.torch.device("cpu"))
    assert (report["n_patients"], report["n_outer_swaps"]) == (58, 616)
    for k in range(5):
        dev, outer = set(cases[(k+1) % 5]), set(cases[k])
        assert events[3*k:3*k+3] == [("infer", k, dev), ("fit", k, dev), ("infer", k, outer)]
    assert (out / "summary.json").is_file() and (out / "risk_curve.png").is_file()
