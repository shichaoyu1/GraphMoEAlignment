"""Fit constant and disagreement-adaptive MRI mixtures on frozen UTSW development folds."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from glioma.cli.idea3_anchor import atomic_csv
from glioma.cli.idea3_authority import metadata_splits
from glioma.cli.idea3_external_swap import RESIDUAL_SHA256
from glioma.cli.idea3_role_states import scopeless_models
from glioma.cli.idea3_rule_mri_swap import load_models, replay_fold, swap_fold, verify_fold_archive, verify_replay
from glioma.cli.utsw_authority import loader, save_json, seed_all
from glioma.data.utsw_authority import RULE_VERSION
from glioma.eval.idea3_mri_mix import frozen_probabilities, score_outer, summarize, write_results
from glioma.eval.idea3_rule_mri_swap import EXPECTED_FOLD_SIZES
from glioma.objectives.idea3_mri_mix import BUDGETS, fit_development


# Full, scope-masked three-state run_20261126_101228_7zvSUO. Other archives
# can be omitted altogether: outer inference uses the same code as development.
OOF_SHA256 = "bb103c4e970a9db455cadd752688656d9b0b75c433e309cf1afad0c891323e02"


def active_rows(split, patient_fold, smoke):
    rows = [r for r in split if r["active"]]
    if len(rows) != EXPECTED_FOLD_SIZES[patient_fold]:
        raise ValueError(f"Unexpected number of active patients in patient fold {patient_fold}")
    return rows[:3] if smoke else rows


def predict_split(models, source, rows, fold, device, batch_size, archive=None):
    ids = [r["subject_id"] for r in rows]
    batches = list(loader(rows, source, batch_size=batch_size))
    own = replay_fold(models, batches, device, fold)
    if archive is not None:
        saved = archive[archive.fold.eq(fold) & archive.subject_id.isin(ids) & archive.image_subject_id.isin(ids)]
        try:
            errors = verify_replay(own, saved[saved.subject_id.eq(saved.image_subject_id)])
        except ValueError as exc:
            print(f"Fold {fold}: {exc}. Recomputing outer swaps with the development runtime.", flush=True)
        else:
            wide, audit = frozen_probabilities(saved, rows, fold)
            return wide, dict(audit, prediction_source="three_state_archive", own_replay_max_abs_error=errors)
    donor = swap_fold(models, batches, ids, device, fold, batch_size)
    wide, audit = frozen_probabilities(pd.concat([own, donor], ignore_index=True), rows, fold)
    return wide, dict(audit, prediction_source="frozen_inference")


def run(source, residual, output, device, batch_size=8, smoke=False, oof_pairs=None):
    source, residual, output = source.resolve(), residual.resolve(), output.resolve()
    if any(path == output or path in output.parents for path in (source, residual)):
        raise ValueError("Choose an output directory outside the frozen archives")
    output.mkdir(parents=True, exist_ok=True)
    if any(entry.name not in {"run.log", "run.pid"} for entry in output.iterdir()):
        raise FileExistsError(f"Expected an unused output directory: {output}")
    (output / ".started").touch(exist_ok=False)
    seed_all()
    raw = (source / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    image_audit = json.loads((source / "image_audit.json").read_text(encoding="utf-8"))
    if manifest["rule_version"] != RULE_VERSION or not image_audit["complete"] or image_audit["errors"]:
        raise ValueError("Expected the completed UTSW ROI cache and registered grading rule")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    archive = None
    if oof_pairs is not None:
        if oof_pairs.is_file() and hashlib.sha256(oof_pairs.read_bytes()).hexdigest() == OOF_SHA256:
            all_states = pd.read_csv(oof_pairs)
            archive = all_states[all_states.state.eq("A") & all_states.active]
        else:
            print(f"Registered OOF archive unavailable at {oof_pairs}. Using frozen outer inference.", flush=True)

    frozen, predictions, coefficients, audit = [], [], [], {}
    for fold in ([0] if smoke else range(5)):
        dev_fold = (fold + 1) % 5
        split = metadata_splits(source, manifest, fold)
        dev = active_rows(split["validation"], dev_fold, smoke)
        outer = active_rows(split["test"], fold, smoke)
        checkpoint = residual / f"fold{fold}"
        verify_fold_archive(source, checkpoint, digest, fold)
        residual_hash = hashlib.sha256((checkpoint / "best.pt").read_bytes()).hexdigest()
        if residual_hash != RESIDUAL_SHA256[fold]:
            raise ValueError(f"Fold {fold} residual checkpoint differs from the three-state study")
        models = scopeless_models(load_models(source, checkpoint, manifest, split["train"], fold, device))
        print(f"Checkpoint {fold}: predicting development fold {dev_fold} ({len(dev)} patients)", flush=True)
        development, dev_audit = predict_split(models, source, dev, fold, device, batch_size)
        fits = fit_development(development)
        coefficients.extend(dict(fit, fold=fold, development_fold=dev_fold) for fit in fits)
        # Only after freezing the coefficients, score this checkpoint's outer patients.
        test, test_audit = predict_split(models, source, outer, fold, device, batch_size, archive)
        train_labels = [r["label"] for r in split["train"] if r["active"]]
        prior = float(np.mean(np.asarray(train_labels) == 1))
        predictions.append(score_outer(test, fits, prior))
        frozen.extend((development.assign(split="development", patient_fold=dev_fold),
                       test.assign(split="outer", patient_fold=fold)))
        audit[str(fold)] = dict(development_fold=dev_fold, development=dev_audit, outer=test_audit,
                                residual_sha256=residual_hash, n_active_train=len(train_labels), train_grade3_prior=prior)
        print(f"Checkpoint {fold}: fitted all budgets; scored {len(outer)} outer patients "
              f"({test_audit['prediction_source']})", flush=True)

    scores = pd.concat(predictions, ignore_index=True)
    patients, report = summarize(scores, 100 if smoke else 10_000, smoke=smoke)
    report.update(study_mode="smoke" if smoke else "full", rule_version=RULE_VERSION, budgets=BUDGETS,
                  n_outer_swaps=sum(v["outer"]["n_swaps"] for v in audit.values()),
                  development_budget="swap Brier <= report-only Brier + budget; fit only on validation (k+1)%5",
                  folds=audit, fitted_parameters=coefficients, manifest_sha256=digest,
                  oof_archive_sha256=OOF_SHA256 if archive is not None else None,
                  device=str(device), batch_size=batch_size, torch_version=str(torch.__version__))
    atomic_csv(pd.concat(frozen, ignore_index=True), output / "frozen_pairs.csv")
    atomic_csv(pd.DataFrame(coefficients), output / "parameters.csv")
    atomic_csv(scores, output / "oof_predictions.csv")
    atomic_csv(patients, output / "patient_results.csv")
    write_results(report, output)
    save_json(output / "summary.json", report)
    print(f"Completed {report['study_mode']}: {report['n_patients']} OOF patients; {output / 'summary.json'}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-source", required=True, type=Path)
    parser.add_argument("--residual-source", required=True, type=Path)
    parser.add_argument("--oof-pairs", type=Path, help="Optional registered three-state mri_pair_predictions.csv")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="Three development and three outer patients, checkpoint 0")
    args = parser.parse_args()
    run(args.authority_source, args.residual_source, args.output, torch.device(args.device),
        args.batch_size, args.smoke, args.oof_pairs)


if __name__ == "__main__":
    main()
