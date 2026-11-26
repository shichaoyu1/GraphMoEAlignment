"""Compare three linear grade readouts with the same frozen UTSW MRI features."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import pandas as pd
import sklearn
import torch

from glioma.cli.idea3_anchor import atomic_csv
from glioma.cli.idea3_authority import metadata_splits
from glioma.cli.idea3_record_value_readout import record_value_fold
from glioma.cli.utsw_authority import save_json, seed_all
from glioma.data.utsw_authority import FIELDS, features
from glioma.eval.idea3_record_value import report_markdown, summarize
from glioma.eval.idea3_role_leakage import TARGET
from glioma.models.idea3 import ReadOnlyGradeResidual


def smoke_sample(rows, training):
    # Cover unrecorded, negative and positive values, plus each training grade.
    counts, sample = Counter(), []
    for row in rows:
        value = row["record"][TARGET]
        key = (value if value in FIELDS[TARGET] else None, row["label"] if training else None)
        if counts[key] < 2:
            sample.append(row)
            counts[key] += 1
    return sample


def run(source, residual, output, device, batch_size=8, smoke=False):
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    if any(output == path or path in output.parents or output in path.parents for path in (source, residual)):
        raise ValueError("Use a separate output directory outside the input archives")
    output.mkdir(parents=True, exist_ok=True)
    if any(p.name not in {"run.log", "run.pid"} for p in output.iterdir()):
        raise FileExistsError(f"Output directory already in use: {output}")
    (output / ".started").touch(exist_ok=False)

    raw = (source / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    rows = [row for row in manifest["rows"] if row["label"] is not None]
    if any(row["features"] != features(row["record"]) for row in rows):
        raise ValueError("Cached molecular feature encoding disagrees with the archived record")
    seed_all()
    fold_info, predictions = {}, []
    for fold in ([0] if smoke else range(5)):
        split = metadata_splits(source, manifest, fold)
        run_dir = residual / f"fold{fold}"
        archived = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        if (archived["manifest_hash"] != digest or archived["fold"] != fold or archived["seed"] != 101
                or not (run_dir / "done.json").is_file()):
            raise ValueError(f"Residual fold {fold} is incomplete or belongs to a different cohort")
        checkpoint = run_dir / "best.pt"
        model = ReadOnlyGradeResidual()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
        model.to(device).eval().requires_grad_(False)
        train, test = split["train"], split["test"]
        if smoke:
            train, test = smoke_sample(train, True), smoke_sample(test, False)
        print(f"Fold {fold}: encoding {len(train)} training and {len(test)} held-out MRIs", flush=True)
        frame = record_value_fold(model, train, test, source, fold, device, batch_size)
        predictions.append(frame)
        fold_info[str(fold)] = {"n_train": len(train), "n_test": len(test),
                                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        print(f"Fold {fold}: completed X, X+M and X+M+V", flush=True)

    predictions = pd.concat(predictions, ignore_index=True)
    report = summarize(predictions, repetitions=100 if smoke else 10_000, seed=2026)
    if not smoke and (report["n_patients"], report["n_measured"], report["n_positive"]) != (618, 330, 63):
        raise ValueError("The full study requires 618 OOF patients, 330 measured and 63 codeleted")
    report.update(study_mode="smoke" if smoke else "full", folds=fold_info, manifest_sha256=digest,
                  device=str(device), torch_version=str(torch.__version__), sklearn_version=sklearn.__version__,
                  inputs={"X": "28 frozen MRI features + IDH, MGMT and IDH testing method (9 columns)",
                          "M": "1 for an explicit positive or negative 1p/19q record, 0 otherwise",
                          "V": "1 for recorded codeletion, 0 otherwise; interpreted jointly with M"},
                  fit_recipe="StandardScaler fitted on training patients, L2 logistic regression C=1, "
                             "lbfgs, max_iter=2000, random_state=101; one row per training patient per readout")
    atomic_csv(predictions, output / "oof_predictions.csv")
    (output / "results.md").write_text(report_markdown(report), encoding="utf-8")
    save_json(output / "summary.json", report)
    print(f"Completed {report['study_mode']} readout comparison: {output / 'summary.json'}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-source", type=Path, required=True, help="Archived manifest, splits and ROI cache")
    parser.add_argument("--residual-source", type=Path, required=True, help="Archived experiment2 checkpoints")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="Check one small fold-0 subset")
    args = parser.parse_args()
    run(args.authority_source.resolve(), args.residual_source.resolve(), args.output.resolve(),
        torch.device(args.device), args.batch_size, args.smoke)


if __name__ == "__main__":
    main()
