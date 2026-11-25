"""Replay archived UTSW grading and swap MRIs within the active rule cohort."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch

from glioma.cli.idea3_anchor import atomic_csv
from glioma.cli.idea3_authority import checkpoint, metadata_splits, molecular_control
from glioma.cli.utsw_authority import loader, save_json, seed_all
from glioma.data.utsw_authority import RULE_VERSION, compile_grade, evidence_from_batch
from glioma.eval.idea3_rule_mri_swap import (
    EXPECTED_FOLD_SIZES, MODELS, VIOLATION_TOLERANCE, patient_results,
    report_markdown, summarize, validate_pairs,
)
from glioma.eval.utsw_authority import prediction_frame
from glioma.models.idea3 import ReadOnlyGradeResidual


# Archived cross-environment replays already differ by up to 5.56e-5.
REPLAY_TOLERANCE = 1e-4
REPLAY_WARNING_TOLERANCE = 1e-5
PROBABILITIES = ["raw_p2", "raw_p3", "raw_p4", "p2", "p3", "p4"]
REPLAY_KEYS = ["fold", "model", "subject_id"]


def load_models(source, residual_run, manifest, train, fold, device):
    residual = ReadOnlyGradeResidual().to(device)
    saved = torch.load(residual_run / "best.pt", map_location=device, weights_only=True)
    residual.load_state_dict(saved["model"])
    return {
        "molecular_projected": molecular_control(train),
        "learned_joint": checkpoint(source, manifest, "learned_joint", fold, device),
        "acf": checkpoint(source, manifest, "acf", fold, device),
        "read_only_residual": residual.eval(),
    }


def verify_fold_archive(source, residual_run, manifest_hash, fold):
    config = json.loads((residual_run / "config.json").read_text(encoding="utf-8"))
    expected = {"manifest_hash": manifest_hash, "fold": fold, "seed": 101}
    for model in ("acf", "learned_joint"):
        checkpoint_file = source / f"fold{fold}" / model / "best.pt"
        expected[f"{model}_checkpoint_hash"] = hashlib.sha256(checkpoint_file.read_bytes()).hexdigest()
    if any(config.get(key) != value for key, value in expected.items()) or not (residual_run / "done.json").is_file():
        raise ValueError(f"Residual checkpoint does not match the completed grading archive in fold {fold}")
    return config["batch_size"]


def predict_batch(models, evidence, ids, labels, image_ids, fold):
    rule = compile_grade(evidence)
    outputs = []
    for model, predict in models.items():
        observed = predict(evidence)
        if any(not torch.equal(observed[key], rule[key]) for key in ("active", "allowed", "status")):
            raise ValueError(f"Compiled rule changed for {model}, fold {fold}")
        for key in ("p0", "p_star"):
            if not torch.isfinite(observed[key]).all():
                raise ValueError(f"Nonfinite {key} for {model}, fold {fold}")
        if (observed["p_star"][~rule["allowed"]].abs() > VIOLATION_TOLERANCE).any():
            raise ValueError(f"Projected prediction violated the rule for {model}, fold {fold}")
        frame = prediction_frame(ids, labels, observed["p0"].cpu().numpy(), observed["p_star"].cpu().numpy(),
                                 rule["active"].cpu().numpy(), rule["status"].cpu().numpy(), fold, model)
        frame["image_subject_id"] = image_ids
        outputs.append(frame)
    return outputs


@torch.inference_mode()
def replay_fold(models, batches, device, fold):
    frames = []
    for batch in batches:
        ids = batch["subject_id"]
        frames.extend(predict_batch(models, evidence_from_batch(batch, device), ids, batch["label"].numpy(),
                                    ids, fold))
    return pd.concat(frames, ignore_index=True)


def verify_replay(replayed, archived):
    observed = replayed.set_index(REPLAY_KEYS).sort_index()
    reference = archived.set_index(REPLAY_KEYS).sort_index()
    if not observed.index.is_unique or not observed.index.equals(reference.index):
        raise ValueError("Clean replay does not match archived patients, folds and models")
    for column in ("label", "reference_grade", "active", "status", "allowed_grades",
                   "reference_conflict", "image_complete", "image_subject_id"):
        if not observed[column].eq(reference[column]).all():
            raise ValueError(f"Clean replay disagrees with archived {column}")
    errors = (observed[PROBABILITIES] - reference[PROBABILITIES]).abs()
    invalid = ~np.isfinite(errors) | errors.gt(REPLAY_TOLERANCE)
    if invalid.any(axis=None):
        fold, model, subject = invalid.index[invalid.any(axis=1)][0]
        raise ValueError(f"Clean replay exceeds {REPLAY_TOLERANCE:g} at fold={fold} model={model} "
                         f"patient={subject}, max absolute error={errors.loc[(fold, model, subject)].max():.6g}")
    maximum_errors = errors.max(axis=1).groupby(level="model").max().to_dict()
    if errors.gt(REPLAY_WARNING_TOLERANCE).any(axis=None):
        warnings.warn(f"Cross-environment clean replay drift within {REPLAY_TOLERANCE:g} tolerance: "
                      f"{maximum_errors}", RuntimeWarning)
    return maximum_errors


@torch.inference_mode()
def swap_fold(models, batches, expected_ids, device, fold, batch_size):
    batches = list(batches)
    ids = [sid for batch in batches for sid in batch["subject_id"]]
    if ids != expected_ids:
        raise ValueError(f"Swap MRI cache does not match verified patients in fold {fold}")
    images = torch.cat([batch["images"] for batch in batches])
    fields = {key: torch.cat([batch[key] for batch in batches]) for key in ("molecular", "available", "active", "label")}
    pairs = [(recipient, donor) for recipient in range(len(ids)) for donor in range(len(ids)) if recipient != donor]
    frames = []
    for start in range(0, len(pairs), batch_size):
        recipient, donor = np.asarray(pairs[start:start + batch_size]).T
        report_batch = {key: values[recipient] for key, values in fields.items()}
        evidence = evidence_from_batch({**report_batch, "images": images[recipient]}, device)
        swapped = replace(evidence, images=images[donor].to(device))
        if not swapped.available[:, :4].all() or not compile_grade(swapped)["active"].all():
            raise ValueError(f"Expected complete MRI and an active rule for every pair in fold {fold}")
        frames.extend(predict_batch(models, swapped, [ids[i] for i in recipient], report_batch["label"].numpy(),
                                    [ids[i] for i in donor], fold))
    return pd.concat(frames, ignore_index=True)


def verify_invariance(frame):
    own = frame[frame.condition == "own"].set_index(REPLAY_KEYS)
    donor = frame[frame.condition == "donor"].set_index(REPLAY_KEYS)
    for model, columns in (("molecular_projected", PROBABILITIES), ("read_only_residual", ["raw_p4"])):
        swapped = donor.xs(model, level="model")[columns]
        baseline = own.xs(model, level="model")[columns].reindex(swapped.index)
        if not np.allclose(swapped.to_numpy(), baseline.to_numpy(), rtol=0, atol=VIOLATION_TOLERANCE):
            raise ValueError(f"MRI swaps changed protected probabilities for {model}")


def run(source, residual, output, device, swap_batch_size):
    if swap_batch_size < 1:
        raise ValueError("Swap batch size must be positive")
    if any(root == output or root in output.parents for root in (source, residual)):
        raise ValueError("Use an output directory outside the source archives")
    if (output / "summary.json").exists():
        raise FileExistsError(f"Completed MRI swap results already exist in {output}")
    seed_all()
    raw = (source / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    manifest_hash = hashlib.sha256(raw).hexdigest()
    audit = json.loads((source / "image_audit.json").read_text(encoding="utf-8"))
    if manifest["rule_version"] != RULE_VERSION or not audit["complete"] or audit["errors"]:
        raise ValueError("MRI swap requires a complete audited ROI cache and matching rule version")
    archive = pd.read_csv(residual / "authority_oof_predictions.csv")
    archive = archive[archive.condition == "clean"]
    output.mkdir(parents=True, exist_ok=True)
    replay_audit = {
        "tolerance": REPLAY_TOLERANCE, "warning_tolerance": REPLAY_WARNING_TOLERANCE,
        "torch_version": str(torch.__version__), "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "by_fold": {},
    }
    splits, replays = {}, []
    for fold in range(len(EXPECTED_FOLD_SIZES)):
        residual_run = residual / f"fold{fold}"
        batch_size = verify_fold_archive(source, residual_run, manifest_hash, fold)
        splits[fold] = metadata_splits(source, manifest, fold)
        models = load_models(source, residual_run, manifest, splits[fold]["train"], fold, device)
        replay = replay_fold(models, loader(splits[fold]["test"], source, batch_size=batch_size), device, fold)
        errors = verify_replay(replay, archive[archive.fold == fold])
        replay_audit["by_fold"][str(fold)] = errors
        save_json(output / "replay_audit.json", replay_audit)
        replays.append(replay)
        print(f"Clean replay verified for fold {fold} ({len(splits[fold]['test'])} patients); "
              f"max_abs_error={max(errors.values()):.6g}", flush=True)
    clean = pd.concat(replays, ignore_index=True)
    reproduction = verify_replay(clean, archive)
    cohort = clean[(clean.model == "read_only_residual") & clean.active]
    if not cohort.image_complete.all() or cohort.reference_conflict.any():
        raise ValueError("The archived active cohort contains incomplete MRI or a reference conflict")
    patients = {fold: sorted(cohort.loc[cohort.fold == fold, "subject_id"]) for fold in splits}
    if tuple(map(len, patients.values())) != EXPECTED_FOLD_SIZES:
        raise ValueError("The active cohort does not match the prespecified five-fold counts")
    frames = [clean[clean.active]]
    for fold, subsets in splits.items():
        models = load_models(source, residual / f"fold{fold}", manifest, subsets["train"], fold, device)
        ids = set(patients[fold])
        eligible = [row for row in subsets["test"] if row["subject_id"] in ids]
        frame = swap_fold(models, loader(eligible, source, batch_size=swap_batch_size), patients[fold], device,
                          fold, swap_batch_size)
        frames.append(frame)
        print(f"MRI swaps complete for fold {fold} ({len(frame) // len(MODELS)} directed pairs)", flush=True)
    predictions = pd.concat(frames, ignore_index=True)
    validate_pairs(predictions, patients)
    predictions, per_patient = patient_results(predictions)
    verify_invariance(predictions)
    report = summarize(per_patient)
    report.update(rule_version=RULE_VERSION, authority_manifest_sha256=manifest_hash,
                  clean_replay_max_abs_error=reproduction, clean_replay_tolerance=REPLAY_TOLERANCE,
                  replay_audit=replay_audit,
                  violation_tolerance=VIOLATION_TOLERANCE, active_n=len(cohort),
                  total_clean_n=int(clean.model.eq("read_only_residual").sum()),
                  fold_sizes={str(fold): len(ids) for fold, ids in patients.items()},
                  device=str(device), swap_batch_size=swap_batch_size)
    output.mkdir(parents=True, exist_ok=True)
    atomic_csv(predictions, output / "pair_predictions.csv")
    atomic_csv(per_patient, output / "patient_results.csv")
    table = output / "results.tmp.md"
    table.write_text(report_markdown(report), encoding="utf-8")
    table.replace(output / "results.md")
    save_json(output / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-source", type=Path, required=True, help="Completed Idea 4 grading archive")
    parser.add_argument("--residual-source", type=Path, required=True, help="Completed Idea 3 experiment2 archive")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--swap-batch-size", type=int, default=8)
    args = parser.parse_args()
    report = run(args.authority_source.resolve(), args.residual_source.resolve(), args.output.resolve(),
                 torch.device(args.device), args.swap_batch_size)
    print(f"Rule-matched MRI swaps complete: {report['n_patients']} patients; summary={args.output / 'summary.json'}",
          flush=True)


if __name__ == "__main__":
    main()
