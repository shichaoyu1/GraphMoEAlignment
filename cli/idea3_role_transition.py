"""Evaluate absent, complementary and rule-qualified 1p/19q on the archived UTSW folds."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from glioma.cli.idea3_anchor import atomic_csv
from glioma.cli.idea3_authority import metadata_splits
from glioma.cli.idea3_role_readout import matched_readout_fold
from glioma.cli.idea3_role_states import TARGET_FIELD, build_state, scopeless_models
from glioma.cli.idea3_rule_mri_swap import (
    PROBABILITIES, REPLAY_TOLERANCE, load_models, predict_batch, replay_fold,
    verify_fold_archive, verify_invariance, verify_replay,
)
from glioma.cli.utsw_authority import loader, save_json, seed_all
from glioma.data.utsw_authority import FIELDS, RULE_VERSION, compile_grade
from glioma.eval.idea3_role_leakage import audit_leakage
from glioma.eval.idea3_role_transition import (
    analyze_states, compare_swap_states, paired_effect, report_markdown,
)
from glioma.eval.idea3_rule_mri_swap import (
    EXPECTED_FOLD_SIZES, MODELS, VIOLATION_TOLERANCE, patient_results, summarize, validate_pairs,
)


def measured(row):
    return row["record"][TARGET_FIELD] in FIELDS[TARGET_FIELD]


def smoke_rows(rows):
    """Exercise both directions, inactive controls and unmeasured records, without grade selection."""
    counts, selected = Counter(), []
    for row in rows:
        group = (row["active"], row["record"][TARGET_FIELD] if measured(row) else "unknown")
        if counts[group] < (2 if row["active"] else 1):
            selected.append(row)
        counts[group] += 1
    # Include one same-scope, different-value field pair to exercise that control.
    donors = field_donors(rows)
    if donors:
        recipient = sorted(donors)[0]
        sample_ids = {r["subject_id"] for r in selected} | {recipient, donors[recipient]}
        return [r for r in rows if r["subject_id"] in sample_ids]
    return selected


def field_donors(rows):
    """Choose an opposite measured value in the same held-out fold, scope, IDH and method."""
    groups = {}
    for row in rows:
        if measured(row):
            rec = row["record"]
            group = (rec["scope_eligible"], rec["IDH"], rec["IDH_Testing Method"])
            groups.setdefault(group, {}).setdefault(rec[TARGET_FIELD], []).append(row["subject_id"])
    recipients = {}
    for values in groups.values():
        if len(values) == 2:
            negative, positive = FIELDS[TARGET_FIELD]
            for value, others in ((negative, positive), (positive, negative)):
                pool = sorted(values[others])
                for index, sid in enumerate(sorted(values[value])):
                    recipients[sid] = pool[index % len(pool)]
    return recipients


def role_frame(models, evidence, ids, labels, image_ids, fold, state):
    frame = pd.concat(predict_batch(models, evidence, ids, labels, image_ids, fold), ignore_index=True)
    frame["state"] = state
    return frame


@torch.inference_mode()
def evaluate_own(models, batches, rows, device, fold):
    lookup = {r["subject_id"]: r for r in rows}
    donor_map = field_donors(rows)
    own, controls, mark = [], [], []
    for batch in batches:
        ids = batch["subject_id"]
        records = [lookup[s]["record"] for s in ids]
        labels = batch["label"].numpy()
        first = {}
        for role in ("C", "A", "C", "O", "A"):
            evidence = build_state(batch, records, role, device)
            frame = role_frame(models, evidence, ids, labels, ids, fold, role)
            if role in first:
                old = first[role]
                if not old[["subject_id", "model", "allowed_grades"]].equals(
                        frame[["subject_id", "model", "allowed_grades"]]):
                    raise ValueError(f"Restoring {role} changed the allowed set or patient order")
                err = (old[PROBABILITIES] - frame[PROBABILITIES]).abs().max(axis=1)
                if err.max() > VIOLATION_TOLERANCE:
                    raise ValueError(f"Restoring {role} changed predictions in fold {fold}")
                old["restore_max_abs_error"] = err
                continue
            frame["restore_max_abs_error"] = np.nan
            first[role] = frame
            own.append(frame)
        for role in ("O", "C", "A"):
            evidence = build_state(batch, records, role, device, hide_mgmt=True)
            allowed = compile_grade(evidence)["active"].cpu().tolist()
            expected = [role == "A" and lookup[s]["active"] for s in ids]
            if allowed != expected:
                raise ValueError(f"MGMT masking or qualification changed an unexpected rule in {fold}/{role}")
            for sid, active in zip(ids, allowed):
                row = lookup[sid]
                mark.append(dict(subject_id=sid, fold=fold, state=role, measured=measured(row),
                                 target_value=row["record"][TARGET_FIELD] if role != "O" and measured(row) else "unknown",
                                 target_visible=role != "O" and measured(row), soft_use=role != "O" and measured(row),
                                 rule_permitted=role == "A", scope_eligible=row["record"]["scope_eligible"],
                                 archive_active=row["active"], mgmt_hidden_active=active,
                                 field_donor_id=donor_map.get(sid) if role == "C" else None))
        positions = [i for i, sid in enumerate(ids) if sid in donor_map]
        if positions:
            subset = {key: batch[key][positions] for key in ("images", "available")}
            chosen = [ids[i] for i in positions]
            changed = [dict(lookup[s]["record"], **{TARGET_FIELD: lookup[donor_map[s]]["record"][TARGET_FIELD]})
                       for s in chosen]
            evidence = build_state(subset, changed, "C", device)
            control = role_frame(models, evidence, chosen, labels[positions], chosen, fold, "C_alternate_value")
            control["field_donor_id"] = control.subject_id.map(donor_map)
            controls.append(control)
    frame = pd.concat(own, ignore_index=True)
    frame["codeletion_measured"] = frame.subject_id.map({s: measured(r) for s, r in lookup.items()})
    frame["archive_active"] = frame.subject_id.map({s: r["active"] for s, r in lookup.items()})
    return frame, pd.DataFrame(mark), pd.concat(controls, ignore_index=True) if controls else None


@torch.inference_mode()
def evaluate_donors(models, rows, source, batch_size, fold, device):
    # At most 12 complete held-out MRIs per fold. The recipient supplies every non-image input.
    cache = list(loader(rows, source, batch_size=batch_size))
    ids = [sid for item in cache for sid in item["subject_id"]]
    if ids != [r["subject_id"] for r in rows] or len(ids) < 2:
        raise ValueError(f"Invalid active MRI cohort in fold {fold}")
    images = torch.cat([item["images"] for item in cache])
    available = torch.cat([item["available"] for item in cache])
    if not available[:, :4].all():
        raise ValueError("An active donor has an incomplete MRI")
    pairs = [(i, j) for i in range(len(rows)) for j in range(len(rows)) if i != j]
    frames = []
    for start in range(0, len(pairs), batch_size):
        recipients, donors = zip(*pairs[start:start + batch_size])
        cases = [rows[i] for i in recipients]
        batch = {"images": images[list(donors)], "available": available[list(recipients)]}
        records = [row["record"] for row in cases]
        for state in ("A", "C"):
            evidence = build_state(batch, records, state, device)
            frame = role_frame(models, evidence, [row["subject_id"] for row in cases],
                               [row["label"] for row in cases], [ids[j] for j in donors], fold, state)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def negative_controls(predictions, c_pairs):
    """Inputs coincide for these pairs, regardless of held-out reference grade."""
    keys = ["model", "fold", "subject_id"]
    controls = {}
    for name, subset, other in (("unmeasured_O_vs_C", ~predictions.codeletion_measured, "O"),
                                ("inactive_A_vs_C", ~predictions.archive_active, "A")):
        rows = predictions[subset]
        c = rows[rows.state.eq("C")].set_index(keys)
        compare = rows[rows.state.eq(other)].set_index(keys).reindex(c.index)
        if len(c) == 0 or not c.allowed_grades.equals(compare.allowed_grades):
            raise ValueError(f"Unexpected allowed grades for negative control {name}")
        controls[name] = float(np.abs(compare[PROBABILITIES] - c[PROBABILITIES]).to_numpy().max())
    c = c_pairs[c_pairs.model.eq("molecular_projected")]
    own = c[c.subject_id.eq(c.image_subject_id)].set_index(keys)[PROBABILITIES]
    donor = c[c.subject_id.ne(c.image_subject_id)].set_index(keys)[PROBABILITIES]
    controls["C_report_only_across_MRI"] = float(np.abs(donor.to_numpy() - own.reindex(donor.index).to_numpy()).max())
    for name, error in controls.items():
        if not np.isfinite(error) or error > VIOLATION_TOLERANCE:
            raise ValueError(f"Negative control {name} changed probabilities by {error}")
    return controls


def run(source, residual, output, device, batch_size=8, smoke=False):
    if batch_size < 1 or any(output == p or p in output.parents or output in p.parents for p in (source, residual)):
        raise ValueError("Use a positive batch size and a result directory separate from both input archives")
    if output.exists() and any(p.name not in {"run.log", "run.pid"} for p in output.iterdir()):
        raise FileExistsError(f"Result directory already contains an earlier run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    # Also protect concurrent callers using the Python entry point directly.
    with (output / ".started").open("x", encoding="utf-8") as marker:
        marker.write("See summary.json for successful completion.\n")
    raw = (source / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    audit = audit_leakage(manifest)
    save_json(output / "leakage_audit.json", audit)
    if ((audit["n_labeled"], audit["n_target_measured"]) != (618, 330)
            or audit["accepted_on_labeled"] != {"true": 618}
            or audit["assay_validity_on_labeled"] != {"unreported": 618}):
        raise ValueError("Unexpected cohort, acceptance or assay-validity profile for this registered protocol")
    image_audit = json.loads((source / "image_audit.json").read_text(encoding="utf-8"))
    if manifest["rule_version"] != RULE_VERSION or not image_audit["complete"] or image_audit["errors"]:
        raise ValueError("The source ROI cache or grading rule failed archive checks")
    seed_all()
    archive_file = residual / "authority_oof_predictions.csv"
    archive = pd.read_csv(archive_file)
    clean = archive[archive.condition.eq("clean")]
    folds = [0] if smoke else list(range(5))
    repetitions = 50 if smoke else 10_000
    config = dict(smoke=smoke, folds=folds, device=str(device), torch_version=str(torch.__version__),
                  authority_source=str(source), residual_source=str(residual), rule_version=RULE_VERSION,
                  manifest_sha256=digest, oof_sha256=hashlib.sha256(archive_file.read_bytes()).hexdigest(),
                  batch_size=batch_size, replay_tolerance=REPLAY_TOLERANCE, rule_tolerance=VIOLATION_TOLERANCE,
                  bootstrap_draws=repetitions, bootstrap_seed=2026)
    save_json(output / "config.json", config)
    own, changes, alternate, swapped, matched, replays = [], [], [], [], [], {}
    patients = {}
    for fold in folds:
        split = metadata_splits(source, manifest, fold)
        test = smoke_rows(split["test"]) if smoke else split["test"]
        run_dir = residual / f"fold{fold}"
        replay_bs = verify_fold_archive(source, run_dir, digest, fold)
        models = load_models(source, run_dir, manifest, split["train"], fold, device)
        frame = replay_fold(models, loader(test, source, batch_size=replay_bs), device, fold)
        reference = clean[clean.fold.eq(fold)]
        if smoke:
            reference = reference[reference.subject_id.isin([r["subject_id"] for r in test])]
        errors = verify_replay(frame, reference)
        replays[str(fold)] = dict(max_abs_error=errors, n=len(test), batch_size=replay_bs,
                                  residual_sha256=hashlib.sha256((run_dir / "best.pt").read_bytes()).hexdigest())
        save_json(output / "replay_audit.json", replays)
        print(f"Fold {fold}: archived predictions verified on {len(test)} patients", flush=True)
        wrapped = scopeless_models(models)
        own_frame, marks, controls = evaluate_own(wrapped, loader(test, source, batch_size=batch_size),
                                                  test, device, fold)
        own.append(own_frame)
        changes.append(marks)
        if controls is not None:
            alternate.append(controls)
        active_rows = [r for r in test if r["active"]]
        if (smoke and len(active_rows) < 2) or (not smoke and len(active_rows) != EXPECTED_FOLD_SIZES[fold]):
            raise ValueError(f"Active patient count changed in fold {fold}")
        patients[fold] = [r["subject_id"] for r in active_rows]
        swapped.append(evaluate_donors(wrapped, active_rows, source, batch_size, fold, device))
        train = [r for r in split["train"] if measured(r)]
        if smoke:
            by_grade = [[row for row in train if row["label"] == label] for label in range(3)]
            train = [row for group in by_grade for row in group[:2]]
        readout = matched_readout_fold(models["read_only_residual"], train, [r for r in test if measured(r)],
                                       source, fold, device, batch_size)
        readout["n_train_patients"] = len(train)
        matched.append(readout)
        print(f"Fold {fold}: states, swaps and paired-state readout complete", flush=True)

    predictions = pd.concat(own, ignore_index=True)
    per_patient, report = analyze_states(predictions, repetitions=repetitions)
    own_active = predictions[predictions.archive_active & predictions.state.isin(["A", "C"])]
    pairs = pd.concat([own_active, *swapped], ignore_index=True)
    a, c = (pairs[pairs.state.eq(s)].copy() for s in ("A", "C"))
    validate_pairs(a, patients)
    a, swap_patients = patient_results(a)
    verify_invariance(a)
    exploratory, diff = compare_swap_states(a, c, repetitions=repetitions)
    report["a_minus_c_swap"] = diff
    report["a_mri_swap"] = summarize(swap_patients, repetitions=repetitions)
    report["negative_controls_max_abs_error"] = negative_controls(predictions, c)
    report["restore_max_abs_error"] = float(predictions.restore_max_abs_error.max())
    report["archive_replay_max_abs_error"] = max(max(r["max_abs_error"].values()) for r in replays.values())
    report["leakage_audit"] = audit
    report["config"] = config
    readout = pd.concat(matched, ignore_index=True)
    key = ["fold", "subject_id"]
    scores = readout.pivot(index=key, columns="state", values="brier")
    expected = predictions[predictions.codeletion_measured].set_index(key).index.unique().sort_values()
    if not scores.index.equals(expected) or scores.isna().any().any():
        raise ValueError("Matched readout omitted a measured patient or state")
    report["paired_state_linear_readout"] = paired_effect(scores["O"] - scores["C"], repetitions=repetitions)
    report["study_mode"] = "smoke" if smoke else "full"
    report["interpretation"] = ("Scope is excluded from all predictive report inputs. Frozen O/C contrasts are "
                                "sensitivity results conditional on the archived fits. The additional train-fold "
                                "readout sees both states. C simulates limited qualification using final values.")
    if alternate:
        controls = pd.concat(alternate, ignore_index=True)
        index = ["model", *key]
        original = predictions[predictions.state.eq("C")].set_index(index)["brier"]
        difference = controls.set_index(index)["brier"]
        difference = difference - original.loc[difference.index]
        report["alternate_minus_own_field_brier3"] = {
            model: paired_effect(difference.xs(model, level="model"), repetitions=repetitions) for model in MODELS}
        atomic_csv(controls, output / "alternate_field_predictions.csv")
    atomic_csv(predictions, output / "state_predictions.csv")
    atomic_csv(per_patient, output / "patient_results.csv")
    atomic_csv(pd.concat(changes), output / "interventions.csv")
    atomic_csv(pairs, output / "mri_pair_predictions.csv")
    atomic_csv(swap_patients, output / "a_swap_patients.csv")
    atomic_csv(exploratory, output / "a_minus_c_swap_patients.csv")
    atomic_csv(readout, output / "paired_state_readout_predictions.csv")
    table = output / "results.tmp.md"
    table.write_text(report_markdown(report), encoding="utf-8")
    table.replace(output / "results.md")
    save_json(output / "summary.json", report)
    print(f"Completed {report['study_mode']} evaluation: {output / 'summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-source", type=Path, required=True)
    parser.add_argument("--residual-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true", help="Small real-checkpoint test on fold 0")
    args = parser.parse_args()
    run(args.authority_source.resolve(), args.residual_source.resolve(), args.output.resolve(),
        torch.device(args.device), args.batch_size, args.smoke)


if __name__ == "__main__":
    main()
