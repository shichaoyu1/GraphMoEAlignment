"""Constrained grading with the archived UTSW folds, models and ROI cache."""

from dataclasses import replace
import hashlib
import json

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from glioma.cli.idea3_anchor import atomic_csv
from glioma.cli.utsw_authority import loader, save_json, seed_all
from glioma.data.utsw_authority import RULE_VERSION, compile_grade, evidence_from_batch
from glioma.eval.idea3 import summarize_grade
from glioma.eval.utsw_authority import prediction_frame
from glioma.models.idea3 import ReadOnlyGradeResidual
from glioma.models.utsw_authority import UTSWAuthorityGrade, grade_loss
from glioma.modules.authority_projection import project_allowed_simplex


def metadata_splits(source, manifest, fold):
    split = json.loads((source / "splits" / f"fold{fold}.json").read_text(encoding="utf-8"))
    rows = {r["subject_id"]: r for r in manifest["rows"] if r["label"] is not None}
    assignment = manifest["folds"]
    expected = {"test": {sid for sid in rows if assignment[sid] == fold},
                "validation": {sid for sid in rows if assignment[sid] == (fold + 1) % 5}}
    expected["train"] = set(rows) - expected["validation"] - expected["test"]
    if any(set(split[role]) != ids or len(split[role]) != len(ids) for role, ids in expected.items()):
        raise ValueError(f"Archived patient splits are inconsistent in fold {fold}")
    return {role: [rows[sid] for sid in sorted(split[role])] for role in expected}


def build_derangement(n, fold):
    if n < 2:
        raise ValueError("Same-fold MRI shuffling requires at least two held-out patients")
    ordered = np.random.default_rng(2026 + fold).permutation(n)
    shuffled = np.empty(n, dtype=int)
    shuffled[ordered] = np.roll(ordered, 1)
    return shuffled


def checkpoint(source, manifest, variant, fold, device):
    run = source / f"fold{fold}" / variant
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    expected = {"variant": variant, "fold": fold, "seed": 101, "rule_version": RULE_VERSION,
                "source_hash": manifest["source_hash"]}
    if any(config.get(key) != value for key, value in expected.items()) or not (run / "DONE.json").is_file():
        raise ValueError(f"Checkpoint does not belong to the archived UTSW protocol: {run}")
    model = UTSWAuthorityGrade(variant).to(device)
    saved = torch.load(run / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(saved["model"])
    return model.eval()


@torch.no_grad()
def residual_batches(model, batches, device):
    features = []
    for batch in batches:
        data = evidence_from_batch(batch, device)
        # Supply only evidence metadata to the readout. The empty trailing dimensions
        # still preserve the batch length required by the existing rule compiler.
        features.append((replace(data, images=data.images.new_empty((len(data.images), 0))), model.encode_mri(data.images),
                         batch["label"].to(device)))
    return features


def fit_read_only(acf, subsets, source, out, device, epochs, batch_size):
    model = ReadOnlyGradeResidual().to(device)
    model.encoders.load_state_dict(acf.encoders.state_dict())
    model.adapters.load_state_dict(acf.adapters.state_dict())
    model.encoders.requires_grad_(False).eval()
    model.adapters.requires_grad_(False).eval()
    train = residual_batches(model, loader(subsets["train"], source, batch_size=batch_size), device)
    validation = residual_batches(model, loader(subsets["validation"], source, batch_size=batch_size), device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(101)
    best = float("inf")
    for epoch in range(epochs):
        for index in rng.permutation(len(train)):
            evidence, features, labels = train[index]
            optimizer.zero_grad(set_to_none=True)
            output = model.predict_from_features(evidence, features)
            loss = grade_loss(output, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite residual training loss")
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            brier, n = 0., 0
            for evidence, features, labels in validation:
                output = model.predict_from_features(evidence, features)
                square = (output["p_star"] - torch.nn.functional.one_hot(labels, 3)).square()
                brier += square.sum().item()
                n += square.numel()
            brier /= n
        if not np.isfinite(brier):
            raise FloatingPointError("Nonfinite residual validation loss")
        if brier < best:
            best = brier
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "val_brier": best}, out / "best.tmp.pt")
            (out / "best.tmp.pt").replace(out / "best.pt")
    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=True)["model"])
    return model.eval()


def molecular_control(rows):
    predictor = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=101)
    predictor.fit([r["features"] for r in rows], [r["label"] for r in rows])
    if not np.array_equal(predictor.classes_, [0, 1, 2]):
        raise ValueError("Archived training fold lacks a reference grade")

    def predict(evidence):
        compiled = compile_grade(evidence)
        raw = torch.as_tensor(predictor.predict_proba(evidence.molecular.cpu().numpy()), device=evidence.images.device)
        p = project_allowed_simplex(raw, compiled["allowed"])
        return {"p0": raw, "p_star": p, **compiled}

    return predict


def withdraw_report(evidence):
    molecular = evidence.molecular.clone()
    molecular[:, :12] = molecular.new_tensor([0, 0, 1] * 4)
    molecular[:, 12] = 0
    available = evidence.available.clone()
    available[:, 4] = False
    return replace(evidence, molecular=molecular, available=available, active=torch.zeros_like(evidence.active))


@torch.no_grad()
def evaluate_interventions(model, variant, test_batches, device, fold):
    images = torch.cat([batch["images"] for batch in test_batches])
    ids = [sid for batch in test_batches for sid in batch["subject_id"]]
    donors = build_derangement(len(images), fold)
    output, start = [], 0
    for batch in test_batches:
        evidence = evidence_from_batch(batch, device)
        stop = start + len(evidence.images)
        occluded = evidence.images.clone()
        height, width = occluded.shape[-2:]
        occluded[..., height // 3:2 * height // 3, width // 3:2 * width // 3] = 0
        interventions = {
            "clean": (evidence, batch["subject_id"]),
            "permuted": (replace(evidence, images=images[donors[start:stop]].to(device)), [ids[i] for i in donors[start:stop]]),
            "occluded": (replace(evidence, images=occluded), batch["subject_id"]),
            "withdrawn": (withdraw_report(evidence), batch["subject_id"]),
        }
        direct = model(evidence)
        for condition, (changed, image_ids) in interventions.items():
            observed = direct if condition == "clean" else model(changed)
            active = observed["active"]
            if condition == "withdrawn":
                if active.any():
                    raise AssertionError("Report withdrawal failed to revoke the rule")
            elif not torch.equal(direct["allowed"], observed["allowed"]):
                raise AssertionError("Changing MRI changed the compiled rule")
            if active.any() and (observed["p_star"][active, 2] > 1e-6).any():
                raise AssertionError("Projected prediction violates the activated rule")
            if variant == "read_only_residual" and condition in ("permuted", "occluded"):
                if not torch.equal(direct["designated_state"], observed["designated_state"]) or not torch.equal(
                    direct["p0"][active, 2], observed["p0"][active, 2]
                ):
                    raise AssertionError("MRI changed the read-only report state or its Grade 4 mass")
            if not torch.isfinite(observed["p_star"]).all():
                raise FloatingPointError("Nonfinite grading predictions")
            frame = prediction_frame(batch["subject_id"], batch["label"].numpy(), observed["p0"].cpu().numpy(),
                                     observed["p_star"].cpu().numpy(), active.cpu().numpy(),
                                     observed["status"].cpu().numpy(), fold, variant)
            frame["condition"], frame["image_subject_id"] = condition, image_ids
            output.append(frame)
        start = stop
    return pd.concat(output, ignore_index=True)


def grade_table(report, path):
    lines = ["# Constrained UTSW grading", "",
             "| Model | MRI/report condition | Active / total | Residual n | Residual Brier | Raw AVR | Projected AVR |",
             "|---|---|---:|---:|---:|---:|---:|"]

    def number(value):
        return "N/A" if value is None else f"{value:.4f}"

    for key, values in report["metrics"].items():
        model, condition = key.split("/")
        active = f"{values['active_n']} / {values['n']}"
        scores = [number(values[key]) for key in ("residual_brier", "avr_raw", "avr")]
        lines.append(f"| {model} | {condition} | {active} | {values['residual_n']} | {' | '.join(scores)} |")
    lines.extend(["", "## Paired residual Brier differences", "",
                  "Differences are read-only residual minus control. Negative values favor the residual model.", "",
                  "| Condition / control | Paired n | Difference | 95% CI |", "|---|---:|---:|---:|"])
    for key, values in report["residual_brier_candidate_minus_control"].items():
        low, high = values["ci95"]
        lines.append(f"| {key} | {values['n']} | {values['estimate']:+.4f} | [{low:+.4f}, {high:+.4f}] |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_authority(source, out, device, epochs, batch_size, repetitions):
    seed_all()
    manifest_file = source / "manifest.json"
    raw = manifest_file.read_bytes()
    manifest = json.loads(raw)
    audit = json.loads((source / "image_audit.json").read_text(encoding="utf-8"))
    if manifest["rule_version"] != RULE_VERSION or not audit["complete"] or audit["errors"]:
        raise ValueError("Authority run requires a complete audited ROI cache and matching rule version")
    out.mkdir(parents=True, exist_ok=True)
    results, reproduction = [], {}
    for fold in range(5):
        seed_all()
        subsets = metadata_splits(source, manifest, fold)
        for role in subsets:
            if not all((source / "cache" / f'{row["subject_id"]}.npz').is_file() for row in subsets[role]):
                raise ValueError(f"Archived ROI cache is incomplete in fold {fold} {role}")
        run = out / f"fold{fold}"
        run.mkdir(exist_ok=True)
        config = {"manifest_hash": hashlib.sha256(raw).hexdigest(), "fold": fold, "epochs": epochs,
                  "batch_size": batch_size, "device": str(device), "seed": 101}
        for variant in ("acf", "learned_joint"):
            checkpoint_path = source / f"fold{fold}" / variant / "best.pt"
            config[f"{variant}_checkpoint_hash"] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if (run / "config.json").exists() and json.loads((run / "config.json").read_text(encoding="utf-8")) != config:
            raise ValueError(f"Existing grading run configuration differs at {run}")
        if (run / "done.json").is_file():
            reproduction[str(fold)] = json.loads((run / "done.json").read_text(encoding="utf-8"))
            results.append(pd.read_csv(run / "predictions.csv"))
            continue
        save_json(run / "config.json", config)
        test = list(loader(subsets["test"], source, batch_size=batch_size))
        acf = checkpoint(source, manifest, "acf", fold, device)
        candidate = fit_read_only(acf, subsets, source, run, device, epochs, batch_size)
        joint = checkpoint(source, manifest, "learned_joint", fold, device)
        models = {"molecular_projected": molecular_control(subsets["train"]), "learned_joint": joint,
                  "acf": acf, "read_only_residual": candidate}
        frames = [evaluate_interventions(model, variant, test, device, fold) for variant, model in models.items()]
        frame = pd.concat(frames, ignore_index=True)
        reproduced = {}
        for variant in ("acf", "learned_joint"):
            previous = pd.read_csv(source / f"fold{fold}" / variant / "predictions.csv").set_index("subject_id")
            observed = frame[(frame.model == variant) & (frame.condition == "clean")].set_index("subject_id")
            if not observed.index.is_unique or set(previous.index) != set(observed.index):
                raise ValueError(f"Archived clean predictions do not cover the same patients for {variant}, fold {fold}")
            p_cols = ["p2", "p3", "p4"]
            error = float((previous[p_cols] - observed[p_cols]).abs().max().max())
            if not np.isfinite(error) or error > .005:
                raise ValueError(f"Archived clean predictions did not reproduce for {variant}, fold {fold}: {error:.6f}")
            reproduced[variant] = error
        atomic_csv(frame, run / "predictions.csv")
        save_json(run / "done.json", reproduced)
        reproduction[str(fold)] = reproduced
        results.append(frame)
        print(f"Authority fold={fold} complete", flush=True)
    frame = pd.concat(results, ignore_index=True)
    report = summarize_grade(frame, repetitions)
    report.update(rule_version=RULE_VERSION, clean_checkpoint_max_abs_error=reproduction,
                  mri_encoder="frozen fold-specific ACF encoders and adapters", model_seed=101)
    atomic_csv(frame, out / "authority_oof_predictions.csv")
    save_json(out / "authority_summary.json", report)
    grade_table(report, out / "authority_results.md")
    return report
