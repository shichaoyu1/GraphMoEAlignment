"""Prepare, validate, train, and aggregate the fixed UTSW authority protocol."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import platform

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression

from glioma.data.utsw_authority import (cohort, fixed_folds, prepare_case, CachedGradeDataset,
    evidence_from_batch, RULE_VERSION, compile_grade)
from glioma.models.utsw_authority import UTSWAuthorityGrade, grade_loss
from glioma.modules.authority_projection import project_allowed_simplex
from glioma.eval.utsw_authority import metrics, prediction_frame, paired_bootstrap


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def seed_all(seed=101):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)


def source_hash():
    root = Path(__file__).resolve().parents[1]
    paths = ["data/utsw_authority.py", "data/utsw_dataset.py", "models/utsw_authority.py",
             "models/encoders.py", "models/authority_fusion.py", "modules/authority_projection.py",
             "modules/hierarchical_spd_fusion.py", "eval/utsw_authority.py", "cli/utsw_authority.py"]
    return hashlib.sha256(b"".join((root/p).read_bytes() for p in paths)).hexdigest()


def prepare(args):
    out = Path(args.output)
    rows = cohort(args.metadata)
    assignments = fixed_folds(rows)
    manifest = {"rule_version": RULE_VERSION, "rows": rows, "folds": assignments,
                "metadata_sha256": hashlib.sha256(Path(args.metadata).read_bytes()).hexdigest(),
                "split_seed": 2026, "model_seed": 101, "source_hash": source_hash()}
    existing = out / "manifest.json"
    if existing.exists():
        previous = json.loads(existing.read_text())
        comparable = {k:v for k,v in previous.items() if k != "source_hash"}
        if comparable != {k:v for k,v in manifest.items() if k != "source_hash"}:
            raise ValueError("Existing manifest differs; use a new output directory")
        if previous["source_hash"] != manifest["source_hash"]:
            if any(out.glob("fold*/*/config.json")):
                raise ValueError("Source changed after formal runs started; use a new output directory")
            for stale in (out/"smoke.json", out/"cheap_predictions.csv"):
                if stale.exists():
                    stale.replace(stale.with_suffix(".previous"))
    save_json(existing, manifest)
    audit = pd.DataFrame([{**{k:v for k,v in r.items() if k not in ("record", "features")},
                           **r["record"], "fold": assignments.get(r["subject_id"])} for r in rows])
    audit.to_csv(out / "rule_audit.csv", index=False)
    audit.loc[audit.molecular_candidate & ~audit.active].to_csv(out / "ambiguous_candidates.csv", index=False)
    for fold in range(5):
        save_json(out / "splits" / f"fold{fold}.json", {
            kind: [r["subject_id"] for r in rows if r["label"] is not None and condition(assignments[r["subject_id"]])]
            for kind, condition in {"train": lambda f: f not in (fold, (fold+1)%5),
                                    "validation": lambda f: f == (fold+1)%5,
                                    "test": lambda f: f == fold}.items()})
    counts = dict(patients=len(rows), labeled=sum(r["label"] is not None for r in rows),
                  grades={str(g+2): sum(r["label"] == g for r in rows) for g in range(3)},
                  active=sum(r["active"] for r in rows),
                  ambiguous=sum(r["molecular_candidate"] and not r["active"] for r in rows),
                  reference_conflicts=sum(r["reference_conflict"] for r in rows))
    save_json(out / "cohort_counts.json", counts)
    print(json.dumps(counts), flush=True)
    if args.audit_only:
        return
    errors, provenance = [], []
    candidates = rows[:args.limit] if args.limit else rows
    for index, row in enumerate(candidates):
        try:
            provenance.append(prepare_case(args.data_root, row["subject_id"], out / "cache", headers_only=args.grids_only))
        except (OSError, ValueError) as error:
            errors.append(dict(subject_id=row["subject_id"], error=str(error)))
        if (index+1) % 10 == 0:
            print(f"Prepared {index+1}/{len(candidates)}; errors={len(errors)}", flush=True)
    audit_name = "grid_audit.json" if args.grids_only else "image_audit.json"
    save_json(out / audit_name, dict(provenance=provenance, errors=errors, complete=not args.limit))
    if errors:
        raise RuntimeError(f"{len(errors)} image checks failed; see {audit_name}. Cohort was not silently reduced.")


def load_protocol(args):
    out = Path(args.output)
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["source_hash"] != source_hash():
        raise ValueError("Source differs from prepared protocol; prepare a new output directory")
    return out, manifest, [r for r in manifest["rows"] if r["label"] is not None]


def loader(rows, out, train=False, batch_size=8):
    return DataLoader(CachedGradeDataset(rows, out/"cache", augment=train), batch_size=batch_size,
                      shuffle=train, num_workers=0, generator=torch.Generator().manual_seed(101))


@torch.no_grad()
def predict(model, batches, device, fold, variant, intervention=False):
    model.eval()
    frames, arrays = [], {key: [] for key in ("subject_id", "designated_state", "mask", "adjacency", "active")}
    if intervention:
        arrays.update({f"{kind}_state": [] for kind in ("noise", "occlusion")})
    for batch in batches:
        evidence = evidence_from_batch(batch, device)
        output = model(evidence)
        cpu = lambda key: output[key].detach().cpu().numpy()
        frame = prediction_frame(batch["subject_id"], batch["label"].numpy(), cpu("p0"), cpu("p_star"),
                                 cpu("active"), cpu("status"), fold, variant)
        if intervention:
            for kind in ("noise", "occlusion"):
                images = evidence.images.clone()
                if kind == "noise":
                    noise = []
                    for sid in batch["subject_id"]:
                        gen = torch.Generator().manual_seed(int(hashlib.sha256(sid.encode()).hexdigest()[:8], 16))
                        noise.append(torch.randn(images.shape[1:], generator=gen))
                    images = (images + .1*torch.stack(noise).to(device)).clamp(0, 1)
                else:
                    images[..., 32:64, 32:64] = 0
                shifted = model(replace(evidence, images=images))
                if not torch.equal(output["mask"], shifted["mask"]):
                    raise AssertionError("Intervention changed compilation")
                frame[f"{kind}_shift"] = (shifted["designated_state"]-output["designated_state"]).abs().amax(-1).cpu().numpy()
                arrays[f"{kind}_state"].append(shifted["designated_state"].cpu().numpy())
                for index, grade in enumerate((2, 3, 4)):
                    frame[f"{kind}_p{grade}"] = shifted["p_star"][:, index].cpu().numpy()
                    frame[f"{kind}_raw_p{grade}"] = shifted["p0"][:, index].cpu().numpy()
            # Withdraw IDH and its testing method, then recompile; not a state-shift comparison.
            molecular = evidence.molecular.clone()
            molecular[:, :3] = molecular.new_tensor([0, 0, 1])
            molecular[:, 9:12] = molecular.new_tensor([0, 0, 1])
            withdrawn = model(replace(evidence, molecular=molecular))
            frame["withdrawn_active"] = withdrawn["active"].cpu().numpy()
            if withdrawn["active"].any():
                raise AssertionError("Withdrawal failed to revoke rule")
        frames.append(frame)
        arrays["subject_id"].extend(batch["subject_id"])
        for key in ("designated_state", "mask", "adjacency", "active"):
            arrays[key].append(cpu(key))
    return pd.concat(frames, ignore_index=True), {k: np.asarray(v) if k == "subject_id" else np.concatenate(v)
                                                 for k, v in arrays.items()}


def train(args):
    seed_all()
    out, manifest, rows = load_protocol(args)
    audit = json.loads((out/"image_audit.json").read_text())
    if audit["errors"] or not audit["complete"]:
        raise ValueError("Formal training requires complete successful image audit")
    if not (out/"smoke.json").exists() or not json.loads((out/"smoke.json").read_text())["passed"]:
        raise ValueError("Run and pass the small-batch learning check before formal training")
    run = out / f"fold{args.fold}" / args.variant
    run.mkdir(parents=True, exist_ok=True)
    config = dict(fold=args.fold, variant=args.variant, epochs=50, batch_size=8, lr=3e-4,
                  weight_decay=1e-4, seed=101, source_hash=source_hash(), rule_version=RULE_VERSION)
    if (run/"config.json").exists() and json.loads((run/"config.json").read_text()) != config:
        raise ValueError("Run configuration changed")
    save_json(run/"config.json", config)
    if (run/"DONE.json").exists():
        print(f"Already complete: {run}", flush=True)
        return
    split = json.loads((out/"splits"/f"fold{args.fold}.json").read_text())
    subsets = {key: [r for r in rows if r["subject_id"] in ids] for key, ids in split.items()}
    batches = loader(subsets["train"], out, train=True)
    validation = loader(subsets["validation"], out)
    device = torch.device(args.device)
    model = UTSWAuthorityGrade(args.variant).to(device)
    initial = hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    save_json(run/"environment.json", dict(torch=torch.__version__, python=platform.python_version(),
              device=str(device), gpu=torch.cuda.get_device_name() if device.type=="cuda" else None, initial_sha256=initial))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best, start, history = float("inf"), 0, []
    if (run/"last.pt").exists():
        state = torch.load(run/"last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, best, history = state["epoch"]+1, state["best"], state["history"]
        torch.set_rng_state(state["torch_rng"])
        if device.type=="cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        batches.generator.set_state(state["loader_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
    for epoch in range(start, 50):
        model.train()
        total = 0.
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            output = model(evidence_from_batch(batch, device))
            loss = grade_loss(output, batch["label"].to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Nonfinite gradient")
            optimizer.step()
            total += loss.item()*len(batch["label"])
        frame, _ = predict(model, validation, device, args.fold, args.variant)
        stats = metrics(frame)
        history.append(dict(epoch=epoch+1, train_loss=total/len(subsets["train"]), **stats))
        if stats["brier"] < best:
            best = stats["brier"]
            torch.save(dict(model=model.state_dict(), epoch=epoch+1, val_brier=best), run/"best.tmp.pt")
            (run/"best.tmp.pt").replace(run/"best.pt")
        save_json(run/"history.json", history)
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch, best=best,
                        history=history, torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if device.type=="cuda" else [],
                        loader_rng=batches.generator.get_state(), python_rng=random.getstate(), numpy_rng=np.random.get_state()),
                   run/"last.tmp.pt")
        (run/"last.tmp.pt").replace(run/"last.pt")
        print(f'{args.variant} fold={args.fold} epoch={epoch+1}/50 val_Brier={stats["brier"]:.5f} '
              f'val_accuracy={stats["accuracy"]:.3f}', flush=True)
    model.load_state_dict(torch.load(run/"best.pt", map_location=device, weights_only=False)["model"])
    val_frame, _ = predict(model, validation, device, args.fold, args.variant)
    val_frame.to_csv(run/"validation.csv", index=False)
    raw_classes = val_frame[[f"raw_p{g}" for g in (2,3,4)]].to_numpy().argmax(1)
    if val_frame.predicted_grade.nunique() == 1 or len(np.unique(raw_classes)) == 1:
        save_json(run/"VALIDATION_REVIEW_REQUIRED.json", dict(reason="Selected model predicts only one class before or after projection",
                  metrics=metrics(val_frame)))
        raise RuntimeError("Majority collapse: inspect validation before running tests or subsequent folds")
    test_frame, arrays = predict(model, loader(subsets["test"], out), device, args.fold, args.variant, intervention=True)
    if args.variant == "acf" and (test_frame.loc[test_frame.active, ["noise_shift", "occlusion_shift"]].to_numpy() > 1e-6).any():
        raise AssertionError("Protected-state invariance failed")
    test_frame.to_csv(run/"predictions.csv", index=False)
    np.savez_compressed(run/"states.npz", **arrays)
    save_json(run/"DONE.json", metrics(test_frame))


def cheap_controls(args):
    out, manifest, rows = load_protocol(args)
    all_frames = []
    for fold in range(5):
        split = json.loads((out/"splits"/f"fold{fold}.json").read_text())
        train_rows = [r for r in rows if r["subject_id"] in split["train"]]
        test_rows = [r for r in rows if r["subject_id"] in split["test"]]
        y = np.array([r["label"] for r in train_rows])
        target = np.array([r["label"] for r in test_rows])
        prior = np.tile(np.bincount(y, minlength=3)/len(y), (len(target), 1))
        logistic = LogisticRegression(C=1., solver="lbfgs", max_iter=2000, random_state=101)
        logistic.fit(np.array([r["features"] for r in train_rows]), y)
        molecular = logistic.predict_proba(np.array([r["features"] for r in test_rows]))
        active = np.array([r["active"] for r in test_rows])
        allowed = np.ones((len(target), 3), dtype=bool)
        allowed[:, 2] = ~active
        for name, p0 in (("prior", prior), ("prior_projected", prior), ("molecular_projected", molecular)):
            p = p0 if name == "prior" else project_allowed_simplex(torch.tensor(p0), torch.tensor(allowed)).numpy()
            all_frames.append(prediction_frame([r["subject_id"] for r in test_rows], target, p0, p, active,
                                              np.where(active, 0, 1), fold, name))
    pd.concat(all_frames).to_csv(out/"cheap_predictions.csv", index=False)


def smoke(args):
    seed_all()
    out, manifest, rows = load_protocol(args)
    # Only fold-0 training patients; balanced fixed IDs, never validation/test examples.
    allowed = set(json.loads((out/"splits"/"fold0.json").read_text())["train"])
    chosen = []
    for label in range(3):
        chosen += [r for r in rows if r["label"]==label and r["subject_id"] in allowed][:2]
    for row in chosen:
        prepare_case(args.data_root, row["subject_id"], out/"cache")
    batch = next(iter(loader(chosen, out)))
    device = torch.device(args.device)
    e = evidence_from_batch(batch, device)
    labels = batch["label"].to(device)
    model = UTSWAuthorityGrade("acf").to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    curve = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(e)
        loss = grade_loss(output, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite smoke loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Nonfinite smoke gradient")
        optimizer.step()
        curve.append(loss.item())
        if (step+1)%10==0:
            print(f"smoke step={step+1} loss={loss.item():.6f}", flush=True)
    frame, _ = predict(model, loader(chosen, out), device, -1, "smoke_only", intervention=True)
    passed = bool(frame.predicted_grade.nunique()==3 and metrics(frame)["accuracy"]>=.95 and curve[-1]<curve[0]*.5)
    save_json(out/"smoke.json", dict(passed=passed, patient_ids=[r["subject_id"] for r in chosen],
                                    steps=args.steps, losses=curve, metrics=metrics(frame),
                                    use="Implementation check on six training patients; not experimental performance"))
    frame.to_csv(out/"smoke_predictions.csv", index=False)
    if not passed:
        raise RuntimeError("Small-batch learning check failed; inspect implementation before formal training")


def aggregate(args):
    out, manifest, rows = load_protocol(args)
    frames = []
    for fold in range(5):
        for variant in ("acf", "learned_joint"):
            run = out/f"fold{fold}"/variant
            if not (run/"DONE.json").exists():
                raise ValueError(f"Incomplete formal run: {run}")
            frames.append(pd.read_csv(run/"predictions.csv"))
        a = json.loads((out/f"fold{fold}"/"acf"/"environment.json").read_text())
        b = json.loads((out/f"fold{fold}"/"learned_joint"/"environment.json").read_text())
        if a["initial_sha256"] != b["initial_sha256"]:
            raise AssertionError("Initialization was not matched")
    frames.append(pd.read_csv(out/"cheap_predictions.csv"))
    frame = pd.concat(frames, ignore_index=True)
    expected_rows = {r["subject_id"]:r for r in rows}
    for name, group in frame.groupby("model"):
        if group.subject_id.duplicated().any() or set(group.subject_id)!=set(manifest["folds"]):
            raise AssertionError(f"OOF coverage failure: {name}")
        for row in group.itertuples():
            if row.fold != manifest["folds"][row.subject_id]:
                raise AssertionError("Patient evaluated on wrong test fold")
            reference = expected_rows[row.subject_id]
            if row.label != reference["label"] or bool(row.active) != reference["active"]:
                raise AssertionError("Export differs from locked labels or rule activation")
        p = group[[f"p{g}" for g in (2,3,4)]].to_numpy(float)
        p0 = group[[f"raw_p{g}" for g in (2,3,4)]].to_numpy(float)
        if not np.isfinite(p).all() or (p < -1e-7).any() or not np.allclose(p.sum(1),1,atol=1e-6):
            raise AssertionError("Invalid probability export")
        if not np.allclose(group.brier, ((p-np.eye(3)[group.label.to_numpy(int)])**2).mean(1), atol=1e-7):
            raise AssertionError("Brier cannot be independently reproduced")
        if name != "prior":
            allowed = np.ones_like(p,dtype=bool)
            allowed[:,2] = ~group.active.to_numpy(bool)
            projected = project_allowed_simplex(torch.tensor(p0),torch.tensor(allowed)).numpy()
            if not np.allclose(projected,p,atol=1e-6):
                raise AssertionError("Projection cannot be independently reproduced")
    frame.to_csv(out/"oof_predictions.csv", index=False)
    frame.loc[frame.predicted_grade != frame.reference_grade].to_csv(out/"failure_cases.csv", index=False)
    stats = {name: metrics(group) for name, group in frame.groupby("model")}
    save_json(out/"summary.json", stats)
    save_json(out/"fold_metrics.json", {f"{name}/fold{fold}": metrics(group)
              for (name, fold), group in frame.groupby(["model", "fold"])})
    intervention_stats = {}
    for name in ("acf","learned_joint"):
        for kind in ("noise","occlusion"):
            perturbed = frame.loc[frame.model==name].copy()
            for grade in (2,3,4):
                perturbed[f"p{grade}"] = perturbed[f"{kind}_p{grade}"]
                perturbed[f"raw_p{grade}"] = perturbed[f"{kind}_raw_p{grade}"]
            intervention_stats[f"{name}/{kind}"] = metrics(perturbed)
    save_json(out/"controlled_intervention_metrics.json",intervention_stats)
    pd.DataFrame({k: {a:b for a,b in v.items() if a!="confusion_matrix"} for k,v in stats.items()}).T.to_csv(out/"summary.csv")
    for name, stat in stats.items():
        pd.DataFrame(stat["confusion_matrix"], index=[2,3,4], columns=[2,3,4]).to_csv(out/f"confusion_{name}.csv")
    acf = frame.loc[frame.model=="acf"]
    save_json(out/"paired_bootstrap.json", {name: paired_bootstrap(acf, group, args.bootstrap)
              for name, group in frame.groupby("model") if name!="acf"})
    from glioma.visualization.utsw_authority import export_results
    export_results(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "smoke", "train", "controls", "aggregate"])
    parser.add_argument("--data-root")
    parser.add_argument("--metadata")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--variant", choices=["acf", "learned_joint"], default="acf")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--grids-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    {"prepare": prepare, "smoke": smoke, "train": train, "controls": cheap_controls, "aggregate": aggregate}[args.command](args)


if __name__ == "__main__":
    main()
