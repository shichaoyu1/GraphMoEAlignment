"""Training and export for the opt-in revision protocol."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess

import numpy as np
import torch
from torch.utils.data import DataLoader

from glioma.data.case_discovery import discover_semantic_cases
from glioma.data.utsw_dataset import UTSWROIPatientDataset, patient_cross_validation_split
from glioma.paper2_revision.evaluation import gate_interventions, summarize
from glioma.paper2_revision.losses import (family_supervision_loss, observed_field_classification_loss,
                                        proxy_anchor_loss, retrieval_loss, target_masks)
from glioma.paper2_revision.model import RevisionModel
from glioma.paper2_revision.protocol import (NODES, audit_split, fit_family_prior, fit_frequency,
                                          targets_for_subjects, validate_splits)
from glioma.semantic.losses import anchor_center_loss
from glioma.semantic.topology import anchor_family_ids
from glioma.semantic.vocab import build_anchor_vocab


METHODS = ("direct", "global", "multilabel", "frequency", "node_frequency",
           "calibration", "factorized", "mean_positive_proxy", "proxy_anchor_multilabel")


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def source_manifest():
    root = Path(__file__).resolve().parents[1]
    files = list((root / "paper2_revision").glob("*.py")) + [
        root / "cli" / "train_paper2_revision.py", root / "cli" / "run_paper2_revision.py",
        root / "cli" / "aggregate_paper2_revision.py", root / "anchors" / "__init__.py",
        root / "cli" / "evaluate_paper2_transfer.py",
        root / "data" / "utsw_dataset.py", root / "data" / "case_discovery.py",
        root / "models" / "encoders.py", root / "models" / "glioma_topomoe_net.py",
        root / "semantic" / "vocab.py", root / "semantic" / "losses.py",
        root / "semantic" / "topology.py", root / "objectives" / "__init__.py",
    ]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
        revision = result.stdout.strip() if result.returncode == 0 else None
    except FileNotFoundError:
        revision = None  # Source hashes remain available on servers without Git.
    return {"git_head": revision, "source_sha256": hashes,
            "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}


def resolve_splits(cases, args):
    if args.splits_file:
        membership = json.loads(Path(args.splits_file).read_text(encoding="utf-8"))
        lookup = {str(case["subject_id"]): case for case in cases}
        if set(membership) != {"train", "val", "test"}:
            raise ValueError("Splits file must contain train/val/test")
        splits = {name: [lookup[str(subject)] for subject in ids] for name, ids in membership.items()}
        if {str(c["subject_id"]) for values in splits.values() for c in values} != set(lookup):
            raise ValueError("Splits file must cover the discovered cohort exactly")
    else:
        splits = patient_cross_validation_split(cases, args.cv_folds, args.cv_fold, args.cv_seed)
    validate_splits(splits)
    return splits


def load_reference(model, args, splits, vocab):
    root = Path(args.reference_run)
    old_splits = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    current = {name: sorted(str(c["subject_id"]) for c in values) for name, values in splits.items()}
    if {name: sorted(map(str, ids)) for name, ids in old_splits.items()} != current:
        raise ValueError("Reference patient splits do not match")
    if json.loads((root / "anchor_vocab.json").read_text(encoding="utf-8")) != vocab:
        raise ValueError("Reference anchor vocabulary does not match")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    for key in ("z_slices", "roi_size", "feat_dim", "shared_dim", "target_policy", "temperature"):
        if config.get(key) != getattr(args, key):
            raise ValueError("Reference configuration mismatch: " + key)
    if bool(config.get("prefer_registered", False)) != args.prefer_registered:
        raise ValueError("Reference configuration mismatch: prefer_registered")
    if config.get("node_mode", "regions") != "regions" or config.get("include_clinical_anchors", False) or config.get("experiment_profile") != "direct_only":
        raise ValueError("Reference must be a region-mode direct_only run without clinical anchors")
    checkpoint = root / args.reference_checkpoint
    if checkpoint.parent.resolve() != root.resolve():
        raise ValueError("Reference checkpoint must be directly inside reference_run")
    # A user-selected local experiment checkpoint, not downloaded external weights.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("anchor_vocab") != vocab:
        raise ValueError("Vocabulary embedded in the reference checkpoint does not match")
    model.encoder.load_state_dict(payload["model"], strict=True)
    model.bank.load_state_dict(payload["bank"], strict=True)
    return {"checkpoint": str(checkpoint.resolve()), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "selection": payload.get("selection"), "scope": "frozen direct encoder and prototypes"}


def make_loaders(splits, args):
    loaders = {}
    for offset, (name, cases) in enumerate(splits.items()):
        dataset = UTSWROIPatientDataset(cases, roi_size=args.roi_size, z_slices=args.z_slices,
                                       prefer_registered=args.prefer_registered,
                                       augment=name == "train" and args.augment, cache=args.cache)
        generator = torch.Generator().manual_seed(args.seed + offset)
        loaders[name] = DataLoader(dataset, batch_size=args.batch_size, shuffle=name == "train",
                                   num_workers=args.num_workers, generator=generator)
    return loaders


def run_epoch(model, loader, lookup, vocab, family_ids, args, device, stage, optimizer=None):
    training = optimizer is not None
    model.train(training)
    if stage == "router":
        model.encoder.eval()
        model.bank.eval()
    totals, active = Counter(), 0
    for batch in loader:
        targets = targets_for_subjects(batch["subject_id"], lookup, args.target_policy, vocab)
        positives, candidates = target_masks(targets, len(vocab), device, args.candidate_policy)
        if not positives.any():
            continue
        with torch.set_grad_enabled(training):
            output = model(batch["images"].to(device), batch["region_masks"].to(device), stage)
            logits = output["scores"].reshape(-1, len(vocab))
            if model.method == "multilabel":
                loss = observed_field_classification_loss(output["classification_logits"].reshape_as(logits), targets, vocab)
            elif model.method == "proxy_anchor_multilabel":
                loss = proxy_anchor_loss(output["cosine"].reshape_as(logits), positives, candidates, args.proxy_alpha, args.proxy_margin)
            else:
                loss = retrieval_loss(logits, positives, candidates, model.method == "mean_positive_proxy")
            terms = {"retrieval": loss}
            if model.method not in {"multilabel", "proxy_anchor_multilabel", "mean_positive_proxy"}:
                center = anchor_center_loss(output["shared"].reshape(-1, args.shared_dim), [list(t.positives) for t in targets], model.bank())
                loss = loss + args.lambda_anchor * center
                terms["anchor"] = center
            if model.head is not None and stage != "direct":
                if args.family_supervision == "on":
                    family_loss = family_supervision_loss(output["gates"].reshape(-1, model.head.num_families), positives,
                                                         torch.as_tensor(family_ids, device=device))
                    loss = loss + args.lambda_family * family_loss
                    terms["family"] = family_loss
                if args.lambda_within:
                    within = []
                    cond = output["conditional"].reshape_as(logits)
                    for family in sorted(set(family_ids)):
                        ids = torch.as_tensor([i for i, f in enumerate(family_ids) if f == family], device=device)
                        if positives[:, ids].any():
                            within.append(retrieval_loss(cond[:, ids], positives[:, ids], candidates[:, ids]))
                    if within:
                        within_loss = torch.stack(within).mean()
                        loss = loss + args.lambda_within * within_loss
                        terms["within"] = within_loss
                # No hidden specialization/balance/distillation terms in this protocol.
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite objective; refusing to skip a failed batch")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
                optimizer.step()
                totals["gradient_norm"] += float(norm)
            for name, value in {**terms, "total": loss}.items():
                totals[name] += float(value.detach())
            active += 1
    if not active:
        raise ValueError("No valid positive queries in this epoch")
    return {"n_batches": active, "losses": {key: value / active for key, value in totals.items()}}


@torch.no_grad()
def collect(model, loader, lookup, vocab, args, device, stage="joint"):
    model.eval()
    rows, direct_rows, gates, conditional, subjects, nodes, targets, mask_counts = [], [], [], [], [], [], [], []
    for batch in loader:
        output = model(batch["images"].to(device), batch["region_masks"].to(device), stage)
        rows.extend(output["scores"].reshape(-1, len(vocab)).cpu().tolist())
        direct_rows.extend((output["cosine"] / args.temperature).reshape(-1, len(vocab)).cpu().tolist())
        if "gates" in output:
            gates.extend(output["gates"].reshape(-1, output["gates"].shape[-1]).cpu().tolist())
            conditional.extend(output["conditional"].reshape(-1, len(vocab)).cpu().tolist())
        subjects.extend(str(subject) for subject in batch["subject_id"] for _ in NODES)
        nodes.extend(node for _ in batch["subject_id"] for node in NODES)
        targets.extend(targets_for_subjects(batch["subject_id"], lookup, args.target_policy, vocab))
        mask_counts.extend(batch["region_masks"].sum(dim=(2, 3, 4)).reshape(-1).tolist())
    return {"scores": rows, "direct_scores": direct_rows, "targets": targets, "subjects": subjects, "nodes": nodes,
            "gates": gates, "conditional": conditional, "empty_region_queries": int(sum(value == 0 for value in mask_counts))}


def export_records(path, records, vocab, family_ids, family_names, lookup, args, split):
    grades = {subject: str(lookup[subject]["metadata"].get("Tumor Grade", "unknown")) for subject in records["subjects"]}
    metrics = summarize(records["scores"], records["targets"], records["subjects"], records["nodes"], grades, family_ids)
    payload = {"schema": "paper2_revision_v1", "split": split, "method": args.method,
               "target_policy": args.target_policy, "candidate_policy": args.candidate_policy,
               "anchor_keys": [a["key"] for a in vocab], "family_ids": family_ids, "family_names": family_names,
               "subject_ids": records["subjects"], "node_names": records["nodes"], "grades": grades,
               "query_targets": [list(t.positives) for t in records["targets"]],
               "observed_candidates": [list(t.observed_candidates) for t in records["targets"]],
               "scores": records["scores"], "routing_weights": records["gates"],
               "family_log_probs": records["conditional"], "empty_region_queries": records["empty_region_queries"],
               "metrics": metrics}
    if "direct_scores" in records and args.method in {"calibration", "factorized"}:
        payload["direct_scores"] = records["direct_scores"]
        payload["direct_metrics"] = summarize(records["direct_scores"], records["targets"], records["subjects"], records["nodes"], grades, family_ids)
        payload["direct_score_scope"] = "same-checkpoint direct score replay; separate training baseline only when calibration uses a frozen reference"
    if split == "test" and records["gates"]:
        interventions, audit = gate_interventions(records["gates"], records["conditional"], records["subjects"], records["nodes"], family_ids, args.seed)
        payload["intervention_audit"] = audit
        payload["intervention_scores"] = {name: scores.tolist() for name, scores in interventions.items()}
        payload["intervention_metrics"] = {name: summarize(scores, records["targets"], records["subjects"], records["nodes"], grades, family_ids) for name, scores in interventions.items()}
    save_json(path, payload)
    return metrics


def train(args):
    if args.method not in METHODS or args.cv_folds < 3:
        raise ValueError("Unsupported method or insufficient patient folds")
    if args.epochs <= 0 or args.patience < 0 or args.batch_size <= 0 or args.temperature <= 0 or args.lr <= 0 or args.router_epochs < 0 or args.joint_epochs < 0:
        raise ValueError("Invalid training configuration")
    if min(args.lambda_anchor, args.lambda_family, args.lambda_within) < 0:
        raise ValueError("Loss weights must be nonnegative")
    numeric = (args.lr, args.weight_decay, args.lambda_anchor, args.lambda_family, args.lambda_within,
               args.grad_clip, args.proxy_alpha, args.proxy_margin, args.temperature)
    if not all(np.isfinite(value) for value in numeric) or args.weight_decay < 0 or args.grad_clip <= 0 or args.proxy_alpha <= 0 or args.proxy_margin < 0:
        raise ValueError("Nonfinite or invalid loss/optimizer parameter")
    if args.roi_size < 8 or min(args.z_slices, args.feat_dim, args.shared_dim) <= 0 or args.num_workers < 0:
        raise ValueError("Invalid input/model dimensions or worker count")
    if args.method == "calibration" and (not args.reference_run or args.projection != "shared" or args.gate == "dynamic" or args.router_epochs <= 0):
        raise ValueError("Calibration requires a direct reference, shared scores and static gate")
    if args.reference_run and args.method not in {"calibration", "factorized"}:
        raise ValueError("Reference reuse is supported only by calibration and factorized heads")
    if args.method == "factorized" and args.router_epochs <= 0:
        raise ValueError("Factorized training requires a router stage")
    out = Path(args.out_dir)
    if out.exists():
        raise FileExistsError("Use a new output directory; existing experiment artifacts are never overwritten: " + str(out))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cases = discover_semantic_cases(args.data_root, args.metadata_tsv, args.max_cases, args.cv_seed)
    lookup = {str(case["subject_id"]): case for case in cases}
    splits = resolve_splits(cases, args)
    vocab, _ = build_anchor_vocab(splits["train"])
    if len(vocab) < 2:
        raise ValueError("At least two training anchors required")
    family_ids, family_names = anchor_family_ids(vocab, include_residual=False)
    initial_gate = fit_family_prior(splits["train"], args.target_policy, vocab, family_ids, args.gate == "node")
    manifest = source_manifest()
    model, reference = None, None
    if args.method not in {"frequency", "node_frequency"}:
        model = RevisionModel(vocab, family_ids, args.method, args.projection, args.gate, args.z_slices,
                              args.feat_dim, args.shared_dim, args.temperature, initial_gate)
        if args.reference_run:
            reference = load_reference(model, args, splits, vocab)
    out.mkdir(parents=True)
    save_json(out / "config.json", vars(args))
    save_json(out / "source_manifest.json", manifest)
    save_json(out / "splits.json", {name: [str(c["subject_id"]) for c in values] for name, values in splits.items()})
    save_json(out / "anchor_vocab.json", vocab)
    audit = {name: audit_split(values, args.target_policy, vocab) for name, values in splits.items()}
    save_json(out / "protocol_audit.json", {"target_policy": args.target_policy, "candidate_policy": args.candidate_policy,
              "splits": audit, "label_source": "patient metadata with explicit region rules; no regional pathology annotation",
              "unknown_labels": "not positive; full-gallery competition retained unless observed_fields is enabled",
              "negative_sampling": "none; all allowed anchors compete", "tie_policy": "stable anchor-vocabulary order",
              "region_masks": "provided segmentation labels 1/2/4; empty masks retained and counted",
              "segmentation_source": args.segmentation_source, "initial_gate_train_only": initial_gate.tolist(),
              "specialization_balance_distillation_weights": [0, 0, 0]})
    grades = {subject: str(case["metadata"].get("Tumor Grade", "unknown")) for subject, case in lookup.items()}
    if args.method in {"frequency", "node_frequency"}:
        probabilities = fit_frequency(splits["train"], args.target_policy, vocab, args.method == "node_frequency")
        for name in ("val", "test"):
            subjects = [str(c["subject_id"]) for c in splits[name] for _ in NODES]
            targets = targets_for_subjects([str(c["subject_id"]) for c in splits[name]], lookup, args.target_policy, vocab)
            scores = np.log(probabilities[np.tile(np.arange(3), len(splits[name])) if len(probabilities) == 3 else np.zeros(len(subjects), dtype=int)])
            records = {"scores": scores.tolist(), "targets": targets, "subjects": subjects,
                       "nodes": list(NODES) * len(splits[name]), "gates": [], "conditional": [], "empty_region_queries": None}
            export_records(out / (name + "_records.json"), records, vocab, family_ids, family_names, lookup, args, name)
        save_json(out / "frequency_fit.json", {"fit_split": "train", "probabilities": probabilities.tolist()})
        save_json(out / "completed.json", {"status": "complete", "kind": "train-only metadata baseline"})
        return out
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if reference is not None:
        save_json(out / "reference_manifest.json", reference)
    model.to(device)
    loaders = make_loaders(splits, args)
    routed = model.head is not None
    schedule = [] if args.reference_run else [("direct" if routed else "joint", args.epochs, args.lr)]
    if routed:
        schedule.append(("router", args.router_epochs, args.lr))
        if args.method != "calibration" and args.joint_epochs:
            schedule.append(("joint", args.joint_epochs, args.lr * 0.1))
    history, selections = [], {}
    previous_checkpoint = None
    for stage, epochs, lr in schedule:
        if previous_checkpoint:
            model.load_state_dict(torch.load(previous_checkpoint, map_location=device, weights_only=True))
        for parameter in model.parameters():
            parameter.requires_grad_(stage != "router")
        if model.head is not None:
            for parameter in model.head.parameters():
                parameter.requires_grad_(stage != "direct")
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=args.weight_decay)
        best_score, stale = -float("inf"), 0
        checkpoint = out / ("best_" + stage + ".pt")
        for epoch in range(1, epochs + 1):
            train_log = run_epoch(model, loaders["train"], lookup, vocab, family_ids, args, device, stage, optimizer)
            records = collect(model, loaders["val"], lookup, vocab, args, device, stage)
            metrics = summarize(records["scores"], records["targets"], records["subjects"], records["nodes"], grades, family_ids)
            score = metrics[args.candidate_policy]["metrics"]["map"]
            if score is None:
                raise ValueError("No valid validation queries; cannot select a checkpoint")
            history.append({"stage": stage, "epoch": epoch, "train": train_log,
                            "validation_map": score, "selection_protocol": args.candidate_policy})
            if score > best_score:
                best_score, stale = score, 0
                torch.save(model.state_dict(), checkpoint)
                selections[stage] = {"epoch": epoch, "validation_map": score, "file": checkpoint.name}
            else:
                stale += 1
            save_json(out / "history.json", history)
            print(f"{args.method} {stage} epoch={epoch} validation mAP={score:.6f}", flush=True)
            if args.patience and stale >= args.patience:
                break
        previous_checkpoint = checkpoint
    # Select router versus joint by validation only; no test-based stage choice.
    eligible = [stage for stage in selections if not routed or stage != "direct"]
    selected = max(eligible, key=lambda stage: selections[stage]["validation_map"])
    model.load_state_dict(torch.load(out / selections[selected]["file"], map_location=device, weights_only=True))
    save_json(out / "checkpoint_manifest.json", {"primary_stage": selected, "selection_split": "val", "stages": selections})
    for name in ("val", "test"):
        export_records(out / (name + "_records.json"), collect(model, loaders[name], lookup, vocab, args, device),
                       vocab, family_ids, family_names, lookup, args, name)
    save_json(out / "completed.json", {"status": "complete", "device": str(device),
                                      "n_parameters": sum(p.numel() for p in model.parameters())})
    return out
