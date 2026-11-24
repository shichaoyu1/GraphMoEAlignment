"""Run the Idea 3 checkpoint control or the fixed-input full matrix."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from glioma.cli.diagnose_idea3_anchor import aggregate_supported, diagnose, field_ap, resample_by_fold
from glioma.cli.idea3_anchor import anchor_frame, atomic_csv, manifest_data
from glioma.cli.replay_idea3_anchor import raw_probe
from glioma.cli.utsw_authority import save_json
from glioma.data.idea3 import ANCHORS
from glioma.eval.idea3 import TARGET_COLS, check_pairs, interval, tradeoff_pass
from glioma.models.idea3 import AnchorPhenotypeNet
from glioma.training.idea3 import forward_batches, linear_probes, loader, targets
from glioma.training.idea3_matched import fit


def run_point(prepared, out, train, val, test, fold, seed, weight, preference, args, images, provenance):
    run = out / f"seed{seed}" / f"fold{fold}" / f"w{weight:.2f}"
    config = {**provenance, "fold": fold, "seed": seed, "weight": weight, "preference": preference,
              "epochs": args.epochs, "batch_size": args.batch_size, "device": args.device}
    config_path = run / "config.json"
    if config_path.is_file() and json.loads(config_path.read_bytes()) != config:
        raise ValueError(f"Configuration changed for {run}. Use a fresh output directory.")
    output = run / "predictions.csv"
    if output.is_file():
        if not config_path.is_file():
            raise ValueError(f"Missing run configuration in {run}")
        return pd.read_csv(output, float_precision="round_trip")
    save_json(config_path, config)
    device = torch.device(args.device)
    cache = prepared / "cache"
    model, selected, history = fit(train, val, cache, images, device, seed, weight, preference,
                                   args.epochs, args.batch_size)
    save_json(run / "validation_history.json", history)
    snapshots = {"final": model.state_dict(), "selected": selected}
    training = loader(train, cache, images, args.batch_size, device)
    heldout = loader(test, cache, images, args.batch_size, device)
    frames = []
    for label, state in snapshots.items():
        torch.save(state, run / f"{label}.tmp.pt")
        (run / f"{label}.tmp.pt").replace(run / f"{label}.pt")
        readout = AnchorPhenotypeNet().to(device)
        readout.load_state_dict(state)
        logits, phenotype, embedding, _, _ = forward_batches(readout, heldout, preference, device)
        _, _, train_embedding, _, _ = forward_batches(readout, training, preference, device)
        direct = torch.column_stack((logits.sigmoid(), phenotype)).numpy()
        probe = linear_probes(train_embedding.numpy(), train, embedding.numpy())
        if not np.isfinite(direct).all() or not np.isfinite(probe).all():
            raise FloatingPointError(f"Nonfinite held-out predictions for {run}")
        frames.extend(anchor_frame(test, scores, fold, seed, weight, label + suffix)
                      for scores, suffix in ((direct, ""), (probe, "_probe")))
        if args.matrix == "verify":
            scores = raw_probe(train_embedding.numpy(), targets(train)[0], embedding.numpy())
            frames.append(anchor_frame(test, np.column_stack((scores, direct[:, -1])),
                                       fold, seed, weight, label + "_raw_probe"))
    frame = pd.concat(frames, ignore_index=True)
    atomic_csv(frame, output)
    print(f"Completed seed={seed} fold={fold} w={weight:.2f}", flush=True)
    return frame


def full_gate(predictions, seeds, repetitions):
    """Evaluate the prespecified deployed readout with shared patient draws across seeds."""
    final = predictions[predictions.method == "final"]
    ref = final[final.seed.eq(seeds[0]) & final.weight.eq(.05)].sort_values("subject_id")
    folds = ref.fold.to_numpy()
    draws = resample_by_fold(folds, repetitions)
    eligible = ref.phenotype.notna().to_numpy()
    labels = {field: ref[field].fillna(-1).to_numpy(dtype=int) for field in ANCHORS}
    contrasts = []
    for seed in seeds:
        endpoints = []
        for weight in (.05, .95):
            rows = final[final.seed.eq(seed) & final.weight.eq(weight)].sort_values("subject_id")
            check_pairs(ref, rows, TARGET_COLS)
            fields = []
            for field in ANCHORS:
                scores = rows[f"p_{field}"].to_numpy()
                per_fold = [field_ap(labels[field][mask], scores[mask], draws[:, mask])["ap_balanced"]
                            for mask in (folds == fold for fold in np.unique(folds))]
                fields.append(aggregate_supported(per_fold))
            errors = (rows.phenotype - rows.pred_phenotype).abs().to_numpy()[eligible]
            count = draws[:, eligible].sum(axis=1)
            mae = np.divide(draws[:, eligible] @ errors, count,
                            out=np.full(len(count), np.nan), where=count > 0)
            endpoints.append(np.stack((np.mean(fields, axis=0), mae)))
        contrasts.append(endpoints[1] - endpoints[0])
    means = np.mean(contrasts, axis=0)
    summary = {name: interval(values) for name, values in zip(("anchor_ap", "phenotype_mae"), means)}
    return {"seeds": list(seeds), "method": "final", "n_patients": len(ref), "n_phenotype": int(eligible.sum()),
            "high_minus_low": summary, "passed": tradeoff_pass(summary)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matrix", choices=("verify", "full"), default="verify")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.bootstrap) < 1:
        parser.error("epochs, batch size and bootstrap must be positive")
    if args.prepared.resolve() == args.output.resolve():
        parser.error("Use a separate output directory for training")
    manifest, digest = manifest_data(args.prepared)
    source = Path(__file__).resolve().parents[1]
    modules = ("cli/train_idea3_matched.py", "training/idea3_matched.py", "training/idea3.py",
               "models/idea3.py", "data/idea3.py", "eval/idea3.py", "cli/diagnose_idea3_anchor.py")
    source_hash = hashlib.sha256(b"".join((source / path).read_bytes() for path in modules)).hexdigest()
    provenance = {"matrix": args.matrix, "manifest_sha256": digest, "source_sha256": source_hash}
    by_id = {r["subject_id"]: r for r in manifest["rows"]}
    seeds, weights = ((42,), (.05, .95)) if args.matrix == "verify" else ((42, 43, 44), (.05, .50, .95))
    frames = []
    for fold, split in enumerate(manifest["splits"]):
        train, val, test = ([by_id[sid] for sid in split[role]] for role in ("train", "val", "test"))
        images = {}
        for seed in seeds:
            for weight in weights:
                preference = weight if args.matrix == "verify" else .50
                frames.append(run_point(args.prepared, args.output, train, val, test, fold, seed, weight,
                                        preference, args, images, provenance))
    predictions = pd.concat(frames, ignore_index=True)
    atomic_csv(predictions, args.output / "oof_predictions.csv")
    if args.matrix == "full":
        primary = full_gate(predictions, seeds, args.bootstrap)
        complete = predictions[predictions.phenotype.notna()]
        primary["phenotype_subset_sensitivity"] = full_gate(complete, seeds, args.bootstrap)
        save_json(args.output / "preference_gate.json", primary)
    for seed in seeds:
        pairs = [("final", "final_probe")]
        if args.matrix == "verify":
            pairs.extend([("selected", "final"), ("selected_probe", "final_probe"),
                          ("selected_raw_probe", "final_raw_probe"), ("final", "final_raw_probe")])
        for left, right in pairs:
            report, detail = diagnose(predictions, seed, args.bootstrap, left, right)
            folder = args.output / f"diagnosis_seed{seed}_{left}_vs_{right}"
            save_json(folder / "diagnosis.json", report)
            atomic_csv(detail, folder / "by_field_fold.csv")


if __name__ == "__main__":
    main()
