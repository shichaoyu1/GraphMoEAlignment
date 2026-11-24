"""Refit molecular readouts on frozen Idea 3 checkpoints without training the MRI encoder."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from glioma.cli.idea3_anchor import anchor_frame, atomic_csv, manifest_data
from glioma.cli.utsw_authority import save_json, seed_all
from glioma.eval.idea3 import PROB_COLS, TARGET_COLS, check_pairs
from glioma.models.idea3 import AnchorPhenotypeNet
from glioma.training.idea3 import forward_batches, linear_probes, loader, targets


def check_prepared(pilot, prepared):
    keys = ("subject_id", "anchors", "phenotype", "manual_mask", "phenotype_status", "images")
    rows = lambda manifest: {r["subject_id"]: tuple(r[k] for k in keys) for r in manifest["rows"]}
    same_settings = all(pilot[k] == prepared[k] for k in ("grid", "splits", "fields"))
    same_inputs = all(pilot["inputs"][k] == prepared["inputs"][k] for k in ("matrix_size", "metadata_sha256"))
    if not (same_settings and same_inputs and rows(pilot) == rows(prepared)):
        raise ValueError("Prepared data do not match the archived cohort, labels and five-fold protocol")


def raw_probe(embedding, anchors, test_embedding):
    predictions = []
    for y in anchors.T:
        observed = np.isfinite(y)
        model = LogisticRegression(C=1.0, max_iter=1000)
        model.fit(embedding[observed], y[observed])
        predictions.append(model.predict_proba(test_embedding)[:, 1])
    return np.column_stack(predictions)


def run(pilot, prepared, out, folds, device, batch_size):
    if out.resolve() in (pilot.resolve(), prepared.resolve()):
        raise ValueError("Choose a separate replay output directory")
    seed_all(42)
    manifest_path = pilot / "manifest.json"
    pilot_manifest = json.loads(manifest_path.read_bytes())
    prepared_manifest, _ = manifest_data(prepared)
    check_prepared(pilot_manifest, prepared_manifest)
    source = pd.read_csv(pilot / "anchor_oof_predictions.csv", float_precision="round_trip")
    original = source[source.seed.eq(42) & source.method.isin(["fixed", "probe"])]
    lookup = {r["subject_id"]: r for r in prepared_manifest["rows"]}
    model = AnchorPhenotypeNet().to(device).eval()
    report, frames = [], []
    for fold in folds:
        ids = prepared_manifest["splits"][fold]
        train, test = [[lookup[sid] for sid in sorted(ids[role])] for role in ("train", "test")]
        images = {}
        train_batches, test_batches = (loader(rows, prepared / "cache", images, batch_size, device)
                                       for rows in (train, test))
        for weight in (.05, .95):
            path = pilot / "runs" / "seed42" / f"fold{fold}" / f"w{weight:.2f}" / "best.pt"
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            _, _, train_embedding, _, _ = forward_batches(model, train_batches, weight, device)
            logits, phenotype, test_embedding, _, _ = forward_batches(model, test_batches, weight, device)
            joint = torch.column_stack((logits.sigmoid(), phenotype)).numpy()
            standardized = linear_probes(train_embedding.numpy(), train, test_embedding.numpy())
            raw = raw_probe(train_embedding.numpy(), targets(train)[0], test_embedding.numpy())
            scores = {"fixed": joint, "probe": standardized,
                      "refit_raw": np.column_stack((raw, joint[:, -1]))}
            record = {"fold": fold, "weight": weight, "train_n": len(train), "test_n": len(test),
                      "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for method, probabilities in scores.items():
                frame = anchor_frame(test, probabilities, fold, 42, weight, method)
                if method in ("fixed", "probe"):
                    archived = original[original.method.eq(method) & original.fold.eq(fold) & original.weight.eq(weight)]
                    archived = archived.sort_values("subject_id")
                    check_pairs(archived, frame, TARGET_COLS)
                    error = float(np.max(np.abs(archived[PROB_COLS].to_numpy() - probabilities)))
                    if not np.isfinite(error) or error > .002:
                        raise ValueError(f"{method} replay differs from the archive on fold {fold}, w={weight}: {error:.6g}")
                    record[f"{method}_max_abs_delta"] = error
                frames.append(frame)
            report.append(record)
            print(f"Replay fold={fold} w={weight:.2f} passed", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    atomic_csv(pd.concat(frames, ignore_index=True), out / "replay_predictions.csv")
    save_json(out / "replay_check.json", {"device": str(device), "checks": report})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True, help="Archived pilot experiment1 directory")
    parser.add_argument("--prepared", type=Path, help="Prepared experiment1 directory on this machine")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), action="append", help="Defaults to all five folds")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=12)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    run(args.pilot, args.prepared or args.pilot, args.output, sorted(set(args.fold or range(5))),
        torch.device(args.device), args.batch_size)


if __name__ == "__main__":
    main()
