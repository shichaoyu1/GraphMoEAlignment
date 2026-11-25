"""Run the seed-42 endpoint subset of the Idea 3 full matrix."""

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from glioma.cli.train_idea3_matched import atomic_csv, full_gate, manifest_data, run_point, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="Root containing the full matrix and a separate screen summary")
    args = parser.parse_args()
    manifest, digest = manifest_data(args.prepared)
    source = Path(__file__).resolve().parents[1]
    # Match train_idea3_matched's fingerprint so full reuses these runs, while
    # preserving the runner and fingerprints of the completed verify matrix.
    modules = ("cli/train_idea3_matched.py", "training/idea3_matched.py", "training/idea3.py",
               "models/idea3.py", "data/idea3.py", "eval/idea3.py", "cli/diagnose_idea3_anchor.py")
    source_hash = hashlib.sha256(b"".join((source / path).read_bytes() for path in modules)).hexdigest()
    provenance = {"matrix": "full", "manifest_sha256": digest, "source_sha256": source_hash}
    settings = argparse.Namespace(matrix="full", epochs=20, batch_size=12, device="cuda")
    by_id = {r["subject_id"]: r for r in manifest["rows"]}
    frames = []
    for fold, split in enumerate(manifest["splits"]):
        train, val, test = ([by_id[sid] for sid in split[role]] for role in ("train", "val", "test"))
        images = {}
        for weight in (.05, .95):
            frames.append(run_point(args.prepared, args.output / "full", train, val, test, fold, 42,
                                    weight, .50, settings, images, provenance))
    predictions = pd.concat(frames, ignore_index=True)
    report = full_gate(predictions, (42,), 2000)
    report["phenotype_subset_sensitivity"] = full_gate(predictions[predictions.phenotype.notna()], (42,), 2000)
    report["stage"] = "single_seed_screen"
    folder = args.output / "screen"
    folder.mkdir(parents=True, exist_ok=True)
    atomic_csv(predictions, folder / "oof_predictions.csv")
    save_json(folder / "screen_gate.json", report)
    print(f"Screen complete: passed={report['passed']}; summary={folder / 'screen_gate.json'}", flush=True)


if __name__ == "__main__":
    main()
