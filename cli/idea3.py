"""Run the Idea 3 anchor-phenotype pilot and constrained-grading experiment."""

import argparse
import hashlib
import json
from pathlib import Path

from glioma.cli.utsw_authority import save_json
from glioma.data.idea3 import (ANCHORS, IMAGE_SIZE, anchor_targets, case_signature, image_sources,
                               manual_masks, phenotype_target, prepare_images)
from glioma.data.utsw_dataset import load_utsw_metadata, parse_utsw_label, patient_cross_validation_split


def prepare_anchor(data_root, metadata_path, out):
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    metadata = load_utsw_metadata(str(metadata_path))
    rows, audit, errors = [], [], []
    reference = None
    manifest_file = out / "manifest.json"
    previous = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.exists() else None
    inputs = {"data_root": str(data_root.resolve()), "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
              "matrix_size": IMAGE_SIZE}
    if previous and previous["inputs"] != inputs:
        raise ValueError("The prepared inputs differ. Choose a new output directory.")
    prior_rows = {row["subject_id"]: row for row in previous["rows"]} if previous else {}
    for sid, record in sorted(metadata.items()):
        folder = data_root / sid
        masks = manual_masks(folder)
        anchors = anchor_targets(record)
        base = {"subject_id": sid, "anchors": anchors, "grade": parse_utsw_label(record, "grade")}
        if not masks:
            audit.append({**base, "status": "no_manual_mask" if folder.is_dir() else "source_missing"})
            continue
        try:
            paths, volumes, current_grid = image_sources(folder, reference)
            reference = current_grid
            signature = hashlib.sha256(repr(case_signature(paths + masks)).encode()).hexdigest()
            if previous and prior_rows.get(sid, {}).get("source_signature") != signature:
                raise ValueError("Source files differ from the prepared cohort. Choose a new output directory.")
            phenotype = phenotype_target(masks, reference)
            prepare_images(paths, volumes, cache / f"{sid}.npz")
            row = {**base, **phenotype, "source_signature": signature, "images": [p.name for p in paths]}
            rows.append(row)
            audit.append({**row, "status": phenotype["phenotype_status"]})
        except (ValueError, OSError) as error:
            errors.append({**base, "status": "invalid_source", "error": str(error)})
        if rows and rows[-1]["subject_id"] == sid and len(rows) % 50 == 0:
            print(f"Prepared {len(rows)} manual cases", flush=True)
    save_json(out / "cohort_audit.json", {"cases": audit + errors, "errors": errors})
    if errors or len(rows) < 5:
        raise ValueError(f"Could not certify the manual-mask cohort. See {out / 'cohort_audit.json'}")
    cases = [{"subject_id": row["subject_id"], "label": row["grade"], "metadata": metadata[row["subject_id"]]}
             for row in rows]
    splits = [{role: [row["subject_id"] for row in subset] for role, subset in
               patient_cross_validation_split(cases, fold=fold).items()} for fold in range(5)]
    manifest = {"inputs": inputs, "grid": reference, "rows": rows, "splits": splits, "split_seed": 2026,
                "fields": list(ANCHORS)}
    if previous and previous != manifest:
        raise ValueError("The prepared cohort or splits differ. Choose a new output directory.")
    save_json(manifest_file, manifest)
    print(f"Prepared {len(rows)} manual cases on a common grid", flush=True)


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "anchor", "authority", "all"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--authority-source", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--anchor-epochs", type=int, default=20)
    parser.add_argument("--authority-epochs", type=int, default=50)
    parser.add_argument("--anchor-batch-size", type=int, default=12)
    parser.add_argument("--authority-batch-size", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=2000)
    arguments = parser.parse_args()
    if min(arguments.anchor_epochs, arguments.authority_epochs, arguments.anchor_batch_size,
           arguments.authority_batch_size, arguments.bootstrap) < 1:
        parser.error("epochs, batch sizes and bootstrap draws must be positive")
    first = arguments.output / "experiment1"
    second = arguments.output / "experiment2"
    device = torch.device(arguments.device)
    if arguments.command in ("authority", "all") and arguments.authority_source is None:
        parser.error("authority and all require --authority-source")
    if arguments.command in ("prepare", "all"):
        if arguments.data_root is None or arguments.metadata is None:
            parser.error("prepare and all require --data-root and --metadata")
        prepare_anchor(arguments.data_root, arguments.metadata, first)
    if arguments.command in ("anchor", "all"):
        from glioma.cli.idea3_anchor import run_anchor
        result = run_anchor(first, device, arguments.anchor_epochs, arguments.anchor_batch_size, arguments.bootstrap)
        print(f'Preference stage: {result["preference_stage"]}', flush=True)
    if arguments.command in ("authority", "all"):
        from glioma.cli.idea3_authority import run_authority
        if second.resolve() == arguments.authority_source.resolve():
            parser.error("The authority output directory must differ from the archived source")
        run_authority(arguments.authority_source, second, device, arguments.authority_epochs,
                      arguments.authority_batch_size, arguments.bootstrap)


if __name__ == "__main__":
    main()
