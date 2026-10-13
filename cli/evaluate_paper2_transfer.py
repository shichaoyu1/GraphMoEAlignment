"""Frozen transfer to an independent cohort; no fitting or checkpoint selection."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from glioma.data.case_discovery import discover_semantic_cases
from glioma.data.utsw_dataset import UTSWROIPatientDataset
from glioma.paper2_revision.model import RevisionModel
from glioma.paper2_revision.protocol import NODES, audit_split, targets_for_subjects
from glioma.paper2_revision.training import collect, export_records, save_json, source_manifest
from glioma.semantic.topology import anchor_family_ids


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_run", required=True, help="Completed revision training run")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--metadata_tsv")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--segmentation_source", default="not_documented")
    parser.add_argument("--cpu", action="store_true")
    request = parser.parse_args(argv)
    source, out = Path(request.source_run), Path(request.out_dir)
    if out.exists():
        raise FileExistsError("External evaluation requires a new output directory")
    if not (source / "completed.json").is_file():
        raise ValueError("Source training run is incomplete")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    args = argparse.Namespace(**config)
    vocab = json.loads((source / "anchor_vocab.json").read_text(encoding="utf-8"))
    membership = json.loads((source / "splits.json").read_text(encoding="utf-8"))
    original_subjects = {str(subject) for subjects in membership.values() for subject in subjects}
    cases = discover_semantic_cases(request.data_root, request.metadata_tsv)
    lookup = {str(case["subject_id"]): case for case in cases}
    if not cases or len(lookup) != len(cases):
        raise ValueError("External cohort must contain unique patients with metadata")
    if original_subjects.intersection(lookup):
        raise ValueError("External and source patient IDs overlap; use verified globally unique IDs")
    family_ids, family_names = anchor_family_ids(vocab, include_residual=False)
    audit = audit_split(cases, args.target_policy, vocab)
    if audit["empty_target_queries"] == audit["n_queries"]:
        raise ValueError("No external positives in the frozen source vocabulary")
    checkpoint = None
    if args.method in {"frequency", "node_frequency"}:
        fitted = json.loads((source / "frequency_fit.json").read_text(encoding="utf-8"))
        probabilities = np.asarray(fitted["probabilities"])
        subjects = [str(case["subject_id"]) for case in cases for _ in NODES]
        rows = np.tile(np.arange(3), len(cases)) if len(probabilities) == 3 else np.zeros(len(subjects), dtype=int)
        records = {"scores": np.log(probabilities[rows]).tolist(), "subjects": subjects,
                   "nodes": list(NODES) * len(cases), "targets": targets_for_subjects(list(lookup), lookup, args.target_policy, vocab),
                   "gates": [], "conditional": [], "empty_region_queries": None}
    else:
        selected = json.loads((source / "checkpoint_manifest.json").read_text(encoding="utf-8"))
        checkpoint = source / selected["stages"][selected["primary_stage"]]["file"]
        if checkpoint.parent.resolve() != source.resolve():
            raise ValueError("Checkpoint must belong to source_run")
        model = RevisionModel(vocab, family_ids, args.method, args.projection, args.gate, args.z_slices,
                              args.feat_dim, args.shared_dim, args.temperature)
        device = torch.device("cuda" if torch.cuda.is_available() and not request.cpu else "cpu")
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.to(device).eval()
        dataset = UTSWROIPatientDataset(cases, roi_size=args.roi_size, z_slices=args.z_slices,
                                       prefer_registered=args.prefer_registered, augment=False, cache=False)
        records = collect(model, DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                            num_workers=args.num_workers), lookup, vocab, args, device)
    out.mkdir(parents=True)
    save_json(out / "transfer_manifest.json", {"source_run": str(source.resolve()),
              "mode": "frozen inference; no external fitting, calibration or selection",
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint else None,
              "source_config": config, "external_segmentation_source": request.segmentation_source,
              "cohort_audit": audit, "patient_overlap": 0, "vocabulary": "fixed source-training vocabulary"})
    save_json(out / "source_manifest.json", source_manifest())
    export_records(out / "external_records.json", records, vocab, family_ids, family_names, lookup, args, "external")
    save_json(out / "completed.json", {"status": "complete", "kind": "frozen external evaluation"})
    return out


if __name__ == "__main__":
    main()
