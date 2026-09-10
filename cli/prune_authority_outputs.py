"""Safely prune regenerable artifacts from completed authority runs."""

import argparse
import json
from pathlib import Path


ROOT_MARKERS = ("s0.json", "pipeline.status", "development_manifest.json", "frozen_protocol.json")


def _validated_root(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Authority output root does not exist: {root}")
    if root == Path(root.anchor):
        raise ValueError("Refusing to prune a filesystem root")
    if not any((root / marker).exists() for marker in ROOT_MARKERS):
        raise ValueError(f"No authority protocol marker found in {root}")
    return root


def completed_run_directories(root):
    root = _validated_root(root)
    for done in sorted(root.glob("*/*/DONE.json")):
        run = done.parent
        if (run / "RUNNING.lock").exists():
            continue
        config_path = run / "config.json"
        if not config_path.exists():
            raise ValueError(f"Completed run has no config: {run}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        result = json.loads(done.read_text(encoding="utf-8"))
        if result.get("config_hash") != config.get("config_hash"):
            raise ValueError(f"Config/DONE hash mismatch: {run}")
        yield run


def prune(root, remove_event_arrays=False, remove_last_checkpoints=False,
          remove_best_checkpoints=False, apply=False):
    if not any((remove_event_arrays, remove_last_checkpoints, remove_best_checkpoints)):
        raise ValueError("Select at least one artifact class to prune")
    targets = []
    for run in completed_run_directories(root):
        if remove_event_arrays:
            targets.extend(path for path in (run / "events").glob("*.npz") if path.is_file())
        if remove_last_checkpoints and (run / "last.pt").is_file():
            targets.append(run / "last.pt")
        if remove_best_checkpoints and (run / "best.pt").is_file():
            targets.append(run / "best.pt")
    bytes_total = sum(path.stat().st_size for path in targets)
    if apply:
        for path in targets:
            path.unlink()
    return {"mode": "applied" if apply else "dry-run", "files": len(targets),
            "bytes": bytes_total, "gib": bytes_total / 1024 ** 3,
            "sample": [str(path) for path in targets[:20]]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--completed-only", action="store_true",
                        help="Accepted for clarity; pruning is always restricted to verified completed runs")
    parser.add_argument("--remove-event-arrays", action="store_true")
    parser.add_argument("--remove-last-checkpoints", action="store_true")
    parser.add_argument("--remove-best-checkpoints", action="store_true")
    parser.add_argument("--apply", action="store_true", help="Delete listed files; omission is a dry run")
    args = parser.parse_args(argv)
    result = prune(args.root, args.remove_event_arrays, args.remove_last_checkpoints,
                   args.remove_best_checkpoints, args.apply)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
