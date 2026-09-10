"""Server protocol: development selection, frozen manifests, resume and sharding."""

import argparse
import json
import os
from pathlib import Path

import torch

from glioma.cli.train_authority import DEFAULTS, atomic_json, canonical_hash, run_training, source_hash
from glioma.data.authority_benchmarks import AuthoritySynthetic, s0_check
from glioma.eval.authority_diagnostics import ARTIFACT_LEVELS, evaluate_suite
from glioma.models.authority_fusion import AuthorityFusion, CORE_VARIANTS, TRAINED_VARIANTS


LEARNING_RATES = (.001, .0003, .0001)


def job_id(c):
    return f'{c["task"]}_{c["variant"]}_{c["attribution"]}_d{c["data_seed"]}_m{c["model_seed"]}'


def development_jobs():
    jobs = []
    for task in ("s1", "s2"):
        for variant in TRAINED_VARIANTS:
            for lr in LEARNING_RATES:
                config = dict(DEFAULTS, task=task, variant=variant, lr=lr)
                jobs.append(dict(id=job_id(config) + f"_lr{lr:g}", config=config))
    return jobs


def formal_jobs(phase, selected=None):
    if phase not in ("main", "attribution"):
        raise ValueError("Expected main or attribution")
    factors = [(variant, "base", {}) for variant in TRAINED_VARIANTS] if phase == "main" else [
        (variant, tag, setting) for tag, setting in (
            ("euclidean", {"geometry": "euclidean"}), ("no_anchors", {"anchors": False}),
            ("low_sample", {"train_n": 2048})) for variant in ("acf", "learned_joint")]
    if phase == "attribution":
        factors.append(("acf", "role_shift", {"mask_policy": "role_shift"}))
    jobs = []
    for task in ("s1", "s2"):
        for variant, attribution, override in factors:
            for data_seed in range(42, 47):
                for model_seed in range(101, 104):
                    config = dict(DEFAULTS, task=task, variant=variant, phase=phase, attribution=attribution,
                                  data_seed=data_seed, model_seed=model_seed, **override)
                    config["lr"] = selected[f"{task}/{variant}"]["lr"] if selected else None
                    jobs.append(dict(id=job_id(config), config=config))
    assert len(jobs) == (300 if phase == "main" else 210)
    assert len({j["id"] for j in jobs}) == len(jobs)
    return jobs


def freeze_development(root):
    root = Path(root)
    candidates = {}
    current_hash = source_hash()
    for job in development_jobs():
        directory = root / "development" / job["id"]
        done = directory / "DONE.json"
        if not done.exists():
            raise ValueError("All 60 equal-budget development trials must finish before freezing: " + job["id"])
        result = json.loads(done.read_text(encoding="utf-8"))
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if (result["config_hash"] != config["config_hash"] or
                config["config_hash"] != canonical_hash({k: v for k, v in config.items() if k != "config_hash"})):
            raise ValueError("Development result/configuration identity mismatch")
        if config["source_hash"] != current_hash or result["source_hash"] != current_hash:
            raise ValueError("Development results use a different source revision")
        for key, value in job["config"].items():
            if config[key] != value:
                raise ValueError("Development budget/configuration mismatch: " + key)
        key = f'{config["task"]}/{config["variant"]}'
        candidates.setdefault(key, []).append(dict(lr=config["lr"], validation_brier=result["best_validation_brier"],
                                                  config_hash=result["config_hash"], trial_id=job["id"]))
    selected = {key: min(values, key=lambda x: (x["validation_brier"], x["lr"])) for key, values in candidates.items()}
    frozen = dict(source_hash=current_hash, development_data_seed=41, development_model_seed=100,
                  selection_metric="minimum validation Brier; ties choose smaller learning rate",
                  selected=selected, candidates=candidates,
                  training_defaults=DEFAULTS, formal_trainings=510, main=300, attribution=210,
                  rule_version="authority-v1", clinical_authority_validation=False)
    frozen["protocol_hash"] = canonical_hash(frozen)
    path = root / "frozen_protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != frozen:
        raise ValueError("Existing frozen protocol differs; use a new root rather than overwrite it")
    atomic_json(path, frozen)
    for phase in ("main", "attribution"):
        atomic_json(root / f"{phase}_manifest.json", formal_jobs(phase, selected))
    return frozen


def load_frozen(root):
    path = Path(root) / "frozen_protocol.json"
    if not path.exists():
        raise ValueError("Run development and freeze before formal training; no automatic default LR")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if frozen["protocol_hash"] != canonical_hash({k: v for k, v in frozen.items() if k != "protocol_hash"}):
        raise ValueError("Frozen protocol was edited")
    if frozen["source_hash"] != source_hash():
        raise ValueError("Source differs from the frozen development version")
    return frozen


def rule_only_jobs(root, device, shard, shards, artifact_level="summary", minimum_free_gb=0):
    for index, (task, seed) in enumerate((task, seed) for task in ("s1", "s2") for seed in range(42, 47)):
        if index % shards != shard:
            continue
        directory = root / "rule_only" / f"{task}_d{seed}"
        config = dict(DEFAULTS, task=task, variant="rule_only", data_seed=seed, model_seed=0, phase="rule_only")
        config["source_hash"] = source_hash(); config["config_hash"] = canonical_hash(config)
        if (directory / "DONE.json").exists():
            completed = json.loads((directory / "DONE.json").read_text(encoding="utf-8"))
            if completed["config_hash"] != config["config_hash"]:
                raise ValueError("Rule-only configuration/source changed; use a new root")
            continue
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / "config.json", config)
        summary = evaluate_suite(AuthorityFusion(task, "rule_only").to(device),
                                  AuthoritySynthetic(task, seed, "test", 6000), directory / "events", config,
                                  device=device, artifact_level=artifact_level, minimum_free_gb=minimum_free_gb)
        atomic_json(directory / "DONE.json", dict(config_hash=config["config_hash"], source_hash=source_hash(),
                                                  clean=summary["clean"], train_seconds=0, parameters=0,
                                                  artifact_level=artifact_level, checkpoint_retention="none"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("plan", "smoke", "development", "freeze", "main", "attribution", "rule-only"), required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "reports/paper4_authority_v1")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Run a prefix of this shard; resume later with the same command")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--artifact-level", choices=ARTIFACT_LEVELS, default="summary")
    parser.add_argument("--checkpoint-retention", choices=("all", "best", "none"), default="none")
    parser.add_argument("--minimum-free-gb", type=float, default=2)
    args = parser.parse_args(argv)
    if not 0 <= args.shard < args.shards:
        parser.error("Need 0 <= shard < shards")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    args.root.mkdir(parents=True, exist_ok=True)
    atomic_json(args.root / "s0.json", s0_check())
    if args.phase == "plan":
        atomic_json(args.root / "development_manifest.json", development_jobs())
        for phase in ("main", "attribution"):
            atomic_json(args.root / f"{phase}_planned.json", formal_jobs(phase))
        print("Prepared 60 development trials + 300 main + 210 attribution; planned formal LR remains null until frozen.")
        return
    if args.phase == "freeze":
        frozen = freeze_development(args.root)
        print("Frozen protocol " + frozen["protocol_hash"])
        return
    if args.phase == "rule-only":
        load_frozen(args.root)
        rule_only_jobs(args.root, args.device, args.shard, args.shards, args.artifact_level, args.minimum_free_gb)
        return
    if args.phase == "smoke":
        jobs = []
        for task in ("s1", "s2"):
            for variant in CORE_VARIANTS:
                config = dict(DEFAULTS, task=task, variant=variant, phase="smoke", train_n=1024, val_n=256,
                              test_n=256, epochs=8, lr=.001)
                jobs.append(dict(id=job_id(config), config=config))
    elif args.phase == "development":
        jobs = development_jobs()
    else:
        frozen = load_frozen(args.root)
        jobs = formal_jobs(args.phase, frozen["selected"])
    pending = [job for i, job in enumerate(jobs) if i % args.shards == args.shard]
    if args.limit is not None:
        # The limit counts unfinished jobs, so repeated limited runs make forward progress.
        completed = args.root / args.phase
        pending = [j for j in pending if not (completed / j["id"] / "DONE.json").exists()][:args.limit]
    errors = []
    for job in pending:
        directory = args.root / args.phase / job["id"]
        lock = directory / "RUNNING.lock"
        directory.mkdir(parents=True, exist_ok=True)
        try:
            with lock.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(pid=os.getpid(), job=job["id"])))
        except FileExistsError as error:
            raise RuntimeError(f"Job is locked: {lock}. Check the recorded process before removing a stale lock.") from error
        try:
            run_training(job["config"], directory, args.device, evaluate=args.phase != "development",
                         artifact_level=args.artifact_level, checkpoint_retention=args.checkpoint_retention,
                         minimum_free_gb=args.minimum_free_gb)
        except Exception as error:
            atomic_json(directory / "FAILED.json", dict(type=type(error).__name__, message=str(error)))
            errors.append(job["id"])
            if not args.continue_on_error:
                raise
        finally:
            lock.unlink(missing_ok=True)
    if errors:
        raise RuntimeError("Failed jobs: " + ", ".join(errors))


if __name__ == "__main__":
    main()
