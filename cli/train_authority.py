"""Independent, resumable authority training; no changes to the MRI entry points."""

import argparse
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from glioma.data.authority_benchmarks import AuthoritySynthetic
from glioma.eval.authority_diagnostics import evaluate_suite, metrics, predict, synchronize
from glioma.models.authority_fusion import AuthorityFusion, TRAINED_VARIANTS, authority_loss


DEFAULTS = dict(task="s1", variant="acf", data_seed=41, model_seed=100, train_n=12000, val_n=3000,
                test_n=6000, epochs=50, batch_size=256, lr=.001, weight_decay=.0001,
                token_count=16, token_dim=8, spd_dim=8, hidden_dim=64, layers=2, cross_mass=.35,
                geometry="spd", anchors=True, mask_policy="authority", projection_backend="analytic",
                phase="development", attribution="base")
SOURCE_FILES = ("data/authority_benchmarks.py", "models/authority_fusion.py", "modules/authority_rules.py",
                "modules/authority_projection.py", "modules/hierarchical_spd_fusion.py",
                "eval/authority_diagnostics.py", "cli/train_authority.py",
                "cli/run_authority_protocol.py", "cli/aggregate_authority.py")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_hash():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for filename in SOURCE_FILES:
        digest.update(filename.encode())
        # Archives/git may normalize line endings; scientific code is otherwise exact.
        digest.update((root / filename).read_text(encoding="utf-8").replace("\r\n", "\n").encode())
    return digest.hexdigest()


def atomic_json(path, content):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_checkpoint(path, content):
    temporary = Path(str(path) + ".tmp")
    torch.save(content, temporary)
    os.replace(temporary, path)


def select_dominant_source(train, validation):
    """Fit equal ridge probes on training data, select source by validation Brier only."""
    scores = []
    for source in range(4):
        def features(batch):
            e = batch.evidence
            content = e.tokens[:, 0, source].mean(1).double()
            q = e.target_id.double()[:, None]
            statement = e.statements[:, source].double()
            return torch.cat([torch.ones(len(q), 1, dtype=torch.double), content, statement, q,
                              content*q, statement*q], -1)
        x, v = features(train), features(validation)
        y = F.one_hot(train.labels.long(), 4).double() if train.evidence.task == "s1" else train.labels.double()
        vy = F.one_hot(validation.labels.long(), 4).double() if train.evidence.task == "s1" else validation.labels.double()
        coefficients = torch.linalg.solve(x.T @ x + .01*torch.eye(x.shape[1], dtype=x.dtype), x.T @ y)
        predictions = (v @ coefficients).clamp(0, 1)
        if train.evidence.task == "s1":
            predictions = predictions / predictions.sum(-1, keepdim=True).clamp_min(1e-10)
        scores.append(float((predictions - vy).square().mean()))
    return int(np.argmin(scores)), scores


def build_model(config, dominant_source=0):
    keys = ("task", "variant", "token_dim", "spd_dim", "hidden_dim", "layers", "cross_mass", "geometry", "anchors", "mask_policy", "projection_backend")
    return AuthorityFusion(**{key: config[key] for key in keys}, dominant_source=dominant_source)


def run_training(settings, directory, device="cpu", full_interventions=True, evaluate=True, stop_after=None):
    """Resume at epoch boundaries, replaying any incomplete epoch with saved RNG state.

    stop_after is for interruption tests only and never changes the registered epoch budget.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config = dict(DEFAULTS, **settings)
    if min(config[k] for k in ("train_n", "val_n", "test_n", "epochs", "batch_size")) < 1:
        raise ValueError("Sample counts, epochs and batch size must be positive")
    config["source_hash"] = source_hash()
    config["rule_version"] = "authority-v1"
    config["evaluation_mode"] = "full" if evaluate and full_interventions else "clean" if evaluate else "none"
    config["config_hash"] = canonical_hash({k: v for k, v in config.items() if k != "config_hash"})
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous["config_hash"] != config["config_hash"]:
            raise ValueError(f"Refusing to mix changed code/configuration in {directory}; choose a new output root")
    else:
        atomic_json(config_path, config)
    completion = directory / "DONE.json"
    if completion.exists():
        result = json.loads(completion.read_text(encoding="utf-8"))
        if result["config_hash"] != config["config_hash"]:
            raise ValueError("Completion/config mismatch")
        return result
    seed = config["model_seed"]
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Fail on unsupported nondeterministic kernels rather than silently changing pairing.
    torch.use_deterministic_algorithms(True)
    def data(split, count):
        return AuthoritySynthetic(config["task"], config["data_seed"], split, count,
                                  config["token_count"], config["token_dim"])
    train = data("train", config["train_n"]).materialize()
    validation = data("val", config["val_n"]).materialize()
    dominant, probe_scores = select_dominant_source(train, validation) if config["variant"] == "dominant" else (0, None)
    model = build_model(config, dominant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    environment = dict(python=platform.python_version(), platform=platform.platform(), torch=torch.__version__,
                       numpy=np.__version__, cuda=torch.version.cuda, device=str(device),
                       accelerator=torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else platform.processor(),
                       threads=torch.get_num_threads(), dominant_source=dominant, probe_validation_brier=probe_scores,
                       parameters=sum(p.numel() for p in model.parameters()),
                       deterministic_algorithms=True, cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    atomic_json(directory / "environment.json", environment)
    latest = directory / "last.pt"
    start_epoch, best_brier, best_epoch, history, train_seconds = 0, float("inf"), 0, [], 0.
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved["config_hash"] != config["config_hash"]:
            raise ValueError("Checkpoint/config mismatch")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        start_epoch, best_brier, best_epoch = saved["epoch"], saved["best_brier"], saved["best_epoch"]
        history, train_seconds = saved["history"], saved["train_seconds"]
        torch.set_rng_state(saved["torch_rng"])
        np.random.set_state(saved["numpy_rng"]); random.setstate(saved["python_rng"])
        if torch.device(device).type == "cuda" and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, config["epochs"]):
        synchronize(device)
        started = time.perf_counter()
        model.train()
        order = torch.randperm(len(train.labels))
        total_loss, count = 0., 0
        for indices in order.split(config["batch_size"]):
            batch = train.take(indices).to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch.evidence)
            loss = authority_loss(output, batch.labels, config["task"], config["variant"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch+1}")
            loss.backward()
            if any(not bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None):
                raise FloatingPointError(f"Nonfinite gradient at epoch {epoch+1}")
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices); count += len(indices)
        synchronize(device)
        elapsed = time.perf_counter() - started
        train_seconds += elapsed
        validation_metrics = metrics(predict(model, validation, config["batch_size"], device), validation)
        brier = validation_metrics["brier"]
        if brier is None or not np.isfinite(brier):
            raise FloatingPointError("Validation Brier is unavailable/nonfinite")
        if brier < best_brier:
            best_brier, best_epoch = brier, epoch + 1
            atomic_checkpoint(directory / "best.pt", dict(model=model.state_dict(), epoch=best_epoch,
                              validation_brier=best_brier, config_hash=config["config_hash"], dominant_source=dominant))
        history.append(dict(epoch=epoch+1, loss=total_loss/count, validation_brier=brier, training_seconds=elapsed))
        atomic_checkpoint(latest, dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch+1,
                           best_brier=best_brier, best_epoch=best_epoch, history=history, train_seconds=train_seconds,
                           config_hash=config["config_hash"], torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                           python_rng=random.getstate(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
        atomic_json(directory / "history.json", history)
        print(json.dumps(dict(run=directory.name, epoch=epoch+1, validation_brier=brier)), flush=True)
        if stop_after is not None and epoch+1 >= stop_after:
            return {"status": "interrupted_for_test", "epoch": epoch+1}
    best = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"])
    result = dict(config_hash=config["config_hash"], source_hash=config["source_hash"], best_epoch=best_epoch,
                  best_validation_brier=best_brier, train_seconds=train_seconds,
                  peak_cuda_memory_bytes=int(torch.cuda.max_memory_allocated(device)) if torch.device(device).type == "cuda" else None,
                  parameters=environment["parameters"], evaluation="full" if evaluate and full_interventions else "clean" if evaluate else "none")
    if evaluate:
        summaries = evaluate_suite(model, data("test", config["test_n"]), directory / "events", config,
                                   config["batch_size"], device, full=full_interventions)
        result["clean"] = summaries["clean"]
    atomic_json(completion, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--task", choices=("s1", "s2"))
    parser.add_argument("--variant", choices=TRAINED_VARIANTS)
    for key in ("data_seed", "model_seed", "train_n", "val_n", "test_n", "epochs", "batch_size"):
        parser.add_argument("--" + key.replace("_", "-"), type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--clean-only", action="store_true")
    parser.add_argument("--no-evaluation", action="store_true", help="Development LR selection: validation only")
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    settings = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    for key in DEFAULTS:
        if getattr(args, key, None) is not None:
            settings[key] = getattr(args, key)
    result = run_training(settings, args.output, args.device, not args.clean_only, not args.no_evaluation)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
