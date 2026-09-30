"""Run the Idea4.5 v3 phenomenon and mechanism experiment."""

import argparse
import hashlib
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

from glioma.data.idea4_5 import (
    INFORMATION_CONDITIONS, load_availability_npz, make_availability_splits, partition_training,
)
from glioma.io.artifacts import save_json
from glioma.models.idea4_5 import Idea45AvailabilityModel
from glioma.training.idea4_5_audit import audit_pair, brier_loss, masked_view
from glioma.training.engine import set_seed


def build_parser():
    parser = argparse.ArgumentParser(description="Idea4.5 v3 paired availability audit")
    parser.add_argument("--dataset", choices=["synthetic", "npz"], default="synthetic")
    parser.add_argument("--data_path")
    parser.add_argument("--label_mode", choices=["class", "sign", "sign_nozero"], default="class")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--information_condition", choices=INFORMATION_CONDITIONS, default="novel_signal")
    parser.add_argument("--extra_information", choices=["unique", "redundant"], help=argparse.SUPPRESS)
    parser.add_argument("--signal_form", choices=["mean", "covariance"], default="mean")
    parser.add_argument("--num_samples", type=int, default=6000)
    parser.add_argument("--token_count", type=int, default=12)
    parser.add_argument("--token_dim", type=int, default=8)
    parser.add_argument("--sparse_modalities", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--edge_target", type=int, default=0)
    parser.add_argument("--edge_source", type=int, default=1)
    parser.add_argument("--geometry", choices=["spd", "euclidean"], default="euclidean")
    parser.add_argument("--moment_mode", choices=["mean_augmented", "covariance"], default="mean_augmented")
    parser.add_argument("--readout_mode", choices=["nodewise", "pooled"], default="nodewise")
    parser.add_argument("--local_topology", choices=["uniform", "identity", "learned"], default="uniform")
    parser.add_argument("--edge_normalization", choices=["dynamic", "fixed"], default="dynamic")
    parser.add_argument("--shared_dim", type=int, default=32)
    parser.add_argument("--spd_dim", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--rich_fraction", type=float, default=0.8)
    parser.add_argument("--audit_size", type=int, help="Compatibility override for singleton groups")
    parser.add_argument("--audit_source_batches", type=int, default=20)
    parser.add_argument("--audit_batch_size", type=int, default=64)
    parser.add_argument("--audit_checkpoints", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    parser.add_argument("--audit_optimizers", choices=["mechanism_sgd", "stateful"], nargs="+",
                        default=["mechanism_sgd", "stateful"])
    parser.add_argument("--mechanism_lr", type=float, default=1e-3)
    parser.add_argument("--training_optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_norm", type=float, default=1.0, help="Set <=0 to disable training clipping")
    parser.add_argument("--data_seed", type=int, default=1001)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--train_seed", type=int, default=101)
    parser.add_argument("--model_seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--init_seed", type=int)
    parser.add_argument("--order_seed", type=int)
    parser.add_argument("--mask_seed", type=int)
    parser.add_argument("--audit_seed", type=int)
    parser.add_argument("--cpu", action="store_true")
    return parser


def _complete(dataset, modalities):
    observed = dataset.modality_mask[:, list(modalities)].all(dim=-1)
    tokens = (dataset.token_mask[:, :, list(modalities)].sum(dim=-1) >= 2).all(dim=-1).all(dim=-1)
    if not bool((observed & tokens).all()):
        raise ValueError("Paired protocol requires every rich-view modality and at least two valid tokens")


def _evaluate(model, dataset, modalities, batch_size, device):
    loss_sum = correct = count = 0
    model.eval()
    for batch in DataLoader(dataset, batch_size=batch_size):
        view = masked_view(batch, modalities, device)
        with torch.no_grad():
            logits = model(view["tokens"], view["modality_mask"], view["token_mask"])["logits"]
            loss_sum += float(brier_loss(logits, view["label"]).cpu()) * len(view["label"])
            correct += int((logits.argmax(dim=-1) == view["label"]).sum())
        count += len(view["label"])
    return {"brier": loss_sum / count, "accuracy": correct / count, "n": count}


def _full_batch(dataset):
    return default_collate([dataset[index] for index in range(len(dataset))])


def _audit_batches(dataset, count, batch_size, rng):
    if batch_size > len(dataset):
        raise ValueError("audit_batch_size exceeds the fit role")
    for _ in range(count):
        indices = rng.choice(len(dataset), size=batch_size, replace=False)
        yield default_collate([dataset[int(index)] for index in indices])


def _checkpoint_epochs(epochs, fractions):
    result = set()
    for value in fractions:
        if not 0 < value <= 1:
            raise ValueError("audit_checkpoints must be fractions in (0,1]")
        result.add(min(epochs, max(1, int(round(value * epochs)))))
    return result


def _rng_snapshot():
    payload = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        payload["cuda"] = torch.cuda.get_rng_state_all()
    return payload


def _save_audit_objects(path, audit):
    arrays = {name: np.asarray(values, dtype=np.float32) for name, values in audit.pop("per_object").items()}
    arrays["measure_sample_ids"] = np.asarray(audit["measure_sample_ids"])
    arrays["source_sample_ids"] = np.asarray(audit["source_sample_ids"])
    np.savez_compressed(path, **arrays)
    audit["per_object_file"] = os.path.basename(path)
    audit["measure_object_count"] = len(audit.pop("measure_sample_ids"))
    audit["source_object_count"] = len(audit.pop("source_sample_ids"))


def _source_hashes():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = (
        "cli/idea4_5.py", "cli/aggregate_idea4_5.py", "data/idea4_5.py",
        "models/idea4_5.py", "training/idea4_5_audit.py", "modules/idea4_5_path.py",
    )
    result = {}
    for relative in paths:
        with open(os.path.join(root, relative), "rb") as handle:
            result[relative] = hashlib.sha256(handle.read()).hexdigest()
    return result


def _validate(args):
    if args.dataset == "npz" and not args.data_path:
        raise ValueError("--data_path is required for NPZ input")
    if args.dataset == "synthetic" and args.label_mode != "class":
        raise ValueError("Synthetic labels already use the registered binary class task")
    if not 0 <= args.rich_fraction <= 1 or args.epochs < 1:
        raise ValueError("rich_fraction must be in [0,1] and epochs positive")
    if args.audit_source_batches < 1 or args.audit_batch_size < 1:
        raise ValueError("Audit batch counts and sizes must be positive")
    if args.edge_normalization == "fixed" and args.local_topology == "learned":
        raise ValueError("Fixed edge normalization requires uniform or identity topology")


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.extra_information:
        args.information_condition = {"unique": "novel_signal", "redundant": "repeat_measurement"}[
            args.extra_information
        ]
    if args.model_seed is not None:
        args.train_seed = args.model_seed
    for offset, name in enumerate(("init_seed", "order_seed", "mask_seed", "audit_seed"), start=1):
        if getattr(args, name) is None:
            setattr(args, name, args.train_seed * 10 + offset)
    _validate(args)
    random.seed(args.init_seed)
    set_seed(args.init_seed)
    if args.dataset == "synthetic":
        splits = make_availability_splits(
            args.num_samples, args.token_count, args.token_dim, args.information_condition,
            args.signal_form, data_seed=args.data_seed, split_seed=args.split_seed,
        )
    else:
        splits = load_availability_npz(args.data_path, args.label_mode)
    train = splits["train"]
    _, groups, modalities, _, token_dim = train.tokens.shape
    if groups != 1:
        raise ValueError("Idea4.5 v3 expects one fusion group")
    sparse, rich = tuple(args.sparse_modalities), tuple(range(modalities))
    if len(sparse) < 2 or len(set(sparse)) != len(sparse) or not set(sparse).issubset(rich) or len(sparse) == len(rich):
        raise ValueError("sparse_modalities must name at least two, but not all, distinct modalities")
    edge = (args.edge_target, args.edge_source)
    if edge[0] == edge[1] or not set(edge).issubset(sparse):
        raise ValueError("The audited edge must connect modalities present in both views")
    for dataset in splits.values():
        _complete(dataset, rich)
    roles, role_indices = partition_training(
        train, args.split_seed, synthetic=args.dataset == "synthetic", audit_size=args.audit_size
    )
    if args.audit_batch_size > len(roles["fit"]):
        raise ValueError("audit_batch_size exceeds fit role")

    labels = set(train.labels.reshape(-1).tolist())
    num_classes = len(labels)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = Idea45AvailabilityModel(
        token_dim, modalities, num_classes, shared_dim=args.shared_dim, spd_dim=args.spd_dim,
        geometry=args.geometry, local_topology=args.local_topology, moment_mode=args.moment_mode,
        readout_mode=args.readout_mode, edge_normalization=args.edge_normalization,
    ).to(device)
    if args.training_optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.0, weight_decay=0.0)

    os.makedirs(args.out_dir, exist_ok=True)
    config = vars(args) | {
        "rich_modalities": rich, "num_classes": num_classes, "device": str(device),
        "audit_model_mode": "eval", "source_sha256": _source_hashes(),
    }
    save_json(os.path.join(args.out_dir, "config.json"), config)
    save_json(os.path.join(args.out_dir, "data_roles.json"), {
        "indices": role_indices,
        "sample_ids": {role: [str(train.sample_ids[index]) for index in indices]
                       for role, indices in role_indices.items()},
        "group_ids": {role: [str(value) for value in sorted(set(train.group_ids[indices]))]
                      for role, indices in role_indices.items()},
        "validation_sample_ids": splits["val"].sample_ids.tolist(),
        "test_sample_ids": splits["test"].sample_ids.tolist(),
    })
    save_json(os.path.join(args.out_dir, "run_status.json"), {"status": "running", "started": time.time()})

    measure_batch = _full_batch(roles["measure"])
    checkpoints = _checkpoint_epochs(args.epochs, args.audit_checkpoints)
    audit_rng = np.random.default_rng(args.audit_seed)
    history, audits, training_steps = [], [], 0
    started = time.perf_counter()
    clip_norm = None if args.clip_norm <= 0 else args.clip_norm
    for epoch in range(1, args.epochs + 1):
        loader = DataLoader(
            roles["fit"], batch_size=args.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(args.order_seed + epoch), drop_last=False,
        )
        batch_count = len(loader)
        rich_count = int(round(batch_count * args.rich_fraction))
        contexts = np.asarray(["rich"] * rich_count + ["sparse"] * (batch_count - rich_count))
        np.random.default_rng(args.mask_seed + epoch).shuffle(contexts)
        model.train()
        loss_sum = object_count = clipped_steps = 0
        seen = {"rich": 0, "sparse": 0}
        for batch, context in zip(loader, contexts):
            view = masked_view(batch, rich if context == "rich" else sparse, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(view["tokens"], view["modality_mask"], view["token_mask"])["logits"]
            loss = brier_loss(logits, view["label"])
            loss.backward()
            if clip_norm is not None:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
                clipped_steps += int(float(norm) > clip_norm)
            optimizer.step()
            training_steps += 1
            count = len(view["label"])
            loss_sum += float(loss.detach().cpu()) * count
            object_count += count
            seen[str(context)] += count
        validation = {
            name: _evaluate(model, splits["val"], view, args.batch_size, device)
            for name, view in (("rich", rich), ("sparse", sparse))
        }
        history.append({
            "epoch": epoch, "train_brier": loss_sum / object_count, "exposures": seen,
            "context_batches": {"rich": rich_count, "sparse": batch_count - rich_count},
            "clipped_steps": clipped_steps, "validation": validation,
            "elapsed_seconds": time.perf_counter() - started,
        })
        if epoch in checkpoints:
            torch.save({
                "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "rng": _rng_snapshot(), "audit_rng": audit_rng.bit_generator.state, "config": config,
            }, os.path.join(args.out_dir, f"checkpoint_epoch_{epoch}.pt"))
            for source_index, update_batch in enumerate(_audit_batches(
                roles["fit"], args.audit_source_batches, args.audit_batch_size, audit_rng
            )):
                for audit_optimizer in args.audit_optimizers:
                    audit_started = time.perf_counter()
                    audit = audit_pair(
                        model, optimizer, update_batch, measure_batch, sparse, rich, edge,
                        audit_optimizer=audit_optimizer, mechanism_lr=args.mechanism_lr,
                        clip_norm=clip_norm if audit_optimizer == "stateful" else None,
                    )
                    audit.update({"epoch": epoch, "source_batch": source_index,
                                  "edge": {"target": edge[0], "source": edge[1]},
                                  "elapsed_seconds": time.perf_counter() - audit_started})
                    object_path = os.path.join(
                        args.out_dir, f"audit_objects_e{epoch}_b{source_index}_{audit_optimizer}.npz"
                    )
                    _save_audit_objects(object_path, audit)
                    audits.append(audit)
            save_json(os.path.join(args.out_dir, "audits.json"), audits)
        save_json(os.path.join(args.out_dir, "history.json"), history)

    tests = {
        name: _evaluate(model, splits["test"], view, args.batch_size, device)
        for name, view in (("rich", rich), ("sparse", sparse))
    }
    save_json(os.path.join(args.out_dir, "test_metrics.json"), tests)
    by_optimizer = {}
    for name in args.audit_optimizers:
        subset = [audit for audit in audits if audit["audit_optimizer"] == name]
        by_optimizer[name] = {
            metric: float(np.mean([audit[metric] for audit in subset]))
            for metric in ("delta_abs_rich_to_sparse", "delta_rel_rich_vs_sparse", "A_r", "A_s", "J")
        }
    run_summary = {"config": config, "audit_means": by_optimizer, "test": tests,
                   "training_steps": training_steps,
                   "audit_elapsed_seconds": float(sum(audit["elapsed_seconds"] for audit in audits)),
                   "elapsed_seconds": time.perf_counter() - started}
    save_json(os.path.join(args.out_dir, "run_summary.json"), run_summary)
    save_json(os.path.join(args.out_dir, "run_status.json"), {
        "status": "completed", "completed": time.time(), "elapsed_seconds": run_summary["elapsed_seconds"]
    })
    return {"audits": audits, "test": tests}


if __name__ == "__main__":
    main()
