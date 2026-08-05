"""Train the task-agnostic Paper 4 diagnostic benchmark model."""

import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from glioma.data.paper4_benchmarks import load_npz_splits, make_synthetic_splits
from glioma.eval.paper4_diagnostics import edge_recovery
from glioma.io.artifacts import save_json
from glioma.models.paper4_benchmark import Paper4BenchmarkModel
from glioma.training.engine import set_seed


def build_parser():
    parser = argparse.ArgumentParser(description="Paper 4 general multimodal diagnostic benchmark")
    parser.add_argument("--dataset", choices=["synthetic", "avmnist", "mosei"], required=True)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--geometry", choices=["spd", "euclidean"], default="spd")
    parser.add_argument(
        "--local_topology", choices=["learned", "identity", "uniform"], default="learned"
    )
    parser.add_argument("--local_cross_mass", type=float, default=0.35)
    parser.add_argument("--shared_dim", type=int, default=64)
    parser.add_argument("--spd_dim", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=None)
    parser.add_argument("--model_seed", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=600)
    parser.add_argument("--num_modalities", type=int, choices=[4, 6], default=4)
    parser.add_argument("--token_count", type=int, default=12)
    parser.add_argument("--token_dim", type=int, default=8)
    parser.add_argument("--planted_graph", choices=["chain", "star", "two_community"], default="chain")
    parser.add_argument(
        "--synthetic_regime",
        choices=["geometry_only", "exchangeable", "topology_relevant"],
        default="topology_relevant",
    )
    parser.add_argument("--missing_rate", type=float, default=0.0)
    parser.add_argument("--eval_missing_rates", nargs="*", type=float, default=[0.0, 0.25, 0.50])
    parser.add_argument("--eval_noise_stds", nargs="*", type=float, default=[0.0, 0.25])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser


def resolve_seeds(args):
    if args.split_seed is None:
        args.split_seed = int(args.seed)
    if args.model_seed is None:
        args.model_seed = int(args.seed)
    return args


def _target(labels, dataset):
    labels = labels.reshape(-1)
    if dataset == "mosei":
        return (labels >= 0).long()
    return labels.long()


def _binary_f1(targets, predictions):
    targets = np.asarray(targets) == 1
    predictions = np.asarray(predictions) == 1
    true_positive = np.logical_and(targets, predictions).sum()
    denominator = 2 * true_positive + np.logical_and(~targets, predictions).sum() + np.logical_and(targets, ~predictions).sum()
    return float(2 * true_positive / denominator) if denominator else 0.0


def _perturb_batch(batch, missing_rate, noise_std, generator):
    tokens = batch["tokens"].clone()
    modality_mask = batch["modality_mask"].clone()
    token_mask = batch["token_mask"].clone()
    if missing_rate > 0:
        drop = torch.rand(modality_mask.shape, generator=generator) < float(missing_rate)
        modality_mask &= ~drop
        empty = torch.where(modality_mask.sum(dim=-1) == 0)[0]
        if len(empty):
            original = batch["modality_mask"][empty]
            restore = original.to(torch.int64).argmax(dim=-1)
            modality_mask[empty, restore] = True
        token_mask &= modality_mask[:, None, :, None]
    if noise_std > 0:
        noise = torch.randn(tokens.shape, generator=generator, dtype=tokens.dtype)
        tokens = tokens + float(noise_std) * noise * token_mask.unsqueeze(-1).to(tokens.dtype)
    return tokens, modality_mask, token_mask


def evaluate(model, loader, device, dataset, topology_override=None, missing_rate=0.0, noise_std=0.0, seed=0):
    model.eval()
    targets = []
    original_labels = []
    predictions = []
    margins = []
    adjacency = []
    generator = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        for batch in loader:
            tokens, modality_mask, token_mask = _perturb_batch(
                batch, missing_rate, noise_std, generator
            )
            output = model(
                tokens.to(device),
                modality_mask=modality_mask.to(device),
                token_mask=token_mask.to(device),
                topology_override=topology_override,
            )
            logits = output["logits"]
            target = _target(batch["label"], dataset)
            targets.extend(target.tolist())
            original_labels.extend(batch["label"].reshape(-1).tolist())
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            if logits.shape[-1] == 2:
                margins.extend((logits[:, 1] - logits[:, 0]).cpu().tolist())
            adjacency.append(output["fusion"]["local_adjacency"].mean(dim=(0, 1)).cpu().numpy())
    metrics = {"accuracy": float(np.mean(np.asarray(targets) == np.asarray(predictions)))}
    if set(targets).issubset({0, 1}):
        metrics["binary_f1"] = _binary_f1(targets, predictions)
    if dataset == "mosei" and margins:
        labels = np.asarray(original_labels, dtype=float)
        score = np.asarray(margins, dtype=float)
        metrics["mae"] = float(np.mean(np.abs(score - labels)))
        correlation = np.corrcoef(score, labels)[0, 1]
        metrics["correlation"] = float(correlation) if np.isfinite(correlation) else float("nan")
    return metrics, np.mean(adjacency, axis=0)


def _profile_flops(model, batch, device):
    try:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU], with_flops=True) as profiler:
            model(
                batch["tokens"].to(device),
                modality_mask=batch["modality_mask"].to(device),
                token_mask=batch["token_mask"].to(device),
            )
        return int(sum(event.flops for event in profiler.key_averages()))
    except (ImportError, RuntimeError):
        return None


def main(argv=None):
    args = resolve_seeds(build_parser().parse_args(argv))
    if args.dataset != "synthetic" and not args.data_path:
        raise ValueError("--data_path is required for AV-MNIST and MOSEI")
    set_seed(args.model_seed)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.dataset == "synthetic":
        splits, planted = make_synthetic_splits(
            num_samples=args.num_samples,
            num_modalities=args.num_modalities,
            token_count=args.token_count,
            token_dim=args.token_dim,
            graph_type=args.planted_graph,
            regime=args.synthetic_regime,
            missing_rate=args.missing_rate,
            seed=args.split_seed,
        )
    else:
        splits = load_npz_splits(args.data_path)
        planted = None
    sample = splits["train"].tokens
    _, groups, modalities, _, token_dim = sample.shape
    if groups != 1:
        raise ValueError("The public benchmark scaffold currently expects one fusion group")
    default_classes = {"synthetic": 2, "avmnist": 10, "mosei": 2}
    num_classes = args.num_classes or default_classes[args.dataset]
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=args.num_workers,
            generator=torch.Generator().manual_seed(args.model_seed),
        )
        for name, dataset in splits.items()
    }
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = Paper4BenchmarkModel(
        token_dim=token_dim,
        num_modalities=modalities,
        num_classes=num_classes,
        shared_dim=args.shared_dim,
        spd_dim=args.spd_dim,
        geometry=args.geometry,
        local_topology=args.local_topology,
        local_cross_mass=args.local_cross_mass,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best_accuracy = -float("inf")
    best_state = None
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            output = model(
                batch["tokens"].to(device),
                modality_mask=batch["modality_mask"].to(device),
                token_mask=batch["token_mask"].to(device),
            )
            target = _target(batch["label"], args.dataset).to(device)
            loss = F.cross_entropy(output["logits"], target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation, _ = evaluate(
            model, loaders["val"], device, args.dataset, seed=args.model_seed + epoch
        )
        history.append(
            {"epoch": epoch, "train_loss": float(np.mean(losses)), "val": validation}
        )
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    test_metrics, adjacency = evaluate(
        model, loaders["test"], device, args.dataset, seed=args.model_seed
    )
    interventions = {}
    for topology in ("identity", "uniform"):
        metrics, _ = evaluate(
            model,
            loaders["test"],
            device,
            args.dataset,
            topology_override=topology,
            seed=args.model_seed,
        )
        interventions[topology] = {
            **metrics,
            "delta_accuracy": float(metrics["accuracy"] - test_metrics["accuracy"]),
        }
    robustness = {}
    for missing_rate in args.eval_missing_rates:
        for noise_std in args.eval_noise_stds:
            key = f"missing_{missing_rate:g}_noise_{noise_std:g}"
            robustness[key], _ = evaluate(
                model,
                loaders["test"],
                device,
                args.dataset,
                missing_rate=missing_rate,
                noise_std=noise_std,
                seed=args.split_seed + 1009,
            )
    first_batch = next(iter(loaders["test"]))
    diagnostics = {
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "forward_flops": _profile_flops(model, first_batch, device),
        "training_seconds": float(elapsed),
    }
    if planted is not None:
        diagnostics.update(edge_recovery(adjacency, planted))
        save_json(os.path.join(args.out_dir, "planted_adjacency.json"), planted.tolist())
    config = vars(args) | {
        "num_classes_resolved": num_classes,
        "num_modalities_resolved": modalities,
        "token_dim_resolved": token_dim,
    }
    save_json(os.path.join(args.out_dir, "config.json"), config)
    save_json(os.path.join(args.out_dir, "history.json"), history)
    save_json(os.path.join(args.out_dir, "test_metrics.json"), test_metrics)
    save_json(os.path.join(args.out_dir, "intervention_metrics.json"), interventions)
    save_json(os.path.join(args.out_dir, "robustness_metrics.json"), robustness)
    save_json(os.path.join(args.out_dir, "adjacency.json"), adjacency.tolist())
    save_json(os.path.join(args.out_dir, "diagnostics.json"), diagnostics)
    torch.save({"model": model.state_dict(), "config": config}, os.path.join(args.out_dir, "best_benchmark.pt"))
    return {"metrics": test_metrics, "diagnostics": diagnostics}


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "resolve_seeds", "evaluate", "main"]
