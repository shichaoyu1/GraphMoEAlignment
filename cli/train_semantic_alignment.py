"""Command-line entry point for glioma semantic-alignment experiments."""

import argparse
import os


DEFAULT_OUT_DIR = "output/semantic_alignment_experiment"
DEFAULT_VALIDATION_OUTPUT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "validation"))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Glioma pathology-anchored semantic-unit alignment experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--metadata_tsv", type=str, default=None)
    parser.add_argument(
        "--variant",
        default="full",
        choices=["full", "clip", "medclip_style", "dcca", "graph_shared_only", "hgt", "no_anchor", "graph_only", "modality_vector", "no_private", "no_graph"],
    )
    parser.add_argument("--paper_config", default="none", choices=["none", "paper1", "paper2", "paper3"])
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--validation_output_root", default=DEFAULT_VALIDATION_OUTPUT_ROOT)

    parser.add_argument("--roi_size", type=int, default=96)
    parser.add_argument("--z_slices", type=int, default=7)
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--prefer_registered", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--feat_dim", type=int, default=256)
    parser.add_argument("--node_mode", default="regions", choices=["regions", "modalities"])
    parser.add_argument("--graph_type", default="learnable", choices=["no_graph", "fixed", "similarity", "learnable", "random"])
    parser.add_argument("--shared_dim", type=int, default=128)
    parser.add_argument("--private_dim", type=int, default=128)
    parser.add_argument("--diffusion_T", type=int, default=20)
    parser.add_argument("--graph_warmup_epochs", type=int, default=5)
    parser.add_argument("--graph_ema_momentum", type=float, default=0.95)
    parser.add_argument("--graph_ema_blend", type=float, default=0.5)
    parser.add_argument("--diffusion_init_alpha", type=float, default=0.05)
    parser.add_argument("--shared_private_mix_init", type=float, default=0.05)
    parser.add_argument("--classifier_private_scale_init", type=float, default=0.05)
    parser.add_argument("--diffusion_max_ratio", type=float, default=0.5)
    parser.add_argument("--moe_module", default="none", choices=["none", "semantic_moe", "graph_moe", "diffusion_moe", "topo_moe"])
    parser.add_argument("--topomoe_version", default="v2", choices=["v1", "v2"])
    parser.add_argument("--topo_mode", default="prior_plus_learned", choices=["prior_only", "learned_only", "prior_plus_learned"])
    parser.add_argument("--topo_epsilon", type=float, default=1e-4)
    parser.add_argument("--topo_temperature", type=float, default=1.0)
    parser.add_argument("--topo_beta_init", type=float, default=0.1)
    parser.add_argument("--route_mixture", default="log_prob", choices=["log_prob", "product"])
    parser.add_argument("--disable_topology_refinement", action="store_true")
    parser.add_argument("--disable_family_balanced_route", action="store_true")
    parser.add_argument("--specialize_margin", type=float, default=0.05)
    parser.add_argument("--no_private", action="store_true")
    parser.add_argument("--no_diffusion", action="store_true")

    parser.add_argument("--target_policy", default="region_rules", choices=["region_rules", "all_patient_anchors"])
    parser.add_argument("--exclude_pathology_anchors", action="store_true")
    parser.add_argument("--exclude_molecular_anchors", action="store_true")
    parser.add_argument("--include_clinical_anchors", action="store_true")
    parser.add_argument("--align_max_cases", type=int, default=None)
    parser.add_argument("--graph_top_k", type=int, default=3)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--alignment_objective", default="clip", choices=["clip", "medclip", "dcca"])
    parser.add_argument("--dcca_reg", type=float, default=1e-3)
    parser.add_argument("--dcca_clip_weight", type=float, default=0.2)
    parser.add_argument("--lambda_anchor", type=float, default=0.05)
    parser.add_argument("--lambda_cons", type=float, default=0.05)
    parser.add_argument("--lambda_decouple", type=float, default=0.01)
    parser.add_argument("--lambda_leak", type=float, default=0.02)
    parser.add_argument("--lambda_diff", type=float, default=0.05)
    parser.add_argument("--lambda_diff_norm", type=float, default=0.02)
    parser.add_argument("--lambda_gate_entropy", type=float, default=0.0)
    parser.add_argument("--lambda_load_balance", type=float, default=0.0)
    parser.add_argument("--lambda_route", type=float, default=1.0, help="Legacy v1 route loss weight")
    parser.add_argument("--lambda_topo", type=float, default=0.05, help="Legacy v1 topology loss weight")
    parser.add_argument("--lambda_route_balance", type=float, default=0.01, help="Legacy v1 balance weight")
    parser.add_argument("--lambda_route_sparse", type=float, default=0.0, help="Legacy v1 sparse weight")
    parser.add_argument("--lambda_family_route", type=float, default=0.3)
    parser.add_argument("--lambda_within_anchor", type=float, default=0.3)
    parser.add_argument("--lambda_topo_prior", type=float, default=0.05)
    parser.add_argument("--lambda_topo_delta", type=float, default=0.001)
    parser.add_argument("--lambda_specialize", type=float, default=0.1)
    parser.add_argument("--lambda_anchor_family_balance", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_interventions", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser


def _metric_score(metrics, fallback):
    import numpy as np

    for name in ("map", "mrr"):
        value = metrics.get(name, float("nan")) if metrics else float("nan")
        if not np.isnan(value):
            return float(value), name
    return float(fallback), "loss_fallback"


def _checkpoint_selection(metrics, fallback, epoch):
    score, metric = _metric_score(metrics, fallback)
    return score, {
        "metric": metric,
        "score": score,
        "epoch": int(epoch),
        "fallback": metric == "loss_fallback",
    }


def _checkpoint_payload(model, bank, args, anchor_vocab, epoch, selection):
    return {
        "model": model.state_dict(),
        "bank": bank.state_dict(),
        "args": vars(args),
        "anchor_vocab": anchor_vocab,
        "epoch": int(epoch),
        "selection": selection,
    }


def main(args=None):
    if args is None:
        args = build_parser().parse_args()
    import torch

    from glioma.config.semantic_alignment_config import apply_paper_profile, apply_variant
    from glioma.data.case_discovery import discover_semantic_cases
    from glioma.data.loaders import make_loader
    from glioma.data.utsw_dataset import describe_cases, stratified_split
    from glioma.eval.semantic_alignment_eval import collect_alignment_records, evaluate_and_save, metrics_from_records
    from glioma.io.artifacts import save_json
    from glioma.models.glioma_graph_diffusion_net import GliomaGraphDiffusionNet
    from glioma.models.glioma_topomoe_net import GliomaTopoMoENet
    from glioma.objectives import SemanticPrototypeBank
    from glioma.semantic.topology import anchor_family_ids, build_cooccurrence_prior
    from glioma.semantic.vocab import build_anchor_vocab, build_medclip_ignore_ids
    from glioma.training.engine import run_epoch, set_seed

    args = apply_paper_profile(apply_variant(args))
    if args.paper_config != "none" and args.out_dir == DEFAULT_OUT_DIR:
        args.out_dir = os.path.join(args.validation_output_root, args.paper_config)
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    cases = discover_semantic_cases(
        args.data_root,
        metadata_tsv=args.metadata_tsv,
        max_cases=args.max_cases,
        seed=args.seed,
        include_clinical=args.include_clinical_anchors,
    )
    if len(cases) < 2:
        raise ValueError(f"Need at least 2 semantic cases; found {len(cases)}")
    splits = stratified_split(cases, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed)
    if not splits["val"]:
        splits["val"] = list(splits["test"])
    if not splits["test"]:
        splits["test"] = list(splits["val"] or splits["train"])

    anchor_vocab, key_to_id = build_anchor_vocab(
        splits["train"],
        include_pathology=not args.exclude_pathology_anchors,
        include_molecular=not args.exclude_molecular_anchors,
        include_clinical=args.include_clinical_anchors,
    )
    if len(anchor_vocab) < 2:
        raise ValueError("Need at least two train-set semantic anchors")

    family_ids, family_names = anchor_family_ids(anchor_vocab)
    topo_prior = build_cooccurrence_prior(splits["train"], anchor_vocab, key_to_id) if args.moe_module == "topo_moe" else None
    case_lookup = {case["subject_id"]: case for case in cases}
    loaders = {name: make_loader(split_cases, args, name) for name, split_cases in splits.items()}
    loss_context = {
        "medclip_ignore_ids": build_medclip_ignore_ids(anchor_vocab),
        "family_ids": family_ids,
        "family_names": family_names,
        "residual_index": len(family_names) - 1,
    }

    is_v2 = args.moe_module == "topo_moe" and args.topomoe_version == "v2"
    if is_v2:
        model = GliomaTopoMoENet(
            z_slices=args.z_slices,
            node_mode=args.node_mode,
            feat_dim=args.feat_dim,
            shared_dim=args.shared_dim,
            graph_type=args.graph_type,
            topo_prior=topo_prior,
            anchor_family_ids=family_ids,
            num_families=len(family_names),
            topo_mode=args.topo_mode,
            align_temperature=args.temperature,
            topomoe_version="v2",
            topo_epsilon=args.topo_epsilon,
            topo_temperature=args.topo_temperature,
            topo_beta_init=args.topo_beta_init,
            route_mixture=args.route_mixture,
            refine_prototypes=not args.disable_topology_refinement,
            specialize_margin=args.specialize_margin,
        ).to(device)
    else:
        model = GliomaGraphDiffusionNet(
            num_classes=1,
            z_slices=args.z_slices,
            node_mode=args.node_mode,
            feat_dim=args.feat_dim,
            shared_dim=args.shared_dim,
            private_dim=args.private_dim,
            graph_type=args.graph_type,
            diffusion_T=args.diffusion_T,
            graph_ema_momentum=args.graph_ema_momentum,
            graph_ema_blend=args.graph_ema_blend,
            diffusion_init_alpha=args.diffusion_init_alpha,
            shared_private_mix_init=args.shared_private_mix_init,
            classifier_private_scale_init=args.classifier_private_scale_init,
            diffusion_max_ratio=args.diffusion_max_ratio,
            moe_module=args.moe_module,
            use_anchor=False,
            use_private=not args.no_private,
            use_diffusion=not args.no_diffusion,
            topo_prior=topo_prior,
            anchor_family_ids=family_ids if topo_prior is not None else None,
            num_families=len(family_names) if topo_prior is not None else 0,
            topo_mode=args.topo_mode,
            align_temperature=args.temperature,
        ).to(device)

    bank = SemanticPrototypeBank(len(anchor_vocab), args.shared_dim).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(bank.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    print(f"Semantic alignment cases: {len(cases)} | split labels: {describe_cases(cases)}")
    for split_name in ("train", "val", "test"):
        print(f"{split_name}: {len(splits[split_name])} | distribution: {describe_cases(splits[split_name])}")

    save_json(os.path.join(args.out_dir, "config.json"), vars(args))
    save_json(os.path.join(args.out_dir, "anchor_vocab.json"), anchor_vocab)
    save_json(os.path.join(args.out_dir, "anchor_families.json"), {"family_names": family_names, "family_ids": family_ids})
    save_json(os.path.join(args.out_dir, "splits.json"), {name: [case["subject_id"] for case in values] for name, values in splits.items()})

    history = []
    if is_v2:
        checkpoint_paths = {
            "direct": os.path.join(args.out_dir, "best_direct_alignment.pt"),
            "routed": os.path.join(args.out_dir, "best_routed_topomoe.pt"),
        }
        best = {"direct": -float("inf"), "routed": -float("inf")}
        checkpoint_manifest = {"primary": "routed", "direct": {}, "routed": {}}
    else:
        checkpoint_paths = {"direct": os.path.join(args.out_dir, "best_semantic_alignment.pt")}
        best = {"direct": -float("inf")}
        checkpoint_manifest = {"primary": "direct", "direct": {}}

    for epoch in range(1, args.epochs + 1):
        train_epoch = run_epoch(model, bank, loaders["train"], optimizer, device, args, case_lookup, key_to_id, epoch, loss_context)
        val_epoch = run_epoch(model, bank, loaders["val"], None, device, args, case_lookup, key_to_id, epoch, loss_context)
        val_records = collect_alignment_records(model, bank, loaders["val"], device, args, case_lookup, key_to_id, anchor_vocab)
        val_metrics = metrics_from_records(val_records, anchor_vocab, checkpoint_type="validation")
        scheduler.step()

        history.append({"epoch": epoch, "train": train_epoch, "val": {**val_epoch, "direct": val_metrics["direct"], "routed": val_metrics["routed"]}})
        save_json(os.path.join(args.out_dir, "history.json"), history)

        direct_score, direct_selection = _checkpoint_selection(
            val_metrics["direct"], -val_epoch["losses"]["alignment"], epoch
        )
        if direct_score > best["direct"]:
            best["direct"] = direct_score
            torch.save(_checkpoint_payload(model, bank, args, anchor_vocab, epoch, direct_selection), checkpoint_paths["direct"])
            checkpoint_manifest["direct"] = {**direct_selection, "path": os.path.basename(checkpoint_paths["direct"])}

        if is_v2:
            routed_score, routed_selection = _checkpoint_selection(val_metrics["routed"], direct_score, epoch)
            if routed_score > best["routed"]:
                best["routed"] = routed_score
                torch.save(_checkpoint_payload(model, bank, args, anchor_vocab, epoch, routed_selection), checkpoint_paths["routed"])
                checkpoint_manifest["routed"] = {**routed_selection, "path": os.path.basename(checkpoint_paths["routed"])}
        save_json(os.path.join(args.out_dir, "checkpoint_manifest.json"), checkpoint_manifest)

    def load_checkpoint(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        bank.load_state_dict(checkpoint["bank"])

    if is_v2:
        load_checkpoint(checkpoint_paths["direct"])
        direct_records = collect_alignment_records(
            model, bank, loaders["test"], device, args, case_lookup, key_to_id, anchor_vocab, max_cases=args.align_max_cases
        )
        direct_checkpoint_metrics = metrics_from_records(direct_records, anchor_vocab, checkpoint_type="direct_best")
        save_json(os.path.join(args.out_dir, "test_metrics_direct_checkpoint.json"), direct_checkpoint_metrics)

        load_checkpoint(checkpoint_paths["routed"])
        metrics = evaluate_and_save(
            model,
            bank,
            loaders["test"],
            device,
            args,
            case_lookup,
            key_to_id,
            anchor_vocab,
            args.out_dir,
            checkpoint_type="routed_best",
            run_interventions=not args.skip_interventions,
            figure_context={
                "seed": args.seed,
                "checkpoint_type": "routed_best",
                "topology_mode": args.topo_mode,
                "history_path": os.path.join(args.out_dir, "history.json"),
                "checkpoint_comparison": {
                    "direct_best": direct_checkpoint_metrics,
                },
            },
        )
    else:
        load_checkpoint(checkpoint_paths["direct"])
        figure_context = None
        if args.paper_config == "paper2" and getattr(model, "use_topo_moe", False):
            figure_context = {
                "seed": args.seed,
                "checkpoint_type": "direct_best_v1",
                "topology_mode": args.topo_mode,
                "history_path": os.path.join(args.out_dir, "history.json"),
            }
        metrics = evaluate_and_save(
            model,
            bank,
            loaders["test"],
            device,
            args,
            case_lookup,
            key_to_id,
            anchor_vocab,
            args.out_dir,
            checkpoint_type="direct_best_v1" if figure_context else "direct_best",
            run_interventions=False,
            figure_context=figure_context,
        )
    return metrics


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "main"]
