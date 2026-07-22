"""Training loop for semantic-alignment experiments."""

from collections import Counter

import numpy as np
import torch

from glioma.data.loaders import build_query_targets
from glioma.semantic.losses import (
    anchor_center_loss,
    dcca_alignment_loss,
    geodesic_path_semantic_loss,
    masked_multi_positive_nll,
    medclip_multi_positive_loss,
    multi_positive_contrastive_loss,
    topomoe_family_balanced_losses,
)


def node_names_for_mode(node_mode):
    if node_mode == "regions":
        return ["Necrotic/Core", "Edema", "Enhancing"]
    return ["T1", "T1ce", "T2", "FLAIR"]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize_tensor(tensor, nan=0.0, posinf=0.0, neginf=0.0):
    return torch.nan_to_num(tensor, nan=nan, posinf=posinf, neginf=neginf)


def graph_cons_scale(epoch, warmup_epochs):
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch) / float(warmup_epochs))


def geodesic_loss_scale(epoch, warmup_epochs):
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch) / float(warmup_epochs))


def _alignment_loss(args, queries, target_ids, prototypes, loss_context):
    if args.alignment_objective == "medclip":
        return (
            medclip_multi_positive_loss(
                queries,
                target_ids,
                prototypes,
                ignore_ids_by_anchor=loss_context["medclip_ignore_ids"],
                temperature=args.temperature,
            ),
            0.0,
        )
    if args.alignment_objective == "dcca":
        clip_loss = multi_positive_contrastive_loss(
            queries,
            target_ids,
            prototypes,
            temperature=args.temperature,
        )
        try:
            dcca_loss = dcca_alignment_loss(queries, target_ids, prototypes, reg=args.dcca_reg)
            return dcca_loss + args.dcca_clip_weight * clip_loss, 0.0
        except RuntimeError:
            return clip_loss, 1.0
    return (
        multi_positive_contrastive_loss(
            queries,
            target_ids,
            prototypes,
            temperature=args.temperature,
        ),
        0.0,
    )


def _loss_or_zero(losses, name, zero):
    value = losses.get(name)
    return zero if value is None else value


def run_epoch(model, bank, loader, optimizer, device, args, case_lookup, key_to_id, epoch, loss_context):
    training = optimizer is not None
    model.train(training)
    bank.train(training)
    node_names = node_names_for_mode(args.node_mode)
    totals = Counter()
    cons_scale = graph_cons_scale(epoch, args.graph_warmup_epochs)
    freeze_graph = training and (epoch <= args.graph_warmup_epochs)
    is_topomoe = bool(getattr(model, "use_topo_moe", False))
    is_topomoe_v2 = is_topomoe and getattr(model, "topomoe_version", "v1") == "v2"
    is_geodesic = bool(getattr(model, "use_geodesic_fusion", False))
    geo_scale = geodesic_loss_scale(epoch, args.geo_warmup_epochs) if is_geodesic else 0.0

    for batch in loader:
        images = batch["images"].to(device)
        region_masks = batch.get("region_masks")
        if region_masks is not None:
            region_masks = region_masks.to(device)
        subject_ids = batch["subject_id"]

        with torch.set_grad_enabled(training):
            prototypes = sanitize_tensor(bank())
            output = model(
                images,
                region_masks=region_masks,
                return_extras=True,
                freeze_graph=freeze_graph,
                anchor_prototypes=prototypes
                if getattr(model, "requires_anchor_prototypes", is_topomoe)
                else None,
            )
            shared = sanitize_tensor(output["extras"]["shared"])
            queries = shared.reshape(-1, shared.shape[-1])
            target_ids = build_query_targets(subject_ids, node_names, case_lookup, key_to_id, args)
            alignment_loss, dcca_fallback = _alignment_loss(
                args, queries, target_ids, prototypes, loss_context
            )
            anchor_loss = anchor_center_loss(queries, target_ids, prototypes)
            zero = alignment_loss * 0
            losses = output["losses"]

            legacy_route_loss = zero
            family_route_loss = zero
            within_anchor_loss = zero
            routed_scores = output["extras"].get("routed_scores")
            if routed_scores is None:
                routed_scores = output["extras"].get("routed_logits")

            if is_topomoe_v2:
                if not args.disable_family_balanced_route:
                    family_route_loss, within_anchor_loss = topomoe_family_balanced_losses(
                        output["extras"]["routing_weights"],
                        output["extras"]["family_log_probs"],
                        target_ids,
                        loss_context["family_ids"],
                        loss_context["residual_index"],
                    )
                if args.disable_family_balanced_route or output["extras"].get("routed_log_probs") is None:
                    within_anchor_loss = masked_multi_positive_nll(
                        routed_scores.reshape(-1, routed_scores.shape[-1]),
                        target_ids,
                    )
            elif is_topomoe and routed_scores is not None:
                legacy_route_loss = masked_multi_positive_nll(
                    routed_scores.reshape(-1, routed_scores.shape[-1]),
                    target_ids,
                )

            path_semantic_loss = zero
            if is_geodesic and output["extras"].get("fusion_interior_paths") is not None:
                path_semantic_loss = geodesic_path_semantic_loss(
                    output["extras"]["fusion_interior_paths"],
                    output["extras"]["fusion_pair_valid"],
                    target_ids,
                    prototypes,
                    temperature=args.temperature,
                )

            total = (
                alignment_loss
                + args.lambda_anchor * anchor_loss
                + args.lambda_cons * cons_scale * _loss_or_zero(losses, "cons", zero)
                + args.lambda_decouple * _loss_or_zero(losses, "decouple", zero)
                + args.lambda_leak * _loss_or_zero(losses, "leak", zero)
                + args.lambda_diff * _loss_or_zero(losses, "diff", zero)
                + args.lambda_diff_norm * _loss_or_zero(losses, "diff_norm", zero)
                + args.lambda_gate_entropy * _loss_or_zero(losses, "gate_entropy", zero)
                + args.lambda_load_balance * _loss_or_zero(losses, "load_balance", zero)
            )
            if is_topomoe_v2:
                total = (
                    total
                    + args.lambda_family_route * family_route_loss
                    + args.lambda_within_anchor * within_anchor_loss
                    + args.lambda_topo_prior * _loss_or_zero(losses, "topo_prior", zero)
                    + args.lambda_topo_delta * _loss_or_zero(losses, "topo_delta", zero)
                    + args.lambda_specialize * _loss_or_zero(losses, "specialize", zero)
                    + args.lambda_anchor_family_balance * _loss_or_zero(losses, "route_balance", zero)
                )
            else:
                total = (
                    total
                    + args.lambda_route * legacy_route_loss
                    + args.lambda_topo * _loss_or_zero(losses, "topo", zero)
                    + args.lambda_specialize * _loss_or_zero(losses, "specialize", zero)
                    + args.lambda_route_balance * _loss_or_zero(losses, "route_balance", zero)
                    + args.lambda_route_sparse * _loss_or_zero(losses, "route_sparse", zero)
                )
            if is_geodesic:
                total = (
                    total
                    + args.lambda_geo_energy * geo_scale * _loss_or_zero(losses, "geo_energy", zero)
                    + args.lambda_path_semantic * geo_scale * path_semantic_loss
                    + args.lambda_spd_condition * _loss_or_zero(losses, "spd_condition", zero)
                    + args.lambda_manifold_topology
                    * _loss_or_zero(losses, "manifold_topology", zero)
                )

            if not torch.isfinite(total):
                totals["nonfinite_batches"] += 1
                if training:
                    optimizer.zero_grad(set_to_none=True)
                continue

            topo_grad_norm = 0.0
            geopath_grad_norm = 0.0
            if training:
                optimizer.zero_grad()
                total.backward()
                topo_module = getattr(model, "topo_moe", None)
                if topo_module is not None and topo_module.A_raw.grad is not None:
                    topo_grad_norm = float(topo_module.A_raw.grad.detach().norm().cpu())
                fusion_module = getattr(model, "fusion", None)
                if fusion_module is not None:
                    squared_norm = zero.detach()
                    gradient_parameters = (
                        fusion_module.parameters()
                        if getattr(model, "use_manifold_fusion", False)
                        else fusion_module.geopath_net.parameters()
                    )
                    for parameter in gradient_parameters:
                        if parameter.grad is not None:
                            squared_norm = squared_norm + parameter.grad.detach().square().sum()
                    geopath_grad_norm = float(squared_norm.sqrt().cpu())
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(bank.parameters()), args.grad_clip)
                optimizer.step()

        batch_size = images.shape[0]
        totals["n"] += batch_size
        scalar_losses = {
            "total": total,
            "alignment": alignment_loss,
            "anchor": anchor_loss,
            "legacy_route": legacy_route_loss,
            "family_route": family_route_loss,
            "within_anchor": within_anchor_loss,
            "geo_energy": _loss_or_zero(losses, "geo_energy", zero),
            "path_semantic": path_semantic_loss,
            "spd_condition": _loss_or_zero(losses, "spd_condition", zero),
            "manifold_topology": _loss_or_zero(losses, "manifold_topology", zero),
            "cons": _loss_or_zero(losses, "cons", zero),
            "decouple": _loss_or_zero(losses, "decouple", zero),
            "leak": _loss_or_zero(losses, "leak", zero),
            "diff": _loss_or_zero(losses, "diff", zero),
            "diff_norm": _loss_or_zero(losses, "diff_norm", zero),
            "topo_prior": _loss_or_zero(losses, "topo_prior", _loss_or_zero(losses, "topo", zero)),
            "topo_delta": _loss_or_zero(losses, "topo_delta", zero),
            "specialize": _loss_or_zero(losses, "specialize", zero),
            "route_balance": _loss_or_zero(losses, "route_balance", zero),
            "route_sparse": _loss_or_zero(losses, "route_sparse", zero),
            "dcca_fallback": torch.as_tensor(dcca_fallback, device=device),
        }
        for name, value in scalar_losses.items():
            totals[f"loss:{name}"] += float(value.detach().cpu()) * batch_size

        routing = output["extras"].get("routing_weights")
        if routing is not None:
            family_names = loss_context.get("family_names", [])
            usage = routing.detach().mean(dim=(0, 1)).cpu().tolist()
            for idx, value in enumerate(usage):
                name = family_names[idx] if idx < len(family_names) else f"family_{idx}"
                totals[f"routing:usage_{name}"] += float(value) * batch_size
            entropy = -(routing * torch.log(routing.clamp(min=1e-8))).sum(dim=-1).mean()
            totals["routing:entropy"] += float(entropy.detach().cpu()) * batch_size

        diagnostics = output["extras"].get("topology_diagnostics") or {}
        for name, value in diagnostics.items():
            totals[f"topology:{name}"] += float(value.detach().cpu()) * batch_size
        totals["topology:gradient_norm"] += topo_grad_norm * batch_size
        fusion_diagnostics = output["extras"].get("fusion_diagnostics") or {}
        for name, value in fusion_diagnostics.items():
            totals[f"fusion:{name}"] += float(value.detach().cpu()) * batch_size
        totals["fusion:geopath_gradient_norm"] += geopath_grad_norm * batch_size
        totals["fusion:loss_scale"] += geo_scale * batch_size
        totals["representation:shared_norm"] += float(output["extras"]["shared_norm"].cpu()) * batch_size
        totals["representation:private_norm"] += float(output["extras"]["private_norm"].cpu()) * batch_size
        totals["representation:diffusion_residual_norm"] += float(
            output["extras"]["diffusion_residual_norm"].cpu()
        ) * batch_size

    n_samples = max(totals["n"], 1)

    def group(prefix):
        marker = f"{prefix}:"
        return {
            key[len(marker) :]: value / n_samples
            for key, value in totals.items()
            if key.startswith(marker)
        }

    loss_values = group("loss")
    loss_values["nonfinite_batches"] = float(totals.get("nonfinite_batches", 0.0))
    return {
        "losses": loss_values,
        "routing": group("routing"),
        "topology": group("topology"),
        "fusion": group("fusion"),
        "representation": group("representation"),
    }


__all__ = [
    "set_seed",
    "sanitize_tensor",
    "graph_cons_scale",
    "geodesic_loss_scale",
    "run_epoch",
]
