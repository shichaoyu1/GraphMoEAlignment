"""Paired finite-step audits for Idea4.5 v3."""

import copy

import torch
import torch.nn.functional as F


def masked_view(batch, modalities, device):
    allowed = torch.zeros(batch["modality_mask"].shape[-1], dtype=torch.bool)
    allowed[list(modalities)] = True
    modality_mask = batch["modality_mask"] & allowed
    if not modality_mask.any(dim=-1).all():
        raise ValueError("A requested view has no observed modality")
    result = {
        "modality_mask": modality_mask.to(device),
        "label": batch["label"].reshape(-1).long().to(device),
    }
    if "tokens" in batch:
        token_mask = batch["token_mask"] & modality_mask[:, None, :, None]
        result.update({"tokens": batch["tokens"].to(device), "token_mask": token_mask.to(device)})
    elif "images" in batch:
        result["images"] = batch["images"].to(device)
        if "region_masks" in batch:
            result["region_masks"] = batch["region_masks"].to(device)
    else:
        raise ValueError("Batch must provide tokens or raw images")
    for key in ("sample_id", "group_id"):
        if key in batch:
            result[key] = list(batch[key])
    return result


def forward_view(model, view, gradient_edge=None):
    if "tokens" in view:
        return model(
            view["tokens"], view["modality_mask"], view["token_mask"],
            gradient_edge=gradient_edge,
        )
    return model(
        view["images"], modality_mask=view["modality_mask"],
        region_masks=view.get("region_masks"), gradient_edge=gradient_edge,
    )


def brier_values(logits, target):
    probabilities = logits.softmax(dim=-1)
    return (probabilities - F.one_hot(target, logits.shape[-1])).square().sum(dim=-1)


def brier_loss(logits, target):
    return brier_values(logits, target).mean()


def batch_risk_values(model, view):
    if isinstance(view, (list, tuple)):
        return torch.cat([batch_risk_values(model, part) for part in view])
    with torch.no_grad():
        output = forward_view(model, view)
        return brier_values(output["logits"], view["label"]).detach().cpu()


def batch_risk(model, view):
    return float(batch_risk_values(model, view).mean())


def _block_name(name):
    if name.startswith("classifier") or ".classifier" in name:
        return "classifier"
    if "token_adapters" in name:
        return "adapters"
    if "readout" in name:
        return "readout"
    if "relation" in name or "mix" in name:
        return "graph"
    return "other"


def _matches_prefix(name, prefix):
    return name == prefix or name.startswith(f"{prefix}.") or name.startswith(f"availability.{prefix}")


def _norms(named_tensors):
    squared = {}
    for name, tensor in named_tensors:
        if tensor is None:
            continue
        block = _block_name(name)
        squared[block] = squared.get(block, 0.0) + float(tensor.detach().square().sum().cpu())
    return {name: value ** 0.5 for name, value in squared.items()} | {
        "all": sum(squared.values()) ** 0.5
    }


def _stateful_optimizer(branch, optimizer):
    if len(optimizer.param_groups) != 1:
        raise ValueError("Idea4.5 v3 currently requires one optimizer parameter group")
    lr = optimizer.param_groups[0]["lr"]
    if isinstance(optimizer, torch.optim.AdamW):
        clone = torch.optim.AdamW(branch.parameters(), lr=lr)
    elif isinstance(optimizer, torch.optim.SGD):
        clone = torch.optim.SGD(branch.parameters(), lr=lr)
    else:
        raise TypeError("Only AdamW and SGD are supported by the reproducible audit")
    clone.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    return clone


def virtual_update(
    model,
    optimizer,
    view,
    gradient_edge=None,
    lr_scale=1.0,
    audit_optimizer="stateful",
    mechanism_lr=None,
    clip_norm=1.0,
    zero_parameter_prefixes=(),
):
    """Take one branch update from a complete cloned state and report block movement."""
    branch = copy.deepcopy(model).eval()
    if audit_optimizer == "stateful":
        branch_optimizer = _stateful_optimizer(branch, optimizer)
    elif audit_optimizer == "mechanism_sgd":
        branch_optimizer = torch.optim.SGD(
            branch.parameters(), lr=float(mechanism_lr or optimizer.param_groups[0]["lr"]),
            momentum=0.0, weight_decay=0.0,
        )
    else:
        raise ValueError("audit_optimizer must be stateful or mechanism_sgd")
    for group in branch_optimizer.param_groups:
        group["lr"] *= float(lr_scale)
    branch_optimizer.zero_grad(set_to_none=True)
    logits = forward_view(branch, view, gradient_edge=gradient_edge)["logits"]
    loss = brier_loss(logits, view["label"])
    loss.backward()
    for name, parameter in branch.named_parameters():
        if parameter.grad is not None and any(
            _matches_prefix(name, prefix) for prefix in zero_parameter_prefixes
        ):
            parameter.grad.zero_()
    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
           for parameter in branch.parameters()):
        raise FloatingPointError("Nonfinite gradient in virtual update")
    gradient_norms = _norms((name, parameter.grad) for name, parameter in branch.named_parameters())
    clip_scale = 1.0
    if clip_norm is not None:
        total = torch.nn.utils.clip_grad_norm_(branch.parameters(), float(clip_norm), error_if_nonfinite=True)
        clip_scale = min(1.0, float(clip_norm) / max(float(total), 1e-12))
    before = {name: parameter.detach().clone() for name, parameter in branch.named_parameters()}
    branch_optimizer.step()
    displacement_norms = _norms(
        (name, parameter.detach() - before[name]) for name, parameter in branch.named_parameters()
    )
    return branch, {
        "source_loss": float(loss.detach().cpu()),
        "gradient_norms": gradient_norms,
        "displacement_norms": displacement_norms,
        "clip_scale": clip_scale,
        "lr_scale": float(lr_scale),
        "zero_parameter_prefixes": list(zero_parameter_prefixes),
    }


def optimizer_background(model, optimizer, measured):
    """Run a stateful optimizer step with explicit zero gradients."""
    branch = copy.deepcopy(model).eval()
    branch_optimizer = _stateful_optimizer(branch, optimizer)
    before = {name: parameter.detach().clone() for name, parameter in branch.named_parameters()}
    branch_optimizer.zero_grad(set_to_none=True)
    for parameter in branch.parameters():
        parameter.grad = torch.zeros_like(parameter)
    branch_optimizer.step()
    return {
        "risk": {name: batch_risk(branch, view) for name, view in measured.items()},
        "displacement_norms": _norms(
            (name, parameter.detach() - before[name]) for name, parameter in branch.named_parameters()
        ),
    }


def audit_pair(
    model,
    optimizer,
    update_batch,
    measure_batch,
    sparse,
    rich,
    edge,
    audit_optimizer="stateful",
    mechanism_lr=None,
    clip_norm=1.0,
):
    """Evaluate four paired branches and matched alternative explanations."""
    device = next(model.parameters()).device
    views = {
        name: masked_view(update_batch, modalities, device)
        for name, modalities in (("rich", rich), ("sparse", sparse))
    }
    measure_batches = measure_batch if isinstance(measure_batch, (list, tuple)) else [measure_batch]
    measured = {
        name: [masked_view(batch, modalities, device) for batch in measure_batches]
        for name, modalities in (("rich", rich), ("sparse", sparse))
    }
    was_training = model.training
    model.eval()
    baseline_values = {name: batch_risk_values(model, view) for name, view in measured.items()}
    baseline = {name: float(values.mean()) for name, values in baseline_values.items()}
    results = {}
    per_object = {f"baseline_{name}": values.tolist() for name, values in baseline_values.items()}
    for context in ("rich", "sparse"):
        for gate in ("on", "off"):
            branch, diagnostics = virtual_update(
                model, optimizer, views[context], None if gate == "on" else (*edge, 0.0),
                audit_optimizer=audit_optimizer, mechanism_lr=mechanism_lr, clip_norm=clip_norm,
            )
            risks = {}
            for target, view in measured.items():
                values = batch_risk_values(branch, view)
                risks[target] = float(values.mean())
                per_object[f"{context}_{gate}_{target}"] = values.tolist()
            results[f"{context}_{gate}"] = {"risk": risks} | diagnostics

    rich_on = results["rich_on"]["risk"]["sparse"]
    rich_off = results["rich_off"]["risk"]["sparse"]
    sparse_on = results["sparse_on"]["risk"]["sparse"]
    sparse_off = results["sparse_off"]["risk"]["sparse"]
    reverse_edge = (edge[1], edge[0])
    controls = {"matched_displacement_lr": {}, "reverse_edge": {}, "parameter_zero": {}}
    for context in ("rich", "sparse"):
        on_norm = results[f"{context}_on"]["displacement_norms"]["all"]
        off_norm = results[f"{context}_off"]["displacement_norms"]["all"]
        scale = off_norm / on_norm if on_norm > 0 else 1.0
        matched, diagnostics = virtual_update(
            model, optimizer, views[context], lr_scale=scale,
            audit_optimizer=audit_optimizer, mechanism_lr=mechanism_lr, clip_norm=clip_norm,
        )
        controls["matched_displacement_lr"][context] = {
            "risk": {name: batch_risk(matched, view) for name, view in measured.items()}
        } | diagnostics
        reverse, diagnostics = virtual_update(
            model, optimizer, views[context], (*reverse_edge, 0.0),
            audit_optimizer=audit_optimizer, mechanism_lr=mechanism_lr, clip_norm=clip_norm,
        )
        controls["reverse_edge"][context] = {
            "risk": {name: batch_risk(reverse, view) for name, view in measured.items()}
        } | diagnostics
        for control_name, prefixes in (("head", ("classifier",)), ("encoder", ("fusion.token_adapters",))):
            controlled, diagnostics = virtual_update(
                model, optimizer, views[context], audit_optimizer=audit_optimizer,
                mechanism_lr=mechanism_lr, clip_norm=clip_norm, zero_parameter_prefixes=prefixes,
            )
            controls["parameter_zero"].setdefault(control_name, {})[context] = {
                "risk": {name: batch_risk(controlled, view) for name, view in measured.items()}
            } | diagnostics
    if audit_optimizer == "stateful":
        controls["optimizer_background"] = optimizer_background(model, optimizer, measured)
    model.train(was_training)
    return {
        "audit_optimizer": audit_optimizer,
        "baseline": baseline,
        "branches": results,
        "delta_abs_rich_to_sparse": rich_on - baseline["sparse"],
        "delta_rel_rich_vs_sparse": rich_on - sparse_on,
        "delta_abs_sparse_to_rich": results["sparse_on"]["risk"]["rich"] - baseline["rich"],
        "A_r": rich_on - rich_off,
        "A_s": sparse_on - sparse_off,
        "J": (rich_on - rich_off) - (sparse_on - sparse_off),
        "controls": controls,
        "compute_counts": {
            "virtual_updates": 12,
            "optimizer_background_steps": int(audit_optimizer == "stateful"),
            "measure_forward_passes": 28 if audit_optimizer == "stateful" else 26,
        },
        "source_sample_ids": views["rich"].get("sample_id", []),
        "measure_sample_ids": [
            sample_id for part in measured["rich"] for sample_id in part.get("sample_id", [])
        ],
        "per_object": per_object,
    }


__all__ = [
    "masked_view", "forward_view", "brier_values", "brier_loss", "batch_risk_values", "batch_risk",
    "virtual_update", "optimizer_background", "audit_pair",
]
