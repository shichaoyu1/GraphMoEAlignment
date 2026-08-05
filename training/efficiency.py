"""Lightweight, dependency-free efficiency profiling for paper comparisons."""

import time

import torch
import torch.nn as nn


def profile_model_efficiency(model, bank, loader, device, repetitions=10):
    """Estimate Conv/Linear FLOPs and measure patient-batch inference latency."""
    batch = next(iter(loader))
    images = batch["images"].to(device)
    region_masks = batch.get("region_masks")
    if region_masks is not None:
        region_masks = region_masks.to(device)
    prototypes = bank()
    kwargs = {
        "region_masks": region_masks,
        "return_extras": True,
        "anchor_prototypes": prototypes
        if getattr(model, "requires_anchor_prototypes", getattr(model, "use_topo_moe", False))
        else None,
    }

    operations = {"value": 0, "active": True}

    def hook(module, inputs, output):
        if not operations["active"]:
            return
        if isinstance(module, nn.Linear):
            operations["value"] += int(output.numel() * module.in_features * 2)
        elif isinstance(module, nn.Conv2d):
            kernel = module.kernel_size[0] * module.kernel_size[1]
            per_output = kernel * (module.in_channels // module.groups) * 2
            operations["value"] += int(output.numel() * per_output)

    handles = [
        module.register_forward_hook(hook)
        for module in model.modules()
        if isinstance(module, (nn.Linear, nn.Conv2d))
    ]
    model.eval()
    bank.eval()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        model(images, **kwargs)
        operations["active"] = False
        for _ in range(2):
            model(images, **kwargs)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(max(int(repetitions), 1)):
            model(images, **kwargs)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    for handle in handles:
        handle.remove()

    total_parameters = sum(parameter.numel() for parameter in model.parameters()) + sum(
        parameter.numel() for parameter in bank.parameters()
    )
    router = getattr(model, "topo_moe", None)
    router_parameters = sum(parameter.numel() for parameter in router.parameters()) if router is not None else 0
    batch_size = int(images.shape[0])
    peak_memory = (
        float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if device.startswith("cuda") else None
    )
    return {
        "batch_size": batch_size,
        "parameters_total": int(total_parameters),
        "parameters_router": int(router_parameters),
        "flops_per_patient_approx": float(operations["value"] / max(batch_size, 1)),
        "latency_ms_per_patient": float(elapsed * 1000 / (max(int(repetitions), 1) * max(batch_size, 1))),
        "peak_gpu_memory_mb": peak_memory,
        "flops_scope": "Conv2d and Linear multiply-adds; reported as an implementation-level approximation",
    }


__all__ = ["profile_model_efficiency"]
