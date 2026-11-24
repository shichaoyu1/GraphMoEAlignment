"""Controlled Idea 3 training with paired validation-selected and final checkpoints."""

from copy import deepcopy

import numpy as np
import torch

from glioma.cli.diagnose_idea3_anchor import field_ap
from glioma.cli.utsw_authority import seed_all
from glioma.data.idea3 import ANCHORS
from glioma.models.idea3 import AnchorPhenotypeNet
from glioma.training.idea3 import control_parameters, forward_batches, loader, objective


def fit(train, val, cache, images, device, seed, weight, preference, epochs, batch_size):
    seed_all(seed)
    _, _, scales = control_parameters(train)
    eligible = [r for r in train if any(v is not None for v in r["anchors"]) or r["phenotype"] is not None]
    batches = loader(eligible, cache, images, batch_size, device, seed=seed)
    validation = loader(val, cache, images, batch_size, device)
    model = AnchorPhenotypeNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    history, best, best_state = [], float("inf"), None
    for epoch in range(1, epochs + 1):
        model.train()
        for image, anchors, target in batches:
            optimizer.zero_grad(set_to_none=True)
            logits, phenotype, _ = model(image.to(device=device, dtype=torch.float32, non_blocking=True), preference)
            loss = objective(logits, phenotype, anchors.to(device), target.to(device), weight, scales)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training objective")
            loss.backward()
            optimizer.step()
        logits, phenotype, _, anchors, target = forward_batches(model, validation, preference, device)
        val_loss = float(objective(logits, phenotype, anchors, target, weight, scales))
        if not torch.isfinite(torch.tensor(val_loss)):
            raise FloatingPointError("Nonfinite validation objective")
        record = {"epoch": epoch, "validation_loss": val_loss, "validation_mae": None}
        valid = target.isfinite()
        if valid.any():
            record["validation_mae"] = float((phenotype[valid] - target[valid]).abs().mean())
        for index, field in enumerate(ANCHORS):
            label, logit = anchors[:, index], logits[:, index]
            observed = label >= 0
            record[f"{field}_bce"] = (float(torch.nn.functional.binary_cross_entropy_with_logits(
                logit[observed], label[observed])) if observed.any() else None)
            ap = field_ap(label.numpy().astype(int), logit.sigmoid().numpy(), np.ones((1, len(label)), dtype=int))
            record[f"{field}_ap"] = float(ap["ap_balanced"][0]) if np.isfinite(ap["ap_balanced"][0]) else None
        history.append(record)
        if val_loss < best:
            best, best_state = val_loss, deepcopy(model.state_dict())
    return model, best_state, history
