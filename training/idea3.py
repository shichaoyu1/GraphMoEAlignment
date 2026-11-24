"""Shared training and train-only controls for the anchor-phenotype experiment."""

from copy import deepcopy
from time import perf_counter

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from glioma.cli.utsw_authority import seed_all
from glioma.data.idea3 import AnchorPhenotypeDataset
from glioma.models.idea3 import AnchorPhenotypeNet, task_losses


WEIGHTS = (.05, .25, .50, .75, .95)
INTERMEDIATE = (.15, .375, .625, .85)


def targets(rows):
    a = np.array([row["anchors"] for row in rows], dtype=float)
    b = np.array([row["phenotype"] for row in rows], dtype=float)
    return a, b


def control_parameters(rows):
    anchors, phenotype = targets(rows)
    for field in anchors.T:
        if set(field[np.isfinite(field)]) != {0, 1}:
            raise ValueError("Each molecular field needs both classes in the training fold")
    if not np.isfinite(phenotype).any():
        raise ValueError("Training fold has no nonempty phenotype target")
    frequency = np.nanmean(anchors, axis=0)
    median = np.nanmedian(phenotype)
    entropy = -(frequency * np.log(frequency) + (1 - frequency) * np.log1p(-frequency))
    scales = (max(float(entropy.mean()), .05), max(float(np.nanmean(abs(phenotype - median))), .05))
    return frequency, median, scales


def molecular_features(anchors):
    return np.eye(3)[np.nan_to_num(anchors, nan=2).astype(int)].reshape(len(anchors), -1)


def phenotype_prediction(x, y, test):
    has_y = np.isfinite(y)
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(x[has_y], y[has_y])
    return model.predict(test).clip(0, 1)


def linear_probes(embedding, train_rows, test_embedding):
    anchors, phenotype = targets(train_rows)
    scores = []
    for labels in anchors.T:
        observed = np.isfinite(labels)
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
        model.fit(embedding[observed], labels[observed])
        scores.append(model.predict_proba(test_embedding)[:, 1])
    scores.append(phenotype_prediction(embedding, phenotype, test_embedding))
    return np.column_stack(scores)


def loader(rows, cache, images, batch_size, device, seed=None):
    return DataLoader(AnchorPhenotypeDataset(rows, cache, images), batch_size=batch_size,
                      shuffle=seed is not None, pin_memory=device.type == "cuda",
                      generator=None if seed is None else torch.Generator().manual_seed(seed))


@torch.no_grad()
def forward_batches(model, batches, weight, device):
    model.eval()
    results = []
    for image, anchors, target in batches:
        features = model(image.to(device=device, dtype=torch.float32, non_blocking=True), weight)
        results.append([*(x.cpu() for x in features), anchors, target])
    return [torch.cat(parts) for parts in zip(*results)]


def objective(logits, phenotype, anchors, target, weight, scales):
    aa, pp = task_losses(logits, phenotype, anchors, target)
    return weight * aa / scales[0] + (1 - weight) * pp / scales[1]


def fit_anchor_model(train, val, cache, images, device, seed, weight, epochs, batch_size):
    seed_all(seed)
    selected_weights = WEIGHTS if weight is None else (weight,)
    pref_rng = np.random.default_rng(seed)
    _, _, scales = control_parameters(train)
    training = [r for r in train if any(v is not None for v in r["anchors"]) or r["phenotype"] is not None]
    batches = loader(training, cache, images, batch_size, device, seed=seed)
    validation = loader(val, cache, images, batch_size, device)
    model = AnchorPhenotypeNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_score, best_state, best_epoch = float("inf"), None, None
    start = perf_counter()
    for epoch in range(epochs):
        model.train()
        for image, anchors, target in batches:
            w = float(pref_rng.choice(selected_weights))
            optimizer.zero_grad(set_to_none=True)
            logits, phenotype, _ = model(image.to(device=device, dtype=torch.float32, non_blocking=True), w)
            loss = objective(logits, phenotype, anchors.to(device), target.to(device), w, scales)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite anchor-phenotype training loss")
            loss.backward()
            optimizer.step()
        scores = []
        for w in selected_weights:
            logits, phenotype, _, anchors, target = forward_batches(model, validation, w, device)
            scores.append(float(objective(logits, phenotype, anchors, target, w, scales)))
        score = float(np.mean(scores))
        if not np.isfinite(score):
            raise FloatingPointError("Nonfinite anchor-phenotype validation loss")
        if score < best_score:
            best_score, best_state, best_epoch = score, deepcopy(model.state_dict()), epoch + 1
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "validation_loss": best_score, "loss_scales": scales,
                   "train_seconds": perf_counter() - start, "parameters": sum(p.numel() for p in model.parameters())}
