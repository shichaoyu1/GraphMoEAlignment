"""Train-only fixed evidence masks and paired, class-balanced MRI diagnostics."""

import json

import numpy as np
import pandas as pd
import torch

from glioma.data.idea3_pareto import balanced_weights


FIELDS = ('idh', '1p19q', 'mgmt', 'phenotype')
PREFERENCES = (0., .25, .5, .75, 1.)


@torch.no_grad()
def choose_masks(tokens, targets, reader, k, proposals, local_steps, seed):
    """Choose one global mask per preference using training tensors only."""
    device = tokens.device
    generator = torch.Generator(device=device).manual_seed(seed)
    weights = balanced_weights(targets)
    scale = max(float((targets[:, 3] - targets[:, 3].median()).abs().mean()), .05)

    def risk(mask):
        prediction = reader(tokens, mask.expand(len(tokens), -1))
        molecular = ((prediction[:, :3] - targets[:, :3].clamp_min(0)).square() * weights).sum(1).mean()
        phenotype = (prediction[:, 3] - targets[:, 3]).abs().mean()
        return torch.stack([molecular / .25, phenotype / scale])

    candidates = torch.rand(proposals, tokens.shape[1], generator=generator, device=device).topk(k, -1).indices
    losses = torch.stack([risk(candidate) for candidate in candidates])
    masks = {}
    for preference in PREFERENCES:
        coefficients = losses.new_tensor([preference, 1 - preference])
        costs = losses @ coefficients
        current, best = candidates[costs.argmin()].clone(), costs.min()
        for _ in range(local_steps if k < tokens.shape[1] else 0):
            available = torch.ones(tokens.shape[1], device=device).scatter(0, current, 0.)
            candidate = current.clone()
            slot = torch.randint(k, (), generator=generator, device=device)
            candidate[slot] = torch.multinomial(available, 1, generator=generator)[0]
            cost = risk(candidate) @ coefficients
            if cost < best:
                current, best = candidate, cost
        masks[preference] = current.cpu().tolist()
    return masks


@torch.no_grad()
def predict_role(tokens, targets, subject_ids, reader, masks, train_targets, role, fold, training_seed,
                 random_repeats, seed):
    """Write every random draw; constant references have an empty mask."""
    device = tokens.device
    generator = torch.Generator(device=device).manual_seed(seed)
    prevalence = torch.stack([(train_targets[:, j] == 1).sum() / (train_targets[:, j] >= 0).sum()
                              for j in range(3)])
    median = train_targets[:, 3].median()
    records = []
    has_classes = all(bool((targets[:, j] == label).any()) for j in range(3) for label in (0, 1))
    molecular_weights = balanced_weights(targets) if has_classes else None
    target_rows = targets.cpu().tolist()

    def append(method, preference, selected, repeats=1, constant=None):
        patients = torch.arange(len(tokens), device=device).repeat_interleave(repeats)
        for offset in range(0, len(patients), 256):
            ids = patients[offset:offset + 256]
            indices = selected[offset:offset + len(ids)]
            prediction = reader(tokens[ids], indices) if constant is None else constant.expand(len(ids), -1)
            if not torch.isfinite(prediction).all():
                raise ValueError('Nonfinite reader predictions')
            anchor_values = (((prediction[:, :3] - targets[ids, :3].clamp_min(0)).square() *
                              molecular_weights[ids]).sum(1).cpu().tolist() if molecular_weights is not None else [None] * len(ids))
            for index, (patient, values, indices_row) in enumerate(zip(ids.tolist(), prediction.cpu().tolist(), indices.cpu().tolist())):
                row = {'method': method, 'role': role, 'fold': fold, 'seed': training_seed,
                       'subject_id': subject_ids[patient], 'weight': preference,
                       'sample': (offset + index) % repeats, 'indices': json.dumps(indices_row)}
                for field, value, target in zip(FIELDS, values, target_rows[patient]):
                    row['prediction_' + field], row['target_' + field] = value, target
                row['phenotype_loss'] = abs(values[3] - row['target_phenotype'])
                row['anchor_loss'] = anchor_values[index]
                records.append(row)

    for preference, mask in masks.items():
        append('fixed8_train_selected', preference, torch.tensor(mask, device=device).expand(len(tokens), -1))
    random = torch.rand(len(tokens) * random_repeats, tokens.shape[1], generator=generator,
                        device=device).topk(len(next(iter(masks.values()))), -1).indices
    append('random8', .5, random, random_repeats)
    append('all_tokens_reference', .5, torch.arange(tokens.shape[1], device=device).expand(len(tokens), -1))
    empty = torch.empty(len(tokens), 0, dtype=torch.long, device=device)
    append('train_frequency_median', .5, empty, constant=torch.cat([prevalence, median[None]]))
    append('constant_half_train_median', .5, empty, constant=torch.cat([prevalence.new_full((3,), .5), median[None]]))
    frame = pd.DataFrame(records)
    frame['anchor_loss'] = pd.to_numeric(frame.anchor_loss, errors='coerce')
    return frame


def patient_errors(frame):
    keys = ['role', 'fold', 'seed', 'method', 'weight', 'subject_id']
    if frame.duplicated(keys + ['sample']).any():
        raise ValueError('Duplicate patient/sample predictions')
    for field in FIELDS:
        if frame.groupby(['role', 'fold', 'subject_id'])['target_' + field].nunique().gt(1).any():
            raise ValueError('Conflicting patient targets')
    frame = frame.copy()
    for field in FIELDS[:3]:
        frame['error_' + field] = (frame['prediction_' + field] - frame['target_' + field].clip(lower=0)) ** 2
    columns = [f'{prefix}_{field}' for prefix in ('target', 'error') for field in FIELDS[:3]] + ['phenotype_loss']
    return frame.groupby(keys, as_index=False)[columns].mean()


def risks(rows):
    losses = []
    counts = {}
    for field in FIELDS[:3]:
        counts[field] = {}
        for label in (0, 1):
            selected = rows['target_' + field] == label
            counts[field][str(label)] = int(selected.sum())
            losses.append(float(rows.loc[selected, 'error_' + field].mean()) if selected.any() else None)
    return {'anchor_brier': float(np.mean(losses)) if all(v is not None for v in losses) else None,
            'phenotype_mae_pp': float(rows.phenotype_loss.mean() * 100), 'molecular_counts': counts,
            'patients': len(rows)}


def endpoint_contrast(patients, repetitions, seed):
    """Pooled OOF class-balanced Brier; bootstrap patients within each fold."""
    selected = patients[(patients.method == 'fixed8_train_selected') & patients.weight.isin([0., 1.])]
    if selected.empty:
        raise ValueError('Missing fixed-mask endpoints')
    rng = np.random.default_rng(seed)
    delta = np.zeros((repetitions + 1, 2))
    pooled_numerator = np.zeros((repetitions + 1, 6))
    pooled_denominator = np.zeros_like(pooled_numerator)
    folds = []
    total = selected[selected.weight == 0].shape[0]
    for fold, group in selected.groupby('fold'):
        left = group[group.weight == 0].sort_values('subject_id')
        right = group[group.weight == 1].sort_values('subject_id')
        if left.empty or tuple(left.subject_id) != tuple(right.subject_id):
            raise ValueError('Unpaired endpoint patients')
        n = len(left)
        counts = np.vstack([np.ones(n), rng.multinomial(n, np.full(n, 1 / n), repetitions)])
        fold_values = np.zeros_like(delta)
        for field_index, field in enumerate(FIELDS[:3]):
            if not np.array_equal(left['target_' + field].values, right['target_' + field].values):
                raise ValueError('Endpoint labels differ')
            difference = right['error_' + field].to_numpy() - left['error_' + field].to_numpy()
            for label in (0, 1):
                observed = left['target_' + field].to_numpy() == label
                denominator = counts @ observed
                numerator = counts @ (difference * observed)
                pooled_denominator[:, field_index * 2 + label] += denominator
                pooled_numerator[:, field_index * 2 + label] += numerator
                fold_values[:, 0] += np.divide(numerator, denominator, out=np.full(repetitions + 1, np.nan),
                                               where=denominator > 0) / 6
        phenotype = right.phenotype_loss.to_numpy() - left.phenotype_loss.to_numpy()
        fold_values[:, 1] = counts @ phenotype / n * 100
        delta[:, 1] += fold_values[:, 1] * n / total
        folds.append({'fold': int(fold), 'patients': n,
                      'anchor_delta': finite(fold_values[0, 0]), 'phenotype_delta_pp': float(fold_values[0, 1]),
                      'expected_direction': bool(fold_values[0, 0] < 0 < fold_values[0, 1])})
    delta[:, 0] = np.divide(pooled_numerator, pooled_denominator, out=np.full_like(pooled_numerator, np.nan),
                            where=pooled_denominator > 0).mean(1)
    valid = np.isfinite(delta[1:]).all(1)
    interval = np.quantile(delta[1:][valid], [.025, .975], axis=0) if valid.any() else None
    return {'patients': total, 'folds': folds, 'anchor_delta': finite(delta[0, 0]),
            'phenotype_delta_pp': float(delta[0, 1]),
            'anchor_ci95': interval[:, 0].tolist() if interval is not None else None,
            'phenotype_ci95_pp': interval[:, 1].tolist() if interval is not None else None,
            'valid_bootstrap': int(valid.sum()), 'invalid_bootstrap': int((~valid).sum()),
            'expected_direction_folds': sum(f['expected_direction'] for f in folds),
            'intervals_support_expected_direction': bool(interval is not None and interval[1, 0] < 0 < interval[0, 1]),
            'interval_scope': 'paired patients within fold, conditional on fitted readers and train-selected masks; unadjusted'}


def finite(value):
    return float(value) if np.isfinite(value) else None


def summarize(frame, repetitions, seed):
    patients = patient_errors(frame)
    curves = []
    for (role, fold, method, preference), rows in patients.groupby(['role', 'fold', 'method', 'weight']):
        curves.append({'role': role, 'fold': int(fold), 'method': method, 'weight': float(preference), **risks(rows)})
    endpoints = {role: endpoint_contrast(rows, repetitions, seed) for role, rows in patients.groupby('role')}
    return curves, endpoints
