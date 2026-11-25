"""Independent cached-token and exact-oracle environments for Idea 3."""

from itertools import combinations
import json
from pathlib import Path

import numpy as np
import torch


def balanced_weights(targets):
    weights = torch.zeros_like(targets[:, :3])
    for field in range(3):
        for value in (0, 1):
            selected = targets[:, field] == value
            count = int(selected.sum())
            if count == 0:
                raise ValueError(f"Molecular field {field} has no class {value}")
            weights[selected, field] = len(targets) / (6 * count)
    return weights


class TokenEnvironment:
    def __init__(self, tokens, targets, reader, device, subject_ids):
        self.tokens, self.targets = tokens.to(device), targets.to(device)
        self.reader = reader.to(device).eval().requires_grad_(False)
        self.weights = balanced_weights(self.targets)
        self.context = self.tokens.mean(1)
        self.subject_ids = subject_ids
        self.items = self.tokens.shape[1]
        self.scales = torch.tensor([.25, max(float((self.targets[:, 3] - self.targets[:, 3].median()).abs().mean()), .05)], device=device)

    def losses(self, ids, indices):
        scores = self.reader(self.tokens[ids], indices)
        target = self.targets[ids]
        anchor = ((scores[:, :3] - target[:, :3].clamp_min(0)).square() * self.weights[ids]).sum(-1)
        return torch.stack([anchor, (scores[:, 3] - target[:, 3]).abs()], -1)


class ControlledEnvironment:
    def __init__(self, task, device):
        self.context = torch.eye(2, device=device)
        self.subject_ids = ['context0', 'context1']
        self.items = 8
        self.scales = torch.ones(2, device=device)
        self.masks = torch.tensor(list(combinations(range(8), 4)), device=device)
        mask = torch.zeros(70, 8, device=device).scatter_(1, self.masks, 1)
        if task == 'sensors':
            precision_a = mask.new_tensor([[8, 4, 2, 1, 0, 0, 0, 0], [1, 2, 4, 8, 0, 0, 0, 0]])
            precision_c = mask.new_tensor([[0, 0, 0, 0, 1, 2, 4, 8], [0, 0, 0, 0, 8, 4, 2, 1]])
            self.risk_table = torch.stack([1 / (1 + precision_a @ mask.T), 1 / (1 + precision_c @ mask.T)], -1)
        elif task == 'nonconvex':
            points = mask.new_tensor([[0, 1], [.25, .9375], [.5, .75], [.75, .4375], [1, 0], [1, 1]])
            self.risk_table = points[(torch.arange(70, device=device) // 10).clamp_max(5)][None].repeat(2, 1, 1)
        elif task == 'common':
            values = torch.arange(70, device=device).float() / 70
            self.risk_table = values[None, :, None].repeat(2, 1, 2)
        else:
            raise ValueError(task)
        codes = (2 ** self.masks).sum(-1)
        self.lookup = torch.full((256,), -1, dtype=torch.long, device=device)
        self.lookup[codes] = torch.arange(70, device=device)

    def losses(self, ids, indices):
        action = self.lookup[(2 ** indices).sum(-1)]
        if (action < 0).any():
            raise ValueError('Invalid controlled evidence action')
        return self.risk_table[ids, action]


def cohort_rows(prepared, authority, fold):
    rows = json.loads((Path(prepared) / 'manifest.json').read_text(encoding='utf-8'))['rows']
    split = json.loads((Path(authority) / 'splits' / f'fold{fold}.json').read_text(encoding='utf-8'))
    result = {role: [r for r in rows if r['subject_id'] in set(ids) and r['phenotype'] is not None]
              for role, ids in split.items()}
    groups = [set(r['subject_id'] for r in result[role]) for role in ('train', 'validation', 'test')]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError('Authority folds overlap')
    return result
