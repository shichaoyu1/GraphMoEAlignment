"""Paired, label-separated synthetic tasks for authority mechanism experiments."""

from dataclasses import dataclass, fields, replace
from typing import List

import numpy as np
import torch


@dataclass
class AuthorityEvidence:
    tokens: torch.Tensor
    source_available: torch.Tensor
    verified_validity: torch.Tensor  # 0 invalid, 1 valid, 2 unknown
    source_quality: torch.Tensor
    statements: torch.Tensor
    authority_sources: torch.Tensor  # [batch, relation], positions in source axis
    target_id: torch.Tensor
    conflict: torch.Tensor
    task: str
    rule_version: str = "authority-v1"

    def to(self, device):
        return replace(self, **{f.name: getattr(self, f.name).to(device)
                                for f in fields(self) if isinstance(getattr(self, f.name), torch.Tensor)})

    def take(self, indices):
        return replace(self, **{f.name: getattr(self, f.name)[indices]
                                for f in fields(self) if isinstance(getattr(self, f.name), torch.Tensor)})


@dataclass
class AuthorityBatch:
    evidence: AuthorityEvidence
    labels: torch.Tensor
    latent_probabilities: torch.Tensor
    sample_ids: List[str]
    event_ids: List[str]

    def to(self, device):
        return replace(self, evidence=self.evidence.to(device), labels=self.labels.to(device),
                       latent_probabilities=self.latent_probabilities.to(device))

    def take(self, indices):
        ids = indices.tolist() if isinstance(indices, (torch.Tensor, np.ndarray)) else list(indices)
        return AuthorityBatch(self.evidence.take(indices), self.labels[indices],
                              self.latent_probabilities[indices], [self.sample_ids[i] for i in ids],
                              [self.event_ids[i] for i in ids])


def s0_check():
    observations = [(x1, x2, a, (x1, x2)[a]) for x1 in (0, 1) for x2 in (0, 1) for a in (0, 1)]
    errors = sum(min(sum(row[3] == y for row in observations if row[:2] == (x1, x2))
                     for y in (0, 1)) for x1 in (0, 1) for x2 in (0, 1))
    return {"cases": observations, "policy_aware_error": 0.0, "policy_blind_bayes_error": errors / 8.0}


class AuthoritySynthetic:
    """Latent random draws are stable across intervention calls and nested sample sizes."""

    def __init__(self, task, data_seed=41, split="train", count=12000, token_count=16, token_dim=8):
        if task not in ("s1", "s2") or split not in ("train", "val", "test"):
            raise ValueError("Unknown authority task or split")
        if count < 1 or token_count < 2 or token_dim < 6:
            raise ValueError("Need positive samples, >=2 tokens and >=6 token dimensions")
        self.task, self.seed, self.split, self.count = task, int(data_seed), split, int(count)
        split_id = {"train": 1, "val": 2, "test": 3}[split]
        # Independent named streams ensure a smaller training set is an exact prefix.
        def rng(stream):
            return np.random.default_rng(np.random.SeedSequence([data_seed, split_id, 1 if task == "s1" else 2, stream]))
        self.noise = rng(0).normal(0, .5, (count, 4, token_count, token_dim)).astype(np.float32)
        self.roles = np.argsort(rng(1).random((count, 4)), axis=1)
        self.bits = rng(2).integers(0, 2, (count, 4))
        self.claims = rng(3).integers(0, 2, (count, 4, 2))
        self.uniforms = rng(4).random((count, 4, 6))
        self.label_uniforms = rng(5).random((count, 6))
        self.prob_latents = rng(6).uniform(-1, 1, (count, 6))
        self.q = rng(7).integers(0, 2, count)

    def materialize(self, event="clean", eta=0.0, rb=.8, kappa=.1, strength=1.0):
        n = self.count
        row = np.arange(n)
        tokens = self.noise.copy()
        claims = self.claims.copy()
        q = self.q.copy()
        if event == "target_switch":
            q = 1 - q
        available = np.ones((n, 4), dtype=bool)
        validity = np.ones((n, 4), dtype=np.int64)
        quality = np.full((n, 4), .6, dtype=np.float32)
        if self.task == "s1":
            # q is allowed context; switching q changes the protocol's coarse target.
            a, b = self.bits[:, 0], self.bits[:, 1]
            owners = self.roles[:, :1].copy()
            owner, fine_source = owners[:, 0], self.roles[:, 1]
            statements = claims[:, :, :1]
            statements[row, owner, 0] = a ^ (self.uniforms[:, 0, 0] < eta)
            fine_observed = b ^ (self.uniforms[:, 1, 0] >= rb)
            for source in range(4):
                is_owner = owner == source
                coarse = a ^ (self.uniforms[:, source, 1] < kappa)
                coarse = np.where(is_owner, statements[:, source, 0], coarse)
                # Only auxiliary coarse content receives the conflict amplitude change.
                scale = np.where(is_owner, 1., strength)
                tokens[:, source, :, 0] += ((2 * coarse - 1) * scale)[:, None]
                fine = np.where(fine_source == source, fine_observed, self.bits[:, 2])
                tokens[:, source, :, 1] += (2 * fine - 1)[:, None]
                quality[:, source] = np.where(fine_source == source, rb, .6)
            labels = 2 * (a ^ q) + b
            probabilities = np.eye(4, dtype=np.float32)[labels]
        else:
            owners = self.roles[:, :2].copy()
            owners[q == 1] = owners[q == 1, ::-1]
            statements = claims
            directions = np.stack([claims[row, owners[:, r], r] for r in range(2)], axis=1)
            probabilities = .5 + .35 * self.prob_latents
            for relation in range(2):
                high = .75 + .1 * self.prob_latents[:, 2 * relation]
                low = .25 + .1 * self.prob_latents[:, 2 * relation + 1]
                direction = directions[:, relation]
                probabilities[:, 2 * relation] = np.where(direction == 1, high, low)
                probabilities[:, 2 * relation + 1] = np.where(direction == 1, low, high)
            labels = (self.label_uniforms < probabilities).astype(np.float32)
            # Tokens depend on latent variables and statements, never on sampled labels.
            for source in range(4):
                tokens[:, source, :, :2] += (2 * claims[:, source, :] - 1)[:, None, :]
                tokens[:, source, :, 2:4] += self.prob_latents[:, None, :2]
                strength_b = (rb - .5) * 3
                # Reliability interventions must hold both admitted sources fixed.
                # The owner union is invariant to q switching the two relations.
                auxiliary = (self.roles[:, :2] != source).all(-1)
                tokens[:, source, :, 4:6] += auxiliary[:, None, None] * strength_b * self.prob_latents[:, None, 4:6]
                quality[:, source] = np.where(auxiliary, rb, .6)
            # Accepted wrong statements leave true probabilities / labels fixed.
            for relation in range(2):
                flip = self.uniforms[:, relation, 0] < eta
                statements[row, owners[:, relation], relation] ^= flip
        original_owners = owners.copy()
        for relation in range(owners.shape[1]):
            owner = original_owners[:, relation]
            if event in ("invalid", "unknown"):
                validity[row, owner] = 0 if event == "invalid" else 2
            elif event == "missing":
                available[row, owner] = False
            elif event == "wrong_statement":
                statements[row, owner, relation] ^= 1
        if event == "wrong_policy":
            owners = (owners + 1) % 4
        elif event == "shuffled_policy":
            owners = owners[np.roll(row, 1)]
        elif event == "all_missing":
            available[:] = False
        known = {"clean", "invalid", "unknown", "missing", "wrong_statement", "wrong_policy",
                 "shuffled_policy", "all_missing", "target_switch", "conflict", "bypass", "aux_conflict", "reliability"}
        if event not in known:
            raise ValueError("Unknown intervention: " + event)
        ev = AuthorityEvidence(
            torch.from_numpy(tokens[:, None]), torch.from_numpy(available), torch.from_numpy(validity),
            torch.from_numpy(quality), torch.from_numpy(statements.copy()), torch.from_numpy(owners.copy()),
            torch.from_numpy(q), torch.full((n,), event == "conflict", dtype=torch.bool), self.task)
        ids = [f"{self.task}/d{self.seed}/{self.split}/{i}" for i in range(n)]
        tag = f"{event}:eta={eta:g}:rb={rb:g}:kappa={kappa:g}:strength={strength:g}"
        return AuthorityBatch(ev, torch.from_numpy(labels), torch.from_numpy(probabilities.astype(np.float32)),
                              ids, [f"{sample}/{tag}" for sample in ids])


def intervention_grid(task):
    events = [(name, {"event": name}) for name in
              ("invalid", "unknown", "missing", "all_missing", "wrong_statement", "wrong_policy",
               "shuffled_policy", "target_switch", "conflict", "bypass")]
    if task == "s1":
        for eta in (0., .05, .15):
            for rb in (.6, .8, .95):
                for kappa in (0., .5, 1.):
                    events.append((f"grid_e{eta:g}_r{rb:g}_k{kappa:g}",
                                   {"eta": eta, "rb": rb, "kappa": kappa}))
        for strength in (1., 2., 4., 8.):
            events.append((f"conflict_strength_{strength:g}", {"event": "aux_conflict", "kappa": 1., "strength": strength}))
    else:
        for rb in (.6, .8, .95):
            events.append((f"reliability_{rb:g}", {"event": "reliability", "rb": rb}))
    return events
