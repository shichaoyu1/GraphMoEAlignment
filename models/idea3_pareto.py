"""Small MRI token reader and preference-conditioned evidence policies."""

import math
import torch
from torch import nn
from glioma.data.utsw_authority import compile_grade, FEATURE_DIM
from glioma.modules.authority_projection import project_allowed_simplex


class TokenGridPool(nn.Module):
    """Adaptive 4x4x4 averages using reductions with deterministic CUDA backward."""

    def forward(self, values):
        # Adaptive pooling bins may overlap, including when an axis is smaller than 4.
        # Separable slice means preserve those bins without CUDA's atomic pool backward.
        for axis in (-3, -2, -1):
            size = values.shape[axis]
            bins = []
            for index in range(4):
                start = index * size // 4
                end = ((index + 1) * size + 3) // 4
                bins.append(values.narrow(axis, start, end - start).mean(axis, keepdim=True))
            values = torch.cat(bins, dim=axis)
        return values


class TokenReader(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv3d(4, 16, 5, 2, 2), nn.GroupNorm(4, 16), nn.SiLU(),
                                     nn.Conv3d(16, 32, 3, 2, 1), nn.GroupNorm(4, 32), nn.SiLU(),
                                     nn.Conv3d(32, width, 3, 2, 1), nn.SiLU(), TokenGridPool())
        self.head = nn.Sequential(nn.Linear(width * 2, 64), nn.SiLU(), nn.Linear(64, 4))

    def encode(self, images):
        return self.encoder(images).flatten(2).transpose(1, 2)

    def forward(self, tokens, indices):
        selected = tokens.gather(1, indices[..., None].expand(-1, -1, tokens.shape[-1]))
        pooled = torch.cat([selected.mean(1), selected.std(1, unbiased=False)], -1)
        return self.head(pooled).sigmoid()


class EvidencePolicy(nn.Module):
    def __init__(self, items=64, context=64, width=64, experts=1, direct=False):
        super().__init__()
        self.items = items
        self.direct = direct
        self.greedy = False
        self.selection_k = 4
        self.input = nn.Linear(items + context + 5, width)
        self.experts = nn.ModuleList(nn.Sequential(nn.Linear(width, width), nn.SiLU(),
                                                  nn.Linear(width, items)) for _ in range(experts))
        self.router = nn.Linear(width, experts)

    def forward(self, state, context, preference, time):
        pref = torch.stack([preference, 1 - preference, torch.sin(math.pi * preference),
                            torch.cos(math.pi * preference), time], -1)
        hidden = torch.nn.functional.silu(self.input(torch.cat([state, context, pref], -1)))
        routes = self.router(hidden).softmax(-1)
        predictions = torch.stack([expert(hidden) for expert in self.experts], 1)
        return (predictions * routes[..., None]).sum(1), routes

    def sample(self, context, preference, steps=16, noise=.05):
        state = torch.randn(len(context), self.items, device=context.device)
        if self.direct:
            state.zero_()
            logits, routing = self(state, context, preference, state.new_zeros(len(state)))
            available = torch.ones_like(logits, dtype=torch.bool)
            selected = []
            for _ in range(self.selection_k):
                probabilities = logits.masked_fill(~available, -torch.inf).softmax(-1)
                choice = probabilities.argmax(-1, keepdim=True) if self.greedy else torch.multinomial(probabilities, 1)
                selected.append(choice)
                available.scatter_(1, choice, False)
            indices = torch.cat(selected, 1)
            scores = torch.full_like(logits, -1).scatter_(1, indices, 1)
            return scores, state[:, None], indices, routing[:, None]
        states, actions, routes = [], [], []
        for step in range(steps):
            time = state.new_full((len(state),), step / steps)
            velocity, routing = self(state, context, preference, time)
            action = state + velocity / steps + noise / math.sqrt(steps) * torch.randn_like(state)
            states.append(state)
            actions.append(action)
            routes.append(routing)
            state = action
        return state, torch.stack(states, 1), torch.stack(actions, 1), torch.stack(routes, 1)

    def log_probability(self, states, actions, context, preference, noise):
        if self.direct:
            logits = self(states[:, 0], context, preference, states.new_zeros(len(states)))[0]
            available = torch.ones_like(logits, dtype=torch.bool)
            values = []
            for choice in actions.unbind(1):
                distribution = torch.distributions.Categorical(logits=logits.masked_fill(~available, -torch.inf))
                values.append(distribution.log_prob(choice))
                available = available.scatter(1, choice[:, None], False)
            return torch.stack(values, 1).sum(-1, keepdim=True)
        steps = states.shape[1]
        times = torch.arange(steps, device=states.device).expand(len(states), -1) / steps
        means = states + self(states.flatten(0, 1), context[:, None].expand(-1, steps, -1).flatten(0, 1),
                              preference[:, None].expand(-1, steps).flatten(), times.flatten())[0].reshape_as(states) / steps
        return torch.distributions.Normal(means, noise / math.sqrt(steps)).log_prob(actions).sum(-1)


class TokenGradeReadout(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.report = nn.Sequential(nn.Linear(FEATURE_DIM, 64), nn.SiLU(), nn.Linear(64, 3))
        self.mri = nn.Sequential(nn.Linear(128 + FEATURE_DIM, 64), nn.SiLU(), nn.Linear(64, 3))

    def forward(self, tokens, indices, evidence):
        selected = tokens.gather(1, indices[..., None].expand(-1, -1, tokens.shape[-1]))
        pooled = torch.cat([selected.mean(1), selected.std(1, unbiased=False)], -1)
        report = self.report(evidence.molecular)
        logits = report + self.mri(torch.cat([pooled, evidence.molecular], -1))
        rule = compile_grade(evidence)
        raw = report.softmax(-1) if self.variant == 'report' else logits.softmax(-1)
        if self.variant == 'readonly':
            report_prob = report.softmax(-1)
            restricted = torch.cat([(1 - report_prob[:, 2:]) * logits[:, :2].softmax(-1), report_prob[:, 2:]], -1)
            raw = torch.where(rule['active'][:, None], restricted, raw)
        return {'p0': raw, 'p_star': project_allowed_simplex(raw, rule['allowed']), **rule}
