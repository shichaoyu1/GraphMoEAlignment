"""Mixture-of-experts components for optional semantic routing."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticMoEGate(nn.Module):
    def __init__(self, in_dim: int, num_experts: int, hidden_dim: int = 128, temperature: float = 1.0):
        super().__init__()
        self.num_experts = num_experts
        self.temperature = float(temperature)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, context):
        logits = self.net(context) / max(self.temperature, 1e-6)
        return F.softmax(logits, dim=-1)

    @staticmethod
    def gate_entropy(gate_probs):
        return -(gate_probs * torch.log(gate_probs + 1e-8)).sum(dim=-1).mean()

    @staticmethod
    def load_balance(gate_probs):
        mean_usage = gate_probs.mean(dim=0)
        target = torch.full_like(mean_usage, 1.0 / max(mean_usage.numel(), 1))
        return torch.mean((mean_usage - target) ** 2)


class AnchorExpert(nn.Module):
    def __init__(self, shared_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class RegionExpert(nn.Module):
    def __init__(self, shared_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class GraphExpert(nn.Module):
    def __init__(self, shared_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class DiffusionExpert(nn.Module):
    def __init__(self, shared_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


__all__ = [
    "SemanticMoEGate",
    "AnchorExpert",
    "RegionExpert",
    "GraphExpert",
    "DiffusionExpert",
]
