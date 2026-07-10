"""Graph construction and graph consistency losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticGraphBuilder(nn.Module):
    def __init__(self, shared_dim: int, graph_type: str = 'learnable', tau: float = 0.2):
        super().__init__()
        self.graph_type = graph_type
        self.tau = tau
        self.edge_mlp = nn.Sequential(
            nn.Linear(shared_dim * 4, shared_dim),
            nn.SiLU(),
            nn.Linear(shared_dim, 1),
        )

    def _mask_self(self, scores):
        n_nodes = scores.shape[1]
        eye = torch.eye(n_nodes, dtype=torch.bool, device=scores.device).unsqueeze(0)
        return scores.masked_fill(eye, -1e9)

    def forward(self, shared):
        batch_size, n_nodes, dim = shared.shape
        device = shared.device

        if self.graph_type == 'no_graph':
            return torch.zeros(batch_size, n_nodes, n_nodes, device=device)

        if self.graph_type == 'fixed':
            adjacency = torch.ones(batch_size, n_nodes, n_nodes, device=device)
            adjacency = adjacency - torch.eye(n_nodes, device=device).unsqueeze(0)
            return adjacency / max(n_nodes - 1, 1)

        if self.graph_type == 'random':
            scores = torch.rand(batch_size, n_nodes, n_nodes, device=device)
            return F.softmax(self._mask_self(scores), dim=-1)

        if self.graph_type == 'similarity':
            normalized = F.normalize(shared, dim=-1)
            scores = torch.matmul(normalized, normalized.transpose(1, 2)) / self.tau
            return F.softmax(self._mask_self(scores), dim=-1)

        if self.graph_type == 'learnable':
            source = shared.unsqueeze(2).expand(-1, -1, n_nodes, -1)
            target = shared.unsqueeze(1).expand(-1, n_nodes, -1, -1)
            pair = torch.cat([source, target, (source - target).abs(), source * target], dim=-1)
            scores = self.edge_mlp(pair).squeeze(-1)
            return F.softmax(self._mask_self(scores), dim=-1)

        raise ValueError(f'Unsupported graph_type: {self.graph_type}')


def graph_laplacian_consistency(shared, adjacency):
    if adjacency.abs().sum() == 0:
        return shared.sum() * 0
    adjacency_sym = 0.5 * (adjacency + adjacency.transpose(1, 2))
    diff = shared.unsqueeze(2) - shared.unsqueeze(1)
    energy = 0.5 * adjacency_sym.unsqueeze(-1) * diff.pow(2)
    return energy.sum(dim=(1, 2, 3)).mean()


__all__ = ["SemanticGraphBuilder", "graph_laplacian_consistency"]
