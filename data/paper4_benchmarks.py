"""Token-level datasets for the Paper 4 diagnostic benchmark scaffold."""

import numpy as np
import torch
from torch.utils.data import Dataset


def planted_modality_graph(num_modalities, graph_type):
    if num_modalities < 2:
        raise ValueError("Synthetic benchmarks require at least two modalities")
    adjacency = np.zeros((num_modalities, num_modalities), dtype=np.float32)
    if graph_type == "chain":
        for index in range(num_modalities - 1):
            adjacency[index, index + 1] = adjacency[index + 1, index] = 1.0
    elif graph_type == "star":
        adjacency[0, 1:] = 1.0
        adjacency[1:, 0] = 1.0
    elif graph_type == "two_community":
        midpoint = num_modalities // 2
        for start, stop in ((0, midpoint), (midpoint, num_modalities)):
            adjacency[start:stop, start:stop] = 1.0
        np.fill_diagonal(adjacency, 0.0)
        adjacency[midpoint - 1, midpoint] = adjacency[midpoint, midpoint - 1] = 0.25
    else:
        raise ValueError(f"Unsupported planted graph: {graph_type}")
    return adjacency


class TokenBenchmarkDataset(Dataset):
    def __init__(self, tokens, labels, modality_mask=None, token_mask=None):
        self.tokens = torch.as_tensor(tokens, dtype=torch.float32)
        self.labels = torch.as_tensor(labels)
        if self.tokens.ndim != 5:
            raise ValueError("tokens must have shape [N, G, M, T, C]")
        if len(self.tokens) != len(self.labels):
            raise ValueError("tokens and labels must contain the same number of samples")
        sample_count, groups, modalities, token_count = self.tokens.shape[:4]
        if modality_mask is None:
            modality_mask = np.ones((sample_count, modalities), dtype=bool)
        if token_mask is None:
            token_mask = np.ones((sample_count, groups, modalities, token_count), dtype=bool)
        self.modality_mask = torch.as_tensor(modality_mask, dtype=torch.bool)
        self.token_mask = torch.as_tensor(token_mask, dtype=torch.bool)
        if tuple(self.modality_mask.shape) != (sample_count, modalities):
            raise ValueError("modality_mask must have shape [N, M]")
        if tuple(self.token_mask.shape) != tuple(self.tokens.shape[:-1]):
            raise ValueError("token_mask must have shape [N, G, M, T]")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "tokens": self.tokens[index],
            "label": self.labels[index],
            "modality_mask": self.modality_mask[index],
            "token_mask": self.token_mask[index],
        }


def make_synthetic_splits(
    num_samples=600,
    num_modalities=4,
    token_count=12,
    token_dim=8,
    graph_type="chain",
    regime="topology_relevant",
    missing_rate=0.0,
    seed=42,
):
    if regime not in {"geometry_only", "exchangeable", "topology_relevant"}:
        raise ValueError(f"Unsupported synthetic regime: {regime}")
    if not 0.0 <= float(missing_rate) < 1.0:
        raise ValueError("missing_rate must be in [0, 1)")
    rng = np.random.default_rng(seed)
    graph = planted_modality_graph(num_modalities, graph_type)
    labels = rng.integers(0, 2, size=num_samples, dtype=np.int64)
    innovations = rng.normal(
        size=(num_samples, num_modalities, token_count, token_dim)
    ).astype(np.float32)
    signs = (2 * labels - 1).astype(np.float32)[:, None, None, None]

    if regime == "geometry_only":
        tokens = innovations.copy()
        tokens[..., 0] *= (1.0 + 0.45 * labels[:, None, None]).astype(np.float32)
    elif regime == "exchangeable":
        shared = rng.normal(size=(num_samples, 1, token_count, token_dim)).astype(np.float32)
        tokens = innovations + (0.25 + 0.35 * labels[:, None, None, None]) * shared
    else:
        degree = graph.sum(axis=-1, keepdims=True).clip(min=1.0)
        normalized = graph / degree
        propagated = np.einsum("mn,bntc->bmtc", normalized, innovations)
        tokens = innovations + 0.45 * signs * propagated

    modality_mask = rng.random((num_samples, num_modalities)) >= float(missing_rate)
    empty = np.where(modality_mask.sum(axis=1) == 0)[0]
    if len(empty):
        modality_mask[empty, rng.integers(0, num_modalities, size=len(empty))] = True
    token_mask = np.broadcast_to(
        modality_mask[:, None, :, None],
        (num_samples, 1, num_modalities, token_count),
    ).copy()
    tokens = tokens[:, None]

    order = rng.permutation(num_samples)
    train_stop = int(0.70 * num_samples)
    val_stop = int(0.85 * num_samples)
    indices = {
        "train": order[:train_stop],
        "val": order[train_stop:val_stop],
        "test": order[val_stop:],
    }
    splits = {
        name: TokenBenchmarkDataset(
            tokens[index], labels[index], modality_mask[index], token_mask[index]
        )
        for name, index in indices.items()
    }
    return splits, graph


def load_npz_splits(path):
    """Load pretokenized AV-MNIST or aligned MOSEI features from one NPZ file."""
    archive = np.load(path)
    splits = {}
    for split in ("train", "val", "test"):
        token_key = f"{split}_tokens"
        label_key = f"{split}_labels"
        if token_key not in archive or label_key not in archive:
            raise ValueError(f"NPZ file is missing {token_key} or {label_key}")
        splits[split] = TokenBenchmarkDataset(
            archive[token_key],
            archive[label_key],
            archive[f"{split}_modality_mask"] if f"{split}_modality_mask" in archive else None,
            archive[f"{split}_token_mask"] if f"{split}_token_mask" in archive else None,
        )
    return splits


__all__ = [
    "TokenBenchmarkDataset",
    "planted_modality_graph",
    "make_synthetic_splits",
    "load_npz_splits",
]
