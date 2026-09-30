"""Stable paired observations and group-disjoint roles for Idea4.5 v3."""

import numpy as np
from torch.utils.data import Subset

from glioma.data.paper4_benchmarks import TokenBenchmarkDataset


INFORMATION_CONDITIONS = ("exact_copy", "repeat_measurement", "novel_signal", "null_signal")


class AvailabilityDataset(TokenBenchmarkDataset):
    """Token dataset carrying stable object and independence-group identifiers."""

    def __init__(self, tokens, labels, sample_ids, group_ids, modality_mask=None, token_mask=None):
        super().__init__(tokens, labels, modality_mask, token_mask)
        self.sample_ids = np.asarray(sample_ids).astype(str)
        self.group_ids = np.asarray(group_ids).astype(str)
        if len(self.sample_ids) != len(self) or len(self.group_ids) != len(self):
            raise ValueError("IDs must have one entry per object")
        if len(set(self.sample_ids)) != len(self):
            raise ValueError("sample_ids must be unique")

    def __getitem__(self, index):
        return super().__getitem__(index) | {
            "sample_id": str(self.sample_ids[index]),
            "group_id": str(self.group_ids[index]),
        }


def make_availability_splits(
    num_samples=6000,
    token_count=12,
    token_dim=8,
    extra_information="novel_signal",
    signal_form="mean",
    seed=42,
    data_seed=None,
    split_seed=None,
):
    """Generate condition-paired data without condition-dependent split streams."""
    condition = {"unique": "novel_signal", "redundant": "repeat_measurement"}.get(
        extra_information, extra_information
    )
    if condition not in INFORMATION_CONDITIONS:
        raise ValueError(f"extra_information must be one of {INFORMATION_CONDITIONS}")
    if signal_form not in {"mean", "covariance"}:
        raise ValueError("signal_form must be mean or covariance")
    if num_samples < 30 or token_dim < 2 or token_count < 2:
        raise ValueError("Require >=30 objects and >=2 token dimensions and tokens")
    data_seed = seed if data_seed is None else int(data_seed)
    split_seed = seed if split_seed is None else int(split_seed)

    # Splits and observation components have named, independent random streams.
    order = np.random.default_rng(split_seed).permutation(num_samples)
    latent_rng = np.random.default_rng(np.random.SeedSequence([data_seed, 0]))
    shared, independent, label_noise = latent_rng.normal(size=(3, num_samples))
    labels = (shared + independent + 0.15 * label_noise > 0).astype(np.int64)
    null = np.random.default_rng(np.random.SeedSequence([data_seed, 1])).normal(size=num_samples)
    if condition == "novel_signal":
        third = independent
    elif condition == "null_signal":
        third = null
    else:
        third = shared
    signals = np.stack((shared, shared, third), axis=1)
    tokens = np.random.default_rng(np.random.SeedSequence([data_seed, 2])).normal(
        size=(num_samples, 3, token_count, token_dim)
    ).astype(np.float32)
    if signal_form == "mean":
        tokens[..., 0] += (0.8 * signals[:, :, None]).astype(np.float32)
    else:
        factor = np.random.default_rng(np.random.SeedSequence([data_seed, 3])).normal(
            size=(num_samples, 3, token_count)
        ).astype(np.float32)
        scale = np.exp(0.45 * np.tanh(signals)).astype(np.float32)
        tokens[..., 0] += scale[:, :, None] * factor
        tokens[..., 1] += factor
    if condition == "exact_copy":
        tokens[:, 2] = tokens[:, 0]

    sample_ids = np.asarray([f"synthetic_{data_seed}_{index}" for index in range(num_samples)])
    bounds = (0, int(0.8 * num_samples), int(0.9 * num_samples), num_samples)
    splits = {}
    for name, start, stop in zip(("train", "val", "test"), bounds[:-1], bounds[1:]):
        indices = order[start:stop]
        splits[name] = AvailabilityDataset(
            tokens[indices, None], labels[indices], sample_ids[indices], sample_ids[indices]
        )
    return splits


def load_availability_npz(path, label_mode="class"):
    """Load v3 NPZ data and require provenance IDs for leakage checks."""
    splits = {}
    with np.load(path, allow_pickle=False) as archive:
        for name in ("train", "val", "test"):
            required = [f"{name}_{suffix}" for suffix in ("tokens", "labels", "sample_ids", "group_ids")]
            missing = [key for key in required if key not in archive]
            if missing:
                raise ValueError(f"NPZ missing {missing}; v3 requires object and group provenance")
            labels = archive[f"{name}_labels"].reshape(-1)
            if not np.isfinite(labels).all():
                raise ValueError("Nonfinite labels")
            keep = labels != 0 if label_mode == "sign_nozero" else np.ones(len(labels), dtype=bool)
            if label_mode in {"sign", "sign_nozero"}:
                labels = (labels >= 0).astype(np.int64)
            elif (labels < 0).any() or (labels != np.floor(labels)).any():
                raise ValueError("Class labels must be nonnegative integers")
            modality_mask = archive[f"{name}_modality_mask"][keep] if f"{name}_modality_mask" in archive else None
            token_mask = archive[f"{name}_token_mask"][keep] if f"{name}_token_mask" in archive else None
            splits[name] = AvailabilityDataset(
                archive[f"{name}_tokens"][keep], labels[keep],
                archive[f"{name}_sample_ids"][keep], archive[f"{name}_group_ids"][keep],
                modality_mask, token_mask,
            )
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        for attribute in ("sample_ids", "group_ids"):
            if set(getattr(splits[left], attribute)) & set(getattr(splits[right], attribute)):
                raise ValueError(f"{attribute} overlap between {left} and {right}")
    classes = set(splits["train"].labels.tolist())
    if classes != set(range(len(classes))) or len(classes) < 2:
        raise ValueError("Training labels must cover contiguous classes starting at zero")
    if any(not set(dataset.labels.tolist()).issubset(classes) for dataset in splits.values()):
        raise ValueError("Validation/test contain classes absent from training")
    return splits


def partition_training(dataset, split_seed, synthetic=False, audit_size=None):
    """Split training groups into fit/select/measure without object leakage."""
    groups = np.asarray(sorted(set(dataset.group_ids)))
    np.random.default_rng(np.random.SeedSequence([int(split_seed), 19])).shuffle(groups)
    if len(groups) < 10:
        raise ValueError("At least 10 independent training groups are required")
    measure_count = int(round(len(groups) * (0.125 if synthetic else 0.1)))
    select_count = measure_count
    if audit_size is not None:
        if any(np.sum(dataset.group_ids == group) != 1 for group in groups):
            raise ValueError("audit_size override is allowed only for singleton groups")
        measure_count = select_count = int(audit_size)
    if min(measure_count, select_count) < 1 or measure_count + select_count >= len(groups):
        raise ValueError("Invalid select/measure role sizes")
    role_groups = {
        "measure": groups[:measure_count],
        "select": groups[measure_count:measure_count + select_count],
        "fit": groups[measure_count + select_count:],
    }
    indices = {
        role: np.flatnonzero(np.isin(dataset.group_ids, selected)).tolist()
        for role, selected in role_groups.items()
    }
    return {role: Subset(dataset, values) for role, values in indices.items()}, indices


__all__ = [
    "AvailabilityDataset", "INFORMATION_CONDITIONS", "make_availability_splits",
    "load_availability_npz", "partition_training",
]
