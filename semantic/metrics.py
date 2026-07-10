"""Semantic retrieval and alignment metrics."""

from collections import defaultdict

import numpy as np


def binary_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pos = labels == 1
    neg = labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos_ranks = ranks[pos].sum()
    return float((pos_ranks - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def retrieval_metrics(query_vectors, target_ids, prototypes, gallery_ids=None, ks=(1, 5, 10), subject_ids=None):
    query_vectors = np.asarray(query_vectors, dtype=np.float32)
    prototypes = np.asarray(prototypes, dtype=np.float32)
    query_vectors = np.nan_to_num(query_vectors, nan=0.0, posinf=0.0, neginf=0.0)
    prototypes = np.nan_to_num(prototypes, nan=0.0, posinf=0.0, neginf=0.0)
    if gallery_ids is None:
        gallery_ids = list(range(len(prototypes)))
    gallery_ids = list(gallery_ids)
    if len(query_vectors) == 0 or len(gallery_ids) == 0:
        return {}

    query_norm = query_vectors / (np.linalg.norm(query_vectors, axis=1, keepdims=True) + 1e-8)
    proto_norm = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)
    scores = query_norm @ proto_norm[np.asarray(gallery_ids)].T

    recalls = {k: [] for k in ks}
    reciprocal_ranks = []
    average_precisions = []
    pos_scores = []
    neg_scores = []
    pos_dists = []
    neg_dists = []
    edge_labels = []
    edge_scores = []

    for row, positives in enumerate(target_ids):
        positives = set(positives).intersection(gallery_ids)
        if not positives:
            continue
        ranking = np.argsort(-scores[row])
        ranked_anchor_ids = [gallery_ids[idx] for idx in ranking]
        for k in ks:
            top_k = ranked_anchor_ids[: min(k, len(ranked_anchor_ids))]
            recalls[k].append(float(any(anchor_id in positives for anchor_id in top_k)))
        first_rank = next((rank + 1 for rank, anchor_id in enumerate(ranked_anchor_ids) if anchor_id in positives), None)
        if first_rank is not None:
            reciprocal_ranks.append(1.0 / first_rank)
        precision_hits = []
        hit_count = 0
        for rank_idx, anchor_id in enumerate(ranked_anchor_ids, start=1):
            if anchor_id in positives:
                hit_count += 1
                precision_hits.append(hit_count / rank_idx)
        if precision_hits:
            average_precisions.append(float(np.mean(precision_hits)))

        for col, anchor_id in enumerate(gallery_ids):
            score = float(scores[row, col])
            distance = float(np.linalg.norm(query_norm[row] - proto_norm[anchor_id]))
            is_positive = anchor_id in positives
            edge_scores.append(score)
            edge_labels.append(1 if is_positive else 0)
            if is_positive:
                pos_scores.append(score)
                pos_dists.append(distance)
            else:
                neg_scores.append(score)
                neg_dists.append(distance)

    metrics = {f"recall@{k}": float(np.mean(values)) if values else float("nan") for k, values in recalls.items()}
    map_query = float(np.mean(average_precisions)) if average_precisions else float("nan")
    metrics["map_query"] = map_query
    if subject_ids is not None and len(subject_ids) == len(target_ids):
        patient_ap = defaultdict(list)
        for row, positives in enumerate(target_ids):
            positives = set(positives).intersection(gallery_ids)
            if not positives:
                continue
            ranking = np.argsort(-scores[row])
            ranked_anchor_ids = [gallery_ids[idx] for idx in ranking]
            precision_hits = []
            hit_count = 0
            for rank_idx, anchor_id in enumerate(ranked_anchor_ids, start=1):
                if anchor_id in positives:
                    hit_count += 1
                    precision_hits.append(hit_count / rank_idx)
            if precision_hits:
                patient_ap[str(subject_ids[row])].append(float(np.mean(precision_hits)))
        patient_means = [float(np.mean(values)) for values in patient_ap.values() if values]
        metrics["map"] = float(np.mean(patient_means)) if patient_means else map_query
    else:
        metrics["map"] = map_query
    metrics["mrr"] = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else float("nan")
    metrics["pair_auc"] = binary_auc(edge_labels, edge_scores)
    metrics["average_positive_similarity"] = float(np.mean(pos_scores)) if pos_scores else float("nan")
    metrics["average_negative_similarity"] = float(np.mean(neg_scores)) if neg_scores else float("nan")
    metrics["positive_negative_distance_gap"] = (
        float(np.mean(neg_dists) - np.mean(pos_dists)) if pos_dists and neg_dists else float("nan")
    )
    metrics["anchor_consistency"] = float(np.mean(pos_scores) - np.mean(neg_scores)) if pos_scores and neg_scores else float("nan")

    if edge_scores:
        order = np.argsort(-np.asarray(edge_scores))
        for k in (10, 25, 50):
            top = order[: min(k, len(order))]
            metrics[f"edge_precision@{k}"] = float(np.mean(np.asarray(edge_labels)[top])) if len(top) else float("nan")
    return metrics


def bootstrap_ci(values, seed=42, n_bootstrap=1000):
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(float(np.mean(sample)))
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]

__all__ = ["binary_auc", "retrieval_metrics", "bootstrap_ci"]
