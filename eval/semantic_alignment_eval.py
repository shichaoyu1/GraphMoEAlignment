"""Semantic-alignment and TopoMoE evaluation."""

import json
import os

import numpy as np
import torch

from glioma.data.loaders import build_query_targets
from glioma.io.artifacts import (
    save_json,
    save_manifold_artifacts,
    save_patient_level_records,
    save_routing_records,
    save_topomoe_figure_manifest,
    save_topomoe_topology,
)
from glioma.semantic.metrics import retrieval_metrics
from glioma.semantic.topology import anchor_family_ids as infer_anchor_family_ids
from glioma.visualization.semantic_alignment import (
    node_names_for_mode,
    save_alignment_space_plot,
    save_semantic_unit_graph,
)
from glioma.visualization.geodesic_fusion import (
    build_geodesic_payloads,
    save_geodesic_figures,
)
from glioma.visualization.topomoe import save_topomoe_figures
from glioma.visualization.manifold_fusion import save_manifold_figures


def _metadata_available(value):
    return str(value or "").strip().lower() not in {"", "na", "n/a", "none", "nan", "unknown"}


def _resolved_eval_seed(args):
    model_seed = getattr(args, "model_seed", None)
    return int(getattr(args, "seed", 42) if model_seed is None else model_seed)


def _mix_family_routes(routing, family_log_probs, family_ids):
    routing = np.asarray(routing, dtype=np.float32)
    family_log_probs = np.asarray(family_log_probs, dtype=np.float32)
    mixed = np.full_like(family_log_probs, -1e9)
    for anchor_id, family_id in enumerate(family_ids):
        mixed[:, anchor_id] = (
            np.log(np.clip(routing[:, int(family_id)], 1e-8, None))
            + family_log_probs[:, anchor_id]
        )
    return mixed


def _dataset_routing_controls(query_records, family_log_probs, family_ids, seed):
    if not query_records or len(family_log_probs) != len(query_records):
        return {}
    routing = np.asarray([record.get("routing_weights", []) for record in query_records], dtype=np.float32)
    if routing.ndim != 2 or routing.shape[1] == 0:
        return {}
    index = {
        (str(record["subject_id"]), str(record["node_name"])): row
        for row, record in enumerate(query_records)
    }
    subjects = sorted({str(record["subject_id"]) for record in query_records})
    nodes = sorted({str(record["node_name"]) for record in query_records})
    rng = np.random.default_rng(int(seed))
    patient_sources = dict(zip(subjects, rng.permutation(subjects)))
    node_sources = dict(zip(nodes, rng.permutation(nodes)))
    patient_shuffled = routing.copy()
    node_shuffled = routing.copy()
    for row, record in enumerate(query_records):
        subject = str(record["subject_id"])
        node = str(record["node_name"])
        patient_row = index.get((patient_sources[subject], node))
        node_row = index.get((subject, node_sources[node]))
        if patient_row is not None:
            patient_shuffled[row] = routing[patient_row]
        if node_row is not None:
            node_shuffled[row] = routing[node_row]
    uniform = np.full_like(routing, 1.0 / routing.shape[1])
    return {
        "uniform_routing": _mix_family_routes(uniform, family_log_probs, family_ids),
        "patient_shuffled_routing": _mix_family_routes(
            patient_shuffled, family_log_probs, family_ids
        ),
        "node_shuffled_routing": _mix_family_routes(node_shuffled, family_log_probs, family_ids),
    }


def retrieval_metrics_from_scores(scores, target_ids, gallery_ids=None, ks=(1, 5, 10), subject_ids=None):
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return {}
    gallery_ids = list(range(scores.shape[1])) if gallery_ids is None else list(gallery_ids)
    if not gallery_ids:
        return {}
    gallery_scores = scores[:, np.asarray(gallery_ids)]
    recalls = {k: [] for k in ks}
    reciprocal_ranks = []
    average_precisions = []
    patient_ap = {}
    for row, positives in enumerate(target_ids):
        positives = set(positives).intersection(gallery_ids)
        if not positives:
            continue
        ranking = np.argsort(-gallery_scores[row])
        ranked_ids = [gallery_ids[idx] for idx in ranking]
        for k in ks:
            recalls[k].append(float(any(anchor_id in positives for anchor_id in ranked_ids[:k])))
        first_rank = next((rank + 1 for rank, anchor_id in enumerate(ranked_ids) if anchor_id in positives), None)
        if first_rank is not None:
            reciprocal_ranks.append(1.0 / first_rank)
        hits = 0
        precisions = []
        for rank, anchor_id in enumerate(ranked_ids, start=1):
            if anchor_id in positives:
                hits += 1
                precisions.append(hits / rank)
        if precisions:
            ap = float(np.mean(precisions))
            average_precisions.append(ap)
            if subject_ids is not None and len(subject_ids) == len(target_ids):
                patient_ap.setdefault(str(subject_ids[row]), []).append(ap)
    metrics = {f"recall@{k}": float(np.mean(values)) if values else float("nan") for k, values in recalls.items()}
    metrics["map_query"] = float(np.mean(average_precisions)) if average_precisions else float("nan")
    patient_means = [float(np.mean(values)) for values in patient_ap.values() if values]
    metrics["map"] = float(np.mean(patient_means)) if patient_means else metrics["map_query"]
    metrics["mrr"] = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else float("nan")
    return metrics


def score_queries_against_prototypes(query_vectors, prototypes):
    query_vectors = np.asarray(query_vectors, dtype=np.float32)
    prototypes = np.asarray(prototypes, dtype=np.float32)
    if len(query_vectors) == 0 or len(prototypes) == 0:
        return np.empty((0, len(prototypes)), dtype=np.float32)
    query_norm = query_vectors / (np.linalg.norm(query_vectors, axis=1, keepdims=True) + 1e-8)
    proto_norm = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)
    return query_norm @ proto_norm.T


def build_routing_spectrum(route_entries, family_names):
    if not route_entries:
        return {}

    def summarize(entries):
        weights = np.asarray([entry["weights"] for entry in entries], dtype=np.float32)
        return {
            "count": int(len(entries)),
            "mean": {family: float(weights[:, idx].mean()) for idx, family in enumerate(family_names)},
            "std": {family: float(weights[:, idx].std()) for idx, family in enumerate(family_names)},
        }

    groups = {"by_node": {}, "by_grade": {}, "by_node_grade": {}}
    for entry in route_entries:
        groups["by_node"].setdefault(entry["node_name"], []).append(entry)
        groups["by_grade"].setdefault(entry["grade"], []).append(entry)
        groups["by_node_grade"].setdefault(f"{entry['node_name']}|{entry['grade']}", []).append(entry)
    return {
        "family_names": family_names,
        "overall": summarize(route_entries),
        **{
            name: {key: summarize(entries) for key, entries in values.items()}
            for name, values in groups.items()
        },
    }


def _anchor_family_name(anchor_idx, family_ids, family_names):
    if 0 <= anchor_idx < len(family_ids):
        family_idx = int(family_ids[anchor_idx])
        if 0 <= family_idx < len(family_names):
            return family_names[family_idx]
    return "unknown"


def _top_k_entries(scores, anchor_vocab, positives, family_ids, family_names, k=5):
    scores = np.asarray(scores, dtype=np.float32)
    positives = set(int(anchor_id) for anchor_id in positives)
    if scores.size == 0:
        return []
    entries = []
    for rank, anchor_idx in enumerate(np.argsort(-scores)[: min(k, len(scores))], start=1):
        anchor = anchor_vocab[int(anchor_idx)]
        entries.append(
            {
                "rank": int(rank),
                "anchor_id": int(anchor_idx),
                "label": anchor["label"],
                "source": anchor.get("source", ""),
                "family": _anchor_family_name(int(anchor_idx), family_ids, family_names),
                "score": float(scores[int(anchor_idx)]),
                "is_target": bool(int(anchor_idx) in positives),
            }
        )
    return entries


def build_compact_routing_records(records, anchor_vocab, family_ids, family_names, top_k=5):
    compact = []
    for idx, record in enumerate(records["query_records"]):
        positives = records["query_targets"][idx]
        compact.append(
            {
                "subject_id": record["subject_id"],
                "node_name": record["node_name"],
                "grade": record.get("grade", "unknown"),
                "target_ids": [int(anchor_id) for anchor_id in positives],
                "target_labels": record.get("target_labels", []),
                "label_availability": record.get("label_availability", {}),
                "routing_weights": record.get("routing_weights_by_family", {}),
                "direct_top_k": _top_k_entries(
                    records["direct_scores"][idx], anchor_vocab, positives, family_ids, family_names, top_k
                ),
                "routed_top_k": _top_k_entries(
                    records["routed_scores"][idx], anchor_vocab, positives, family_ids, family_names, top_k
                ) if len(records["routed_scores"]) else [],
            }
        )
    return compact


def _topology_edges(matrix, initial, prior, anchor_vocab, family_ids, family_names):
    edges = []
    for source in range(matrix.shape[0]):
        for target in range(matrix.shape[1]):
            if source == target:
                continue
            effective = float(matrix[source, target])
            baseline = float(initial[source, target])
            prior_weight = float(prior[source, target])
            if effective <= 0:
                continue
            edges.append(
                {
                    "source_id": source,
                    "target_id": target,
                    "source_label": anchor_vocab[source]["label"],
                    "target_label": anchor_vocab[target]["label"],
                    "source_family": _anchor_family_name(source, family_ids, family_names),
                    "target_family": _anchor_family_name(target, family_ids, family_names),
                    "prior_weight": prior_weight,
                    "initial_weight": baseline,
                    "effective_weight": effective,
                    "delta_vs_prior": effective - prior_weight,
                    "delta_vs_initial": effective - baseline,
                    "new_edge": bool(prior_weight <= 0 and effective > baseline + 1e-6),
                }
            )
    edges.sort(key=lambda item: item["effective_weight"], reverse=True)
    return edges


def topomoe_topology_payload(model, anchor_vocab, family_ids, family_names):
    topo = getattr(model, "topo_moe", None)
    if topo is None:
        return {}
    prior = topo.A_prior.detach().cpu().numpy()
    initial = topo.initial_topology().detach().cpu().numpy()
    effective = topo.effective_topology().detach().cpu().numpy()
    delta = topo._symmetric_delta().detach().cpu().numpy()
    diagnostics = {
        key: float(value.detach().cpu())
        for key, value in topo.topology_diagnostics().items()
    }
    return {
        "version": topo.version,
        "topology_mode": topo.topo_mode,
        "family_names": list(family_names),
        "family_ids": [int(value) for value in family_ids],
        "anchor_families": [
            {
                "anchor_id": idx,
                "label": anchor["label"],
                "source": anchor.get("source", ""),
                "family": _anchor_family_name(idx, family_ids, family_names),
            }
            for idx, anchor in enumerate(anchor_vocab)
        ],
        "diagnostics": diagnostics,
        "A_prior": np.nan_to_num(prior).tolist(),
        "A_initial": np.nan_to_num(initial).tolist(),
        "A_effective": np.nan_to_num(effective).tolist(),
        "DeltaA_symmetric": np.nan_to_num(delta).tolist(),
        "top_edges": _topology_edges(effective, initial, prior, anchor_vocab, family_ids, family_names)[:32],
    }


def _deterministic_permutation(length, seed, device):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randperm(length, generator=generator).to(device)


def _wrong_family_routing(target_ids, family_ids, num_anchor_families, residual_index, shape, device):
    override = torch.zeros(shape, device=device)
    for row, positives in enumerate(target_ids):
        positive_families = {int(family_ids[idx]) for idx in positives if int(family_ids[idx]) < num_anchor_families}
        candidates = [family for family in range(num_anchor_families) if family not in positive_families]
        wrong = candidates[0] if candidates else residual_index
        if wrong is None:
            wrong = int((min(positive_families) + 1) % num_anchor_families) if positive_families else 0
        override.reshape(-1, shape[-1])[row, wrong] = 1.0
    return override


def _intervention_outputs(model, shared, prototypes, target_ids, seed, node_names):
    topo = model.topo_moe
    device = shared.device
    effective = topo.effective_topology().detach()
    family_ids = topo.family_ids.detach()
    permutation = _deterministic_permutation(topo.num_anchors, seed + 101, device)
    family_permutation = _deterministic_permutation(topo.num_anchors, seed + 211, device)
    uniform = torch.full_like(effective, 1.0 / topo.num_anchors)
    baseline = model.route_shared_nodes(shared, prototypes)
    baseline_routing = baseline["routing_weights"].detach()
    patient_permutation = _deterministic_permutation(shared.shape[0], seed + 307, device)
    node_permutation = _deterministic_permutation(shared.shape[1], seed + 401, device)
    scenarios = {
        "remove_pathology_expert": {"disabled_family_ids": [0]},
        "remove_molecular_expert": {"disabled_family_ids": [1]} if topo.num_anchor_families > 1 else {},
        "shuffle_topology": {"topology_override": effective[permutation][:, permutation]},
        "identity_topology": {"topology_override": torch.eye(topo.num_anchors, device=device)},
        "uniform_topology": {"topology_override": uniform},
        "force_wrong_family": {
            "routing_override": _wrong_family_routing(
                target_ids,
                family_ids.cpu().tolist(),
                topo.num_anchor_families,
                topo.residual_index,
                (shared.shape[0], shared.shape[1], topo.num_families),
                device,
            )
        },
        "anchor_family_permutation": {"family_ids_override": family_ids[family_permutation]},
        "uniform_routing": {"routing_override": torch.ones_like(baseline_routing)},
        "patient_shuffled_routing": {
            "routing_override": baseline_routing[patient_permutation]
        },
        "node_shuffled_routing": {
            "routing_override": baseline_routing[:, node_permutation]
        },
    }
    if topo.use_residual_expert:
        scenarios["remove_residual_branch"] = {"disabled_family_ids": [topo.residual_index]}
    for node_idx, node_name in enumerate(node_names):
        masked = shared.clone()
        masked[:, node_idx] = 0
        scenarios[f"mask_node_{node_name.lower().replace('/', '_').replace(' ', '_')}"] = {"shared_override": masked}

    outputs = {}
    for name, options in scenarios.items():
        shared_input = options.pop("shared_override", shared)
        routed = model.route_shared_nodes(shared_input, prototypes, **options)["routed_scores"]
        if routed is not None:
            outputs[name] = routed.detach().cpu().numpy()
    return outputs


def collect_alignment_records(
    model,
    bank,
    loader,
    device,
    args,
    case_lookup,
    key_to_id,
    anchor_vocab,
    max_cases=None,
    collect_interventions=False,
    graph_intervention=None,
):
    model.eval()
    bank.eval()
    node_names = node_names_for_mode(args.node_mode)
    topo = getattr(model, "topo_moe", None)
    include_residual = bool(topo is not None and topo.use_residual_expert)
    family_ids, family_names = infer_anchor_family_ids(
        anchor_vocab,
        include_residual=include_residual,
    )
    query_vectors, query_targets, query_records = [], [], []
    routed_scores, route_entries, adjacency_mats = [], [], []
    family_conditionals = []
    intervention_scores = {}
    fusion_adjacencies, fusion_energies, fusion_linear_energies = [], [], []
    fusion_ratios, fusion_deviations = [], []
    representative_paths = []
    fusion_pair_indices = []
    manifold_local_adjacencies, manifold_upper_adjacencies = [], []
    manifold_local_distances, manifold_upper_distances = [], []
    manifold_raw_scales, manifold_raw_traces = [], []
    manifold_eigenvalues, manifold_conditions = [], []
    manifold_local_updates, manifold_upper_updates = [], []
    manifold_identity_l2, manifold_identity_cosine = [], []
    manifold_upper_node_names = []
    manifold_subject_ids = []
    seen_cases = 0
    dropped_nonfinite_queries = 0

    with torch.no_grad():
        for batch in loader:
            if max_cases is not None and seen_cases >= max_cases:
                break
            images = batch["images"].to(device)
            region_masks = batch.get("region_masks")
            if region_masks is not None:
                region_masks = region_masks.to(device)
            subject_ids = list(batch["subject_id"])
            remaining = len(subject_ids) if max_cases is None else min(len(subject_ids), max_cases - seen_cases)
            prototypes_tensor = bank()
            model_kwargs = {
                "region_masks": region_masks,
                "return_extras": True,
                "anchor_prototypes": prototypes_tensor
                if getattr(model, "requires_anchor_prototypes", getattr(model, "use_topo_moe", False))
                else None,
            }
            if graph_intervention is not None:
                model_kwargs["graph_intervention"] = graph_intervention
            output = model(images, **model_kwargs)
            shared_tensor = output["extras"]["shared"]
            shared_np = np.nan_to_num(shared_tensor.detach().cpu().numpy())
            routing = output["extras"].get("routing_weights")
            routing_np = np.nan_to_num(routing.detach().cpu().numpy()) if routing is not None else None
            routed = output["extras"].get("routed_scores")
            if routed is None:
                routed = output["extras"].get("routed_logits")
            routed_np = np.nan_to_num(routed.detach().cpu().numpy()) if routed is not None else None
            family_conditional = output["extras"].get("family_log_probs")
            family_conditional_np = (
                np.nan_to_num(family_conditional.detach().cpu().numpy())
                if family_conditional is not None
                else None
            )
            adjacency = output["extras"].get("adjacency")
            if adjacency is not None:
                adjacency_mats.append(np.nan_to_num(adjacency.detach().cpu().numpy()[:remaining]))

            fusion_adjacency = output["extras"].get("fusion_adjacency")
            if fusion_adjacency is not None:
                fusion_adjacencies.append(
                    np.nan_to_num(fusion_adjacency.detach().cpu().numpy()[:remaining])
                )
                fusion_energies.append(
                    np.nan_to_num(
                        output["extras"]["fusion_geodesic_energy"].detach().cpu().numpy()[:remaining]
                    )
                )
                fusion_linear_energies.append(
                    np.nan_to_num(
                        output["extras"]["fusion_linear_energy"].detach().cpu().numpy()[:remaining]
                    )
                )
                fusion_ratios.append(
                    np.nan_to_num(
                        output["extras"]["fusion_energy_ratio"].detach().cpu().numpy()[:remaining]
                    )
                )
                fusion_deviations.append(
                    np.nan_to_num(
                        output["extras"]["fusion_path_deviation"].detach().cpu().numpy()[:remaining]
                    )
                )
                fusion_pair_indices = output["extras"]["fusion_pair_indices"].detach().cpu().tolist()
                paths = output["extras"].get("fusion_paths")
                if paths is not None and len(representative_paths) < 8:
                    paths_np = np.nan_to_num(paths.detach().cpu().numpy()[:remaining])
                    for sample_idx, subject_id in enumerate(subject_ids[:remaining]):
                        if len(representative_paths) >= 8:
                            break
                        representative_paths.append(
                            {"subject_id": str(subject_id), "paths": paths_np[sample_idx].tolist()}
                        )

            manifold_local = output["extras"].get("manifold_local_adjacency")
            if manifold_local is not None:
                manifold_subject_ids.extend(str(value) for value in subject_ids[:remaining])
                def append_manifold(name, target):
                    value = output["extras"].get(name)
                    if value is not None:
                        target.append(np.nan_to_num(value.detach().cpu().numpy()[:remaining]))

                append_manifold("manifold_local_adjacency", manifold_local_adjacencies)
                append_manifold("manifold_upper_adjacency", manifold_upper_adjacencies)
                append_manifold("manifold_local_distances", manifold_local_distances)
                append_manifold("manifold_upper_distances", manifold_upper_distances)
                append_manifold("manifold_raw_scales", manifold_raw_scales)
                append_manifold("manifold_raw_spd_traces", manifold_raw_traces)
                append_manifold("manifold_spd_eigenvalues", manifold_eigenvalues)
                append_manifold("manifold_condition_numbers", manifold_conditions)
                append_manifold("manifold_local_update_ratio", manifold_local_updates)
                append_manifold("manifold_upper_update_ratio", manifold_upper_updates)
                append_manifold("manifold_identity_l2_shift", manifold_identity_l2)
                append_manifold("manifold_identity_cosine_shift", manifold_identity_cosine)
                manifold_upper_node_names = list(
                    output["extras"].get("manifold_upper_node_names", manifold_upper_node_names)
                )

            batch_targets = build_query_targets(subject_ids, node_names, case_lookup, key_to_id, args)
            batch_interventions = {}
            if collect_interventions and getattr(model, "topomoe_version", "v1") == "v2":
                batch_interventions = _intervention_outputs(
                    model,
                    shared_tensor,
                    prototypes_tensor,
                    batch_targets,
                    _resolved_eval_seed(args),
                    node_names,
                )

            for sample_idx, subject_id in enumerate(subject_ids[:remaining]):
                metadata = case_lookup[str(subject_id)]["metadata"]
                grade = str(metadata.get("Tumor Grade", "unknown"))
                for node_idx, node_name in enumerate(node_names):
                    flat_idx = sample_idx * len(node_names) + node_idx
                    ids = batch_targets[flat_idx]
                    if not ids:
                        continue
                    vector = shared_np[sample_idx, node_idx]
                    if not np.all(np.isfinite(vector)):
                        dropped_nonfinite_queries += 1
                        continue
                    weights = routing_np[sample_idx, node_idx].astype(float).tolist() if routing_np is not None else []
                    weights_by_family = {
                        family_names[idx]: float(value)
                        for idx, value in enumerate(weights)
                        if idx < len(family_names)
                    }
                    if weights:
                        route_entries.append(
                            {
                                "subject_id": str(subject_id),
                                "node_name": node_name,
                                "grade": grade,
                                "weights": weights,
                                "weights_by_family": weights_by_family,
                                "sum": float(np.sum(weights)),
                            }
                        )
                    if routed_np is not None:
                        routed_scores.append(routed_np[sample_idx, node_idx])
                    if family_conditional_np is not None:
                        family_conditionals.append(family_conditional_np[sample_idx, node_idx])
                    for scenario, scores in batch_interventions.items():
                        intervention_scores.setdefault(scenario, []).append(scores[sample_idx, node_idx])
                    query_vectors.append(vector)
                    query_targets.append(ids)
                    query_records.append(
                        {
                            "subject_id": str(subject_id),
                            "node_name": node_name,
                            "grade": grade,
                            "source": "MRI",
                            "target_labels": [anchor_vocab[idx]["label"] for idx in ids],
                            "routing_weights": weights,
                            "routing_weights_by_family": weights_by_family,
                            "label_availability": {
                                "IDH": _metadata_available(metadata.get("IDH", "")),
                                "MGMT": _metadata_available(metadata.get("MGMT", "")),
                                "1p19q": _metadata_available(
                                    metadata.get(
                                        "1p19Q CODEL",
                                        metadata.get("1p19q", metadata.get("1p19Q", "")),
                                    )
                                ),
                            },
                        }
                    )
            seen_cases += remaining

    prototypes = np.nan_to_num(bank().detach().cpu().numpy())
    query_vectors = np.asarray(query_vectors, dtype=np.float32)
    direct_scores = score_queries_against_prototypes(query_vectors, prototypes)
    routed_scores = np.asarray(routed_scores, dtype=np.float32) if routed_scores else np.empty((0, len(anchor_vocab)), dtype=np.float32)
    family_conditionals = (
        np.asarray(family_conditionals, dtype=np.float32)
        if family_conditionals
        else np.empty((0, len(anchor_vocab)), dtype=np.float32)
    )
    intervention_scores.update(
        _dataset_routing_controls(
            query_records,
            family_conditionals,
            family_ids,
            seed=_resolved_eval_seed(args) + 503,
        )
    )
    adjacency = np.concatenate(adjacency_mats).mean(axis=0) if adjacency_mats else None
    records = {
        "query_vectors": query_vectors,
        "query_targets": query_targets,
        "query_subject_ids": [record["subject_id"] for record in query_records],
        "query_records": query_records,
        "direct_scores": direct_scores,
        "routed_scores": routed_scores,
        "route_entries": route_entries,
        "family_names": family_names,
        "family_ids": family_ids,
        "routing_spectrum": build_routing_spectrum(route_entries, family_names),
        "prototypes": prototypes,
        "adjacency": adjacency,
        "case_count": seen_cases,
        "dropped_nonfinite_queries": dropped_nonfinite_queries,
        "topomoe_topology": topomoe_topology_payload(model, anchor_vocab, family_ids, family_names),
        "intervention_scores": {
            name: np.asarray(scores, dtype=np.float32) for name, scores in intervention_scores.items()
        },
        "geodesic_fusion": {
            "pair_indices": fusion_pair_indices,
            "adjacency": np.concatenate(fusion_adjacencies, axis=0)
            if fusion_adjacencies
            else np.empty((0, 3, 4, 4), dtype=np.float32),
            "geodesic_energy": np.concatenate(fusion_energies, axis=0)
            if fusion_energies
            else np.empty((0, 3, 6), dtype=np.float32),
            "linear_energy": np.concatenate(fusion_linear_energies, axis=0)
            if fusion_linear_energies
            else np.empty((0, 3, 6), dtype=np.float32),
            "energy_ratio": np.concatenate(fusion_ratios, axis=0)
            if fusion_ratios
            else np.empty((0, 3, 6), dtype=np.float32),
            "path_deviation": np.concatenate(fusion_deviations, axis=0)
            if fusion_deviations
            else np.empty((0, 3, 6), dtype=np.float32),
            "representative_paths": representative_paths,
        },
        "manifold_fusion": {
            "local_adjacency": np.concatenate(manifold_local_adjacencies, axis=0)
            if manifold_local_adjacencies
            else np.empty((0, 3, 4, 4), dtype=np.float32),
            "upper_adjacency": np.concatenate(manifold_upper_adjacencies, axis=0)
            if manifold_upper_adjacencies
            else np.empty((0, 0, 0), dtype=np.float32),
            "local_distances": np.concatenate(manifold_local_distances, axis=0)
            if manifold_local_distances
            else np.empty((0, 3, 4, 4), dtype=np.float32),
            "upper_distances": np.concatenate(manifold_upper_distances, axis=0)
            if manifold_upper_distances
            else np.empty((0, 0, 0), dtype=np.float32),
            "raw_scales": np.concatenate(manifold_raw_scales, axis=0)
            if manifold_raw_scales
            else np.empty((0, 3, 4), dtype=np.float32),
            "raw_spd_traces": np.concatenate(manifold_raw_traces, axis=0)
            if manifold_raw_traces
            else np.empty((0, 3, 4), dtype=np.float32),
            "spd_eigenvalues": np.concatenate(manifold_eigenvalues, axis=0)
            if manifold_eigenvalues
            else np.empty((0, 3, 4, 0), dtype=np.float32),
            "condition_numbers": np.concatenate(manifold_conditions, axis=0)
            if manifold_conditions
            else np.empty((0, 3, 4), dtype=np.float32),
            "local_update_ratio": np.concatenate(manifold_local_updates, axis=0)
            if manifold_local_updates
            else np.empty((0, 3, 4), dtype=np.float32),
            "upper_update_ratio": np.concatenate(manifold_upper_updates, axis=0)
            if manifold_upper_updates
            else np.empty((0, 0), dtype=np.float32),
            "identity_l2_shift": np.concatenate(manifold_identity_l2, axis=0)
            if manifold_identity_l2
            else np.empty((0, 3), dtype=np.float32),
            "identity_cosine_shift": np.concatenate(manifold_identity_cosine, axis=0)
            if manifold_identity_cosine
            else np.empty((0, 3), dtype=np.float32),
            "upper_node_names": manifold_upper_node_names,
            "subject_ids": manifold_subject_ids,
            "family_prior": (
                model.fusion.family_prior.detach().cpu().numpy()
                if getattr(model, "use_manifold_fusion", False)
                else np.empty((0, 0), dtype=np.float32)
            ),
        },
    }
    records["routing_records"] = build_compact_routing_records(records, anchor_vocab, family_ids, family_names)
    return records


def _subset_score_metrics(scores, records, indices):
    if not indices:
        return {}
    return retrieval_metrics_from_scores(
        scores[indices],
        [records["query_targets"][idx] for idx in indices],
        subject_ids=[records["query_subject_ids"][idx] for idx in indices],
    )


def _metric_delta_payload(baseline, intervention):
    return {
        "baseline": baseline,
        "intervention": intervention,
        "delta_map": float(intervention.get("map", float("nan")) - baseline.get("map", float("nan"))),
        "delta_mrr": float(intervention.get("mrr", float("nan")) - baseline.get("mrr", float("nan"))),
    }


def intervention_metrics(records):
    baseline = retrieval_metrics_from_scores(
        records["routed_scores"],
        records["query_targets"],
        subject_ids=records["query_subject_ids"],
    )
    family_ids = records["family_ids"]
    payload = {"baseline": baseline, "scenarios": {}}
    for name, scores in records.get("intervention_scores", {}).items():
        overall = retrieval_metrics_from_scores(
            scores,
            records["query_targets"],
            subject_ids=records["query_subject_ids"],
        )
        by_family = {}
        has_residual = bool(records["family_names"] and records["family_names"][-1] == "residual")
        family_names = records["family_names"][:-1] if has_residual else records["family_names"]
        for family_idx, family_name in enumerate(family_names):
            indices = [
                idx
                for idx, positives in enumerate(records["query_targets"])
                if any(int(family_ids[anchor_id]) == family_idx for anchor_id in positives)
            ]
            family_baseline = _subset_score_metrics(records["routed_scores"], records, indices)
            family_intervention = _subset_score_metrics(scores, records, indices)
            by_family[family_name] = _metric_delta_payload(family_baseline, family_intervention)
        by_node = {}
        for node_name in sorted({record["node_name"] for record in records["query_records"]}):
            indices = [idx for idx, record in enumerate(records["query_records"]) if record["node_name"] == node_name]
            node_baseline = _subset_score_metrics(records["routed_scores"], records, indices)
            node_intervention = _subset_score_metrics(scores, records, indices)
            by_node[node_name] = _metric_delta_payload(node_baseline, node_intervention)
        payload["scenarios"][name] = {
            "overall": overall,
            "delta_map": float(overall.get("map", float("nan")) - baseline.get("map", float("nan"))),
            "delta_mrr": float(overall.get("mrr", float("nan")) - baseline.get("mrr", float("nan"))),
            "by_target_family": by_family,
            "by_node": by_node,
        }
    return payload


def metrics_from_records(records, anchor_vocab, checkpoint_type=None):
    direct = retrieval_metrics(
        records["query_vectors"],
        records["query_targets"],
        records["prototypes"],
        subject_ids=records["query_subject_ids"],
    )
    routed = retrieval_metrics_from_scores(
        records["routed_scores"],
        records["query_targets"],
        subject_ids=records["query_subject_ids"],
    )
    payload = {
        "checkpoint_type": checkpoint_type,
        "case_count": records["case_count"],
        "query_count": len(records["query_vectors"]),
        "anchor_count": len(anchor_vocab),
        "dropped_nonfinite_queries": records["dropped_nonfinite_queries"],
        "direct": direct,
        "routed": routed,
    }
    direct_scores = records["direct_scores"]
    by_grade = {}
    by_node = {}
    for grade in sorted({record.get("grade", "unknown") for record in records["query_records"]}):
        indices = [
            index for index, record in enumerate(records["query_records"])
            if record.get("grade", "unknown") == grade
        ]
        by_grade[grade] = _subset_score_metrics(direct_scores, records, indices)
    for node in sorted({record["node_name"] for record in records["query_records"]}):
        indices = [
            index for index, record in enumerate(records["query_records"])
            if record["node_name"] == node
        ]
        by_node[node] = _subset_score_metrics(direct_scores, records, indices)
    by_family = {}
    metric_family_names = list(records.get("family_names", []))
    if metric_family_names and metric_family_names[-1] == "residual":
        metric_family_names = metric_family_names[:-1]
    for family_index, family_name in enumerate(metric_family_names):
        gallery = [
            index for index, value in enumerate(records.get("family_ids", []))
            if int(value) == family_index
        ]
        indices = [
            index for index, positives in enumerate(records["query_targets"])
            if any(anchor_id in gallery for anchor_id in positives)
        ]
        if indices and gallery:
            by_family[family_name] = retrieval_metrics_from_scores(
                direct_scores[indices],
                [records["query_targets"][index] for index in indices],
                gallery_ids=gallery,
                subject_ids=[records["query_subject_ids"][index] for index in indices],
            )

    def macro(groups):
        result = {}
        for metric in ("map", "mrr", "recall@1"):
            values = [entry.get(metric, float("nan")) for entry in groups.values()]
            finite = [float(value) for value in values if np.isfinite(value)]
            result[metric] = float(np.mean(finite)) if finite else float("nan")
        return result

    payload["direct_subgroups"] = {
        "by_grade": by_grade,
        "by_node": by_node,
        "by_target_family": by_family,
        "macro": {
            "grade": macro(by_grade),
            "node": macro(by_node),
            "target_family": macro(by_family),
        },
    }
    galleries = {
        "pathology_unavailable": [idx for idx, anchor in enumerate(anchor_vocab) if anchor["source"] != "Pathology"],
        "molecular_unavailable": [idx for idx, anchor in enumerate(anchor_vocab) if anchor["source"] != "Gene"],
    }
    payload["gallery_availability_stress"] = {
        name: retrieval_metrics(
            records["query_vectors"],
            records["query_targets"],
            records["prototypes"],
            gallery_ids=gallery,
            subject_ids=records["query_subject_ids"],
        )
        for name, gallery in galleries.items()
        if gallery
    }
    return payload


def _patient_average_precision(scores, records):
    grouped = {}
    for row, positives in enumerate(records["query_targets"]):
        positives = set(int(value) for value in positives)
        if not positives:
            continue
        order = np.argsort(-np.asarray(scores[row]))
        hits = 0
        precisions = []
        for rank, anchor_id in enumerate(order, start=1):
            if int(anchor_id) in positives:
                hits += 1
                precisions.append(hits / rank)
        if precisions:
            subject = str(records["query_subject_ids"][row])
            grouped.setdefault(subject, []).append(float(np.mean(precisions)))
    return {subject: float(np.mean(values)) for subject, values in grouped.items()}


def _paired_ap_summary(baseline_scores, intervention_scores, records, seed=31):
    baseline = _patient_average_precision(baseline_scores, records)
    intervention = _patient_average_precision(intervention_scores, records)
    subjects = sorted(set(baseline).intersection(intervention))
    deltas = np.asarray([intervention[subject] - baseline[subject] for subject in subjects], dtype=float)
    if not len(deltas):
        return {"n": 0, "mean_delta": float("nan"), "ci95": [float("nan"), float("nan")]}
    rng = np.random.default_rng(seed)
    bootstrap = rng.choice(deltas, size=(5000, len(deltas)), replace=True).mean(axis=1)
    return {
        "n": int(len(deltas)),
        "mean_delta": float(deltas.mean()),
        "std_delta": float(deltas.std(ddof=1 if len(deltas) > 1 else 0)),
        "ci95": [float(np.percentile(bootstrap, 2.5)), float(np.percentile(bootstrap, 97.5))],
        "positive_fraction": float((deltas > 0).mean()),
    }


def paper4_graph_role_metrics(records, intervention_records):
    manifold = records.get("manifold_fusion", {})
    local = np.asarray(manifold.get("local_adjacency", []), dtype=float)
    upper = np.asarray(manifold.get("upper_adjacency", []), dtype=float)
    local_offdiag = local.sum(axis=-1) - np.diagonal(local, axis1=-2, axis2=-1) if local.size else np.asarray([])
    upper_offdiag = upper.sum(axis=-1) - np.diagonal(upper, axis1=-2, axis2=-1) if upper.size else np.asarray([])
    region_family = upper[:, :3, 3:].sum(axis=-1) if upper.ndim == 3 and upper.shape[-1] > 3 else np.asarray([])

    def summary(values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if not values.size:
            return {"mean": float("nan"), "median": float("nan"), "n": 0}
        return {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std(ddof=1 if values.size > 1 else 0)),
            "n": int(values.size),
        }

    baseline_metrics = retrieval_metrics_from_scores(
        records["direct_scores"], records["query_targets"], subject_ids=records["query_subject_ids"]
    )
    payload = {
        "baseline": baseline_metrics,
        "structural": {
            "local_offdiagonal_mass": summary(local_offdiag),
            "upper_offdiagonal_mass": summary(upper_offdiag),
            "region_family_mass": summary(region_family),
            "local_update_ratio": summary(manifold.get("local_update_ratio", [])),
            "upper_update_ratio": summary(manifold.get("upper_update_ratio", [])),
            "identity_l2_shift": summary(manifold.get("identity_l2_shift", [])),
            "identity_cosine_shift": summary(manifold.get("identity_cosine_shift", [])),
            "identity_changed_fraction": float(
                (np.asarray(manifold.get("identity_l2_shift", []), dtype=float) > 1e-8).mean()
            ) if np.asarray(manifold.get("identity_l2_shift", [])).size else float("nan"),
        },
        "interventions": {},
    }
    for name, intervention in intervention_records.items():
        metrics = retrieval_metrics_from_scores(
            intervention["direct_scores"],
            intervention["query_targets"],
            subject_ids=intervention["query_subject_ids"],
        )
        vector_delta = np.linalg.norm(
            np.asarray(intervention["query_vectors"]) - np.asarray(records["query_vectors"]), axis=-1
        )
        payload["interventions"][name] = {
            "metrics": metrics,
            "delta_map": float(metrics.get("map", np.nan) - baseline_metrics.get("map", np.nan)),
            "delta_mrr": float(metrics.get("mrr", np.nan) - baseline_metrics.get("mrr", np.nan)),
            "delta_recall@1": float(
                metrics.get("recall@1", np.nan) - baseline_metrics.get("recall@1", np.nan)
            ),
            "embedding_l2_shift": summary(vector_delta),
            "changed_fraction": float((vector_delta > 1e-8).mean()) if vector_delta.size else float("nan"),
            "patient_ap": _paired_ap_summary(
                records["direct_scores"], intervention["direct_scores"], records
            ),
        }
    return payload


def _history_topology_diagnostics(history_path):
    if not history_path or not os.path.exists(history_path):
        return {}
    with open(history_path, "r", encoding="utf-8") as file:
        history = json.load(file)
    gradient_norms = [
        float(entry.get("train", {}).get("topology", {}).get("gradient_norm", float("nan")))
        for entry in history
    ]
    finite = [value for value in gradient_norms if np.isfinite(value)]
    if not finite:
        return {}
    return {
        "gradient_norm": finite[-1],
        "gradient_norm_max": max(finite),
    }


def _uses_topomoe_artifact_protocol(model, args):
    is_topomoe = bool(getattr(model, "use_topo_moe", False))
    return is_topomoe and (
        getattr(model, "topomoe_version", "v1") == "v2"
        or getattr(args, "paper_config", "none") == "paper2"
    )


def _uses_geodesic_artifact_protocol(model, args):
    return bool(getattr(model, "use_geodesic_fusion", False)) and getattr(
        args, "paper_config", "none"
    ) == "paper4"


def _uses_manifold_artifact_protocol(model, args):
    return bool(getattr(model, "use_manifold_fusion", False)) and getattr(
        args, "paper_config", "none"
    ) == "paper4"


def _uses_published_baseline_protocol(model, args):
    return bool(getattr(model, "use_published_baseline", False)) and getattr(
        args, "paper_config", "none"
    ) == "paper4"


def evaluate_and_save(
    model,
    bank,
    loader,
    device,
    args,
    case_lookup,
    key_to_id,
    anchor_vocab,
    out_dir,
    checkpoint_type=None,
    metrics_filename="test_metrics.json",
    run_interventions=True,
    figure_context=None,
):
    is_v2 = getattr(model, "topomoe_version", "v1") == "v2"
    use_topomoe_artifacts = _uses_topomoe_artifact_protocol(model, args)
    use_geodesic_artifacts = _uses_geodesic_artifact_protocol(model, args)
    use_manifold_artifacts = _uses_manifold_artifact_protocol(model, args)
    use_published_baseline = _uses_published_baseline_protocol(model, args)
    records = collect_alignment_records(
        model,
        bank,
        loader,
        device,
        args,
        case_lookup,
        key_to_id,
        anchor_vocab,
        max_cases=args.align_max_cases,
        collect_interventions=bool(run_interventions and is_v2),
    )
    metrics = metrics_from_records(records, anchor_vocab, checkpoint_type=checkpoint_type)
    graph_role = {}
    if use_manifold_artifacts and run_interventions:
        intervention_records = {
            name: collect_alignment_records(
                model,
                bank,
                loader,
                device,
                args,
                case_lookup,
                key_to_id,
                anchor_vocab,
                max_cases=args.align_max_cases,
                graph_intervention=name,
            )
            for name in ("identity", "uniform", "shuffle", "no_local", "no_upper", "no_region_family")
        }
        graph_role = paper4_graph_role_metrics(records, intervention_records)
    if not use_topomoe_artifacts:
        legacy_metrics = dict(metrics.get("direct", {}))
        legacy_metrics.update(
            {
                "checkpoint_type": checkpoint_type,
                "case_count": metrics["case_count"],
                "query_count": metrics["query_count"],
                "anchor_count": metrics["anchor_count"],
                "dropped_nonfinite_queries": metrics["dropped_nonfinite_queries"],
            }
        )
        if use_manifold_artifacts or use_published_baseline:
            legacy_metrics["subgroups"] = metrics.get("direct_subgroups", {})
        metrics = legacy_metrics
    save_json(os.path.join(out_dir, metrics_filename), metrics)
    save_patient_level_records(records, out_dir)

    if use_topomoe_artifacts:
        context = figure_context or {}
        records.get("topomoe_topology", {}).setdefault("diagnostics", {}).update(
            _history_topology_diagnostics(context.get("history_path"))
        )
        save_json(os.path.join(out_dir, "routing_spectrum.json"), records["routing_spectrum"])
        save_routing_records(records, out_dir)
        save_topomoe_topology(records, out_dir)
        diagnostics = records.get("topomoe_topology", {}).get("diagnostics", {})
        save_json(os.path.join(out_dir, "topomoe_diagnostics.json"), diagnostics)
        interventions = intervention_metrics(records) if run_interventions and is_v2 else {}
        if interventions:
            save_json(os.path.join(out_dir, "intervention_metrics.json"), interventions)
        manifest = save_topomoe_figures(
            records,
            metrics,
            anchor_vocab,
            out_dir,
            interventions=interventions,
            context=context,
        )
        save_topomoe_figure_manifest(manifest, out_dir)
    elif use_manifold_artifacts:
        context = figure_context or {}
        save_manifold_artifacts(records, anchor_vocab, out_dir)
        if graph_role:
            save_json(os.path.join(out_dir, "graph_role_metrics.json"), graph_role)
        figures = save_manifold_figures(
            records, metrics, anchor_vocab, out_dir, graph_role=graph_role
        )
        total_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        fusion_parameters = sum(
            parameter.numel() for parameter in model.fusion.parameters() if parameter.requires_grad
        )
        manifest = {
            "status": "single_seed_prototype",
            "limitation": "Single-seed artifacts are diagnostic; paper claims use aggregate evidence checks.",
            "seed": context.get("seed"),
            "checkpoint_type": checkpoint_type,
            "fusion_backend": context.get("fusion_backend"),
            "spd_geometry": context.get("spd_geometry"),
            "upper_graph": context.get("spd_upper_graph"),
            "anchor_families": context.get("spd_anchor_families"),
            "graph_policy": context.get("spd_graph_policy"),
            "graph_intervention": context.get("graph_intervention"),
            "parameter_count": {
                "total_trainable": int(total_parameters),
                "fusion_trainable": int(fusion_parameters),
            },
            "case_count": records.get("case_count"),
            "source_artifacts": [value for value in [
                "manifold_feature_stats.json",
                "manifold_case_records.json",
                "manifold_graph_records.npz",
                "manifold_topology.json",
                "graph_role_metrics.json" if graph_role else None,
                metrics_filename,
            ] if value is not None],
            "figures": figures,
        }
        save_json(os.path.join(out_dir, "paper4_manifold_figure_manifest.json"), manifest)
    elif use_geodesic_artifacts:
        context = figure_context or {}
        diagnostics, graph = build_geodesic_payloads(records, context)
        save_json(os.path.join(out_dir, "geodesic_diagnostics.json"), diagnostics)
        save_json(os.path.join(out_dir, "fusion_graph.json"), graph)
        figures = save_geodesic_figures(records, graph, out_dir, context)
        manifest = {
            "status": "single_seed",
            "seed": context.get("seed"),
            "checkpoint_type": checkpoint_type,
            "fusion_mode": context.get("fusion_mode"),
            "metric_support": context.get("metric_support"),
            "fusion_graph": context.get("fusion_graph"),
            "case_count": records.get("case_count"),
            "figures": figures,
        }
        save_json(os.path.join(out_dir, "fusion_figure_manifest.json"), manifest)
    elif use_published_baseline:
        total_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        fusion_parameters = sum(
            parameter.numel() for parameter in model.fusion.parameters() if parameter.requires_grad
        )
        save_json(
            os.path.join(out_dir, "paper4_baseline_manifest.json"),
            {
                "status": "single_seed",
                "seed": (figure_context or {}).get("seed"),
                "checkpoint_type": checkpoint_type,
                "baseline": getattr(model, "paper4_baseline", None),
                "case_count": records.get("case_count"),
                "parameter_count": {
                    "total_trainable": int(total_parameters),
                    "fusion_trainable": int(fusion_parameters),
                },
                "source_artifacts": [metrics_filename, "patient_level_records.json"],
            },
        )
    else:
        save_json(os.path.join(out_dir, "semantic_alignment_metrics.json"), metrics)
        save_alignment_space_plot(records, anchor_vocab, out_dir)
        save_semantic_unit_graph(records, anchor_vocab, args, out_dir)
    return metrics


__all__ = [
    "collect_alignment_records",
    "evaluate_and_save",
    "intervention_metrics",
    "metrics_from_records",
    "paper4_graph_role_metrics",
    "retrieval_metrics_from_scores",
    "_uses_geodesic_artifact_protocol",
    "_uses_manifold_artifact_protocol",
    "_uses_published_baseline_protocol",
]
