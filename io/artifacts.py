"""Artifact serialization helpers."""

import json
import os

import numpy as np


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def save_patient_level_records(records, out_dir):
    payload = {
        "subject_ids": [record["subject_id"] for record in records["query_records"]],
        "node_names": [record["node_name"] for record in records["query_records"]],
        "query_targets": records["query_targets"],
        "query_vectors": np.nan_to_num(records["query_vectors"], nan=0.0, posinf=0.0, neginf=0.0).tolist(),
        "prototypes": np.nan_to_num(records["prototypes"], nan=0.0, posinf=0.0, neginf=0.0).tolist(),
    }
    save_json(os.path.join(out_dir, "patient_level_records.json"), payload)


def save_routing_records(records, out_dir):
    payload = {
        "family_names": records.get("family_names", []),
        "records": records.get("routing_records", []),
    }
    save_json(os.path.join(out_dir, "routing_records.json"), payload)


def save_topomoe_topology(records, out_dir):
    payload = records.get("topomoe_topology", {})
    if payload:
        save_json(os.path.join(out_dir, "topomoe_topology.json"), payload)


def save_topomoe_figure_manifest(manifest, out_dir):
    save_json(os.path.join(out_dir, "topomoe_figure_manifest.json"), manifest)


def save_manifold_artifacts(records, anchor_vocab, out_dir):
    from glioma.visualization.manifold_fusion import build_manifold_payloads

    feature_stats, topology, case_records = build_manifold_payloads(records, anchor_vocab)
    save_json(os.path.join(out_dir, "manifold_feature_stats.json"), feature_stats)
    save_json(os.path.join(out_dir, "manifold_case_records.json"), case_records)
    save_json(os.path.join(out_dir, "manifold_topology.json"), topology)
    manifold = records.get("manifold_fusion", {})
    np.savez_compressed(
        os.path.join(out_dir, "manifold_graph_records.npz"),
        local_adjacency=np.nan_to_num(manifold.get("local_adjacency", [])),
        upper_adjacency=np.nan_to_num(manifold.get("upper_adjacency", [])),
        local_distances=np.nan_to_num(manifold.get("local_distances", [])),
        upper_distances=np.nan_to_num(manifold.get("upper_distances", [])),
        raw_scales=np.nan_to_num(manifold.get("raw_scales", [])),
        raw_spd_traces=np.nan_to_num(manifold.get("raw_spd_traces", [])),
        spd_eigenvalues=np.nan_to_num(manifold.get("spd_eigenvalues", [])),
        condition_numbers=np.nan_to_num(manifold.get("condition_numbers", [])),
    )
    return feature_stats, topology, case_records


__all__ = [
    "save_json",
    "save_patient_level_records",
    "save_routing_records",
    "save_topomoe_topology",
    "save_topomoe_figure_manifest",
    "save_manifold_artifacts",
]
