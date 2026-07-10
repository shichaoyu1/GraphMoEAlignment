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


__all__ = [
    "save_json",
    "save_patient_level_records",
    "save_routing_records",
    "save_topomoe_topology",
    "save_topomoe_figure_manifest",
]
