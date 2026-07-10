"""Anchor construction and query-target policies for glioma alignment."""

PATHOLOGY_FIELDS = ("Tumor Grade", "Tumor Type")
MOLECULAR_FIELDS = ("IDH", "MGMT", "1p19Q CODEL")
CLINICAL_FIELDS = ("Age at Histological Diagnosis", "Gender")


def _clean_value(value):
    if value is None:
        return ""
    return str(value).strip()


def _canonical_field(field):
    return field.lower().replace(" ", "_").replace("/", "").replace("-", "_")


def _anchor_source(field):
    if field in PATHOLOGY_FIELDS:
        return "Pathology"
    if field in MOLECULAR_FIELDS:
        return "Gene"
    return "Clinical"


def _anchor_type(field):
    if field == "Tumor Grade":
        return "pathology-grade"
    if field == "Tumor Type":
        return "pathology-diagnosis"
    if field in MOLECULAR_FIELDS:
        return "molecular-marker"
    return "clinical-context"


def _make_anchor(field, value):
    value = _clean_value(value)
    key = f"{_canonical_field(field)}::{value.lower()}"
    if field == "Tumor Grade":
        label = f"Pathology grade {value}"
    elif field == "Tumor Type":
        label = f"Pathology {value}"
    else:
        label = f"{field} {value}"
    return {
        "key": key,
        "label": label,
        "field": field,
        "value": value,
        "source": _anchor_source(field),
        "node_type": _anchor_type(field),
    }


def semantic_anchors(metadata, include_pathology=True, include_molecular=True, include_clinical=False):
    anchors = []
    fields = []
    if include_pathology:
        fields.extend(PATHOLOGY_FIELDS)
    if include_molecular:
        fields.extend(MOLECULAR_FIELDS)
    if include_clinical:
        fields.extend(CLINICAL_FIELDS)

    for field in fields:
        value = _clean_value(metadata.get(field))
        if value and value.lower() not in {"na", "n/a", "nan", "none", "unknown"}:
            anchors.append(_make_anchor(field, value))
    return anchors


def target_anchor_keys(metadata, node_name, policy, include_pathology=True, include_molecular=True, include_clinical=False):
    anchors = {
        anchor["field"]: anchor["key"]
        for anchor in semantic_anchors(
            metadata,
            include_pathology=include_pathology,
            include_molecular=include_molecular,
            include_clinical=include_clinical,
        )
    }
    if policy == "all_patient_anchors":
        return list(anchors.values())

    node = _clean_value(node_name).lower()
    fields = []
    if "enhancing" in node or "t1ce" in node:
        fields = ["Tumor Grade", "MGMT", "Tumor Type"]
    elif "edema" in node or "flair" in node or "t2" in node:
        fields = ["IDH", "Tumor Type", "Tumor Grade"]
    elif "necrotic" in node or "core" in node or "t1" in node:
        fields = ["Tumor Grade", "1p19Q CODEL", "Tumor Type"]
    else:
        fields = ["Tumor Grade", "IDH", "MGMT", "1p19Q CODEL", "Tumor Type"]

    keys = [anchors[field] for field in fields if field in anchors]
    return keys or list(anchors.values())


__all__ = [
    "CLINICAL_FIELDS",
    "MOLECULAR_FIELDS",
    "PATHOLOGY_FIELDS",
    "semantic_anchors",
    "target_anchor_keys",
]
