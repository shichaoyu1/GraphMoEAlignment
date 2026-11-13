"""Grade labels and molecular evidence travel through separate interfaces."""
from dataclasses import dataclass, replace
from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold

from glioma.data.utsw_dataset import (load_utsw_metadata, parse_utsw_label,
    UTSWROIPatientDataset, MODALITIES, find_modality_file, find_segmentation_file,
    _bbox_from_mask, _z_indices)

RULE_VERSION = "who2021_oligo_scope58_v1"
SCOPE_TYPES = {"OLIGODENDROGLIOMA, ANAPLASTIC", "OLIGODENDROGLIOMA, NOS",
               "OLIGODENDROGLIOMA GRADE 2"}
FIELDS = {"IDH": ["wild type", "mutated"], "1p19Q CODEL": ["non co-deleted", "co-deleted"],
          "MGMT": ["unmethylated", "methylated"], "IDH_Testing Method": ["IHC", "NGS"]}
FEATURE_DIM = 13  # four three-way one-hot records, plus externally derived eligibility


def molecular_record(row):
    """Only the explicitly whitelisted fields and a scope bit leave this function."""
    record = {key: str(row.get(key, "")).strip() for key in FIELDS}
    record["scope_eligible"] = str(row.get("Tumor Type", "")).strip().upper() in SCOPE_TYPES
    record["curated_record_accepted"] = True
    record["assay_validity"] = "unreported"
    return record


def features(record):
    values = []
    for key, categories in FIELDS.items():
        value = record[key]
        index = categories.index(value) if value in categories else 2
        values.extend(float(i == index) for i in range(3))
    return values + [float(record["scope_eligible"])]


def rule_active(record):
    return bool(record["curated_record_accepted"] and record["assay_validity"] != "invalid"
                and record["IDH"] == "mutated" and record["1p19Q CODEL"] == "co-deleted"
                and record["IDH_Testing Method"] in ("IHC", "NGS") and record["scope_eligible"])


def cohort(metadata_path):
    result = []
    for sid, row in sorted(load_utsw_metadata(str(metadata_path)).items()):
        record = molecular_record(row)
        label = parse_utsw_label(row, "grade")
        result.append(dict(subject_id=sid, label=label, record=record, features=features(record),
                           active=rule_active(record), molecular_candidate=(record["IDH"] == "mutated"
                           and record["1p19Q CODEL"] == "co-deleted"),
                           reference_conflict=bool(rule_active(record) and label == 2)))
    return result


def fixed_folds(rows):
    labeled = [r for r in rows if r["label"] is not None]
    strata = [f'{r["label"]}_{int(r["active"])}' for r in labeled]
    assignment = {}
    for fold, (_, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=2026).split(labeled, strata)):
        for index in test:
            assignment[labeled[index]["subject_id"]] = fold
    return assignment


@dataclass
class GradeEvidence:
    images: torch.Tensor
    molecular: torch.Tensor
    available: torch.Tensor
    active: torch.Tensor

    def to(self, device):
        return replace(self, **{k: v.to(device) for k, v in vars(self).items()})


def compile_grade(evidence, variant="acf"):
    if variant not in ("acf", "learned_joint"):
        raise ValueError(variant)
    available = torch.cat([evidence.available, torch.ones(len(evidence.images), 4,
                         dtype=torch.bool, device=evidence.images.device)], 1)
    molecular = evidence.molecular
    supported = ((molecular[:, 1] == 1) & (molecular[:, 4] == 1)
                 & ((molecular[:, 9] == 1) | (molecular[:, 10] == 1)) & (molecular[:, 12] == 1))
    active = evidence.active & evidence.available[:, 4] & supported
    allowed = torch.ones(len(active), 3, dtype=torch.bool, device=active.device)
    allowed[:, 2] = ~active
    mask = available[:, :, None] & available[:, None, :]
    for support in range(5, 9):
        mask[:, support] = False
        mask[:, support, support] = True
    if variant == "acf":
        mask[active, 4, :4] = False
    return dict(active=active, allowed=allowed, mask=mask[:, None], available=available[:, None],
                status=torch.where(~evidence.available.any(1), 3, torch.where(active, 0, 1)))


def prepare_case(root, sid, cache_dir, headers_only=False):
    """Validate all source grids before reusing the established ROI implementation."""
    import nibabel as nib
    folder = Path(root) / sid
    paths = [find_modality_file(str(folder), m) for m in MODALITIES]
    volumes = [nib.load(p) for p in paths]
    seg_path = find_segmentation_file(str(folder), volumes[0].shape)
    segmentation = nib.load(seg_path)
    for volume in volumes + [segmentation]:
        if volume.shape != volumes[0].shape or not np.allclose(volume.affine, volumes[0].affine, atol=1e-4):
            raise ValueError(f"{sid}: mismatched shape/affine; registration is required")
        if nib.aff2axcodes(volume.affine) != nib.aff2axcodes(volumes[0].affine):
            raise ValueError(f"{sid}: mismatched orientation")
    if headers_only:
        return dict(subject_id=sid, files=[Path(p).name for p in paths], segmentation=Path(seg_path).name,
                    shape=list(volumes[0].shape), affine=volumes[0].affine.tolist(),
                    orientation=list(nib.aff2axcodes(volumes[0].affine)))
    signature = {"version": 1, "roi": 96, "slices": 7,
                 "files": [(Path(p).name, Path(p).stat().st_size, Path(p).stat().st_mtime_ns)
                           for p in paths + [seg_path]]}
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    destination = Path(cache_dir) / f"{sid}.npz"
    if destination.exists():
        with np.load(destination) as old:
            if str(old["signature"]) == key:
                return json.loads(str(old["provenance"]))
    seg = segmentation.get_fdata(dtype=np.float32)
    if not np.any(seg > 0):
        raise ValueError(f"{sid}: empty tumor segmentation")
    bbox = _bbox_from_mask(seg)
    z_ids = _z_indices(bbox[-1], seg.shape[2], 7)
    case = dict(patient_dir=str(folder), subject_id=sid, label=0)
    images, masks = UTSWROIPatientDataset([case])._load_case(case)
    if not torch.isfinite(images).all() or any(float(x.std()) == 0 for x in images):
        raise ValueError(f"{sid}: unusable image intensities")
    provenance = dict(subject_id=sid, files=[Path(p).name for p in paths],
                      segmentation=Path(seg_path).name, shape=list(volumes[0].shape),
                      affine=volumes[0].affine.tolist(), orientation=list(nib.aff2axcodes(volumes[0].affine)),
                      bbox=[int(v) for v in bbox], z_ids=[int(z) for z in z_ids], signature=key)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.npz")
    np.savez(temporary, images=images.numpy(), masks=masks.numpy(), signature=key,
             provenance=json.dumps(provenance))
    temporary.replace(destination)
    return provenance


class CachedGradeDataset(Dataset):
    def __init__(self, rows, cache_dir, augment=False):
        self.rows, self.cache_dir, self.augment = rows, Path(cache_dir), augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with np.load(self.cache_dir / f'{row["subject_id"]}.npz') as data:
            images = torch.from_numpy(data["images"].copy())
        if self.augment:
            for axis in (-1, -2):
                if torch.rand(()) > .5:
                    images = images.flip(axis)
        report_available = any(row["record"][k] in values for k, values in FIELDS.items())
        return dict(images=images, molecular=torch.tensor(row["features"]),
                    available=torch.tensor([True] * 4 + [report_available]), active=torch.tensor(row["active"]),
                    label=torch.tensor(row["label"]), subject_id=row["subject_id"])


def evidence_from_batch(batch, device):
    return GradeEvidence(**{k: batch[k].to(device) for k in ("images", "molecular", "available", "active")})
