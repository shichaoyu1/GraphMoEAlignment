"""The audited TCGA-LGG roster and the 13 accepted BraTS 2021 ROI caches."""

import csv
import hashlib
import io
import json
from zipfile import ZipFile

from glioma.data.utsw_authority import features, molecular_record, rule_active


BUNDLE_SHA256 = "5b0a9a7021ce535236c33580c74782a45377dfc06c8c65f94ba50b787a49b5bf"
SMOKE_IDS = ("TCGA-CS-5396", "TCGA-CS-6668")


def prepare_tcga(bundle, output, smoke):
    if hashlib.sha256(bundle.read_bytes()).hexdigest() != BUNDLE_SHA256:
        raise ValueError(f"TCGA input bundle differs from the audited 13-case release: {bundle}")
    cache = output / "cache"
    cache.mkdir(exist_ok=True)
    rows, accepted = [], []
    with ZipFile(bundle) as archive:
        image_audit = json.loads(archive.read("image_audit.json"))
        roster = csv.DictReader(io.StringIO(archive.read("cohort.csv").decode("utf-8")))
        for item in roster:
            sid = item["subject_id"]
            if item["include"] != "1" or (smoke and sid not in SMOKE_IDS):
                continue
            codel = item["subtype"] == "LGG_IDHmut-codel" and item["arm_1p"] == item["arm_19q"] == "-1"
            record = molecular_record({
                "Tumor Type": "OLIGODENDROGLIOMA, NOS" if item["histology"] == "Oligodendroglioma" else "",
                "IDH": "mutated" if item["idh_variant"].startswith(("IDH1 p.R132", "IDH2 p.R172")) else "",
                "1p19Q CODEL": "co-deleted" if codel else "",
                "IDH_Testing Method": item["idh_method"],
                "MGMT": item["mgmt"],
            })
            if not rule_active(record):
                raise ValueError(f"TCGA molecular evidence does not activate the registered rule for {sid}")
            grade = int(item["reference_grade"])
            member = f"cache/{sid}.npz"
            (output / member).write_bytes(archive.read(member))
            rows.append(dict(subject_id=sid, label=grade - 2, record=record, features=features(record),
                             active=True, tss=item["tss"], idh_source=item["idh_source"]))
            accepted.append(dict(**image_audit[sid], subject_id=sid, report=record, evidence=item))
    return rows, dict(complete=True, accepted=accepted, errors={}, input_sha256=BUNDLE_SHA256,
                      verification="NIfTI voxel and ROI audit before packing; frozen bundle verified at inference")
