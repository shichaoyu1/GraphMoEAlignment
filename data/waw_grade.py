"""The prespecified WAW-Glioma v3 cohort and its UTSW-compatible ROI inputs."""

import csv
from pathlib import Path
import shutil
import tempfile
from zipfile import BadZipFile, ZipFile

import nibabel as nib
import numpy as np

from glioma.data.utsw_authority import features, molecular_record, prepare_case, rule_active


CLINICAL = "1_0_clinical_molecular_data.csv"
SCANS = "2_1_preop_scans_coreg.zip"
MASKS = "3_1_preop_masks.zip"
GRADE2 = (4, 17, 59, 389, 498, 505, 523, 526, 569, 845, 889)
GRADE3 = (26, 153, 173, 249, 313, 323, 581, 672, 699, 724, 908, 962)
SEQUENCES = {"t1n": "t1", "t1c": "t1ce", "t2w": "t2", "t2f": "flair"}


def cohort(table, smoke=False):
    expected = {str(i): 2 for i in GRADE2} | {str(i): 3 for i in GRADE3}
    if smoke:
        expected = {i: expected[i] for i in ("4", "26")}
    rows = {}
    with table.open(encoding="utf-8-sig", newline="") as file:
        for item in csv.DictReader(file, delimiter=";"):
            sid = item["id"]
            if sid not in expected:
                continue
            diagnosis = item["integrated_diagnosis"].strip().lower()
            scope = diagnosis.startswith("oligodendroglioma") and "phenotype" not in diagnosis
            record = molecular_record({
                "Tumor Type": "OLIGODENDROGLIOMA, NOS" if scope else "",
                "IDH": "mutated" if item["idh1/2"] == "1" else "",
                "1p19Q CODEL": "co-deleted" if item["Co-1p/19q"] == "1" else "",
                "IDH_Testing Method": "NGS" if item["ngs"] == "1" else "",
                "MGMT": {"0": "unmethylated", "1": "methylated"}.get(item["mgmt_met"], ""),
            })
            grade = int(item["who_grade"])
            if sid in rows or not rule_active(record) or grade != expected[sid]:
                raise ValueError(f"WAW {sid}: duplicate or inconsistent registered case")
            rows[sid] = dict(subject_id=sid, record=record, features=features(record),
                             active=True, label=grade - 2)
    if rows.keys() != expected.keys():
        raise ValueError(f"Missing WAW cases: {sorted(expected.keys() - rows.keys())}")
    return [rows[i] for i in sorted(rows, key=int)]


def prepare_waw_case(sid, scans, masks, staging, cache):
    # Materialize only this examination, under names the unchanged UTSW ROI loader accepts.
    folder = staging / sid
    folder.mkdir()
    files = {}
    for waw, utsw in SEQUENCES.items():
        member = f"gliomai_preop_scans_renamed/{sid}/{sid}_0_brain_{waw}.nii.gz"
        files[f"brain_{utsw}.nii.gz"] = (scans, member)
    files["tumor_seg.nii.gz"] = (masks, f"preop_masks/{sid}/{sid}_0_tumor_mask_pre.nii.gz")
    for name, (archive, member) in files.items():
        path = folder / name
        with archive.open(member) as src, path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        image = nib.load(str(path))
        voxels = np.asanyarray(image.dataobj)
        if voxels.ndim != 3 or not np.isfinite(voxels).all() or voxels.max() <= 0:
            raise ValueError(f"{member}: expected a finite, nonempty 3D volume")
        del voxels
    audit = prepare_case(staging, sid, cache)
    audit["archive_members"] = {name: member for name, (_, member) in files.items()}
    return audit


def prepare_images(rows, inputs, output):
    checked, errors = [], {}
    with ZipFile(inputs / SCANS) as scans, ZipFile(inputs / MASKS) as masks:
        for row in rows:
            sid = row["subject_id"]
            try:
                with tempfile.TemporaryDirectory(dir=output) as staging:
                    audit = prepare_waw_case(sid, scans, masks, Path(staging), output / "cache")
                checked.append(dict(**audit, reference_grade=row["label"] + 2, report=row["record"]))
            except (ValueError, OSError, KeyError, EOFError, BadZipFile, nib.filebasedimages.ImageFileError) as exc:
                errors[sid] = str(exc)
    return dict(complete=len(checked) == len(rows), accepted=checked, errors=errors)
