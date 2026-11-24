"""MRI-only inputs and independently computed UTSW phenotype targets."""

import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from glioma.data.utsw_dataset import MODALITIES, find_modality_file, parse_utsw_label, percentile_norm


ANCHORS = ("idh", "1p19q", "mgmt")
MANUAL_MASKS = ("rtumorseg_manual_correction.nii.gz", "tumorseg_manual_correction.nii.gz")
IMAGE_SIZE = 64


def manual_masks(folder):
    return [folder / name for name in MANUAL_MASKS if (folder / name).is_file()]


def grid(volume):
    return {"shape": list(volume.shape), "affine": volume.affine.tolist()}


def same_grid(volume, reference):
    return list(volume.shape) == reference["shape"] and np.allclose(
        volume.affine, reference["affine"], rtol=0, atol=1e-4
    )


def image_sources(folder, reference=None):
    paths = [Path(find_modality_file(str(folder), modality, prefer_registered=True)) for modality in MODALITIES]
    volumes = [nib.load(path) for path in paths]
    if len(volumes[0].shape) != 3 or not np.isfinite(volumes[0].affine).all():
        raise ValueError("Expected a finite three-dimensional MRI grid")
    reference = grid(volumes[0]) if reference is None else reference
    if not all(same_grid(volume, reference) for volume in volumes):
        raise ValueError("MRI sequences must match the fixed registered grid in shape and affine")
    return paths, volumes, reference


def fixed_view(volumes, size=IMAGE_SIZE):
    channels = []
    for volume in volumes:
        values = volume.get_fdata(dtype=np.float32)
        if not np.isfinite(values).all() or not np.any(values > 0):
            raise ValueError("MRI contains invalid or empty intensities")
        channels.append(percentile_norm(values))
    images = torch.from_numpy(np.stack(channels))
    if not torch.all(images.flatten(1).std(dim=1) > 0):
        raise ValueError("MRI contains a constant sequence")
    # The full image lattice is pooled without knowing the segmentation or its location.
    return F.adaptive_avg_pool3d(images, (size, size, size)).numpy().astype(np.float16)


def phenotype_target(masks, reference):
    unmapped = None
    for path in masks:
        volume = nib.load(path)
        if same_grid(volume, reference):
            labels = volume.get_fdata(dtype=np.float32)
            unknown_labels = labels[~np.isin(labels, [0, 1, 2, 4])]
            unknown = len(unknown_labels)
            tumor = int(np.count_nonzero(labels)) if not unknown else None
            edema = int(np.count_nonzero(labels == 2))
            count_3 = int(np.count_nonzero(unknown_labels == 3))
            status = "eligible" if tumor else "empty_mask"
            if unknown:
                status = "unmapped_label_3" if count_3 == unknown else "unmapped_labels"
            result = {"manual_mask": path.name, "tumor_voxels": tumor, "edema_voxels": edema,
                      "unmapped_3_voxels": count_3, "unmapped_voxels": unknown,
                      "phenotype_status": status, "phenotype": edema / tumor if tumor else None}
            if not unknown:
                return result
            if unmapped is None:
                unmapped = result
    if unmapped is not None:
        return unmapped
    raise ValueError("No manual segmentation matches the registered MRI grid")


def case_signature(paths):
    return [(str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns) for path in paths]


def prepare_images(paths, volumes, destination):
    signature = hashlib.sha256(repr(case_signature(paths)).encode()).hexdigest()
    if destination.is_file():
        with np.load(destination) as archive:
            if str(archive["signature"]) == signature:
                return signature
    temporary = destination.with_suffix(".tmp.npz")
    np.savez(temporary, images=fixed_view(volumes), signature=signature)
    temporary.replace(destination)
    return signature


def anchor_targets(metadata):
    return [parse_utsw_label(metadata, field) for field in ANCHORS]


class AnchorPhenotypeDataset(Dataset):
    def __init__(self, rows, cache, images):
        self.rows = rows
        self.cache = Path(cache)
        self.images = images

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        sid = row["subject_id"]
        if sid not in self.images:
            with np.load(self.cache / f"{sid}.npz") as archive:
                self.images[sid] = torch.from_numpy(archive["images"])
        anchors = torch.tensor([x if x is not None else -1 for x in row["anchors"]], dtype=torch.float32)
        phenotype = float("nan") if row["phenotype"] is None else row["phenotype"]
        return self.images[sid], anchors, torch.tensor(phenotype, dtype=torch.float32)
