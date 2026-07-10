"""UTSW-Glioma dataset utilities and patient-level ROI dataset."""

import csv
import glob
import os
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


MODALITIES = ("t1", "t1ce", "t2", "flair")

MODALITY_CANDIDATES = {
    "t1": ["brain_t1.nii.gz", "brain_t1_ants.nii.gz", "*_t1.nii.gz", "*-t1.nii.gz"],
    "flair": ["brain_flair.nii.gz", "brain_fl_ants.nii.gz", "*_flair.nii.gz", "*-flair.nii.gz", "*_fl_*.nii.gz"],
    "t1ce": ["brain_t1ce.nii.gz", "brain_t1ce_ants.nii.gz", "*_t1ce.nii.gz", "*-t1ce.nii.gz", "*_t1gd.nii.gz", "*-t1gd.nii.gz"],
    "t2": ["brain_t2.nii.gz", "brain_t2_ants.nii.gz", "*_t2.nii.gz", "*-t2.nii.gz"],
}

SEGMENTATION_CANDIDATES = [
    "rtumorseg_manual_correction.nii.gz",
    "tumorseg_manual_correction.nii.gz",
    "tumorseg_FeTS.nii.gz",
    "*_seg.nii.gz",
    "*-seg.nii.gz",
    "*seg*.nii.gz",
]

UTSW_METADATA_FILENAME = "UTSW_Glioma_Metadata-2-1.tsv"


def percentile_norm(vol: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    vol = np.asarray(vol, dtype=np.float32)
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    fg = vol[vol > 0]
    if len(fg) == 0:
        return vol.astype(np.float32)
    lo, hi = np.percentile(fg, [p_lo, p_hi])
    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.zeros_like(vol, dtype=np.float32)
    out = np.clip((vol - lo) / (hi - lo + 1e-8), 0, 1)
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return out.astype(np.float32)


def _glob_candidates(folder: str, patterns: list) -> list:
    hits = []
    for pattern in patterns:
        hits.extend(glob.glob(os.path.join(folder, pattern)))
    return sorted(set(hits))


def _prefer_shape_match(paths: list, shape: tuple = None) -> str:
    if not paths:
        return None
    if shape is None:
        return paths[0]
    try:
        import nibabel as nib
    except ImportError:
        return paths[0]
    for path in paths:
        if tuple(nib.load(path).shape) == tuple(shape):
            return path
    return paths[0]


def find_modality_file(folder: str, modality: str, prefer_registered: bool = False) -> str:
    key = modality.lower()
    patterns = MODALITY_CANDIDATES.get(key, [f"*{modality}*.nii.gz"])
    hits = _glob_candidates(folder, patterns)

    if prefer_registered:
        registered = [path for path in hits if "_ants" in os.path.basename(path).lower()]
        if registered:
            hits = registered
    else:
        native = [path for path in hits if "_ants" not in os.path.basename(path).lower()]
        if native:
            hits = native

    if not hits:
        raise FileNotFoundError(f"Cannot find modality '{modality}' under folder={folder}")
    return hits[0]


def find_segmentation_file(folder: str, image_shape: tuple = None) -> str:
    hits = _glob_candidates(folder, SEGMENTATION_CANDIDATES)
    if not hits:
        raise FileNotFoundError(f"Cannot find a segmentation file under folder={folder}")
    return _prefer_shape_match(hits, image_shape)


def find_utsw_metadata(patient_dir: str, metadata_tsv: str = None) -> str:
    if metadata_tsv and os.path.exists(metadata_tsv):
        return metadata_tsv
    current = os.path.abspath(patient_dir)
    for _ in range(4):
        parent = os.path.dirname(current)
        for folder in (current, parent):
            candidate = os.path.join(folder, UTSW_METADATA_FILENAME)
            if os.path.exists(candidate):
                return candidate
        if parent == current:
            break
        current = parent
    return None


def load_utsw_metadata(metadata_tsv: str) -> dict:
    records = {}
    with open(metadata_tsv, "r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            subject_id = row.get("Subject ID")
            if subject_id:
                records[subject_id] = row
    return records


def get_utsw_cases(root_dir: str, metadata_tsv: str = None, require_seg: bool = True) -> list:
    metadata_path = find_utsw_metadata(root_dir, metadata_tsv)
    metadata = load_utsw_metadata(metadata_path) if metadata_path else {}
    cases = []
    for entry in sorted(os.scandir(root_dir), key=lambda item: item.name):
        if not entry.is_dir():
            continue
        if not glob.glob(os.path.join(entry.path, "*.nii.gz")):
            continue
        if require_seg and not _glob_candidates(entry.path, SEGMENTATION_CANDIDATES):
            continue
        info = metadata.get(entry.name, {})
        cases.append(
            {
                "subject_id": entry.name,
                "patient_dir": entry.path,
                "tumor_grade": info.get("Tumor Grade"),
                "tumor_type": info.get("Tumor Type"),
                "metadata": info,
            }
        )
    return cases


def _clean_text(value):
    if value is None:
        return ""
    return str(value).strip().lower()


def parse_utsw_label(metadata: dict, task: str):
    task = task.lower()
    if task == "idh":
        value = _clean_text(metadata.get("IDH"))
        if value in {"mutated", "mutant"}:
            return 1
        if value in {"wild type", "wildtype", "wt"}:
            return 0
        return None
    if task == "mgmt":
        value = _clean_text(metadata.get("MGMT"))
        if value == "methylated":
            return 1
        if value == "unmethylated":
            return 0
        return None
    if task == "1p19q":
        value = _clean_text(metadata.get("1p19Q CODEL"))
        if value in {"co-deleted", "codeleted", "co deleted"}:
            return 1
        if value in {"non co-deleted", "non-codeleted", "non co deleted"}:
            return 0
        return None
    if task == "grade":
        value = _clean_text(metadata.get("Tumor Grade"))
        if value in {"2", "ii"}:
            return 0
        if value in {"3", "iii"}:
            return 1
        if value in {"4", "iv"}:
            return 2
        return None
    raise ValueError(f"Unsupported task: {task}")


def stratified_split(cases, train_ratio=0.7, val_ratio=0.1, seed=42):
    rng = np.random.default_rng(seed)
    grouped = defaultdict(list)
    for case in cases:
        grouped[case["label"]].append(case)
    splits = {"train": [], "val": [], "test": []}
    for label_cases in grouped.values():
        label_cases = list(label_cases)
        rng.shuffle(label_cases)
        n_cases = len(label_cases)
        n_train = max(1, int(round(n_cases * train_ratio))) if n_cases else 0
        n_val = int(round(n_cases * val_ratio)) if n_cases >= 5 else 0
        if n_train + n_val >= n_cases and n_cases > 1:
            n_train = n_cases - 1
            n_val = 0
        splits["train"].extend(label_cases[:n_train])
        splits["val"].extend(label_cases[n_train : n_train + n_val])
        splits["test"].extend(label_cases[n_train + n_val :])
    for split_cases in splits.values():
        split_cases.sort(key=lambda item: item["subject_id"])
    return splits


def describe_cases(cases):
    return dict(sorted(Counter(case["label"] for case in cases).items()))


def _bbox_from_mask(seg, margin=8):
    coords = np.argwhere(seg > 0)
    if len(coords) == 0:
        center = np.array(seg.shape) // 2
        return (0, seg.shape[0] - 1, 0, seg.shape[1] - 1, int(center[2]))
    y_min, x_min, z_min = coords.min(axis=0)
    y_max, x_max, z_max = coords.max(axis=0)
    y_min = max(0, int(y_min) - margin)
    x_min = max(0, int(x_min) - margin)
    y_max = min(seg.shape[0] - 1, int(y_max) + margin)
    x_max = min(seg.shape[1] - 1, int(x_max) + margin)
    z_center = int(round((int(z_min) + int(z_max)) / 2))
    return y_min, y_max, x_min, x_max, z_center


def _z_indices(z_center, depth, z_slices):
    half = z_slices // 2
    indices = [z_center + offset for offset in range(-half, half + 1)]
    if len(indices) > z_slices:
        indices = indices[:z_slices]
    while len(indices) < z_slices:
        indices.append(indices[-1] if indices else z_center)
    return [int(np.clip(index, 0, depth - 1)) for index in indices]


def _resize_stack(stack, roi_size):
    tensor = torch.from_numpy(stack.astype(np.float32)).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(roi_size, roi_size), mode="bilinear", align_corners=False)
    tensor = tensor.squeeze(0)
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / (std + 1e-6)


def _resize_mask_stack(stack, roi_size):
    tensor = torch.from_numpy(stack.astype(np.float32)).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(roi_size, roi_size), mode="nearest")
    return tensor.squeeze(0)


class UTSWROIPatientDataset(Dataset):
    def __init__(
        self,
        cases,
        roi_size: int = 96,
        z_slices: int = 7,
        modalities=MODALITIES,
        prefer_registered: bool = False,
        augment: bool = False,
        cache: bool = False,
    ):
        self.cases = list(cases)
        self.roi_size = roi_size
        self.z_slices = z_slices
        self.modalities = tuple(modalities)
        self.prefer_registered = prefer_registered
        self.augment = augment
        self.cache = cache
        self._cache = {}

    def __len__(self):
        return len(self.cases)

    def _load_case(self, case):
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError("Please install nibabel: pip install nibabel")

        patient_dir = case["patient_dir"]
        first_modality = find_modality_file(patient_dir, self.modalities[0], prefer_registered=self.prefer_registered)
        image_shape = tuple(nib.load(first_modality).shape)
        seg_path = find_segmentation_file(patient_dir, image_shape=image_shape)
        seg = nib.load(seg_path).get_fdata(dtype=np.float32)

        y_min, y_max, x_min, x_max, z_center = _bbox_from_mask(seg)
        z_ids = _z_indices(z_center, seg.shape[2], self.z_slices)

        stacks = []
        for modality in self.modalities:
            path = find_modality_file(patient_dir, modality, prefer_registered=self.prefer_registered)
            volume = nib.load(path).get_fdata(dtype=np.float32)
            volume = percentile_norm(volume)
            slices = []
            for z_idx in z_ids:
                crop = volume[y_min : y_max + 1, x_min : x_max + 1, z_idx]
                slices.append(crop)
            stacks.append(_resize_stack(np.stack(slices, axis=0), self.roi_size))

        region_masks = []
        for label_value in [1, 2, 4]:
            mask_slices = []
            for z_idx in z_ids:
                crop = (seg[y_min : y_max + 1, x_min : x_max + 1, z_idx] == label_value).astype(np.float32)
                mask_slices.append(crop)
            region_masks.append(_resize_mask_stack(np.stack(mask_slices, axis=0), self.roi_size))
        return torch.stack(stacks, dim=0), torch.stack(region_masks, dim=0)

    def __getitem__(self, index):
        case = self.cases[index]
        subject_id = case["subject_id"]
        if self.cache and subject_id in self._cache:
            images, region_masks = self._cache[subject_id]
            images = images.clone()
            region_masks = region_masks.clone()
        else:
            images, region_masks = self._load_case(case)
            if self.cache:
                self._cache[subject_id] = (images.clone(), region_masks.clone())

        if self.augment:
            if torch.rand(()) > 0.5:
                images = torch.flip(images, dims=(-1,))
                region_masks = torch.flip(region_masks, dims=(-1,))
            if torch.rand(()) > 0.5:
                images = torch.flip(images, dims=(-2,))
                region_masks = torch.flip(region_masks, dims=(-2,))

        return {
            "images": images,
            "region_masks": region_masks,
            "label": torch.tensor(case["label"], dtype=torch.long),
            "subject_id": subject_id,
        }


__all__ = [
    "MODALITIES",
    "UTSWROIPatientDataset",
    "describe_cases",
    "find_utsw_metadata",
    "get_utsw_cases",
    "load_utsw_metadata",
    "parse_utsw_label",
    "stratified_split",
]
