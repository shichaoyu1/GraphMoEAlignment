"""DataLoader construction and query-target utilities."""

from torch.utils.data import DataLoader

from glioma.anchors import target_anchor_keys
from glioma.data.utsw_dataset import UTSWROIPatientDataset


def make_loader(cases, args, split_name):
    dataset = UTSWROIPatientDataset(
        cases,
        roi_size=args.roi_size,
        z_slices=args.z_slices,
        prefer_registered=args.prefer_registered,
        augment=split_name == "train" and args.augment,
        cache=args.cache,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=split_name == "train",
        num_workers=args.num_workers,
        drop_last=False,
    )


def build_query_targets(subject_ids, node_names, case_lookup, key_to_id, args):
    target_ids = []
    for subject_id in subject_ids:
        metadata = case_lookup[str(subject_id)]["metadata"]
        for node_name in node_names:
            keys = target_anchor_keys(
                metadata,
                node_name,
                args.target_policy,
                include_pathology=not args.exclude_pathology_anchors,
                include_molecular=not args.exclude_molecular_anchors,
                include_clinical=args.include_clinical_anchors,
            )
            target_ids.append([key_to_id[key] for key in keys if key in key_to_id])
    return target_ids

__all__ = ["make_loader", "build_query_targets"]
