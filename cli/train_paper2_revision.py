"""Explicit entry point for isolated Paper2 revision experiments."""

import argparse


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--metadata_tsv")
    parser.add_argument("--out_dir", required=True, help="A new directory; never overwrites an existing run")
    parser.add_argument("--method", default="factorized", choices=["direct", "global", "multilabel", "frequency", "node_frequency", "calibration", "factorized", "mean_positive_proxy", "proxy_anchor_multilabel"])
    parser.add_argument("--projection", choices=["shared", "separate"], default="shared")
    parser.add_argument("--gate", choices=["global", "node", "dynamic"], default="dynamic")
    parser.add_argument("--target_policy", choices=["region_rules", "all_patient_anchors"], default="region_rules")
    parser.add_argument("--candidate_policy", choices=["full_gallery", "observed_fields"], default="full_gallery")
    parser.add_argument("--family_supervision", choices=["on", "off"], default="on", help="Explicit equal-present-family gate loss; retrieval supervision remains")
    parser.add_argument("--segmentation_source", default="not_documented", help="Describe how provided segmentation was obtained; provenance is not inferred")
    parser.add_argument("--splits_file", help="Optional frozen train/val/test patient membership file")
    parser.add_argument("--reference_run", help="Existing legacy direct_only artifact directory; frozen calibration or head initialization")
    parser.add_argument("--reference_checkpoint", default="best_semantic_alignment.pt")
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--cv_fold", type=int, default=0)
    parser.add_argument("--cv_seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--router_epochs", type=int, default=15)
    parser.add_argument("--joint_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_anchor", type=float, default=0.05)
    parser.add_argument("--lambda_family", type=float, default=0.3)
    parser.add_argument("--lambda_within", type=float, default=0.3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--proxy_alpha", type=float, default=32.0)
    parser.add_argument("--proxy_margin", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--roi_size", type=int, default=96)
    parser.add_argument("--z_slices", type=int, default=7)
    parser.add_argument("--feat_dim", type=int, default=256)
    parser.add_argument("--shared_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_cases", type=int)
    parser.add_argument("--prefer_registered", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main(argv=None):
    from glioma.paper2_revision.training import train
    return train(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
