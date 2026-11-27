"""External A-state matched-MRI evaluation on the audited TCGA-LGG cohort."""

import argparse
from pathlib import Path

import torch

from glioma.cli.idea3_external_swap import run_external
from glioma.data.tcga_lgg_grade import prepare_tcga
from glioma.eval.idea3_tcga_lgg_swap import tcga_diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "authority-source", "residual-source", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="Two registered CS cases and all five frozen fits")
    args = parser.parse_args()
    run_external(args.inputs, args.authority_source, args.residual_source, args.output,
                 torch.device(args.device), args.batch_size, args.smoke, name="TCGA-LGG",
                 release="TCGA-LGG / BraTS 2021 / 2026-11-27", prepare=prepare_tcga,
                 diagnostics=tcga_diagnostics)


if __name__ == "__main__":
    main()
