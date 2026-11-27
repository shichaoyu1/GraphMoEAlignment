"""External A-state MRI swap on the registered WAW-Glioma v3 cases."""

import argparse
from pathlib import Path

import torch

from glioma.cli.idea3_external_swap import RESIDUAL_SHA256 as RESIDUAL_SHA256, run_external
from glioma.data.waw_grade import CLINICAL, cohort, prepare_images


def prepare_waw(inputs, output, smoke):
    rows = cohort(inputs / CLINICAL, smoke)
    return rows, prepare_images(rows, inputs, output)


def run(inputs, source, residual, output, device, batch_size=8, smoke=False):
    return run_external(inputs, source, residual, output, device, batch_size, smoke,
                        name="WAW-Glioma", release="WAW-Glioma v3 / 22205631", prepare=prepare_waw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("waw-inputs", "authority-source", "residual-source", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="Registered cases 4 and 26, all five checkpoints")
    args = parser.parse_args()
    run(args.waw_inputs, args.authority_source, args.residual_source, args.output,
        torch.device(args.device), args.batch_size, args.smoke)


if __name__ == "__main__":
    main()
