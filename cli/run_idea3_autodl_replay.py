"""Prepare server MRI inputs, replay five folds, and export AP diagnostics."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepared = args.output / "prepared" / "experiment1"
    replay = args.output / "replay"
    python = [sys.executable, "-u", "-m"]
    if not (prepared / "manifest.json").is_file():
        subprocess.run([*python, "glioma.cli.idea3", "prepare", "--data-root", args.data_root,
                        "--metadata", args.metadata, "--output", prepared.parent], check=True)
    subprocess.run([*python, "glioma.cli.replay_idea3_anchor", "--pilot", args.pilot, "--prepared", prepared,
                    "--output", replay, "--device", "cuda"], check=True)
    for method in ("probe", "refit_raw"):
        subprocess.run([*python, "glioma.cli.diagnose_idea3_anchor", "--predictions", replay / "replay_predictions.csv",
                        "--candidate-method", method, "--output", args.output / f"diagnosis_{method}"], check=True)


if __name__ == "__main__":
    main()
