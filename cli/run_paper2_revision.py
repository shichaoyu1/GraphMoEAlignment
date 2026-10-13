"""Build a bounded supplemental matrix. Execution is explicit; old runs are read-only."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def experiments(group):
    baseline = [(name, {"method": name}) for name in ("frequency", "node_frequency", "direct", "global", "multilabel")]
    published = [(name, {"method": name}) for name in ("mean_positive_proxy", "proxy_anchor_multilabel")]
    mechanism = [(f"{projection}_{gate}", {"method": "factorized", "projection": projection, "gate": gate})
                 for projection in ("shared", "separate") for gate in ("node", "dynamic")]
    supervision = [(f"targets_{policy}_family_{supervision}", {"method": "factorized", "projection": "separate", "gate": "dynamic",
                                                             "target_policy": policy, "family_supervision": supervision})
                   for policy in ("region_rules", "all_patient_anchors") for supervision in ("on", "off")
                   if not (policy == "region_rules" and supervision == "on")]
    calibration = [(f"calibration_{gate}", {"method": "calibration", "projection": "shared", "gate": gate}) for gate in ("global", "node")]
    groups = {"cheap": baseline[:2], "baselines": baseline, "published": published,
              "mechanism": mechanism, "supervision": supervision, "calibration": calibration,
              "screen": baseline + published + mechanism + supervision}
    return groups[group]


def build_plan(args):
    if args.cv_folds < 3 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Require >=3 folds and unique initialization seeds")
    folds = list(range(args.cv_folds)) if args.folds is None else args.folds
    if not folds or len(set(folds)) != len(folds) or any(fold < 0 or fold >= args.cv_folds for fold in folds):
        raise ValueError("Invalid or duplicate folds")
    if args.group == "calibration" and not args.reference_root:
        raise ValueError("Calibration group requires the existing direct_only reference root")
    plan = []
    for name, flags in experiments(args.group):
        for fold in folds:
            for seed in args.seeds:
                out = Path(args.out_root) / name / f"fold_{fold}" / f"seed_{seed}"
                command = [sys.executable, "-m", "glioma.cli.train_paper2_revision", "--data_root", args.data_root,
                           "--out_dir", str(out), "--cv_folds", str(args.cv_folds), "--cv_fold", str(fold),
                           "--cv_seed", str(args.cv_seed), "--seed", str(seed), "--epochs", str(args.epochs),
                           "--router_epochs", str(args.router_epochs), "--joint_epochs", str(args.joint_epochs),
                           "--batch_size", str(args.batch_size), "--num_workers", str(args.num_workers),
                           "--segmentation_source", args.segmentation_source]
                if args.metadata_tsv:
                    command += ["--metadata_tsv", args.metadata_tsv]
                if args.reference_root:
                    reference = Path(args.reference_root) / f"fold_{fold}" / f"seed_{seed}"
                    command += ["--splits_file", str(reference / "splits.json")]
                    # Target-policy changes must train their own representations.
                    if flags["method"] in {"calibration", "factorized"} and flags.get("target_policy", "region_rules") == "region_rules":
                        command += ["--reference_run", str(reference)]
                if args.cpu:
                    command.append("--cpu")
                for key, value in flags.items():
                    command += ["--" + key, str(value)]
                plan.append({"name": name, "fold": fold, "seed": seed, "output": str(out), "command": command})
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--metadata_tsv")
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--group", choices=["cheap", "baselines", "published", "mechanism", "supervision", "calibration", "screen"], default="cheap")
    parser.add_argument("--reference_root")
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--folds", type=int, nargs="+")
    parser.add_argument("--cv_seed", type=int, default=2026)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--router_epochs", type=int, default=15)
    parser.add_argument("--joint_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--segmentation_source", default="not_documented")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Without this flag prints a plan only")
    parser.add_argument("--skip_completed", action="store_true", help="Skip completed runs only when all resolved configuration values match")
    args = parser.parse_args(argv)
    plan = build_plan(args)
    if not args.execute:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return plan
    from glioma.cli.train_paper2_revision import build_parser
    from glioma.paper2_revision.training import source_manifest
    source = source_manifest()
    # Preflight every output before launching any training.
    pending = []
    for item in plan:
        out = Path(item["output"])
        if out.exists():
            resolved = vars(build_parser().parse_args(item["command"][3:]))
            if args.skip_completed and all((out / name).is_file() for name in ("completed.json", "config.json", "source_manifest.json")):
                saved_source = json.loads((out / "source_manifest.json").read_text(encoding="utf-8"))
                if json.loads((out / "config.json").read_text(encoding="utf-8")) == resolved and saved_source == source:
                    continue
            raise FileExistsError("Existing output or changed configuration: " + str(out))
        pending.append(item)
    for item in pending:
        subprocess.run(item["command"], check=True)
    return plan


if __name__ == "__main__":
    main()
