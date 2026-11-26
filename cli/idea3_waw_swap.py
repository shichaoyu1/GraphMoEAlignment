"""External A-state MRI swap on the registered WAW-Glioma v3 cases."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from glioma.cli.idea3_anchor import atomic_csv
from glioma.cli.idea3_authority import metadata_splits
from glioma.cli.idea3_role_states import scopeless_models
from glioma.cli.idea3_rule_mri_swap import load_models, replay_fold, swap_fold, verify_fold_archive
from glioma.cli.utsw_authority import loader, save_json, seed_all
from glioma.data.utsw_authority import RULE_VERSION
from glioma.data.waw_grade import CLINICAL, cohort, prepare_images
from glioma.eval.idea3_waw_swap import external_results


# SHA-256 recorded by the completed UTSW three-state replay, in checkpoint-fold order.
RESIDUAL_SHA256 = (
    "cf863449c209c6cb0232853ca7552127e535dd7428e068b38c46250012b5d6da",
    "eeb1a8e588a0fe2a192a621d14efc2a4e83e38bbdc969dae4a66d1e2db2f4729",
    "0733ba235cfb8beda080e759a2a5ad18607f940c7eab65f0f758c8467b38b8f0",
    "434666116ae527cb139ff822feadd4285c1bb606996fa73588b0ad96dc6ee566",
    "ecd76ad459e53ab3baaa528f76503767214591a36044500a8ea4d4134e7e7d36",
)


def run(inputs, source, residual, output, device, batch_size=8, smoke=False):
    output.mkdir(parents=True, exist_ok=True)
    if any(entry.name not in {"run.log", "run.pid"} for entry in output.iterdir()):
        raise FileExistsError(f"Expected a new WAW run directory: {output}")
    with (output / ".started").open("x") as marker:
        marker.write("summary.json marks successful completion\n")
    seed_all()
    raw = (source / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    if manifest["rule_version"] != RULE_VERSION or batch_size < 1:
        raise ValueError("Expected the registered UTSW rule and a positive batch size")
    for fold, expected in enumerate(RESIDUAL_SHA256):
        archive = residual / f"fold{fold}"
        verify_fold_archive(source, archive, digest, fold)
        if hashlib.sha256((archive / "best.pt").read_bytes()).hexdigest() != expected:
            raise ValueError(f"Fold {fold}: weights differ from the UTSW three-state replay")

    rows = cohort(inputs / CLINICAL, smoke)
    ids = [r["subject_id"] for r in rows]
    audit = prepare_images(rows, inputs, output)
    save_json(output / "image_audit.json", audit)
    if not audit["complete"]:
        raise ValueError(f"WAW image acceptance failed: {audit['errors']}")
    print(f"Accepted all {len(ids)} WAW examinations; starting frozen inference", flush=True)
    batches = list(loader(rows, output, batch_size=batch_size))
    predictions = []
    for fold in range(5):
        train = metadata_splits(source, manifest, fold)["train"]
        models = scopeless_models(load_models(source, residual / f"fold{fold}", manifest, train, fold, device))
        # The accepted rows already contain precisely the unmasked A-state molecular features.
        predictions.append(replay_fold(models, batches, device, fold))
        predictions.append(swap_fold(models, batches, ids, device, fold, batch_size))
        print(f"Finished checkpoint fold {fold}: {len(ids)} own and {len(ids)*(len(ids)-1)} swaps", flush=True)

    fits, ensemble, patients, report = external_results(pd.concat(predictions, ignore_index=True), ids)
    report.update(study_mode="smoke" if smoke else "full", cohort="WAW-Glioma v3 / 22205631",
                  state="A", rule_version=RULE_VERSION, n_directed_swaps=len(ids)*(len(ids)-1),
                  authority_manifest_sha256=digest, residual_sha256=RESIDUAL_SHA256,
                  grade_counts={str(g): sum(r["label"] + 2 == g for r in rows) for g in (2, 3)},
                  ensemble="mean projected probabilities across five frozen fits before scoring",
                  device=str(device), torch_version=str(torch.__version__), batch_size=batch_size)
    atomic_csv(fits, output / "checkpoint_predictions.csv")
    atomic_csv(ensemble, output / "pair_predictions.csv")
    atomic_csv(patients, output / "patient_results.csv")
    lines = ["# WAW-Glioma A-state MRI swap", "",
             f"{report['study_mode']} | {len(ids)} patients | {report['n_directed_swaps']} directed swaps", "",
             "| Model | Own Brier | Donor Brier | Donor minus own [95% CI] |",
             "|---|---:|---:|---:|"]
    for name, metrics in report["models"].items():
        effect = metrics["own_advantage"]
        lo, hi = effect["ci95"]
        lines.append(f"| {name} | {metrics['own_brier']:.4f} | {metrics['donor_brier']:.4f} | "
                     f"{effect['estimate']:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
    (output / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json(output / "summary.json", report)
    print(f"Completed {report['study_mode']}: {output / 'summary.json'}", flush=True)


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
