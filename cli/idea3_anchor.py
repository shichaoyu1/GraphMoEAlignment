"""Anchor-phenotype matrix, patient-paired figures, and gated preference evaluation."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from glioma.cli.utsw_authority import save_json
from glioma.eval.idea3 import (PROB_COLS, TARGET_COLS, check_pairs, endpoint_contrast, hypervolume,
                               interpolate_predictions, interval, patient_draws, summarize_anchor, tradeoff_pass)
from glioma.training.idea3 import (INTERMEDIATE, WEIGHTS, control_parameters, fit_anchor_model, forward_batches,
                                   linear_probes, loader, molecular_features, phenotype_prediction, targets)


def atomic_csv(frame, path):
    temporary = path.with_suffix(".tmp.csv")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def manifest_data(out):
    manifest_file = out / "manifest.json"
    raw = manifest_file.read_bytes()
    manifest = json.loads(raw)
    if not all((out / "cache" / f'{row["subject_id"]}.npz').is_file() for row in manifest["rows"]):
        raise ValueError("Prepared MRI cache is incomplete")
    return manifest, hashlib.sha256(raw).hexdigest()


def anchor_frame(rows, scores, fold, seed, weight, method):
    anchors, phenotype = targets(rows)
    frame = pd.DataFrame(np.column_stack([anchors, phenotype]), columns=TARGET_COLS)
    frame[PROB_COLS] = scores
    frame["subject_id"] = [r["subject_id"] for r in rows]
    frame["fold"], frame["seed"], frame["weight"], frame["method"] = fold, seed, weight, method
    return frame


def train_fold(out, subsets, fold, seed, weight, device, epochs, batch_size, images, manifest_hash):
    model_name = "conditioned" if weight is None else f"w{weight:.2f}"
    run = out / "runs" / f"seed{seed}" / f"fold{fold}" / model_name
    run.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[1]
    files = ("cli/idea3_anchor.py", "training/idea3.py", "models/idea3.py", "data/idea3.py")
    source_hash = hashlib.sha256(b"".join((source / name).read_bytes() for name in files)).hexdigest()
    config = {"manifest_hash": manifest_hash, "source_hash": source_hash, "epochs": epochs,
              "batch_size": batch_size, "seed": seed, "fold": fold, "weight": weight, "device": str(device)}
    config_path = run / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError(f"Existing run configuration differs at {run}")
    output = run / "predictions.csv"
    if (run / "done.json").exists():
        return pd.read_csv(output)
    save_json(config_path, config)
    cache = out / "cache"
    model, timing = fit_anchor_model(subsets["train"], subsets["val"], cache, images, device,
                                     seed, weight, epochs, batch_size)
    torch.save(model.state_dict(), run / "best.tmp.pt")
    (run / "best.tmp.pt").replace(run / "best.pt")
    test_batch = loader(subsets["test"], cache, images, batch_size, device)
    train_batch = loader(subsets["train"], cache, images, batch_size, device)
    settings = (weight,) if weight is not None else sorted((*WEIGHTS, *INTERMEDIATE))
    result, seconds = [], []
    for w in settings:
        start = perf_counter()
        logits, phenotype, test_embedding, _, _ = forward_batches(model, test_batch, w, device)
        seconds.append(perf_counter() - start)
        if not torch.isfinite(logits).all() or not torch.isfinite(phenotype).all():
            raise FloatingPointError(f"Nonfinite predictions at {run}")
        scores = torch.column_stack((logits.sigmoid(), phenotype)).numpy()
        method = "fixed" if weight is not None else "conditioned"
        result.append(anchor_frame(subsets["test"], scores, fold, seed, w, method))
        if weight is not None:
            _, _, train_embedding, _, _ = forward_batches(model, train_batch, w, device)
            scores = linear_probes(train_embedding.numpy(), subsets["train"], test_embedding.numpy())
            result.append(anchor_frame(subsets["test"], scores, fold, seed, w, "probe"))
    frame = pd.concat(result, ignore_index=True)
    atomic_csv(frame, output)
    save_json(run / "done.json", {**timing, "inference_seconds_per_patient": float(np.mean(seconds)) / len(subsets["test"])})
    print(f"Anchor seed={seed} fold={fold} {model_name} complete ({timing['train_seconds']:.1f}s)", flush=True)
    return frame


def controls(subsets, fold):
    train, test = subsets["train"], subsets["test"]
    frequency, median, _ = control_parameters(train)
    mol_train, y = targets(train)
    mol_test, _ = targets(test)
    molecular = phenotype_prediction(molecular_features(mol_train), y, molecular_features(mol_test))
    base = np.repeat(frequency[None, :], len(test), axis=0)
    return pd.concat([anchor_frame(test, np.column_stack((base, target)), fold, 42, np.nan, name)
                      for name, target in (("frequency_median", np.full(len(test), median)),
                                           ("frequency_molecular", molecular))], ignore_index=True)


def curve_plot(summaries, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), layout="constrained")
    colors = ("#15558c", "#bd4c24", "#16835d", "#794f98", "#945f21")
    for ax, method, title in zip(axes, ("fixed", "probe"), ("Jointly trained heads", "Train-only linear probes")):
        points = [summaries[method][f"{w:.2f}"] for w in WEIGHTS]
        ap = np.array([p["anchor_ap"]["estimate"] for p in points])
        mae = 100 * np.array([p["phenotype_mae"]["estimate"] for p in points])
        ax.plot(ap, mae, color="#8996a0", alpha=.65)
        for index, (w, metrics, color) in enumerate(zip(WEIGHTS, points, colors)):
            x_ci = metrics["anchor_ap"]["ci95"]
            y_ci = metrics["phenotype_mae"]["ci95"]
            if x_ci:
                ax.hlines(mae[index], *x_ci, color=color, alpha=.5)
            if y_ci:
                ax.vlines(ap[index], *(100 * np.array(y_ci)), color=color, alpha=.5)
            ax.scatter(ap[index], mae[index], color=color, zorder=3, label=f"w={w:.2f}")
        ax.set(title=title, xlabel="Molecular macro-AP (higher is better)",
               ylabel="Phenotype MAE, percentage points (lower is better)")
        ax.grid(alpha=.17)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=len(WEIGHTS), frameon=False)
    fig.savefig(out / "anchor_phenotype_curve.pdf")
    fig.savefig(out / "anchor_phenotype_curve.png", dpi=200)
    plt.close(fig)


def groups(frame, method, seed):
    subset = frame[(frame["method"] == method) & (frame.seed == seed)]
    return {w: rows.sort_values("subject_id") for w, rows in subset.groupby("weight", sort=True)}


def dominates_endpoints(points):
    pairs = [(p["anchor_ap"]["estimate"], p["phenotype_mae"]["estimate"]) for p in points]
    if any(a is None or b is None for a, b in pairs):
        return False
    ends = (pairs[0], pairs[-1])
    return any(all(ap >= ea and mae <= em and (ap > ea or mae < em) for ea, em in ends)
               for ap, mae in pairs[1:-1])


def pilot_summary(frame, manifest, repetitions):
    pilot = {name: groups(frame, name, 42) for name in ("fixed", "probe")}
    reference = pilot["fixed"][WEIGHTS[0]]
    if sorted(reference.subject_id) != sorted(row["subject_id"] for row in manifest["rows"]):
        raise ValueError("Pilot predictions do not cover the prepared cohort")
    folds = {sid: fold for fold, split in enumerate(manifest["splits"]) for sid in split["test"]}
    if not reference.fold.eq(reference.subject_id.map(folds)).all():
        raise ValueError("Pilot patient assignments do not match the manifest")
    bootstrap = patient_draws(len(reference), repetitions)
    summaries, contrasts, same_cohort = {}, {}, {}
    phenotype_available = reference.phenotype.notna().to_numpy()
    subset_bootstrap = patient_draws(int(phenotype_available.sum()), repetitions)
    for method, predictions in pilot.items():
        for prediction in predictions.values():
            check_pairs(reference, prediction, TARGET_COLS)
        summaries[method] = {f"{w:.2f}": summarize_anchor(rows, bootstrap) for w, rows in predictions.items()}
        contrast = endpoint_contrast(predictions[WEIGHTS[-1]], predictions[WEIGHTS[0]], bootstrap)
        contrasts[method] = {key: interval(v) for key, v in contrast.items()}
        subset_contrast = endpoint_contrast(predictions[WEIGHTS[-1]].iloc[phenotype_available],
                                            predictions[WEIGHTS[0]].iloc[phenotype_available], subset_bootstrap)
        same_cohort[method] = {key: interval(v) for key, v in subset_contrast.items()}
    for name in ("frequency_median", "frequency_molecular"):
        rows = frame[frame.method == name].sort_values("subject_id")
        check_pairs(reference, rows, TARGET_COLS)
        summaries[name] = summarize_anchor(rows, bootstrap)
    dominates = dominates_endpoints([summaries["fixed"][f"{w:.2f}"] for w in WEIGHTS])
    gate = all(tradeoff_pass(v) for v in contrasts.values()) and not dominates
    by_id = {r["subject_id"]: r for r in manifest["rows"]}
    statuses = pd.Series([r["phenotype_status"] for r in manifest["rows"]]).value_counts().to_dict()
    usable = []
    for fold, split in enumerate(manifest["splits"]):
        counts = {"fold": fold}
        for role, ids in split.items():
            anchors, phenotypes = targets([by_id[sid] for sid in ids])
            counts[role] = {"n": len(ids), "phenotype_n": int(np.isfinite(phenotypes).sum()),
                            **{field + "_n": int(n) for field, n in zip(TARGET_COLS[:-1], np.isfinite(anchors).sum(axis=0))}}
        usable.append(counts)
    return {"summaries": summaries, "endpoints_high_minus_low": contrasts, "pilot_gate_passed": gate,
            "endpoints_on_phenotype_subset": same_cohort, "phenotype_status_counts": statuses,
            "interior_dominates_both_endpoints": dominates, "counts_by_fold": usable}, bootstrap


def preference_summary(frame, bootstrap):
    fixed = groups(frame, "fixed", 42)
    condition = groups(frame, "conditioned", 42)
    interpolated = dict(fixed)
    for w in INTERMEDIATE:
        lo = max(v for v in WEIGHTS if v < w)
        hi = min(v for v in WEIGHTS if v > w)
        interpolated[w] = interpolate_predictions(fixed[lo], fixed[hi], w, lo, hi)
    summary = {}
    for name, points in (("conditioned", condition), ("fixed", fixed), ("interpolation", interpolated)):
        for rows in points.values():
            check_pairs(fixed[WEIGHTS[0]], rows, TARGET_COLS)
        values = {w: summarize_anchor(rows, bootstrap) for w, rows in sorted(points.items())}
        benefits = [(v["anchor_ap"]["estimate"], 1 - v["phenotype_mae"]["estimate"]) for v in values.values()]
        front = [w for w, (a, b) in zip(values, benefits) if not
                 any(x >= a and y >= b and (x > a or y > b) for x, y in benefits)]
        trend = np.diff(benefits, axis=0)
        summary[name] = {"weights": values, "hypervolume": hypervolume(benefits), "nondominated_weights": front,
                         "anchor_correct_direction_fraction": float((trend[:, 0] >= 0).mean()),
                         "phenotype_correct_direction_fraction": float((trend[:, 1] <= 0).mean())}
    for w in INTERMEDIATE:
        interpolated[w]["weight"] = w
        interpolated[w]["method"] = "interpolation"
    return summary, [interpolated[w] for w in INTERMEDIATE]


def run_anchor(out, device, epochs, batch_size, repetitions):
    manifest, signature = manifest_data(out)
    subsets = [{role: [row for row in manifest["rows"] if row["subject_id"] in ids] for role, ids in split.items()}
               for split in manifest["splits"]]
    if any(not rows for split in subsets for rows in split.values()):
        raise ValueError("An anchor-phenotype split is empty")
    cache, frames = {}, []
    for fold, split in enumerate(subsets):
        frames.append(controls(split, fold))
        for weight in WEIGHTS:
            frames.append(train_fold(out, split, fold, 42, weight, device, epochs, batch_size, cache, signature))
    data = pd.concat(frames, ignore_index=True)
    report, bootstrap = pilot_summary(data, manifest, repetitions)
    report["preference_stage"] = "replication_pending" if report["pilot_gate_passed"] else "pilot_gate_not_met"
    save_json(out / "anchor_summary.json", report)
    atomic_csv(data, out / "anchor_oof_predictions.csv")
    curve_plot(report["summaries"], out)
    if not report["pilot_gate_passed"]:
        return report
    for seed in (43, 44):
        for fold, split in enumerate(subsets):
            for weight in (WEIGHTS[0], WEIGHTS[-1]):
                frames.append(train_fold(out, split, fold, seed, weight, device, epochs, batch_size, cache, signature))
    data = pd.concat(frames, ignore_index=True)
    confirmed, confirmation = True, {}
    for method in ("fixed", "probe"):
        all_seeds = []
        ref = groups(data, method, 42)[WEIGHTS[0]]
        for seed in (42, 43, 44):
            predictions = groups(data, method, seed)
            check_pairs(ref, predictions[WEIGHTS[0]], TARGET_COLS)
            difference = endpoint_contrast(predictions[WEIGHTS[-1]], predictions[WEIGHTS[0]], bootstrap)
            all_seeds.append(difference)
        seed_mean = {key: interval(np.mean([v[key] for v in all_seeds], axis=0)) for key in all_seeds[0]}
        confirmation[method] = {"mean_contrast": seed_mean, "per_seed": {str(seed): {k: interval(v) for k, v in diff.items()}
                                                                       for seed, diff in zip((42, 43, 44), all_seeds)}}
        confirmed &= tradeoff_pass(seed_mean) and all(v[0] > 0 for diff in all_seeds for v in diff.values())
    report.update(confirmation=confirmation,
                  preference_stage="conditioned_model_pending" if confirmed else "replication_gate_not_met")
    save_json(out / "anchor_summary.json", report)
    atomic_csv(data, out / "anchor_oof_predictions.csv")
    if not confirmed:
        return report
    for fold, split in enumerate(subsets):
        frames.append(train_fold(out, split, fold, 42, None, device, epochs, batch_size, cache, signature))
    data = pd.concat(frames, ignore_index=True)
    preference, interpolated = preference_summary(data, bootstrap)
    costs = {}
    for name, pattern in (("fixed", "w*"), ("conditioned", "conditioned")):
        runs = [json.loads(path.read_text(encoding="utf-8")) for path in (out / "runs" / "seed42").glob(f"fold*/{pattern}/done.json")]
        costs[name] = {"runs": len(runs), "total_train_seconds": sum(r["train_seconds"] for r in runs),
                       "mean_inference_seconds_per_patient": float(np.mean([r["inference_seconds_per_patient"] for r in runs])),
                       "parameters_per_checkpoint": runs[0]["parameters"]}
    costs["interpolation"] = {"checkpoints_per_intermediate_prediction": 2,
                              "mean_inference_seconds_per_patient_estimate": 2 * costs["fixed"]["mean_inference_seconds_per_patient"]}
    report.update(preference_stage="evaluated", preference=preference, costs=costs)
    save_json(out / "anchor_summary.json", report)
    atomic_csv(pd.concat([data, *interpolated], ignore_index=True), out / "anchor_oof_predictions.csv")
    return report
