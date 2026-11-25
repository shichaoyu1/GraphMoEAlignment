"""Paired, patient-level analysis of rule-matched MRI swaps."""

import numpy as np
import pandas as pd


MODELS = ("molecular_projected", "learned_joint", "acf", "read_only_residual")
EXPECTED_FOLD_SIZES = (12, 12, 11, 11, 12)
VIOLATION_TOLERANCE = 1e-6
PAIR_KEYS = ["fold", "subject_id", "image_subject_id"]
PATIENT_KEYS = ["fold", "subject_id"]


def validate_pairs(frame, patients):
    """Require one clean prediction and every directed swap for each model."""
    expected = {
        (fold, recipient, donor)
        for fold, ids in patients.items()
        for recipient in ids
        for donor in ids
    }
    if set(frame.model) != set(MODELS):
        raise ValueError("MRI swap predictions are missing a model")
    for model, group in frame.groupby("model"):
        actual = set(group[PAIR_KEYS].itertuples(index=False, name=None))
        if actual != expected or len(group) != len(expected):
            raise ValueError(f"Incomplete or duplicate MRI pairs for {model}")
    if not frame.label.isin([0, 1]).all() or not frame.allowed_grades.eq("2|3").all():
        raise ValueError("MRI swap requires active Grade 2/3 patients with feasible references")
    if frame.groupby(PATIENT_KEYS).label.nunique().ne(1).any():
        raise ValueError("Reference grade changed across MRI pairs")


def patient_results(frame):
    frame = frame.copy()
    frame["condition"] = np.where(frame.subject_id == frame.image_subject_id, "own", "donor")
    # Binary Brier on the conditional probability of Grade 3.
    grade3 = frame.p3 / (frame.p2 + frame.p3)
    frame["rule_brier"] = (grade3 - frame.label) ** 2
    if not np.isfinite(frame.rule_brier).all():
        raise ValueError("Nonfinite rule-conditional Brier")
    keys = ["model", *PATIENT_KEYS]
    own = frame[frame.condition == "own"].set_index(keys)
    donor = frame[frame.condition == "donor"].copy()
    donor = donor.join(own[["raw_p4"]].rename(columns={"raw_p4": "own_raw_p4"}), on=keys)
    donor["raw_p4_abs_change"] = (donor.raw_p4 - donor.own_raw_p4).abs()
    donor["projected_violation"] = donor.p4 > VIOLATION_TOLERANCE
    grouped = donor.groupby(keys)
    patients = grouped.agg(n_donors=("rule_brier", "size"), donor_brier=("rule_brier", "mean"),
                           raw_p4_mean_abs_change=("raw_p4_abs_change", "mean"),
                           raw_p4_max_abs_change=("raw_p4_abs_change", "max"),
                           donor_projected_violation_rate=("projected_violation", "mean"))
    patients = patients.join(own[["reference_grade", "rule_brier", "raw_p4", "p4"]].rename(
        columns={"rule_brier": "own_brier", "raw_p4": "own_raw_p4", "p4": "own_projected_p4"}))
    patients["own_projected_violation"] = patients.own_projected_p4 > VIOLATION_TOLERANCE
    patients["own_advantage"] = patients.donor_brier - patients.own_brier
    return frame, patients.reset_index()


def summarize(patients, repetitions=10_000, seed=2026):
    """Use the same fold-stratified patient draws for every model and contrast."""
    own = patients.pivot(index=PATIENT_KEYS, columns="model", values="own_brier")
    advantages = patients.pivot(index=PATIENT_KEYS, columns="model", values="own_advantage")
    folds = own.index.get_level_values("fold").to_numpy()
    rng = np.random.default_rng(seed)
    draws = {
        fold: rng.choice(indices, size=(repetitions, len(indices)))
        for fold in np.unique(folds)
        for indices in [np.flatnonzero(folds == fold)]
    }

    def paired(series):
        values = series.to_numpy()
        boot = sum(values[idx].sum(axis=1) for idx in draws.values()) / len(values)
        by_fold = {str(fold): {"n": int((folds == fold).sum()),
                               "estimate": float(values[folds == fold].mean())} for fold in draws}
        leave_one_out = [float(values[folds != fold].mean()) for fold in draws] if len(draws) > 1 else []
        return {"n": len(values), "estimate": float(values.mean()),
                "ci95": np.quantile(boot, [.025, .975]).tolist(), "by_fold": by_fold,
                "leave_one_fold_out_range": [min(leave_one_out), max(leave_one_out)] if leave_one_out else None}

    metrics = {}
    for model in MODELS:
        rows = patients[patients.model == model]
        metrics[model] = {
            "n_pairs": int(rows.n_donors.sum()),
            "own_brier": float(rows.own_brier.mean()),
            "donor_brier": float(rows.donor_brier.mean()),
            "own_advantage": paired(advantages[model]),
            "raw_p4_mean_abs_change": float(rows.raw_p4_mean_abs_change.mean()),
            "raw_p4_max_abs_change": float(rows.raw_p4_max_abs_change.max()),
            "own_projected_violation_rate": float(rows.own_projected_violation.mean()),
            "donor_projected_violation_rate": float(rows.donor_projected_violation_rate.mean()),
        }
    return {
        "n_patients": len(own), "bootstrap_repetitions": repetitions, "bootstrap_seed": seed,
        "primary": "read_only_residual.own_advantage",
        "models": metrics,
        "own_brier_contrasts": {
            control: paired(own[control] - own.read_only_residual)
            for control in ("molecular_projected", "learned_joint")
        },
    }


def report_markdown(report):
    def effect(row):
        low, high = row["ci95"]
        return f"{row['estimate']:+.4f} [{low:+.4f}, {high:+.4f}]"

    lines = [
        "# Rule-matched MRI swaps", "",
        f"Patient-weighted results for {report['n_patients']} rule-active patients. "
        "Brier is lower when predictions are better. Positive differences favor own MRI "
        "or the read-only residual model. Intervals condition on the saved models and donor pool.", "",
        "| Model | Own MRI Brier | Mean donor Brier | Donor minus own [95% CI] | "
        "Mean absolute raw Grade 4 change | Projected violation rate (own / donor) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, row in report["models"].items():
        lines.append(f"| {model} | {row['own_brier']:.4f} | {row['donor_brier']:.4f} | "
                     f"{effect(row['own_advantage'])} | {row['raw_p4_mean_abs_change']:.2%} | "
                     f"{row['own_projected_violation_rate']:.2%} / {row['donor_projected_violation_rate']:.2%} |")
    lines += ["", "## Own-MRI Brier contrasts", "",
              "| Control minus read-only residual | Paired patients | Difference [95% CI] |", "|---|---:|---:|"]
    for control, row in report["own_brier_contrasts"].items():
        lines.append(f"| {control} | {row['n']} | {effect(row)} |")
    lines += ["", "## Fold sensitivity", "",
              "| Comparison | Fold estimates (0–4) | Leave-one-fold-out range |", "|---|---|---:|"]
    comparisons = {model: row["own_advantage"] for model, row in report["models"].items()}
    comparisons.update({f"{control} minus read-only residual (own MRI)": row
                        for control, row in report["own_brier_contrasts"].items()})
    for name, row in comparisons.items():
        folds = ", ".join(f"{key}: {value['estimate']:+.4f} (n={value['n']})"
                          for key, value in row["by_fold"].items())
        limits = row["leave_one_fold_out_range"]
        leave_one_out = f"[{limits[0]:+.4f}, {limits[1]:+.4f}]" if limits else "N/A"
        lines.append(f"| {name} | {folds} | {leave_one_out} |")
    return "\n".join(lines) + "\n"
