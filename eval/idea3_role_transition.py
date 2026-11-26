"""Patient-paired estimates for the UTSW 1p/19q role intervention."""

import numpy as np
import pandas as pd

from glioma.eval.idea3_rule_mri_swap import MODELS, PAIR_KEYS, PATIENT_KEYS, VIOLATION_TOLERANCE


STATES = ("O", "C", "A")
PATIENT_FIELDS = ["label", "archive_active", "codeletion_measured"]


def _validate(frame):
    if set(frame.model) != set(MODELS) or set(frame.state) != set(STATES):
        raise ValueError("Expected four models and O/C/A predictions")
    if frame[PATIENT_KEYS + PATIENT_FIELDS].isna().any(axis=None):
        raise ValueError("Missing patient identifier, reference label, or cohort flag")
    if frame.groupby("subject_id").fold.nunique().ne(1).any():
        raise ValueError("A patient appears in multiple test folds")
    if frame.duplicated(PATIENT_KEYS + ["model", "state"]).any():
        raise ValueError("Duplicate patient-model-state prediction")
    patients = frame.groupby(PATIENT_KEYS, sort=False)
    if patients.size().ne(len(MODELS) * len(STATES)).any():
        raise ValueError("Missing paired state or model for at least one patient")
    if patients[PATIENT_FIELDS].nunique().ne(1).any(axis=None):
        raise ValueError("Patient reference label or cohort flag changed across predictions")
    if not frame.label.isin([0, 1, 2]).all():
        raise ValueError("Reference grade must be encoded as 0/1/2")
    for flag in ("archive_active", "codeletion_measured", "active"):
        if not frame[flag].isin([True, False]).all():
            raise ValueError(f"Invalid indicator: {flag}")
    if (frame.archive_active & (~frame.codeletion_measured | frame.label.eq(2))).any():
        raise ValueError("Active cases require a measured field and feasible reference grade")
    for columns in (["p2", "p3", "p4"], ["raw_p2", "raw_p3", "raw_p4"]):
        values = frame[columns].to_numpy(dtype=float)
        if (not np.isfinite(values).all() or np.any(values < -VIOLATION_TOLERANCE)
                or np.any(values > 1 + VIOLATION_TOLERANCE)
                or not np.allclose(values.sum(axis=1), 1, atol=1e-5, rtol=0)):
            raise ValueError(f"Invalid probabilities in {columns}")
    expected_active = frame.state.eq("A") & frame.archive_active
    expected_allowed = np.where(expected_active, "2|3", "2|3|4")
    if not np.array_equal(frame.active, expected_active) or not np.array_equal(frame.allowed_grades, expected_allowed):
        raise ValueError("Observed rule/allowed grades disagree with expected state-specific eligibility")
    if frame.loc[expected_active, "p4"].abs().gt(VIOLATION_TOLERANCE).any():
        raise ValueError("Projected Grade 4 probability exceeds the rule tolerance")
    if "status" in frame and not np.array_equal(frame.status, np.where(expected_active, 0, 1)):
        raise ValueError("Unavailable or unexpected prediction status")
    if "image_subject_id" in frame and not frame.subject_id.eq(frame.image_subject_id).all():
        raise ValueError("The soft-field analysis requires own MRI in every state")


def _paired_effects(advantages, repetitions, seed):
    """Resample paired patients within their OOF folds, sharing draws across columns."""
    if repetitions < 1 or advantages.empty:
        raise ValueError("Positive bootstrap repetitions and at least one paired patient required")
    if list(advantages.index.names) != PATIENT_KEYS or advantages.index.has_duplicates:
        raise ValueError("Expected one effect per fold/patient")
    if (advantages.index.get_level_values("subject_id").has_duplicates
            or advantages.index.to_frame(index=False).isna().any(axis=None)
            or not np.isfinite(advantages.to_numpy(dtype=float)).all()):
        raise ValueError("Nonfinite effect or invalid OOF patient identity")
    advantages = advantages.sort_index()
    folds = advantages.index.get_level_values("fold").to_numpy()
    rng = np.random.default_rng(seed)
    fold_rows = {fold: np.flatnonzero(folds == fold) for fold in np.unique(folds)}
    draws = {fold: rng.choice(indices, size=(repetitions, len(indices)))
             for fold, indices in fold_rows.items()}
    estimates = {}
    for model in advantages:
        values = advantages[model].to_numpy()
        boot = sum(values[indices].sum(axis=1) for indices in draws.values()) / len(values)
        by_fold = {str(fold): {"n": len(indices), "estimate": float(values[indices].mean())}
                   for fold, indices in fold_rows.items()}
        dropped = {str(fold): float(values[folds != fold].mean()) for fold in fold_rows if len(fold_rows) > 1}
        estimates[model] = {
            "estimate": float(values.mean()), "n": len(values),
            "ci95": np.quantile(boot, [.025, .975]).tolist(), "by_fold": by_fold,
            "leave_one_fold_out": dropped,
            "leave_one_fold_out_range": [min(dropped.values()), max(dropped.values())] if dropped else None,
        }
    return estimates


def paired_effect(values, repetitions=10_000, seed=2026):
    """Summarize a series of paired effects indexed by (fold, subject_id)."""
    return _paired_effects(values.to_frame("effect"), repetitions, seed)["effect"]


def _brier3(frame):
    labels = frame.label.to_numpy(dtype=int)
    projected = frame[["p2", "p3", "p4"]].to_numpy(dtype=float)
    return ((projected - np.eye(3)[labels]) ** 2).mean(axis=1)


def analyze_states(frame, repetitions=10_000, seed=2026):
    """Return one record per patient/model and the measured-cohort O minus C Brier contrast."""
    if repetitions < 1:
        raise ValueError("Bootstrap repetitions must be positive")
    _validate(frame)
    keys = ["model", *PATIENT_KEYS]
    wide = frame.assign(brier3=_brier3(frame)).pivot(index=keys, columns="state", values="brier3")
    patients = frame.loc[frame.state.eq("O"), keys + PATIENT_FIELDS].set_index(keys)
    patients = patients.join(wide.rename(columns={s: f"brier3_{s}" for s in STATES})).reset_index()
    patients["soft_advantage"] = patients.brier3_O - patients.brier3_C
    measured = patients.loc[patients.codeletion_measured]
    if measured.empty:
        raise ValueError("No patients with measured 1p/19q")
    effects = _paired_effects(measured.pivot(index=PATIENT_KEYS, columns="model", values="soft_advantage"),
                              repetitions, seed)
    cohort = patients[patients.model.eq("read_only_residual")]
    active = cohort[cohort.archive_active]
    models = {}
    for model in MODELS:
        rows = measured[measured.model.eq(model)]
        restricted = frame[frame.model.eq(model) & frame.archive_active]
        p4_profile = {state: {"n": len(group), "mean_raw_p4": float(group.raw_p4.mean()),
                              "mean_projected_p4": float(group.p4.mean())}
                      for state, group in restricted.groupby("state")}
        models[model] = {
            "measured_brier3": {s: float(rows[f"brier3_{s}"].mean()) for s in STATES},
            "soft_advantage": effects[model],
            "active_cohort_p4": p4_profile,
        }
    return patients, {
        "primary": "models.read_only_residual.soft_advantage",
        "primary_definition": "Mean patient Brier3(O) minus Brier3(C) for measured 1p/19q; positive favors C",
        "n_patients": len(cohort), "n_measured": int(cohort.codeletion_measured.sum()),
        "measured_reference_grades": {str(g + 2): int((cohort.codeletion_measured & cohort.label.eq(g)).sum())
                                      for g in range(3)},
        "structure": {
            "n_active": len(active), "n_inactive": len(cohort) - len(active),
            "active_by_fold": {str(fold): int(n) for fold, n in cohort.groupby("fold").archive_active.sum().items()},
            "states_with_expected_allowed_set": list(STATES),
            "allowed_set_mismatches": 0, "forbidden_mass_violations": 0,
            "forbidden_mass_tolerance": VIOLATION_TOLERANCE,
        },
        "bootstrap_repetitions": repetitions, "bootstrap_seed": seed,
        "models": models,
    }


def compare_swap_states(a_pairs, c_pairs, repetitions=10_000, seed=2026):
    """Compare MRI matching advantages in A and C on the identical binary-grade cohort using Brier3."""
    keys = ["model", *PAIR_KEYS]
    a = a_pairs.set_index(keys).sort_index()
    c = c_pairs.set_index(keys).sort_index()
    if a.empty or not a.index.is_unique or not a.index.equals(c.index) or not a.label.equals(c.label):
        raise ValueError("Expected identical, unique and label-matched MRI pairs in both states")
    if set(a.index.get_level_values("model")) != set(MODELS):
        raise ValueError("Missing MRI swap model")
    if not a.active.all() or c.active.any() or not a.label.isin([0, 1]).all():
        raise ValueError("Expected active A and unrestricted C on the same Grade 2/3 cohort")
    scores = pd.DataFrame({"brier3_A": _brier3(a), "brier3_C": _brier3(c)}, index=a.index).reset_index()
    own_mask = scores.subject_id.eq(scores.image_subject_id)
    group_keys = ["model", *PATIENT_KEYS]
    metrics = ["brier3_A", "brier3_C"]
    own = scores[own_mask].set_index(group_keys)[metrics].add_prefix("own_")
    donors = scores[~own_mask].groupby(group_keys, sort=True)
    donor = donors[metrics].mean().add_prefix("mean_donor_")
    if not own.index.sort_values().equals(donor.index) or not own.index.is_unique:
        raise ValueError("Each recipient requires one own MRI and at least one donor")
    patients = own.join(donor)
    patients["n_donors"] = donors.size()
    for state in ("A", "C"):
        patients[f"advantage_{state}"] = patients[f"mean_donor_brier3_{state}"] - patients[f"own_brier3_{state}"]
    patients["a_minus_c_advantage"] = patients.advantage_A - patients.advantage_C
    effects = patients.a_minus_c_advantage.unstack("model")
    return patients.reset_index(), {
        "n_patients": len(effects),
        "definition": "A minus C of mean-donor minus own MRI advantage using Brier3 in both states",
        "models": _paired_effects(effects, repetitions, seed),
    }


def report_markdown(summary):
    mode = summary.get("study_mode", "full")
    lines = [
        "# 1p/19q role intervention", "",
        f"Study mode: **{mode}**.",
    ]
    if "smoke" in mode.lower():
        lines += ["", "Short technical run on a selected subset. Its numerical estimates are for pipeline checks only."]
    lines += ["",
        f"OOF patients: {summary['n_patients']}. Measured 1p/19q: {summary['n_measured']}. "
        f"Rule-active in A: {summary['structure']['n_active']}.",
        "", "## Frozen-checkpoint sensitivity in the measured-field cohort", "",
        "O hides the field, C exposes its value as a soft cue, A grants rule eligibility. "
        "The primary contrast is O minus C for the frozen read-only residual model. "
        "All four frozen readouts receive a neutral scope input even though scope was present during training. "
        "These estimates measure sensitivity to a simulated intervention on finalized report values. "
        "Brier3 averages squared error across the three grades. "
        "The intervals resample patients within folds and condition on the fitted checkpoints.",
        "", "| Model | O Brier3 | C Brier3 | A Brier3 | O minus C [95% CI] |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        row = summary["models"][model]
        means, effect = row["measured_brier3"], row["soft_advantage"]
        low, high = effect["ci95"]
        lines.append(f"| {model} | {means['O']:.4f} | {means['C']:.4f} | {means['A']:.4f} "
                     f"| {effect['estimate']:+.4f} [{low:+.4f}, {high:+.4f}] |")
    lines += ["", "## Soft-field contrast by fold", "",
              "| Model | Fold estimates (n) | Leave-one-fold-out range |", "|---|---|---:|"]
    for model in MODELS:
        effect = summary["models"][model]["soft_advantage"]
        folds = ", ".join(f"{fold}: {row['estimate']:+.4f} ({row['n']})"
                          for fold, row in effect["by_fold"].items())
        limits = effect["leave_one_fold_out_range"]
        interval = f"[{limits[0]:+.4f}, {limits[1]:+.4f}]" if limits else "N/A"
        lines.append(f"| {model} | {folds} | {interval} |")
    if "paired_state_linear_readout" in summary:
        effect = summary["paired_state_linear_readout"]
        low, high = effect["ci95"]
        lines += ["", "## Fold-trained linear readout", "",
                  f"The readout saw paired O and C inputs in training. Its held-out O minus C Brier3 difference "
                  f"is {effect['estimate']:+.4f} [{low:+.4f}, {high:+.4f}] across {effect['n']} measured patients."]
    if "alternate_minus_own_field_brier3" in summary:
        lines += ["", "## Alternate measured-field value in C", "",
                  "Positive values favor the recipient's recorded value.",
                  "", "| Model | Paired patients | Alternate minus own Brier3 [95% CI] |", "|---|---:|---:|"]
        for model, effect in summary["alternate_minus_own_field_brier3"].items():
            low, high = effect["ci95"]
            lines.append(f"| {model} | {effect['n']} | {effect['estimate']:+.4f} [{low:+.4f}, {high:+.4f}] |")
    if summary["structure"]["n_active"]:
        lines += ["", "## Rule-eligible patients across states", "",
                  "All eligible recipients allow grades 2/3/4 in O and C, and grades 2/3 in A.",
                  "", "| Model | State | Patients | Mean raw Grade 4 probability | Mean projected Grade 4 probability |",
                  "|---|---|---:|---:|---:|"]
        for model in MODELS:
            for state in STATES:
                row = summary["models"][model]["active_cohort_p4"][state]
                lines.append(f"| {model} | {state} | {row['n']} | {row['mean_raw_p4']:.4f} "
                             f"| {row['mean_projected_p4']:.4f} |")
    if "a_mri_swap" in summary:
        swap = summary["a_mri_swap"]
        lines += ["", "## A-state MRI swaps", "",
                  f"{swap['n_patients']} recipient patients. Average donors within recipient before comparing. "
                  "Scores use binary Grade 2/3 Brier.", "",
                  "| Model | Own Brier | Mean donor Brier | Donor minus own [95% CI] |",
                  "|---|---:|---:|---:|"]
        for model in MODELS:
            row = swap["models"][model]
            effect = row["own_advantage"]
            low, high = effect["ci95"]
            lines.append(f"| {model} | {row['own_brier']:.4f} | {row['donor_brier']:.4f} "
                         f"| {effect['estimate']:+.4f} [{low:+.4f}, {high:+.4f}] |")
    if "a_minus_c_swap" in summary:
        section = summary["a_minus_c_swap"]
        lines += ["", "## Exploratory MRI matching interaction", "",
                  f"{section['n_patients']} recipients. Both states use three-grade Brier and identical donors.",
                  "", "| Model | A minus C of donor minus own [95% CI] |", "|---|---:|"]
        for model in MODELS:
            effect = section["models"][model]
            low, high = effect["ci95"]
            lines.append(f"| {model} | {effect['estimate']:+.4f} [{low:+.4f}, {high:+.4f}] |")
    return "\n".join(lines) + "\n"
