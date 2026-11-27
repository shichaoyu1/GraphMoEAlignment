"""Patient-level scoring for the A-state MRI mixture comparison."""

import numpy as np
import pandas as pd

from glioma.cli.idea3_rule_mri_swap import PROBABILITIES, verify_invariance
from glioma.eval.idea3_role_transition import paired_effect
from glioma.eval.idea3_rule_mri_swap import (
    PAIR_KEYS, PATIENT_KEYS, VIOLATION_TOLERANCE, patient_results, validate_pairs,
)
from glioma.objectives.idea3_mri_mix import BUDGETS, READ_ONLY


def frozen_probabilities(frame, rows, fold):
    labels = {row["subject_id"]: row["label"] for row in rows}
    validate_pairs(frame, {fold: list(labels)})
    if not frame.label.eq(frame.subject_id.map(labels)).all():
        raise ValueError("Frozen reference grades differ from the manifest")
    for cols in (PROBABILITIES[:3], PROBABILITIES[3:]):
        values = frame[cols].to_numpy()
        if (not np.isfinite(values).all() or values.min() < -VIOLATION_TOLERANCE
                or not np.allclose(values.sum(axis=1), 1, rtol=0, atol=1e-5)):
            raise ValueError("Invalid frozen grade probabilities")
    forbidden = float(frame.p4.abs().max())
    if forbidden > VIOLATION_TOLERANCE:
        raise ValueError("An A-state prediction retains forbidden Grade 4 mass")
    scored, patients = patient_results(frame)
    verify_invariance(scored)
    scored["q"] = scored.p3 / (scored.p2 + scored.p3)
    wide = scored.pivot(index=[*PAIR_KEYS, "label", "condition"], columns="model", values="q").reset_index()
    raw_p4 = scored[scored.model.eq(READ_ONLY)].set_index(PAIR_KEYS).raw_p4
    wide = wide.join(raw_p4.rename("read_only_raw_p4"), on=PAIR_KEYS)
    wide = wide.join(scored.groupby(PAIR_KEYS).p4.max().rename("max_projected_p4"), on=PAIR_KEYS)
    # Use exactly the recipient's report-only prediction for every pairing.
    reports = wide[wide.condition.eq("own")].set_index("subject_id").molecular_projected
    wide["molecular_projected"] = wide.subject_id.map(reports)
    audit = dict(n_patients=len(rows), n_swaps=int(wide.condition.eq("donor").sum()),
                 max_forbidden_probability=forbidden,
                 read_only_raw_p4_max_change=float(patients[patients.model.eq(READ_ONLY)].raw_p4_max_abs_change.max()))
    return wide, audit


def score_outer(rows, fits, prior):
    controls = {"report_only": "molecular_projected", "read_only_full": READ_ONLY,
                "free_full": "learned_joint", "acf_full": "acf", "train_grade3_prior": "train_grade3_prior"}
    rows = rows.assign(train_grade3_prior=prior)
    fixed = [dict(method=m, source=s, budget=budget, a=1., b=1.)
             for budget in BUDGETS for m, s in controls.items()]
    report = rows.molecular_projected.to_numpy()
    frames = []
    for fit in [*fits, *fixed]:
        source = fit["source"]
        delta = rows[source].to_numpy() - report
        weight = fit["a"] + (fit["b"] - fit["a"]) * abs(delta)
        q = rows[source].to_numpy() if fit["method"] in controls else report + weight * delta
        item = rows[[*PAIR_KEYS, "label", "condition"]].assign(
            budget=fit["budget"], method=fit["method"], source=source, report_p3=report,
            source_p3=rows[source], mri_weight=0. if source in ("molecular_projected", "train_grade3_prior") else weight,
            p2=1-q, p3=q, p4=0., brier=(q - rows.label) ** 2,
            brier_minus_report=(q - report) * (q + report - 2 * rows.label))
        frames.append(item)
    return pd.concat(frames, ignore_index=True)


def summarize(predictions, repetitions, smoke=False):
    grouped = predictions.groupby(["budget", "method", *PATIENT_KEYS, "condition"]).brier
    patients = grouped.mean().unstack().rename(columns={"own": "own_brier", "donor": "swap_brier"}).reset_index()
    # Accumulate within-pair differences so an exactly report-only policy has
    # exactly zero excess even at the binding zero-budget decision boundary.
    keys = ["budget", "method", *PATIENT_KEYS]
    delta = predictions[predictions.condition.eq("donor")].groupby(keys).brier_minus_report.mean()
    patients = patients.join(delta.rename("swap_minus_report"), on=keys)
    by_budget, curves = {}, {}
    for name in BUDGETS:
        group = patients[patients.budget.eq(name)]
        own, swap = [group.pivot(index=PATIENT_KEYS, columns="method", values=column)
                     for column in ("own_brier", "swap_brier")]
        excess = group.pivot(index=PATIENT_KEYS, columns="method", values="swap_minus_report")
        by_budget[name] = (own, swap)
        curves[name] = {
            m: {"own_brier": paired_effect(own[m], repetitions),
                "swap_brier": paired_effect(swap[m], repetitions),
                "swap_minus_report": paired_effect(excess[m], repetitions)}
            for m in own
        }
    zero_own, zero_swap = by_budget["0"]
    gain = paired_effect(zero_own.constant - zero_own.adaptive, repetitions)
    excess = curves["0"]["adaptive"]["swap_minus_report"]
    primary = dict(budget=0., constant_minus_adaptive_own=gain, adaptive_swap_minus_report=excess,
                   constant_swap_minus_report=curves["0"]["constant"]["swap_minus_report"],
                   supported=None if smoke else bool(excess["ci95"][1] <= 0 and gain["ci95"][0] > 0))
    free_own, free_swap = by_budget["unlimited"]
    endpoints = {}
    for method in ("constant", "adaptive"):
        improvement = paired_effect(zero_own[method] - free_own[method], repetitions)
        cost = paired_effect(free_swap[method] - zero_swap[method], repetitions)
        endpoints[method] = dict(zero_minus_unlimited_own=improvement, unlimited_minus_zero_swap=cost,
                                 supported=None if smoke else bool(min(improvement["ci95"][0], cost["ci95"][0]) > 0))
    return patients, dict(n_patients=len(zero_own), primary=primary, risks=curves, endpoint_tradeoff=endpoints,
                          bootstrap_repetitions=repetitions, bootstrap_seed=2026)


def write_results(report, output):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    def interval(stat):
        lo, hi = stat["ci95"]
        return f"{stat['estimate']:.4f} [{lo:.4f}, {hi:.4f}]"

    primary = report["primary"]
    lines = ["# UTSW A-state MRI mixture comparison", "",
             f"{report['study_mode']} | {report['n_patients']} OOF patients | zero development budget",
             "", "| Method | Own Brier [95% CI] | Swap Brier [95% CI] | Swap minus report [95% CI] |",
             "|---|---:|---:|---:|"]
    for method, scores in report["risks"]["0"].items():
        lines.append(f"| {method} | " + " | ".join(interval(scores[key]) for key in
                     ("own_brier", "swap_brier", "swap_minus_report")) + " |")
    lines.extend(["", f"Constant minus adaptive own Brier: {interval(primary['constant_minus_adaptive_own'])}.",
                  f"Primary criteria supported: {primary['supported']}.",
                  "Intervals resample recipients within folds, conditional on the fitted models and fixed donor pools."])
    (output / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig = Figure(figsize=(8, 3.6), constrained_layout=True)
    FigureCanvasAgg(fig)
    for ax, metric, title in zip(fig.subplots(1, 2), ("own_brier", "swap_brier"), ("Own MRI", "Swapped MRI")):
        for method, color in (("constant", "#27628c"), ("adaptive", "#b24928")):
            points = [report["risks"][name][method][metric] for name in BUDGETS]
            lo, hi = np.array([p["ci95"] for p in points]).T
            ax.plot(range(len(points)), [p["estimate"] for p in points], "o-", label=method, color=color)
            ax.fill_between(range(len(points)), lo, hi, color=color, alpha=.13)
        ax.axhline(report["risks"]["0"]["report_only"][metric]["estimate"], ls="--", color=".4", label="report only")
        ax.set(xticks=range(len(BUDGETS)), xticklabels=list(BUDGETS), xlabel="Development swap-risk budget",
               ylabel="OOF Brier (lower is better)", title=title)
    fig.axes[0].legend(fontsize="small")
    fig.savefig(output / "risk_curve.png", dpi=160)
