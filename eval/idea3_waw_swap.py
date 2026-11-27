"""Analyze one external donor pool. A checkpoint is not a patient stratum."""

import numpy as np

from glioma.cli.idea3_rule_mri_swap import PROBABILITIES, verify_invariance
from glioma.eval.idea3_rule_mri_swap import patient_results, summarize, validate_pairs


def external_results(frames, ids, stratum="WAW"):
    validate_pairs(frames, {fold: ids for fold in range(5)})
    scored, individual = patient_results(frames)
    verify_invariance(scored)

    keys = ["model", "subject_id", "image_subject_id", "label", "reference_grade", "allowed_grades"]
    averaged = frames.groupby(keys, as_index=False)[PROBABILITIES].mean()
    # The external cohort is a single bootstrap stratum, shared by all five models' fits.
    averaged["fold"] = stratum
    validate_pairs(averaged, {stratum: ids})
    ensemble, patients = patient_results(averaged)
    verify_invariance(ensemble)
    report = summarize(patients)
    for effect in [m["own_advantage"] for m in report["models"].values()] + list(report["own_brier_contrasts"].values()):
        del effect["by_fold"], effect["leave_one_fold_out_range"]
    for name, metrics in report["models"].items():
        subset = patients[patients.model == name]
        metrics["by_reference_grade"] = {
            str(grade): dict(n=len(group), own_brier=float(group.own_brier.mean()),
                             donor_brier=float(group.donor_brier.mean()),
                             own_advantage=float(group.own_advantage.mean()))
            for grade, group in subset.groupby("reference_grade")
        }
        # Omit the same patient from the recipient set and from every remaining donor pool.
        matrix = ensemble[ensemble.model == name].pivot(
            index="subject_id", columns="image_subject_id", values="rule_brier").loc[ids, ids].to_numpy()
        loo = []
        if len(ids) > 2:
            for i in range(len(ids)):
                remaining = np.delete(np.delete(matrix, i, axis=0), i, axis=1)
                own = remaining.diagonal()
                donor = (remaining.sum(axis=1) - own) / (len(own) - 1)
                loo.append(float((donor - own).mean()))
        metrics["leave_one_patient_out_range"] = [min(loo), max(loo)] if loo else None
        metrics["own_advantage_by_checkpoint"] = {
            str(fold): float(group.own_advantage.mean())
            for fold, group in individual[individual.model == name].groupby("fold")
        }
    return (scored.rename(columns={"fold": "checkpoint_fold"}), ensemble.drop(columns="fold"),
            patients.drop(columns="fold"), report)
