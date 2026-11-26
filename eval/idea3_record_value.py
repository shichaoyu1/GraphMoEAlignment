"""Paired comparisons of recording status and measured 1p/19q value."""

import numpy as np

from glioma.eval.idea3_role_transition import _brier3, paired_effect
from glioma.eval.idea3_rule_mri_swap import PATIENT_KEYS


READOUTS = ("X", "X+M", "X+M+V")


def summarize(predictions, repetitions=10_000, seed=2026):
    """Compute patient-level three-class Brier from OOF probabilities for both cohorts."""
    patients = predictions[PATIENT_KEYS + ["label", "codeletion_measured", "codeletion_positive"]]
    patients = patients.drop_duplicates().set_index(PATIENT_KEYS).sort_index()
    if patients.index.has_duplicates or not patients.label.isin([0, 1, 2]).all():
        raise ValueError("Patient reference grade or recording status inconsistent across readouts")
    if not patients.codeletion_positive.notna().eq(patients.codeletion_measured).all():
        raise ValueError("Each measured patient requires a value; an unreported result has unknown value")

    losses = predictions.assign(brier3=_brier3(predictions)).pivot(
        index=PATIENT_KEYS, columns="readout", values="brier3")
    if set(losses.columns) != set(READOUTS) or not np.isfinite(losses.to_numpy()).all():
        raise ValueError("Each patient requires one finite Brier score for all three readouts")
    measured = losses.loc[patients.codeletion_measured]
    value = paired_effect(measured["X+M"] - measured["X+M+V"], repetitions, seed)
    availability = paired_effect(losses["X"] - losses["X+M"], repetitions, seed)
    n_positive = int(patients.codeletion_positive.sum())
    return {
        "n_patients": len(losses), "n_measured": len(measured), "n_positive": n_positive,
        "n_negative": len(measured) - n_positive, "n_unreported": len(losses) - len(measured),
        "primary": "value_given_record",
        "primary_definition": "Measured patients, mean Brier3(X+M) minus Brier3(X+M+V)",
        "secondary_definition": "All labeled patients, mean Brier3(X) minus Brier3(X+M)",
        "value_given_record": value, "record_availability": availability,
        "value_gain_ci_above_zero": value["ci95"][0] > 0,
        "brier3": {name: {r: float(rows[r].mean()) for r in READOUTS}
                   for name, rows in (("all", losses), ("measured", measured))},
        "bootstrap_repetitions": repetitions, "bootstrap_seed": seed,
    }


def report_markdown(summary):
    measured_n = summary["n_measured"]
    value = summary["value_given_record"]
    lo, hi = value["ci95"]
    if summary.get("study_mode") == "smoke":
        finding = "短测用于核对计算和输出，取值的预测收益以五折全量结果评价。"
    elif lo > 0:
        finding = "在本队列的已测患者中，告知具体阳性或阴性值进一步降低了折外三分类 Brier。"
    elif hi < 0:
        finding = "在本队列的已测患者中，加入具体取值后折外三分类 Brier 升高。"
    else:
        finding = "在本队列的已测患者中，具体取值能否进一步降低折外三分类 Brier 仍不确定。"
    lines = ["# 1p/19q 记录与具体取值的增量预测收益", "", finding, "",
             f"评估共 {summary['n_patients']} 名患者。{measured_n} 人有明确记录，其中阳性 "
             f"{summary['n_positive']} 人、阴性 {summary['n_negative']} 人。其余 "
             f"{summary['n_unreported']} 人的取值未知。", "",
             "X 包含 MRI 与其余报告字段。M 标记是否有明确记录，V 结合 M 编码已测结果。"
             "三分类 Brier 取三个等级的概率平方误差均值，分数越低越好。", "",
             "| 评价人群 | 人数 | X | X+M | X+M+V |", "|---|---:|---:|---:|---:|"]
    for name, label, n in (("all", "全部", summary["n_patients"]), ("measured", "有记录", measured_n)):
        means = summary["brier3"][name]
        lines.append(f"| {label} | {n} | " + " | ".join(f"{means[r]:.5f}" for r in READOUTS) + " |")

    lines += ["", "差值为正表示增加相应信息后预测更准。患者等权，按折分层配对重采样 "
              f"{summary['bootstrap_repetitions']:,} 次。区间以当前队列及拟合模型为条件。", "",
              "| 比较 | 人数 | Brier 差值 [95% CI] | 各折差值（人数） | 去一折范围 |",
              "|---|---:|---:|---|---|"]
    for label, effect in (("主比较，有记录人群的 X+M 减 X+M+V", value),
                          ("次比较，全部人群的 X 减 X+M", summary["record_availability"])):
        lo, hi = effect["ci95"]
        folds = "，".join(f"{fold} 折 {row['estimate']:+.5f}（{row['n']}）"
                          for fold, row in effect["by_fold"].items())
        limits = effect["leave_one_fold_out_range"]
        loo = f"[{limits[0]:+.5f}, {limits[1]:+.5f}]" if limits else "单折"
        lines.append(f"| {label} | {effect['n']} | {effect['estimate']:+.5f} "
                     f"[{lo:+.5f}, {hi:+.5f}] | {folds} | {loo} |")
    lines += ["", "次比较反映记录可用性与预测表现的关联，其中可能含有检测选择或记录流程的信息。"]
    return "\n".join(lines) + "\n"
