"""Aggregate independent data seeds first; retain all model-seed pairs."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from glioma.cli.train_authority import atomic_json


METRICS = ("brier", "avr", "residual_brier", "residual_accuracy", "accuracy", "prediction_coverage",
           "protected_shift_max", "projection_mse", "risk_at_coverage_0.8", "inference_ms_per_sample", "train_seconds")


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def cluster_interval(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None, None, None
    mean = float(values.mean())
    if len(values) < 2:
        return mean, None, None
    rng = np.random.default_rng(20260909)
    draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(-1)
    low, high = np.quantile(draws, [.025, .975])
    return mean, float(low), float(high)


def cluster_rows(rows):
    groups = defaultdict(list)
    keys = ("phase", "task", "variant", "attribution", "event", "data_seed")
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    output = []
    for key, items in sorted(groups.items()):
        record = dict(zip(keys, key), model_runs=len(items), model_seeds=";".join(str(x["model_seed"]) for x in items))
        for metric in METRICS:
            values = [row[metric] for row in items if row.get(metric) is not None]
            record[metric] = float(np.mean(values)) if values else None
            record[metric + "_n"] = len(values)
        output.append(record)
    return output


def paired_comparisons(rows):
    """Produce each model-seed pair, then average those pairs within data seed."""
    indexed = {(r["phase"], r["task"], r["variant"], r["attribution"], r["event"], r["data_seed"], r["model_seed"]): r for r in rows}
    pairs = []
    for a in rows:
        comparators = []
        if a["variant"] == "acf":
            for variant in (("learned_joint", "transformer_joint", "graph_only", "learned") if a["attribution"] == "base" else ("learned_joint",)):
                comparators.append((a["phase"], variant, a["attribution"]))
        if a["phase"] == "attribution":
            comparators.append(("main", a["variant"], "base"))
        for phase, variant, attribution in comparators:
            key = (phase, a["task"], variant, attribution, a["event"], a["data_seed"], a["model_seed"])
            b = indexed.get(key)
            if b is None:
                continue
            name = f'{a["phase"]}/{a["variant"]}/{a["attribution"]} minus {phase}/{variant}/{attribution}'
            row = dict(comparison=name, task=a["task"], event=a["event"], data_seed=a["data_seed"], model_seed=a["model_seed"])
            for metric in METRICS:
                row[metric] = a[metric] - b[metric] if a.get(metric) is not None and b.get(metric) is not None else None
            pairs.append(row)
    groups = defaultdict(list)
    for row in pairs:
        groups[(row["comparison"], row["task"], row["event"], row["data_seed"])].append(row)
    clusters = []
    for (comparison, task, event, seed), group in sorted(groups.items()):
        row = dict(comparison=comparison, task=task, event=event, data_seed=seed, paired_model_runs=len(group))
        for metric in METRICS:
            valid = [r[metric] for r in group if r.get(metric) is not None]
            row[metric] = float(np.mean(valid)) if valid else None
        clusters.append(row)
    intervals = []
    for comparison, task, event in sorted({(r["comparison"], r["task"], r["event"]) for r in clusters}):
        group = [r for r in clusters if (r["comparison"], r["task"], r["event"]) == (comparison, task, event)]
        for metric in METRICS:
            values = [r[metric] for r in group if r.get(metric) is not None]
            mean, low, high = cluster_interval(values)
            intervals.append(dict(comparison=comparison, task=task, event=event, metric=metric,
                                  independent_data_seeds=len(values), mean_difference=mean, ci_low=low, ci_high=high))
    return pairs, clusters, intervals


def aggregate(root, phases=("main", "attribution"), allow_incomplete=False):
    from glioma.cli.run_authority_protocol import formal_jobs
    root = Path(root)
    rows, completion, hashes = [], {}, set()
    for phase in phases:
        if phase in ("main", "attribution"):
            expected = {j["id"] for j in formal_jobs(phase)}
        elif phase == "rule_only":
            expected = {f"{task}_d{seed}" for task in ("s1", "s2") for seed in range(42,47)}
        elif phase == "smoke":
            expected = {f"{task}_{variant}_base_d41_m100" for task in ("s1", "s2")
                        for variant in ("learned", "learned_joint", "graph_only", "acf")}
        else:
            raise ValueError("Unsupported reporting phase: " + phase)
        completed = set()
        for done in sorted((root / phase).glob("*/DONE.json")):
            config = json.loads((done.parent / "config.json").read_text(encoding="utf-8"))
            result = json.loads(done.read_text(encoding="utf-8"))
            if result["config_hash"] != config["config_hash"]:
                raise ValueError("Config hash mismatch: " + str(done))
            if expected is not None and done.parent.name not in expected:
                raise ValueError("Unregistered formal job: " + str(done.parent))
            if phase in ("main", "attribution") and result.get("evaluation") != "full":
                raise ValueError("Formal job missing full interventions: " + str(done.parent))
            summaries = json.loads((done.parent / "events/summary.json").read_text(encoding="utf-8"))
            completed.add(done.parent.name); hashes.add(config["source_hash"])
            for event, values in summaries.items():
                identity = {key: config[key] for key in ("task", "variant", "attribution", "data_seed", "model_seed")}
                row = dict(identity, phase=phase, event=event, run=done.parent.name,
                           train_seconds=result["train_seconds"], parameters=result.get("parameters"))
                row.update({k: v for k, v in values.items() if not isinstance(v, (dict, list))})
                rows.append(row)
                if "posthoc_projection" in values:
                    posthoc = dict(row, variant=config["variant"] + "_posthoc")
                    posthoc.update(values["posthoc_projection"])
                    rows.append(posthoc)
        completion[phase] = dict(completed=len(completed), expected=len(expected) if expected else None,
                                 missing=sorted(expected - completed) if expected else [])
    if len(hashes) > 1:
        raise ValueError("Refusing to aggregate different training/evaluation source revisions")
    destination = root / ("summary_" + "_".join(phases))
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(destination / "completeness.json", completion)
    write_csv(destination / "all_runs.csv", rows)
    write_csv(destination / "data_seed_means.csv", cluster_rows(rows))
    pairs, clusters, intervals = paired_comparisons(rows)
    write_csv(destination / "all_paired_runs.csv", pairs)
    write_csv(destination / "paired_data_seed_means.csv", clusters)
    write_csv(destination / "paired_intervals.csv", intervals)
    lines = ["# ACF-SPD authority 实验汇总", "", "这是合成机制实验；不构成患者级临床 authority 验证。", ""]
    for phase, counts in completion.items():
        lines.append(f'- {phase}: {counts["completed"]} / {counts["expected"] if counts["expected"] is not None else "未指定总数"} 个完成任务。')
    lines += ["", "Brier 为各输出维度平方误差的平均；拒答不混入预测分母。AVR 的空分母保留为空。",
              "先在每个数据种子内平均模型种子，再对独立数据种子计算配对差异与 95% 百分位 bootstrap 区间。",
              "正式研究的统计单位是 5 个独立生成数据集；全部 15 次模型配对均单独保存。5 个数据种子的区间应谨慎解释。",
              "单一开发种子不输出有效的跨数据集区间，也不用于论文架构优势结论。", "",
              "| Task | Model | Setting | Data seed | Model runs | Clean Brier | Residual Brier | AVR |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in cluster_rows(rows):
        if row["event"] == "clean":
            def fmt(value):
                return "N/A" if value is None else f"{value:.6f}"
            lines.append(f'| {row["task"]} | {row["variant"]} | {row["attribution"]} | {row["data_seed"]} | {row["model_runs"]} | {fmt(row["brier"])} | {fmt(row["residual_brier"])} | {fmt(row["avr"])} |')
    lines += ["", "文件说明：all_runs.csv 保留每次训练和各项干预；all_paired_runs.csv 保留全部模型配对；",
              "data_seed_means.csv 与 paired_data_seed_means.csv 为数据种子层面的汇总；paired_intervals.csv 为数据集层面的配对区间。",
              "保护状态差异只能在目标、授权来源和关系方向相同且两边均激活时解释；旁路探针是故意引入泄漏的正对照。",
              "错误 authority 单元完整保留真实任务损失。零违反率只证明约束实现，不能证明受保护 graph 的增量价值。"]
    (destination / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if any(v["missing"] for v in completion.values()) and not allow_incomplete:
        raise ValueError("Formal runs are incomplete; diagnostic outputs written with missing list. Use --allow-incomplete for interim summaries.")
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--phases", nargs="+", default=["main", "attribution", "rule_only"])
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)
    print(aggregate(args.root, tuple(args.phases), args.allow_incomplete))


if __name__ == "__main__":
    main()
