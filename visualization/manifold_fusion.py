"""Publication figures for hierarchical SPD manifold fusion."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


MODALITIES = ["T1", "T1ce", "T2", "FLAIR"]
REGIONS = ["Necrotic/Core", "Edema", "Enhancing"]
COLORS = {
    "T1": "#2563EB",
    "T1ce": "#DC2626",
    "T2": "#059669",
    "FLAIR": "#7C3AED",
    "pathology": "#D97706",
    "molecular": "#0891B2",
    "clinical": "#DB2777",
    "residual": "#64748B",
    "region": "#334155",
}
INK = "#172033"
GRID = "#D7DEE8"


def _short_node_name(name):
    return {
        "Necrotic/Core": "Core",
        "Edema": "Ed.",
        "Enhancing": "Enh.",
        "pathology": "Path.",
        "molecular": "Mol.",
        "clinical": "Clin.",
        "residual": "Res.",
    }.get(name, name[:6])


def _save(fig, path):
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return os.path.basename(path)


def _safe_stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1 if values.size > 1 else 0)),
        "n": int(values.size),
    }


def build_manifold_payloads(records, anchor_vocab):
    manifold = records.get("manifold_fusion", {})
    raw_scales = np.asarray(manifold.get("raw_scales", []), dtype=float)
    traces = np.asarray(manifold.get("raw_spd_traces", []), dtype=float)
    conditions = np.asarray(manifold.get("condition_numbers", []), dtype=float)
    eigenvalues = np.asarray(manifold.get("spd_eigenvalues", []), dtype=float)
    local = np.asarray(manifold.get("local_adjacency", []), dtype=float)
    upper = np.asarray(manifold.get("upper_adjacency", []), dtype=float)
    node_names = list(manifold.get("upper_node_names", []))

    feature_stats = {"modality_names": MODALITIES, "by_modality": {}}
    for index, name in enumerate(MODALITIES):
        feature_stats["by_modality"][name] = {
            "raw_scale": _safe_stats(raw_scales[:, :, index] if raw_scales.size else []),
            "raw_spd_trace": _safe_stats(traces[:, :, index] if traces.size else []),
            "condition_number": _safe_stats(conditions[:, :, index] if conditions.size else []),
            "normalized_eigenvalues": _safe_stats(eigenvalues[:, :, index] if eigenvalues.size else []),
        }

    topology = {
        "modality_names": MODALITIES,
        "region_names": REGIONS,
        "upper_node_names": node_names,
        "local_adjacency_mean": np.nan_to_num(local.mean(axis=0)).tolist() if local.size else [],
        "local_adjacency_std": np.nan_to_num(local.std(axis=0)).tolist() if local.size else [],
        "upper_adjacency_mean": np.nan_to_num(upper.mean(axis=0)).tolist() if upper.size else [],
        "upper_adjacency_std": np.nan_to_num(upper.std(axis=0)).tolist() if upper.size else [],
        "family_prior": np.asarray(manifold.get("family_prior", [])).tolist(),
    }

    label_by_id = [anchor.get("label", str(index)) for index, anchor in enumerate(anchor_vocab)]
    case_records = []
    subject_ids = manifold.get("subject_ids", [])
    subject_to_index = {subject: index for index, subject in enumerate(subject_ids)}
    grouped = {}
    direct_scores = np.asarray(records.get("direct_scores", []), dtype=float)
    for query_index, query in enumerate(records.get("query_records", [])):
        subject = query["subject_id"]
        entry = grouped.setdefault(
            subject,
            {"subject_id": subject, "grade": query.get("grade", "unknown"), "regions": []},
        )
        scores = direct_scores[query_index] if query_index < len(direct_scores) else np.asarray([])
        top = np.argsort(scores)[::-1][:5] if scores.size else []
        entry["regions"].append(
            {
                "node_name": query["node_name"],
                "target_labels": query.get("target_labels", []),
                "top_anchors": [
                    {"label": label_by_id[int(anchor)], "score": float(scores[int(anchor)])}
                    for anchor in top
                ],
            }
        )
    for subject, entry in grouped.items():
        index = subject_to_index.get(subject)
        if index is not None and index < len(local):
            entry["local_adjacency"] = local[index].tolist()
            entry["upper_adjacency"] = upper[index].tolist() if index < len(upper) else []
        case_records.append(entry)
    return feature_stats, topology, case_records


def save_manifold_overview(out_dir):
    path = os.path.join(out_dir, "paper4_manifold_overview.png")
    fig, ax = plt.subplots(figsize=(15.5, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    stages = [
        (0.02, 0.17, "Heterogeneous\nfeature maps", "different scale\nand dimension"),
        (0.22, 0.17, "Type-specific\nSPD descriptors", "covariance +\ntrace normalization"),
        (0.42, 0.17, "Local geodesic\ngraphs", "4 modalities ×\n3 regions"),
        (0.62, 0.17, "Upper semantic\ngraph", "regions + anchor\nfamilies + residual"),
        (0.82, 0.17, "Manifold\nreadout", "Log-Euclidean\nbarycenter → retrieval"),
    ]
    for index, (x, y, title, detail) in enumerate(stages):
        color = ["#E8F0FE", "#EAF7F1", "#F3EEFC", "#FFF4E5", "#EEF2F6"][index]
        box = FancyBboxPatch((x, y), 0.16, 0.58, boxstyle="round,pad=0.012,rounding_size=0.015", fc=color, ec="#AAB6C4")
        ax.add_patch(box)
        ax.text(x + 0.08, y + 0.38, title, ha="center", va="center", fontsize=12, weight="bold", color=INK)
        ax.text(x + 0.08, y + 0.16, detail, ha="center", va="center", fontsize=9.5, color="#475569")
        if index < len(stages) - 1:
            ax.annotate("", xy=(x + 0.195, 0.46), xytext=(x + 0.165, 0.46), arrowprops={"arrowstyle": "->", "lw": 1.7, "color": "#64748B"})
    ax.text(0.02, 0.91, "Hierarchical SPD manifold graph fusion", fontsize=17, weight="bold", color=INK)
    ax.text(0.02, 0.83, "Fusion occurs by topology-weighted tangent-space aggregation; no feature-matrix concatenation.", fontsize=10.5, color="#475569")
    return _save(fig, path)


def save_scale_to_manifold(records, out_dir):
    path = os.path.join(out_dir, "paper4_scale_to_manifold.png")
    manifold = records.get("manifold_fusion", {})
    arrays = [
        np.asarray(manifold.get("raw_scales", []), dtype=float),
        np.asarray(manifold.get("raw_spd_traces", []), dtype=float),
        np.asarray(manifold.get("condition_numbers", []), dtype=float),
    ]
    titles = ["Raw token RMS", "SPD trace before normalization", "SPD condition number"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, values, title in zip(axes, arrays, titles):
        if values.size:
            data = [values[:, :, index].reshape(-1) for index in range(4)]
            plot = ax.boxplot(data, patch_artist=True, showfliers=False)
            for patch, name in zip(plot["boxes"], MODALITIES):
                patch.set_facecolor(COLORS[name])
                patch.set_alpha(0.75)
        ax.set_xticks(range(1, 5), MODALITIES)
        ax.set_title(title, fontsize=11, weight="bold", color=INK)
        ax.grid(axis="y", color=GRID, alpha=0.7)
    fig.suptitle("Scale-aware construction of comparable SPD nodes", fontsize=16, weight="bold", color=INK)
    return _save(fig, path)


def _draw_upper_graph(ax, adjacency, names):
    count = len(names)
    if not count or adjacency.size == 0:
        ax.text(0.5, 0.5, "No upper topology", ha="center", va="center")
        ax.axis("off")
        return
    angles = np.linspace(np.pi, -np.pi, count, endpoint=False)
    positions = np.c_[0.5 + 0.36 * np.cos(angles), 0.5 + 0.36 * np.sin(angles)]
    maximum = max(float(np.nanmax(adjacency)), 1e-8)
    for i in range(count):
        for j in range(i + 1, count):
            weight = 0.5 * (adjacency[i, j] + adjacency[j, i])
            if weight <= 0.03 * maximum:
                continue
            ax.plot(*zip(positions[i], positions[j]), color="#94A3B8", lw=0.5 + 5.0 * weight / maximum, alpha=0.65, zorder=1)
    for index, name in enumerate(names):
        color = COLORS.get(name, COLORS["region"] if index < 3 else "#64748B")
        ax.scatter(*positions[index], s=500, color=color, edgecolor="white", linewidth=1.4, zorder=2)
        ax.text(*positions[index], _short_node_name(name), ha="center", va="center", fontsize=7.0, color="white", weight="bold", zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Upper heterogeneous graph", fontsize=11, weight="bold", color=INK)


def save_hierarchical_topology(records, out_dir):
    path = os.path.join(out_dir, "paper4_hierarchical_topology.png")
    manifold = records.get("manifold_fusion", {})
    local = np.asarray(manifold.get("local_adjacency", []), dtype=float)
    upper = np.asarray(manifold.get("upper_adjacency", []), dtype=float)
    names = manifold.get("upper_node_names", [])
    fig = plt.figure(figsize=(15, 8))
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.15])
    local_mean = local.mean(axis=0) if local.size else np.zeros((3, 4, 4))
    vmax = max(float(local_mean.max()), 1e-8)
    for region, name in enumerate(REGIONS):
        ax = fig.add_subplot(grid[0, region])
        image = ax.imshow(local_mean[region], cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(4), MODALITIES, rotation=25)
        ax.set_yticks(range(4), MODALITIES)
        ax.set_title(name, fontsize=11, weight="bold", color=INK)
    fig.colorbar(image, ax=[fig.axes[0], fig.axes[1], fig.axes[2]], fraction=0.02, pad=0.02, label="Mean edge weight")
    ax = fig.add_subplot(grid[1, :])
    _draw_upper_graph(ax, upper.mean(axis=0) if upper.size else np.asarray([]), names)
    fig.suptitle("Hierarchical manifold topology", fontsize=16, weight="bold", color=INK)
    return _save(fig, path)


def save_multiseed_topology(summary, out_dir):
    """Render mean and between-seed SD for the final topology panel."""
    path = os.path.join(out_dir, "paper4_hierarchical_topology.png")
    local = np.asarray(summary.get("local_mean", []), dtype=float)
    local_std = np.asarray(summary.get("local_std", []), dtype=float)
    upper = np.asarray(summary.get("upper_mean", []), dtype=float)
    names = summary.get("upper_node_names", [])
    fig = plt.figure(figsize=(15, 8))
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.15])
    vmax = max(float(np.nanmax(local)) if local.size else 0.0, 1e-8)
    for region, name in enumerate(REGIONS):
        ax = fig.add_subplot(grid[0, region])
        matrix = local[region] if local.size else np.zeros((4, 4))
        deviation = local_std[region] if local_std.size else np.zeros((4, 4))
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
        for row in range(4):
            for col in range(4):
                ax.text(col, row, f"{matrix[row, col]:.2f}\n±{deviation[row, col]:.2f}", ha="center", va="center", fontsize=6.5, color=INK)
        ax.set_xticks(range(4), MODALITIES, rotation=25)
        ax.set_yticks(range(4), MODALITIES)
        ax.set_title(name, fontsize=11, weight="bold", color=INK)
    fig.colorbar(image, ax=[fig.axes[0], fig.axes[1], fig.axes[2]], fraction=0.02, pad=0.02, label="Edge weight: mean ± seed SD")
    ax = fig.add_subplot(grid[1, :])
    _draw_upper_graph(ax, upper, names)
    fig.suptitle("Hierarchical manifold topology across seeds", fontsize=16, weight="bold", color=INK)
    return _save(fig, path)


def _representative_cases(records, limit=3):
    grouped = {}
    scores = np.asarray(records.get("direct_scores", []), dtype=float)
    targets = records.get("query_targets", [])
    for index, query in enumerate(records.get("query_records", [])):
        if index >= len(scores) or not targets[index]:
            continue
        order = np.argsort(scores[index])[::-1]
        rank = min(int(np.where(order == target)[0][0]) + 1 for target in targets[index])
        entry = grouped.setdefault(query["subject_id"], {"grade": query.get("grade", "unknown"), "ranks": []})
        entry["ranks"].append(rank)
    selected = []
    for grade in sorted({entry["grade"] for entry in grouped.values()}):
        candidates = [(subject, np.mean(entry["ranks"])) for subject, entry in grouped.items() if entry["grade"] == grade]
        median = np.median([value for _, value in candidates])
        selected.append(min(candidates, key=lambda item: (abs(item[1] - median), item[0]))[0])
    return selected[:limit]


def save_case_semantic_flow(records, anchor_vocab, out_dir):
    path = os.path.join(out_dir, "paper4_case_semantic_flow.png")
    subjects = _representative_cases(records)
    manifold = records.get("manifold_fusion", {})
    case_subjects = manifold.get("subject_ids", [])
    case_index = {subject: index for index, subject in enumerate(case_subjects)}
    local = np.asarray(manifold.get("local_adjacency", []), dtype=float)
    upper = np.asarray(manifold.get("upper_adjacency", []), dtype=float)
    names = manifold.get("upper_node_names", [])
    query_by_subject = {}
    for index, query in enumerate(records.get("query_records", [])):
        query_by_subject.setdefault(query["subject_id"], []).append((index, query))
    fig, axes = plt.subplots(max(len(subjects), 1), 1, figsize=(15, 3.0 * max(len(subjects), 1)), squeeze=False)
    for row, ax in enumerate(axes[:, 0]):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        if row >= len(subjects):
            ax.text(0.5, 0.5, "No representative case", ha="center", va="center")
            continue
        subject = subjects[row]
        index = case_index.get(subject, 0)
        queries = query_by_subject.get(subject, [])
        grade = queries[0][1].get("grade", "unknown") if queries else "unknown"
        for region in range(3):
            y0 = 0.82 - 0.32 * region
            weights = local[index, region].mean(axis=0) if local.size else np.ones(4) / 4
            for modality, name in enumerate(MODALITIES):
                y = y0 + 0.055 * (modality - 1.5)
                ax.scatter(0.08, y, s=95, color=COLORS[name])
                ax.plot([0.095, 0.31], [y, y0], color=COLORS[name], lw=0.5 + 4 * weights[modality], alpha=0.7)
            ax.scatter(0.33, y0, s=270, color=COLORS["region"])
            ax.text(0.33, y0, _short_node_name(REGIONS[region]), color="white", ha="center", va="center", fontsize=6.5, weight="bold")
        family_positions = np.linspace(0.2, 0.8, max(len(names) - 3, 1))
        for family, name in enumerate(names[3:]):
            y = family_positions[family]
            ax.scatter(0.64, y, s=270, color=COLORS.get(name, "#64748B"))
            ax.text(0.64, y, _short_node_name(name), color="white", ha="center", va="center", fontsize=6.5, weight="bold")
            if upper.size:
                for region in range(3):
                    weight = upper[index, region, 3 + family]
                    ax.plot([0.35, 0.62], [0.82 - 0.32 * region, y], color="#94A3B8", lw=0.5 + 5 * weight, alpha=0.65)
        top_labels = []
        direct = np.asarray(records.get("direct_scores", []), dtype=float)
        for query_index, _ in queries:
            if query_index < len(direct):
                top = int(np.argmax(direct[query_index]))
                top_labels.append(anchor_vocab[top]["label"])
        ax.text(0.82, 0.52, "Top retrieved anchors\n" + "\n".join(top_labels[:3]), ha="left", va="center", fontsize=8.5, color=INK)
        ax.text(0.01, 0.96, f"{subject}  |  grade {grade}  |  median-rank representative", fontsize=9.5, weight="bold", color=INK)
    fig.suptitle("Case-level semantic flow (grade is display-only)", fontsize=16, weight="bold", color=INK)
    return _save(fig, path)


def save_ablation_evidence(metrics, out_dir, aggregate_payload=None):
    path = os.path.join(out_dir, "paper4_ablation_evidence.png")
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    if aggregate_payload and aggregate_payload.get("variants"):
        variants = aggregate_payload["variants"]
        labels = list(variants)
        values = [variants[name]["metrics"].get("map", {}).get("mean", np.nan) for name in labels]
        intervals = [variants[name]["metrics"].get("map", {}).get("ci95", [value, value]) for name, value in zip(labels, values)]
        errors = np.asarray(
            [[value - interval[0] for value, interval in zip(values, intervals)], [interval[1] - value for value, interval in zip(values, intervals)]]
        )
        ax.bar(range(len(labels)), values, yerr=errors, color="#2563EB", alpha=0.8, capsize=4)
        for x, name in enumerate(labels):
            seeds = variants[name]["metrics"].get("map", {}).get("values", [])
            ax.scatter(np.full(len(seeds), x), seeds, color=INK, s=24, zorder=3)
        ax.set_xticks(range(len(labels)), [name.replace("_", " ") for name in labels], rotation=22, ha="right")
        ax.set_ylabel("mAP, mean and bootstrap 95% CI")
    else:
        names = ["R@1", "mAP", "MRR"]
        values = [metrics.get("recall@1", np.nan), metrics.get("map", np.nan), metrics.get("mrr", np.nan)]
        ax.bar(names, values, color=["#2563EB", "#059669", "#D97706"])
        ax.text(0.98, 0.95, "Single-seed diagnostic\nFinal: 5 variants × 3 seeds", transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#64748B")
        ax.set_ylim(0, 1)
    ax.grid(axis="y", color=GRID, alpha=0.7)
    ax.set_title("Manifold-fusion evidence", fontsize=15, weight="bold", color=INK)
    return _save(fig, path)


def save_manifold_figures(records, metrics, anchor_vocab, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    return [
        save_manifold_overview(out_dir),
        save_scale_to_manifold(records, out_dir),
        save_hierarchical_topology(records, out_dir),
        save_case_semantic_flow(records, anchor_vocab, out_dir),
        save_ablation_evidence(metrics, out_dir),
    ]


__all__ = [
    "build_manifold_payloads",
    "save_manifold_figures",
    "save_ablation_evidence",
    "save_manifold_overview",
    "save_scale_to_manifold",
    "save_multiseed_topology",
]
