"""Publication-style TopoMoE visualization utilities."""

import json
import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches


FAMILY_COLORS = {
    "mri": "#2F6FDB",
    "pathology": "#D95F02",
    "molecular": "#1B9E77",
    "clinical": "#7570B3",
    "residual": "#6C757D",
}
NEUTRAL = "#273043"
LIGHT_BG = "#FFFFFF"
PANEL_BG = "#F6F8FA"
GRID = "#D8DEE9"


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _family_color(family):
    return FAMILY_COLORS.get(str(family).lower(), "#4C566A")


def _save(fig, path):
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=LIGHT_BG)
    plt.close(fig)
    return path


def _short_label(label, max_len=28):
    label = str(label).replace("Pathology ", "")
    if len(label) <= max_len:
        return label
    return label[: max_len - 3] + "..."


def _wrap(label, width=18):
    return "\n".join(textwrap.wrap(str(label), width=width)) or str(label)


def _box(ax, xy, width, height, text, color, fontsize=10, lw=1.5):
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=lw,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=NEUTRAL,
    )


def _arrow(ax, start, end, color="#7A869A", lw=1.4):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "lw": lw,
            "shrinkA": 6,
            "shrinkB": 6,
            "mutation_scale": 13,
        },
    )


def _topology_family_matrix(matrix, family_ids, family_names):
    matrix = np.asarray(matrix, dtype=np.float32)
    family_ids = np.asarray(family_ids, dtype=np.int64)
    present = sorted(set(int(idx) for idx in family_ids.tolist()))
    labels = [family_names[idx] if idx < len(family_names) else str(idx) for idx in present]
    family_matrix = np.zeros((len(present), len(present)), dtype=np.float32)
    for row_out, family_i in enumerate(present):
        rows = np.where(family_ids == family_i)[0]
        for col_out, family_j in enumerate(present):
            cols = np.where(family_ids == family_j)[0]
            if len(rows) and len(cols):
                family_matrix[row_out, col_out] = float(matrix[np.ix_(rows, cols)].mean())
    return family_matrix, labels


def _metric_value(metrics, scenario, metric):
    if scenario == "normal":
        return float(metrics.get("direct", {}).get(metric, np.nan))
    return float(metrics.get("gallery_availability_stress", {}).get(scenario, {}).get(metric, np.nan))


def save_topomoe_overview(out_dir, records, anchor_vocab):
    _ensure_dir(out_dir)
    path = os.path.join(out_dir, "topomoe_overview.png")
    family_names = records.get("family_names") or ["pathology", "molecular", "residual"]
    anchor_counts = {}
    for anchor in anchor_vocab:
        key = str(anchor.get("source", "")).lower()
        if key == "gene":
            key = "molecular"
        anchor_counts[key] = anchor_counts.get(key, 0) + 1

    fig, ax = plt.subplots(figsize=(13, 5.8))
    fig.patch.set_facecolor(LIGHT_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.03, 0.94, "TopoMoE-GDI: topology-guided evidence routing", fontsize=17, weight="bold", color=NEUTRAL)
    ax.text(
        0.03,
        0.89,
        "Grade stratifies the anchor vocabulary; inference-time routing is based on MRI semantic units and learned evidence context.",
        fontsize=9.5,
        color="#5F6B7A",
    )

    columns = [
        (0.03, "MRI semantic\nunits", ["Necrotic/Core", "Edema", "Enhancing"], "mri"),
        (0.24, "Disease-anchor\ntopology", ["A_prior", "co-occurrence", "clinical structure"], "pathology"),
        (0.45, "Topology-smoothed\nrouter", ["r_ir = Router\n(u_ir, N(A))", "sparse / balance", "specialize"], "clinical"),
        (0.66, "Anchor\nexperts", [f"{name} expert" for name in family_names], "molecular"),
        (0.83, "Expert-specific\nretrieval", ["routed top-k anchors", "anchor-family evidence", "stress tests"], "residual"),
    ]

    for x, title, items, family in columns:
        color = _family_color(family)
        _box(ax, (x, 0.68), 0.145, 0.08, title, color, fontsize=10.0, lw=1.8)
        for idx, item in enumerate(items[:4]):
            y = 0.56 - idx * 0.105
            _box(ax, (x + 0.012, y), 0.121, 0.06, item, color, fontsize=8.4, lw=1.1)
        if x < 0.82:
            _arrow(ax, (x + 0.145, 0.72), (x + 0.195, 0.72))

    counts_text = [
        f"{label}: {anchor_counts.get(label, 0)} anchors"
        for label in ["pathology", "molecular", "clinical"]
        if anchor_counts.get(label, 0) > 0
    ]
    if counts_text:
        ax.text(0.25, 0.16, "Anchor vocabulary: " + " | ".join(counts_text), fontsize=9, color="#5F6B7A")
    ax.text(0.04, 0.08, "Manuscript role: show evidence routing, not a static semantic-unit graph.", fontsize=10, color=NEUTRAL, weight="bold")
    return _save(fig, path)


def save_routing_spectrum(out_dir, records):
    _ensure_dir(out_dir)
    path = os.path.join(out_dir, "topomoe_routing_spectrum.png")
    route_entries = records.get("route_entries", [])
    family_names = records.get("family_names") or []
    if not route_entries or not family_names:
        return None

    def summarize(group_key):
        grouped = {}
        for entry in route_entries:
            grouped.setdefault(group_key(entry), []).append(entry)
        labels = []
        weights = []
        counts = []
        for key in sorted(grouped):
            entries = grouped[key]
            labels.append(str(key))
            weights.append(np.asarray([entry["weights"] for entry in entries], dtype=np.float32).mean(axis=0))
            counts.append(len(entries))
        return labels, np.asarray(weights), counts

    node_labels, node_weights, node_counts = summarize(lambda entry: entry["node_name"])
    grade_labels, grade_weights, grade_counts = summarize(lambda entry: f"Grade {entry['grade']}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), sharex=True)
    fig.patch.set_facecolor(LIGHT_BG)
    for ax, labels, weights, counts, title in [
        (axes[0], node_labels, node_weights, node_counts, "By MRI semantic unit"),
        (axes[1], grade_labels, grade_weights, grade_counts, "By grade stratum"),
    ]:
        left = np.zeros(len(labels), dtype=np.float32)
        y = np.arange(len(labels))
        for family_idx, family in enumerate(family_names):
            values = weights[:, family_idx]
            ax.barh(y, values, left=left, color=_family_color(family), edgecolor="white", height=0.58, label=family)
            for row, value in enumerate(values):
                if value >= 0.08:
                    ax.text(left[row] + value / 2, row, f"{value:.2f}", ha="center", va="center", fontsize=8, color="white")
            left += values
        ax.set_yticks(y)
        ax.set_yticklabels([f"{label} (n={count})" for label, count in zip(labels, counts)], fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Mean routing weight")
        ax.set_title(title, fontsize=12, weight="bold", color=NEUTRAL)
        ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", frameon=False, ncol=max(len(labels), 1), fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Topology-guided evidence routing spectrum", fontsize=16, weight="bold", color=NEUTRAL, y=1.04)
    fig.text(0.01, -0.08, "Prototype-scale visualization; routing weights are descriptive, not final multi-seed evidence.", fontsize=9, color="#5F6B7A")
    return _save(fig, path)


def _choose_case(routing_records):
    if not routing_records:
        return None

    def score(record):
        routed_hits = sum(1 for item in record.get("routed_top_k", [])[:5] if item.get("is_target"))
        direct_hits = sum(1 for item in record.get("direct_top_k", [])[:5] if item.get("is_target"))
        weights = list(record.get("routing_weights", {}).values())
        specificity = max(weights) - min(weights) if weights else 0.0
        return (routed_hits, direct_hits, specificity)

    return sorted(routing_records, key=score, reverse=True)[0]


def save_case_routing_flow(out_dir, records):
    _ensure_dir(out_dir)
    path = os.path.join(out_dir, "topomoe_case_routing_flow.png")
    record = _choose_case(records.get("routing_records", []))
    if record is None:
        return None

    fig, ax = plt.subplots(figsize=(12.8, 5.8))
    fig.patch.set_facecolor(LIGHT_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.03, 0.94, "Case-level evidence routing flow", fontsize=16, weight="bold", color=NEUTRAL)
    ax.text(
        0.03,
        0.89,
        f"Subject {record['subject_id']} | {record['node_name']} | grade label used for analysis stratum: {record['grade']}",
        fontsize=9.5,
        color="#5F6B7A",
    )

    _box(ax, (0.04, 0.61), 0.18, 0.12, f"MRI unit\n{record['node_name']}", _family_color("mri"), fontsize=11, lw=1.8)
    _arrow(ax, (0.22, 0.67), (0.32, 0.67))

    ax.text(0.34, 0.79, "Router weights", fontsize=11, weight="bold", color=NEUTRAL)
    y = 0.69
    for family, value in record.get("routing_weights", {}).items():
        ax.add_patch(patches.Rectangle((0.34, y - 0.018), 0.22 * float(value), 0.036, color=_family_color(family), alpha=0.88))
        ax.add_patch(patches.Rectangle((0.34, y - 0.018), 0.22, 0.036, fill=False, edgecolor=GRID, linewidth=0.8))
        ax.text(0.57, y, f"{family}: {float(value):.2f}", va="center", ha="left", fontsize=9, color=NEUTRAL)
        y -= 0.08

    _arrow(ax, (0.61, 0.67), (0.70, 0.67))
    ax.text(0.71, 0.79, "Routed top anchors", fontsize=11, weight="bold", color=NEUTRAL)
    for idx, item in enumerate(record.get("routed_top_k", [])[:6]):
        y = 0.70 - idx * 0.075
        color = _family_color(item.get("family", "residual"))
        fill = "#FFF4E5" if item.get("is_target") else "white"
        ax.add_patch(
            patches.FancyBboxPatch(
                (0.71, y - 0.028),
                0.24,
                0.052,
                boxstyle="round,pad=0.006,rounding_size=0.01",
                linewidth=1.4 if item.get("is_target") else 0.8,
                edgecolor=color if item.get("is_target") else GRID,
                facecolor=fill,
            )
        )
        prefix = "*" if item.get("is_target") else " "
        ax.text(0.725, y, f"{prefix} {idx + 1}. {_short_label(item.get('label', ''), 23)}", ha="left", va="center", fontsize=8.5, color=NEUTRAL)
        ax.text(0.94, y, f"{float(item.get('score', 0.0)):.2f}", ha="right", va="center", fontsize=8, color="#5F6B7A")

    targets = ", ".join(_short_label(label, 22) for label in record.get("target_labels", []))
    ax.text(0.04, 0.20, "Target anchors: " + targets, fontsize=9.2, color=NEUTRAL)
    ax.text(0.04, 0.13, "* highlights patient-level positive anchors in the routed ranking.", fontsize=8.8, color="#5F6B7A")
    return _save(fig, path)


def save_topology_consistency(out_dir, records):
    _ensure_dir(out_dir)
    path = os.path.join(out_dir, "topomoe_topology_consistency.png")
    topology = records.get("topomoe_topology") or {}
    if not topology:
        return None

    family_names = topology.get("family_names", [])
    family_ids = topology.get("family_ids", [])
    prior = np.asarray(topology.get("A_prior", []), dtype=np.float32)
    effective = np.asarray(topology.get("A_effective", []), dtype=np.float32)
    if prior.size == 0 or effective.size == 0:
        return None

    prior_family, labels = _topology_family_matrix(prior, family_ids, family_names)
    effective_family, _ = _topology_family_matrix(effective, family_ids, family_names)
    delta_family = effective_family - prior_family
    top_edges = topology.get("top_edges", [])[:8]

    fig = plt.figure(figsize=(13.5, 7))
    fig.patch.set_facecolor(LIGHT_BG)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1.25], wspace=0.35, hspace=0.35)
    matrices = [(prior_family, "Prior family topology"), (effective_family, "Effective family topology"), (delta_family, "Effective - prior")]
    for idx, (matrix, title) in enumerate(matrices):
        ax = fig.add_subplot(gs[0, idx])
        vmax = max(float(np.max(np.abs(matrix))), 1e-6)
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=11, weight="bold", color=NEUTRAL)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", fontsize=8, color=NEUTRAL)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[1, :])
    labels_edge = [_wrap(f"{edge['source_label']} -> {edge['target_label']}", width=38) for edge in top_edges]
    values = [float(edge["effective_weight"]) for edge in top_edges]
    deltas = [float(edge.get("delta_vs_prior", 0.0)) for edge in top_edges]
    y = np.arange(len(labels_edge))
    ax.barh(y, values, color="#8AB6D6", edgecolor="white", height=0.62)
    for row, (value, delta) in enumerate(zip(values, deltas)):
        ax.text(value + 0.005, row, f"{value:.2f} ({delta:+.2f})", va="center", fontsize=8.5, color=NEUTRAL)
    ax.set_yticks(y)
    ax.set_yticklabels(labels_edge, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Effective topology weight; parenthesis shows delta vs prior")
    ax.set_title("Top learned anchor-topology edges", fontsize=11, weight="bold", color=NEUTRAL)
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    fig.suptitle("Disease-anchor topology consistency", fontsize=16, weight="bold", color=NEUTRAL, y=1.02)
    return _save(fig, path)


def save_anchor_availability_stress(out_dir, metrics):
    _ensure_dir(out_dir)
    path = os.path.join(out_dir, "topomoe_anchor_availability_stress.png")
    scenarios = [
        ("normal", "All anchors"),
        ("pathology_unavailable", "No pathology"),
        ("molecular_unavailable", "No molecular"),
    ]
    metric_names = [("recall@1", "R@1"), ("mrr", "MRR"), ("map", "mAP")]
    values = np.asarray(
        [[_metric_value(metrics, scenario, metric) for metric, _ in metric_names] for scenario, _ in scenarios],
        dtype=np.float32,
    )

    fig, ax = plt.subplots(figsize=(10, 5.4))
    fig.patch.set_facecolor(LIGHT_BG)
    x = np.arange(len(metric_names))
    width = 0.24
    scenario_colors = ["#4C78A8", "#F58518", "#54A24B"]
    for idx, (_, label) in enumerate(scenarios):
        offset = (idx - 1) * width
        bars = ax.bar(x + offset, values[idx], width=width, label=label, color=scenario_colors[idx], edgecolor="white")
        for bar in bars:
            height = bar.get_height()
            if np.isfinite(height):
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.015, f"{height:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metric_names], fontsize=10)
    ax.set_ylim(0, max(1.0, np.nanmax(values) + 0.15))
    ax.set_ylabel("Retrieval score")
    ax.set_title("Anchor availability stress test", fontsize=16, weight="bold", color=NEUTRAL)
    ax.text(0.0, -0.22, "Stress test only: removing an anchor family tests retrieval dependence, not a causal intervention.", transform=ax.transAxes, fontsize=9, color="#5F6B7A")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    return _save(fig, path)


def save_topology_dynamics(out_dir, history_path):
    path = os.path.join(out_dir, "topomoe_topology_dynamics.png")
    if not history_path or not os.path.exists(history_path):
        return None
    with open(history_path, "r", encoding="utf-8") as file:
        history = json.load(file)
    if not history:
        return None
    epochs = [entry["epoch"] for entry in history]
    keys = ["effective_prior_fro", "new_edge_mass", "opened_edge_count", "gradient_norm"]
    labels = ["||A_eff-A_prior||F", "New-edge mass", "Opened edges", "Gradient norm"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4))
    fig.patch.set_facecolor(LIGHT_BG)
    for ax, key, label in zip(axes.flat, keys, labels):
        train_values = [entry.get("train", {}).get("topology", {}).get(key, np.nan) for entry in history]
        val_values = [entry.get("val", {}).get("topology", {}).get(key, np.nan) for entry in history]
        ax.plot(epochs, train_values, color="#2F6FDB", lw=1.8, label="train")
        if np.isfinite(np.asarray(val_values, dtype=np.float32)).any():
            ax.plot(epochs, val_values, color="#D95F02", lw=1.5, label="validation")
        ax.set_title(label, fontsize=11, weight="bold", color=NEUTRAL)
        ax.set_xlabel("Epoch")
        ax.grid(color=GRID, alpha=0.7)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Topology learning dynamics", fontsize=16, weight="bold", color=NEUTRAL)
    return _save(fig, path)


def save_checkpoint_comparison(out_dir, metrics, comparison):
    path = os.path.join(out_dir, "topomoe_checkpoint_comparison.png")
    direct_best = (comparison or {}).get("direct_best", {})
    if not direct_best:
        return None
    groups = [
        ("Routed-best\nDirect", metrics.get("direct", {})),
        ("Routed-best\nRouted", metrics.get("routed", {})),
        ("Direct-best\nDirect", direct_best.get("direct", {})),
        ("Direct-best\nRouted", direct_best.get("routed", {})),
    ]
    metric_names = [("recall@1", "R@1"), ("mrr", "MRR"), ("map", "mAP")]
    values = np.asarray([[group.get(name, np.nan) for name, _ in metric_names] for _, group in groups])
    fig, ax = plt.subplots(figsize=(11, 5.6))
    fig.patch.set_facecolor(LIGHT_BG)
    x = np.arange(len(metric_names))
    width = 0.19
    colors = ["#4C78A8", "#F58518", "#72B7B2", "#E45756"]
    for idx, ((label, _), color) in enumerate(zip(groups, colors)):
        ax.bar(x + (idx - 1.5) * width, values[idx], width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metric_names])
    ax.set_ylim(0, max(1.0, float(np.nanmax(values)) + 0.1))
    ax.set_ylabel("Retrieval score")
    ax.set_title("Direct-best vs routed-best checkpoints", fontsize=16, weight="bold", color=NEUTRAL)
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", color=GRID, alpha=0.7)
    return _save(fig, path)


def save_intervention_summary(out_dir, interventions):
    path = os.path.join(out_dir, "topomoe_intervention_summary.png")
    scenarios = (interventions or {}).get("scenarios", {})
    if not scenarios:
        return None
    ordered = sorted(scenarios.items(), key=lambda item: float(item[1].get("delta_map", 0.0)))
    labels = [_wrap(name.replace("_", " "), 24) for name, _ in ordered]
    values = [float(payload.get("delta_map", np.nan)) for _, payload in ordered]
    colors = ["#D95F02" if value < 0 else "#1B9E77" for value in values]
    fig, ax = plt.subplots(figsize=(10.5, max(5.5, 0.48 * len(labels))))
    fig.patch.set_facecolor(LIGHT_BG)
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, edgecolor="white")
    ax.axvline(0, color=NEUTRAL, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Delta routed mAP vs baseline")
    ax.set_title("Intervention validation summary", fontsize=16, weight="bold", color=NEUTRAL)
    ax.grid(axis="x", color=GRID, alpha=0.7)
    return _save(fig, path)


def save_topomoe_figures(records, metrics, anchor_vocab, out_dir, interventions=None, context=None):
    context = context or {}
    figure_paths = []
    for creator in [
        lambda: save_topomoe_overview(out_dir, records, anchor_vocab),
        lambda: save_routing_spectrum(out_dir, records),
        lambda: save_case_routing_flow(out_dir, records),
        lambda: save_topology_consistency(out_dir, records),
        lambda: save_anchor_availability_stress(out_dir, metrics),
        lambda: save_topology_dynamics(out_dir, context.get("history_path")),
        lambda: save_checkpoint_comparison(out_dir, metrics, context.get("checkpoint_comparison")),
        lambda: save_intervention_summary(out_dir, interventions),
    ]:
        path = creator()
        if path:
            figure_paths.append(path)

    manifest = {
        "title": "Paper 2 TopoMoE-GDI v2 figures",
        "status": context.get("status", "single_seed"),
        "seed": context.get("seed"),
        "checkpoint_type": context.get("checkpoint_type"),
        "topology_mode": context.get("topology_mode"),
        "case_count": metrics.get("case_count"),
        "query_count": metrics.get("query_count"),
        "note": "Single-seed figures are diagnostic until multi-seed aggregation is complete.",
        "source_artifacts": [
            "routing_records.json",
            "routing_spectrum.json",
            "topomoe_topology.json",
            "topomoe_diagnostics.json",
            "intervention_metrics.json",
            "test_metrics.json",
        ],
        "figures": [os.path.basename(path) for path in figure_paths],
    }
    return manifest


__all__ = [
    "save_topomoe_figures",
    "save_topomoe_overview",
    "save_routing_spectrum",
    "save_case_routing_flow",
    "save_topology_consistency",
    "save_anchor_availability_stress",
    "save_topology_dynamics",
    "save_checkpoint_comparison",
    "save_intervention_summary",
]
