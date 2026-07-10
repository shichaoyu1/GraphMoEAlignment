"""Semantic-alignment visualization utilities."""

from collections import defaultdict
import os

import matplotlib.pyplot as plt
import numpy as np


def node_names_for_mode(node_mode):
    if node_mode == "regions":
        return ["Necrotic/Core", "Edema", "Enhancing"]
    return ["T1", "T1ce", "T2", "FLAIR"]


def adjacency_to_laplacian(adjacency):
    adjacency = np.asarray(adjacency, dtype=np.float32)
    adjacency_sym = 0.5 * (adjacency + adjacency.T)
    degree = np.diag(adjacency_sym.sum(axis=1))
    return degree - adjacency_sym


def _positions(groups):
    positions = {}
    mri_ids = [idx for idx, group in enumerate(groups) if group == "MRI"]
    anchor_ids = [idx for idx, group in enumerate(groups) if group != "MRI"]

    for order, idx in enumerate(mri_ids):
        y = 0.5 if len(mri_ids) <= 1 else 0.86 - order * (0.72 / max(len(mri_ids) - 1, 1))
        positions[idx] = (0.20, y)
    for order, idx in enumerate(anchor_ids):
        y = 0.5 if len(anchor_ids) <= 1 else 0.90 - order * (0.80 / max(len(anchor_ids) - 1, 1))
        positions[idx] = (0.76, y)
    return positions


def plot_semantic_graph(node_names, adjacency, groups=None, save_path="semantic_graph.png", title="Semantic Unit Graph"):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    adjacency = np.asarray(adjacency, dtype=np.float32)
    groups = groups or ["MRI"] * len(node_names)
    positions = _positions(groups)
    colors = {"MRI": "#5DADE2", "Pathology": "#E67E22", "Gene": "#58D68D"}

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#0b0f19")
    ax.set_facecolor("#0b0f19")
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    max_weight = float(adjacency.max()) if adjacency.size else 1.0
    max_weight = max(max_weight, 1e-6)
    for source in range(len(node_names)):
        for target in range(len(node_names)):
            weight = float(adjacency[source, target])
            if source == target or weight <= 1e-6:
                continue
            if source > target and abs(weight - adjacency[target, source]) < 1e-6:
                continue
            x0, y0 = positions[source]
            x1, y1 = positions[target]
            alpha = 0.20 + 0.70 * weight / max_weight
            line_width = 0.8 + 4.0 * weight / max_weight
            arrow = "->" if abs(weight - adjacency[target, source]) > 1e-6 else "-"
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops={
                    "arrowstyle": arrow,
                    "color": "#c8d6e5",
                    "lw": line_width,
                    "alpha": alpha,
                    "connectionstyle": "arc3,rad=0.08",
                    "shrinkA": 28,
                    "shrinkB": 28,
                },
            )
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.025, f"{weight:.2f}", color="#dfe6e9", fontsize=8, ha="center", va="center", alpha=alpha)

    for idx, name in enumerate(node_names):
        x, y = positions[idx]
        group = groups[idx]
        color = colors.get(group, "#a29bfe")
        ax.scatter([x], [y], s=2200, color=color, alpha=0.25, edgecolor=color, linewidth=2)
        ax.text(x, y, name, color="white", fontsize=10, ha="center", va="center")

    ax.text(0.5, 0.97, title, color="white", fontsize=16, ha="center", va="top", fontweight="bold")
    plt.savefig(save_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_matrix(matrix, node_names, save_path, title, cmap="coolwarm"):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    matrix = np.asarray(matrix, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#0b0f19")
    ax.set_facecolor("#0b0f19")
    vmax = float(np.max(np.abs(matrix))) if matrix.size else 1.0
    vmax = max(vmax, 1e-6)
    image = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(node_names)))
    ax.set_yticks(range(len(node_names)))
    ax.set_xticklabels(node_names, rotation=45, ha="right", color="white", fontsize=8)
    ax.set_yticklabels(node_names, color="white", fontsize=8)
    ax.set_title(title, color="white", fontsize=14, pad=14)
    ax.tick_params(colors="white")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", color="white", fontsize=7)

    cbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="white")
    plt.savefig(save_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_alignment_space_plot(records, anchor_vocab, out_dir, max_edges=160):
    query_vectors = records["query_vectors"]
    prototypes = records["prototypes"]
    if len(query_vectors) == 0:
        return

    points = np.concatenate([query_vectors, prototypes], axis=0)
    points = np.asarray(points, dtype=np.float32)
    points = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
    centered = points - points.mean(axis=0, keepdims=True)
    centered = np.nan_to_num(centered, nan=0.0, posinf=0.0, neginf=0.0)

    def project_to_2d(matrix):
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float32)
        if matrix.shape[1] == 0:
            return np.zeros((matrix.shape[0], 2), dtype=np.float32)
        try:
            _, _, vt = np.linalg.svd(matrix, full_matrices=False)
            if vt.shape[0] >= 2:
                return matrix @ vt[:2].T
            if vt.shape[0] == 1:
                return np.pad(matrix @ vt[:1].T, ((0, 0), (0, 1)))
        except np.linalg.LinAlgError:
            pass
        if matrix.shape[1] >= 2:
            return matrix[:, :2]
        return np.pad(matrix[:, :1], ((0, 0), (0, 1)))

    coords = project_to_2d(centered)
    query_coords = coords[: len(query_vectors)]
    anchor_coords = coords[len(query_vectors) :]

    colors = {"MRI": "#2E86DE", "Pathology": "#E67E22", "Gene": "#27AE60", "Clinical": "#9B59B6"}
    markers = {
        "Necrotic/Core": "^",
        "Edema": "o",
        "Enhancing": "*",
        "T1": "^",
        "T1ce": "*",
        "T2": "s",
        "FLAIR": "o",
        "pathology-grade": "D",
        "pathology-diagnosis": "P",
        "molecular-marker": "X",
        "clinical-context": "h",
    }

    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor("#0b0f19")
    ax.set_facecolor("#101725")

    edge_budget = min(max_edges, len(records["query_targets"]))
    edge_ids = np.linspace(0, len(records["query_targets"]) - 1, edge_budget, dtype=int) if edge_budget else []
    for query_idx in edge_ids:
        for anchor_id in records["query_targets"][query_idx][:1]:
            ax.plot(
                [query_coords[query_idx, 0], anchor_coords[anchor_id, 0]],
                [query_coords[query_idx, 1], anchor_coords[anchor_id, 1]],
                color="#d0d7de",
                alpha=0.12,
                linewidth=0.7,
            )

    seen = set()
    for idx, record in enumerate(records["query_records"]):
        node_name = record["node_name"]
        label = f"MRI: {node_name}"
        ax.scatter(
            query_coords[idx, 0],
            query_coords[idx, 1],
            s=150 if node_name == "Enhancing" else 72,
            c=colors["MRI"],
            marker=markers.get(node_name, "o"),
            edgecolors="white",
            linewidths=0.5,
            alpha=0.78,
            label=label if label not in seen else None,
        )
        seen.add(label)

    for idx, anchor in enumerate(anchor_vocab):
        label = f"{anchor['source']}: {anchor['label']}"
        ax.scatter(
            anchor_coords[idx, 0],
            anchor_coords[idx, 1],
            s=160,
            c=colors.get(anchor["source"], "#95A5A6"),
            marker=markers.get(anchor["node_type"], "D"),
            edgecolors="white",
            linewidths=0.8,
            alpha=0.94,
            label=label if label not in seen else None,
        )
        seen.add(label)

    ax.set_title("Semantic Unit Alignment Space", color="white", fontsize=15, fontweight="bold")
    ax.set_xlabel("PC1 of shared semantic space", color="white")
    ax.set_ylabel("PC2 of shared semantic space", color="white")
    ax.tick_params(colors="#bdc3c7")
    ax.grid(color="white", alpha=0.10)
    for spine in ax.spines.values():
        spine.set_color("#34495e")
    ax.legend(loc="best", fontsize=7, facecolor="#17202a", edgecolor="#566573", labelcolor="white")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "semantic_unit_alignment_space.png"), dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_semantic_unit_graph(records, anchor_vocab, args, out_dir):
    node_names = node_names_for_mode(args.node_mode)
    graph_nodes = list(node_names) + [anchor["label"] for anchor in anchor_vocab]
    groups = ["MRI"] * len(node_names) + [anchor["source"] for anchor in anchor_vocab]
    adjacency = np.zeros((len(graph_nodes), len(graph_nodes)), dtype=np.float32)
    if len(records["query_vectors"]) == 0:
        return

    query_vec = np.nan_to_num(records["query_vectors"], nan=0.0, posinf=0.0, neginf=0.0)
    proto_vec = np.nan_to_num(records["prototypes"], nan=0.0, posinf=0.0, neginf=0.0)
    query_norm = query_vec / (np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-8)
    proto_norm = proto_vec / (np.linalg.norm(proto_vec, axis=1, keepdims=True) + 1e-8)
    scores = query_norm @ proto_norm.T
    score_buckets = defaultdict(list)
    for query_idx, record in enumerate(records["query_records"]):
        node_idx = node_names.index(record["node_name"])
        for anchor_idx in range(len(anchor_vocab)):
            score_buckets[(node_idx, len(node_names) + anchor_idx)].append(float(scores[query_idx, anchor_idx]))

    for (source, target), values in score_buckets.items():
        weight = float(np.mean(values))
        adjacency[source, target] = max(weight, 0.0)
        adjacency[target, source] = max(weight, 0.0)

    positive_edges = []
    for source in range(len(node_names)):
        candidates = [(target, adjacency[source, target]) for target in range(len(node_names), len(graph_nodes))]
        candidates.sort(key=lambda item: item[1], reverse=True)
        positive_edges.extend((source, target, weight) for target, weight in candidates[: args.graph_top_k] if weight > 0)

    sparse = np.zeros_like(adjacency)
    for source, target, weight in positive_edges:
        sparse[source, target] = weight
        sparse[target, source] = weight

    plot_semantic_graph(
        graph_nodes,
        sparse,
        groups,
        save_path=os.path.join(out_dir, "semantic_unit_graph_50patients.png"),
        title=f"Multi-patient Semantic Unit Graph (n={records['case_count']})",
    )
    plot_matrix(
        sparse,
        graph_nodes,
        save_path=os.path.join(out_dir, "semantic_unit_adjacency.png"),
        title="Semantic Unit Adjacency",
        cmap="magma",
    )
    plot_matrix(
        adjacency_to_laplacian(sparse),
        graph_nodes,
        save_path=os.path.join(out_dir, "semantic_unit_laplacian.png"),
        title="Semantic Unit Laplacian",
    )

__all__ = ["save_alignment_space_plot", "save_semantic_unit_graph"]
