"""Paper 4 metric-geodesic fusion diagnostics and figures."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODALITY_NAMES = ["T1", "T1ce", "T2", "FLAIR"]
REGION_NAMES = ["Necrotic/Core", "Edema", "Enhancing"]
COLORS = ["#2563EB", "#DC2626", "#059669", "#7C3AED", "#D97706", "#0891B2"]
BACKGROUND = "#F8FAFC"
INK = "#172033"


def _safe_stats(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1 if finite.size > 1 else 0)),
        "n": int(finite.size),
    }


def build_geodesic_payloads(records, context=None):
    fusion = records.get("geodesic_fusion", {})
    energy = np.asarray(fusion.get("geodesic_energy", []), dtype=np.float64)
    linear = np.asarray(fusion.get("linear_energy", []), dtype=np.float64)
    ratio = np.asarray(fusion.get("energy_ratio", []), dtype=np.float64)
    deviation = np.asarray(fusion.get("path_deviation", []), dtype=np.float64)
    adjacency = np.asarray(fusion.get("adjacency", []), dtype=np.float64)
    pair_indices = fusion.get("pair_indices", [])
    pair_names = [f"{MODALITY_NAMES[i]}-{MODALITY_NAMES[j]}" for i, j in pair_indices]

    diagnostics = {
        "fusion_mode": (context or {}).get("fusion_mode"),
        "metric_support": (context or {}).get("metric_support"),
        "fusion_graph": (context or {}).get("fusion_graph"),
        "case_count": int(records.get("case_count", 0)),
        "geodesic_energy": _safe_stats(energy),
        "linear_energy": _safe_stats(linear),
        "energy_ratio": _safe_stats(ratio),
        "path_deviation": _safe_stats(deviation),
        "by_region": {},
        "by_pair": {},
    }
    for region_idx, region_name in enumerate(REGION_NAMES):
        if energy.ndim == 3 and region_idx < energy.shape[1]:
            diagnostics["by_region"][region_name] = {
                "energy_ratio": _safe_stats(ratio[:, region_idx]),
                "path_deviation": _safe_stats(deviation[:, region_idx]),
            }
    for pair_idx, pair_name in enumerate(pair_names):
        if energy.ndim == 3 and pair_idx < energy.shape[2]:
            diagnostics["by_pair"][pair_name] = {
                "geodesic_energy": _safe_stats(energy[:, :, pair_idx]),
                "linear_energy": _safe_stats(linear[:, :, pair_idx]),
                "energy_ratio": _safe_stats(ratio[:, :, pair_idx]),
                "path_deviation": _safe_stats(deviation[:, :, pair_idx]),
            }

    graph = {
        "modality_names": MODALITY_NAMES,
        "region_names": REGION_NAMES,
        "pair_indices": pair_indices,
        "pair_names": pair_names,
        "adjacency_mean": np.nan_to_num(adjacency.mean(axis=0)).tolist() if adjacency.size else [],
        "adjacency_std": np.nan_to_num(adjacency.std(axis=0)).tolist() if adjacency.size else [],
        "edges": [],
    }
    if adjacency.size and energy.ndim == 3:
        mean_adjacency = adjacency.mean(axis=0)
        for region_idx, region_name in enumerate(REGION_NAMES):
            for pair_idx, (source, target) in enumerate(pair_indices):
                graph["edges"].append(
                    {
                        "region": region_name,
                        "source": MODALITY_NAMES[source],
                        "target": MODALITY_NAMES[target],
                        "mean_weight": float(
                            0.5
                            * (
                                mean_adjacency[region_idx, source, target]
                                + mean_adjacency[region_idx, target, source]
                            )
                        ),
                        "energy_ratio": float(np.nanmean(ratio[:, region_idx, pair_idx])),
                        "path_deviation": float(np.nanmean(deviation[:, region_idx, pair_idx])),
                    }
                )
    return diagnostics, graph


def _save(fig, path):
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)
    return os.path.basename(path)


def _pca(values):
    values = np.asarray(values, dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T
    if basis.shape[1] < 2:
        basis = np.pad(basis, ((0, 0), (0, 2 - basis.shape[1])))
    return centered @ basis


def save_path_projection(records, out_dir, context=None):
    path = os.path.join(out_dir, "geodesic_path_projection.png")
    samples = records.get("geodesic_fusion", {}).get("representative_paths", [])
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    if not samples:
        ax.text(0.5, 0.5, "No explicit path in latent-concat mode", ha="center", va="center")
        ax.set_axis_off()
    else:
        paths = np.asarray(samples[0]["paths"], dtype=np.float64)[0]
        prototypes = np.asarray(records.get("prototypes", []), dtype=np.float64)
        flat_paths = paths.reshape(-1, paths.shape[-1])
        combined = np.concatenate([flat_paths, prototypes], axis=0) if prototypes.size else flat_paths
        projected = _pca(combined)
        projected_paths = projected[: len(flat_paths)].reshape(paths.shape[0], paths.shape[1], 2)
        for pair_idx, line in enumerate(projected_paths):
            ax.plot(line[:, 0], line[:, 1], color=COLORS[pair_idx % len(COLORS)], lw=2)
            ax.scatter(line[[0, -1], 0], line[[0, -1], 1], color=COLORS[pair_idx % len(COLORS)], s=24)
        if prototypes.size:
            proto_xy = projected[len(flat_paths) :]
            ax.scatter(proto_xy[:, 0], proto_xy[:, 1], marker="x", s=28, color="#475569", alpha=0.6)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.grid(color="#CBD5E1", alpha=0.6)
    ax.set_title(
        f"Representative modality paths ({(context or {}).get('fusion_mode', 'unknown')})",
        color=INK,
        weight="bold",
    )
    return _save(fig, path)


def save_modality_graph(graph, out_dir):
    path = os.path.join(out_dir, "modality_geodesic_graph.png")
    adjacency = np.asarray(graph.get("adjacency_mean", []), dtype=np.float64)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.7))
    fig.patch.set_facecolor(BACKGROUND)
    if adjacency.size == 0:
        for ax in axes:
            ax.set_axis_off()
        axes[1].text(0.5, 0.5, "No graph in latent-concat mode", ha="center", va="center")
    else:
        vmax = max(float(np.nanmax(adjacency)), 1e-8)
        for region_idx, ax in enumerate(axes):
            image = ax.imshow(adjacency[region_idx], cmap="Blues", vmin=0, vmax=vmax)
            ax.set_xticks(range(4), MODALITY_NAMES, rotation=30)
            ax.set_yticks(range(4), MODALITY_NAMES)
            ax.set_title(REGION_NAMES[region_idx], weight="bold", color=INK)
        fig.colorbar(image, ax=axes, fraction=0.025, pad=0.03, label="Mean graph weight")
    fig.suptitle("Metric-geodesic modality graph", fontsize=15, weight="bold", color=INK)
    return _save(fig, path)


def save_energy_comparison(records, out_dir):
    path = os.path.join(out_dir, "geodesic_energy_comparison.png")
    fusion = records.get("geodesic_fusion", {})
    energy = np.asarray(fusion.get("geodesic_energy", []), dtype=np.float64)
    linear = np.asarray(fusion.get("linear_energy", []), dtype=np.float64)
    pair_indices = fusion.get("pair_indices", [])
    labels = [f"{MODALITY_NAMES[i]}-{MODALITY_NAMES[j]}" for i, j in pair_indices]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    fig.patch.set_facecolor(BACKGROUND)
    if energy.size:
        x = np.arange(len(labels))
        ax.bar(x - 0.18, energy.mean(axis=(0, 1)), 0.36, label="Learned path", color="#2563EB")
        ax.bar(x + 0.18, linear.mean(axis=(0, 1)), 0.36, label="Euclidean path", color="#94A3B8")
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.legend(frameon=False)
        ax.grid(axis="y", color="#CBD5E1", alpha=0.6)
    else:
        ax.text(0.5, 0.5, "No path-energy records", ha="center", va="center")
        ax.set_axis_off()
    ax.set_ylabel("Data-induced path energy")
    ax.set_title("Path energy by modality pair", fontsize=15, weight="bold", color=INK)
    return _save(fig, path)


def save_geodesic_figures(records, graph, out_dir, context=None):
    os.makedirs(out_dir, exist_ok=True)
    return [
        save_path_projection(records, out_dir, context),
        save_modality_graph(graph, out_dir),
        save_energy_comparison(records, out_dir),
    ]


__all__ = ["build_geodesic_payloads", "save_geodesic_figures"]
