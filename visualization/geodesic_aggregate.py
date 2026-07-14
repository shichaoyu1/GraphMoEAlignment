"""Multi-seed Paper 4 ablation figures."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BACKGROUND = "#F8FAFC"
INK = "#172033"


def _save(fig, path):
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)
    return os.path.basename(path)


def save_ablation_summary(payload, out_dir):
    path = os.path.join(out_dir, "paper4_ablation_summary.png")
    variants = payload.get("variants", {})
    labels = list(variants)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(BACKGROUND)
    if not labels:
        for ax in axes:
            ax.set_axis_off()
        return _save(fig, path)

    x = np.arange(len(labels))
    map_mean = [variants[name]["metrics"].get("map", {}).get("mean", np.nan) for name in labels]
    map_std = [variants[name]["metrics"].get("map", {}).get("std", 0.0) for name in labels]
    axes[0].bar(x, map_mean, yerr=map_std, capsize=4, color="#2563EB")
    axes[0].set_ylabel("Direct mAP, mean +/- SD")
    axes[0].set_title("Paper 4 retrieval ablations", weight="bold", color=INK)

    ratio = [
        variants[name]["diagnostics"].get("energy_ratio", {}).get("mean", np.nan)
        for name in labels
    ]
    deviation = [
        variants[name]["diagnostics"].get("path_deviation", {}).get("mean", np.nan)
        for name in labels
    ]
    axes[1].bar(x - 0.18, ratio, 0.36, label="Energy ratio", color="#059669")
    axes[1].bar(x + 0.18, deviation, 0.36, label="Path deviation", color="#D97706")
    axes[1].set_title("Path diagnostics", weight="bold", color=INK)
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_xticks(x, [label.replace("_", " ") for label in labels], rotation=25, ha="right")
        ax.grid(axis="y", color="#CBD5E1", alpha=0.6)
    return _save(fig, path)


def save_edge_stability(payload, out_dir):
    path = os.path.join(out_dir, "paper4_edge_stability.png")
    variants = payload.get("variants", {})
    selected = variants.get("full_geodesic_graph") or next(iter(variants.values()), {})
    stability = np.asarray(selected.get("graph", {}).get("edge_rank_stability", []), dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7))
    fig.patch.set_facecolor(BACKGROUND)
    if stability.size == 0:
        for ax in axes:
            ax.set_axis_off()
        axes[1].text(0.5, 0.5, "No graph stability records", ha="center", va="center")
    else:
        for region_idx, ax in enumerate(axes):
            image = ax.imshow(stability[region_idx], cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks(range(4), ["T1", "T1ce", "T2", "FLAIR"], rotation=30)
            ax.set_yticks(range(4), ["T1", "T1ce", "T2", "FLAIR"])
            ax.set_title(["Necrotic/Core", "Edema", "Enhancing"][region_idx], weight="bold")
        fig.colorbar(image, ax=axes, fraction=0.025, pad=0.03, label="Top-edge fraction")
    fig.suptitle("Cross-seed modality-edge stability", fontsize=15, weight="bold", color=INK)
    return _save(fig, path)


def save_geodesic_aggregate_figures(payload, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    return [save_ablation_summary(payload, out_dir), save_edge_stability(payload, out_dir)]


__all__ = ["save_geodesic_aggregate_figures"]
