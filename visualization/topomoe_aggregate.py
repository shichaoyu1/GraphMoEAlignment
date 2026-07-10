"""Multi-seed TopoMoE summary figures."""

import os

import matplotlib.pyplot as plt
import numpy as np

from glioma.visualization.topomoe import FAMILY_COLORS, GRID, LIGHT_BG, NEUTRAL


def _save(fig, path):
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=LIGHT_BG)
    plt.close(fig)
    return path


def save_multiseed_routing(payload, out_dir):
    path = os.path.join(out_dir, "topomoe_routing_spectrum_multiseed.png")
    families = payload.get("family_names", [])
    groups = payload.get("by_node", {})
    if not families or not groups:
        return None
    labels = sorted(groups)
    x = np.arange(len(labels))
    width = 0.8 / max(len(families), 1)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor(LIGHT_BG)
    for idx, family in enumerate(families):
        means = [groups[label][family]["mean"] for label in labels]
        stds = [groups[label][family]["std"] for label in labels]
        ax.bar(
            x + (idx - (len(families) - 1) / 2) * width,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=family,
            color=FAMILY_COLORS.get(family, "#4C566A"),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Routing weight, mean +/- SD")
    ax.set_title("Multi-seed routing spectrum", fontsize=16, weight="bold", color=NEUTRAL)
    ax.legend(frameon=False)
    ax.grid(axis="y", color=GRID, alpha=0.7)
    return _save(fig, path)


def save_topology_stability(payload, out_dir):
    path = os.path.join(out_dir, "topomoe_topology_stability.png")
    stability = np.asarray(payload.get("new_edge_stability", []), dtype=np.float32)
    if stability.size == 0:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    fig.patch.set_facecolor(LIGHT_BG)
    image = ax.imshow(stability, cmap="viridis", vmin=0, vmax=1)
    ax.set_xlabel("Target anchor index")
    ax.set_ylabel("Source anchor index")
    ax.set_title("New-edge stability across seeds", fontsize=15, weight="bold", color=NEUTRAL)
    fig.colorbar(image, ax=ax, label="Fraction of seeds opened")
    return _save(fig, path)


def save_multiseed_interventions(payload, out_dir):
    path = os.path.join(out_dir, "topomoe_interventions_multiseed.png")
    scenarios = payload.get("scenarios", {})
    if not scenarios:
        return None
    ordered = sorted(scenarios.items(), key=lambda item: item[1]["delta_map"]["mean"])
    labels = [name.replace("_", " ") for name, _ in ordered]
    means = [entry["delta_map"]["mean"] for _, entry in ordered]
    stds = [entry["delta_map"]["std"] for _, entry in ordered]
    colors = ["#D95F02" if value < 0 else "#1B9E77" for value in means]
    fig, ax = plt.subplots(figsize=(11, max(5.5, len(labels) * 0.45)))
    fig.patch.set_facecolor(LIGHT_BG)
    y = np.arange(len(labels))
    ax.barh(y, means, xerr=stds, capsize=3, color=colors)
    ax.axvline(0, color=NEUTRAL, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Delta routed mAP, mean +/- SD")
    ax.set_title("Multi-seed intervention effects", fontsize=16, weight="bold", color=NEUTRAL)
    ax.grid(axis="x", color=GRID, alpha=0.7)
    return _save(fig, path)


def save_multiseed_figures(routing, topology, interventions, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    paths = [
        save_multiseed_routing(routing, out_dir),
        save_topology_stability(topology, out_dir),
        save_multiseed_interventions(interventions, out_dir),
    ]
    return [os.path.basename(path) for path in paths if path]


__all__ = ["save_multiseed_figures"]
