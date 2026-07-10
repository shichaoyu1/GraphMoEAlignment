"""Synthetic forward checks for TopoMoE wiring.

Run from ``D:\code\pythonProject\PygMonAI``:
    python glioma/scratch/test_topomoe_forward.py
"""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from glioma.models.glioma_graph_diffusion_net import GliomaGraphDiffusionNet
from glioma.modules.topo_moe import TopoMoE


def assert_finite_scalar(value, name):
    assert value.ndim == 0, f"{name} should be scalar, got {tuple(value.shape)}"
    assert torch.isfinite(value), f"{name} should be finite"


def main():
    torch.manual_seed(7)
    batch = 2
    nodes = 3
    shared_dim = 16
    anchors = 5
    family_ids = [0, 0, 1, 1, 2]
    num_families = 4
    prior = torch.eye(anchors)
    prior = prior / prior.sum(dim=1, keepdim=True)
    prototypes = torch.randn(anchors, shared_dim)

    topo_moe = TopoMoE(
        shared_dim=shared_dim,
        topo_prior=prior,
        anchor_family_ids=family_ids,
        num_families=num_families,
    )
    shared_nodes = torch.randn(batch, nodes, shared_dim)
    topo_out = topo_moe(shared_nodes, anchor_prototypes=prototypes)

    routing = topo_out["routing_weights"]
    assert routing.shape == (batch, nodes, num_families)
    assert torch.allclose(routing.sum(dim=-1), torch.ones(batch, nodes), atol=1e-5)
    assert topo_out["routed_log_probs"].shape == (batch, nodes, anchors)
    assert topo_out["routed_log_probs"].reshape(-1, anchors).shape == (batch * nodes, anchors)
    for name in ["topo_prior_loss", "topo_delta_loss", "specialize_loss", "balance_loss", "sparse_loss"]:
        assert_finite_scalar(topo_out[name], name)

    model = GliomaGraphDiffusionNet(
        num_classes=1,
        z_slices=3,
        num_modalities=4,
        node_mode="regions",
        num_regions=nodes,
        feat_dim=16,
        shared_dim=shared_dim,
        private_dim=8,
        graph_type="learnable",
        moe_module="topo_moe",
        use_anchor=False,
        use_private=True,
        use_diffusion=False,
        topo_prior=prior,
        anchor_family_ids=family_ids,
        num_families=num_families,
    )
    images = torch.randn(batch, 4, 3, 32, 32)
    region_masks = torch.ones(batch, nodes, 3, 32, 32)
    model_out = model(images, region_masks=region_masks, return_extras=True, anchor_prototypes=prototypes)
    extras = model_out["extras"]
    losses = model_out["losses"]

    assert extras["routing_weights"].shape == (batch, nodes, num_families)
    assert torch.allclose(extras["routing_weights"].sum(dim=-1), torch.ones(batch, nodes), atol=1e-5)
    assert extras["routed_logits"].shape == (batch, nodes, anchors)
    for name in ["topo", "specialize", "route_balance", "route_sparse"]:
        assert_finite_scalar(losses[name], name)

    print("topomoe forward ok")


if __name__ == "__main__":
    main()
