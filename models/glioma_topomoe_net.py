"""Paper 2 network for topology-guided anchor-family routing."""

import torch
import torch.nn as nn

from glioma.models.encoders import ROI2DEncoder
from glioma.modules.graph import SemanticGraphBuilder, graph_laplacian_consistency
from glioma.modules.topo_moe import TopoMoE


class GliomaTopoMoENet(nn.Module):
    """Shared semantic-unit encoder plus TopoMoE, without graph or diffusion."""

    def __init__(
        self,
        z_slices: int = 7,
        num_modalities: int = 4,
        node_mode: str = "regions",
        num_regions: int = 3,
        feat_dim: int = 256,
        shared_dim: int = 128,
        graph_type: str = "no_graph",
        topo_prior=None,
        anchor_family_ids=None,
        num_families: int = 0,
        topo_mode: str = "prior_plus_learned",
        align_temperature: float = 0.07,
        topomoe_version: str = "v2",
        topo_epsilon: float = 1e-4,
        topo_temperature: float = 1.0,
        topo_beta_init: float = 0.1,
        route_mixture: str = "log_prob",
        refine_prototypes: bool = True,
        specialize_margin: float = 0.05,
    ):
        super().__init__()
        if topo_prior is None or anchor_family_ids is None or num_families < 2:
            raise ValueError("GliomaTopoMoENet requires topology and at least two expert families")
        self.node_mode = node_mode
        self.num_input_modalities = num_modalities
        self.num_regions = num_regions
        self.num_nodes = num_regions if node_mode == "regions" else num_modalities
        self.shared_dim = shared_dim
        self.graph_type = graph_type
        self.use_private = False
        self.use_diffusion = False
        self.use_topo_moe = True
        self.moe_module = "topo_moe"
        self.topomoe_version = topomoe_version

        encoder_channels = z_slices * num_modalities if node_mode == "regions" else z_slices
        self.encoders = nn.ModuleList(
            [ROI2DEncoder(in_ch=encoder_channels, feat_dim=feat_dim) for _ in range(self.num_nodes)]
        )
        self.shared_heads = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(feat_dim, shared_dim), nn.LayerNorm(shared_dim), nn.SiLU())
                for _ in range(self.num_nodes)
            ]
        )
        if graph_type == "no_graph":
            self.graph_builder = None
            self.graph_norm = None
        else:
            self.graph_builder = SemanticGraphBuilder(shared_dim, graph_type=graph_type)
            self.graph_norm = nn.LayerNorm(shared_dim)
        self.topo_moe = TopoMoE(
            shared_dim=shared_dim,
            topo_prior=topo_prior,
            anchor_family_ids=anchor_family_ids,
            num_families=num_families,
            topo_mode=topo_mode,
            temperature=align_temperature,
            version=topomoe_version,
            topo_epsilon=topo_epsilon,
            topo_temperature=topo_temperature,
            topo_beta_init=topo_beta_init,
            route_mixture=route_mixture,
            refine_prototypes=refine_prototypes,
            specialize_margin=specialize_margin,
        )

    def _encode_modalities(self, images):
        features = torch.stack(
            [encoder(images[:, idx]) for idx, encoder in enumerate(self.encoders)],
            dim=1,
        )
        shared = torch.stack(
            [head(features[:, idx]) for idx, head in enumerate(self.shared_heads)],
            dim=1,
        )
        return features, shared

    def _encode_regions(self, images, region_masks):
        if region_masks is None:
            region_masks = torch.ones(
                images.shape[0],
                self.num_regions,
                images.shape[2],
                images.shape[3],
                images.shape[4],
                device=images.device,
                dtype=images.dtype,
            )
        features = []
        for region_idx, encoder in enumerate(self.encoders):
            masked = images * region_masks[:, region_idx : region_idx + 1]
            region_input = masked.reshape(masked.shape[0], -1, masked.shape[-2], masked.shape[-1])
            features.append(encoder(region_input))
        features = torch.stack(features, dim=1)
        shared = torch.stack(
            [head(features[:, idx]) for idx, head in enumerate(self.shared_heads)],
            dim=1,
        )
        return features, shared

    def encode_semantic_units(self, images, region_masks=None, modality_mask=None):
        if modality_mask is not None:
            images = images * modality_mask[:, :, None, None, None]
        if self.node_mode == "regions":
            return self._encode_regions(images, region_masks)
        return self._encode_modalities(images)

    def route_shared_nodes(self, shared_nodes, anchor_prototypes, **intervention):
        return self.topo_moe(
            shared_nodes,
            anchor_prototypes=anchor_prototypes,
            **intervention,
        )

    def forward(
        self,
        images,
        labels=None,
        modality_mask=None,
        region_masks=None,
        return_extras=False,
        freeze_graph=False,
        anchor_prototypes=None,
        topomoe_intervention=None,
    ):
        del labels
        raw_features, shared_raw = self.encode_semantic_units(images, region_masks, modality_mask)
        shared = shared_raw
        adjacency = None
        if self.graph_builder is not None:
            adjacency = self.graph_builder(shared)
            if freeze_graph and self.training:
                adjacency = adjacency.detach()
            shared = self.graph_norm(shared + torch.matmul(adjacency, shared))
        topo_out = self.route_shared_nodes(
            shared,
            anchor_prototypes,
            **(topomoe_intervention or {}),
        )
        losses = {
            "cons": graph_laplacian_consistency(shared_raw, adjacency)
            if adjacency is not None
            else shared_raw.sum() * 0,
            "topo_prior": topo_out["topo_prior_loss"],
            "topo_delta": topo_out["topo_delta_loss"],
            "specialize": topo_out["specialize_loss"],
            "route_balance": topo_out["balance_loss"],
            "route_sparse": topo_out["sparse_loss"],
        }
        output = {"logits": None, "losses": losses}
        if return_extras:
            zero = shared.sum() * 0
            output["extras"] = {
                "raw_features": raw_features,
                "shared_raw": shared_raw,
                "shared": shared,
                "shared_mean": shared.mean(dim=1),
                "shared_norm": shared.norm(dim=-1).mean().detach(),
                "private_norm": zero.detach(),
                "diffusion_residual_norm": zero.detach(),
                "routing_weights": topo_out["routing_weights"],
                "router_logits": topo_out["router_logits"],
                "routed_scores": topo_out["routed_scores"],
                "routed_log_probs": topo_out["routed_log_probs"],
                "family_log_probs": topo_out["family_log_probs"],
                "effective_topology": topo_out["effective_topology"],
                "adjacency": adjacency,
                "topology_diagnostics": topo_out["diagnostics"],
                "residual_contribution": topo_out["residual_contribution"],
            }
        return output


__all__ = ["GliomaTopoMoENet"]
