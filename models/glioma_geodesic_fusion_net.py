"""Paper 4 network for diagnostic metric-geodesic modality fusion."""

import torch
import torch.nn as nn

from glioma.models.encoders import ROI2DEncoder
from glioma.modules.geodesic_fusion import GeodesicModalityGraphFusion
from glioma.modules.hierarchical_spd_fusion import HierarchicalSPDGraphFusion


class GliomaGeodesicFusionNet(nn.Module):
    """Modality-specific encoders followed by an intra-region path graph."""

    def __init__(
        self,
        z_slices: int = 7,
        num_modalities: int = 4,
        num_regions: int = 3,
        feat_dim: int = 256,
        shared_dim: int = 128,
        fusion_mode: str = "geodesic",
        geo_metric_support: str = "case_and_anchors",
        geo_path_steps: int = 5,
        geo_gamma: float = 0.5,
        geo_rho: float = 1e-3,
        geo_metric_alpha: float = 1.0,
        geo_graph_temperature: float = 1.0,
        geo_bend_init: float = 0.1,
        use_fusion_graph: bool = True,
        paper4_fusion_backend: str = "vector_geodesic",
        spd_dim: int = 16,
        spd_geometry: str = "spd",
        spd_eigenvalue_min: float = 1e-4,
        spd_local_temperature: float = 1.0,
        spd_upper_temperature: float = 1.0,
        use_spd_upper_graph: bool = True,
        use_spd_anchor_families: bool = True,
        anchor_family_ids=None,
        anchor_family_names=None,
        family_prior=None,
    ):
        super().__init__()
        self.node_mode = "regions"
        self.num_input_modalities = int(num_modalities)
        self.num_regions = int(num_regions)
        self.num_nodes = int(num_regions)
        self.shared_dim = int(shared_dim)
        self.graph_type = "no_graph"
        self.use_private = False
        self.use_diffusion = False
        self.use_topo_moe = False
        self.use_geodesic_fusion = True
        self.requires_anchor_prototypes = True
        self.fusion_mode = fusion_mode
        self.paper4_fusion_backend = paper4_fusion_backend
        self.use_manifold_fusion = paper4_fusion_backend == "spd_hierarchical"
        if paper4_fusion_backend not in {"vector_geodesic", "spd_hierarchical"}:
            raise ValueError(f"Unsupported Paper 4 fusion backend: {paper4_fusion_backend}")

        self.modality_encoders = nn.ModuleList(
            [ROI2DEncoder(in_ch=z_slices, feat_dim=feat_dim) for _ in range(num_modalities)]
        )
        self.modality_heads = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(feat_dim, shared_dim), nn.LayerNorm(shared_dim), nn.SiLU())
                for _ in range(num_modalities)
            ]
        )
        if self.use_manifold_fusion:
            if anchor_family_ids is None or anchor_family_names is None:
                raise ValueError("SPD hierarchical fusion requires anchor family metadata")
            self.fusion = HierarchicalSPDGraphFusion(
                token_dim=192,
                shared_dim=shared_dim,
                family_ids=anchor_family_ids,
                family_names=anchor_family_names,
                family_prior=family_prior,
                spd_dim=spd_dim,
                num_regions=num_regions,
                num_modalities=num_modalities,
                geometry=spd_geometry,
                use_upper_graph=use_spd_upper_graph,
                use_anchor_families=use_spd_anchor_families,
                path_steps=geo_path_steps,
                eigenvalue_min=spd_eigenvalue_min,
                local_temperature=spd_local_temperature,
                upper_temperature=spd_upper_temperature,
            )
        else:
            self.fusion = GeodesicModalityGraphFusion(
                shared_dim=shared_dim,
                num_modalities=num_modalities,
                fusion_mode=fusion_mode,
                metric_support=geo_metric_support,
                path_steps=geo_path_steps,
                gamma=geo_gamma,
                rho=geo_rho,
                metric_alpha=geo_metric_alpha,
                graph_temperature=geo_graph_temperature,
                bend_init=geo_bend_init,
                use_graph=use_fusion_graph,
            )

    def encode_modality_regions(self, images, region_masks=None, modality_mask=None, return_tokens=False):
        batch, modalities, depth, height, width = images.shape
        if modalities != self.num_input_modalities:
            raise ValueError(f"Expected {self.num_input_modalities} modalities, got {modalities}")
        if region_masks is None:
            region_masks = torch.ones(
                batch,
                self.num_regions,
                depth,
                height,
                width,
                device=images.device,
                dtype=images.dtype,
            )
        if modality_mask is not None:
            images = images * modality_mask[:, :, None, None, None].to(images.dtype)

        raw_by_modality = []
        shared_by_modality = []
        tokens_by_modality = []
        for modality_idx, (encoder, head) in enumerate(zip(self.modality_encoders, self.modality_heads)):
            modality = images[:, modality_idx : modality_idx + 1].unsqueeze(1)
            masked = modality * region_masks[:, :, None]
            region_input = masked.reshape(batch * self.num_regions, depth, height, width)
            if return_tokens:
                raw, tokens = encoder.forward_with_tokens(region_input)
                tokens_by_modality.append(
                    tokens.reshape(batch, self.num_regions, tokens.shape[-2], tokens.shape[-1])
                )
            else:
                raw = encoder(region_input)
            raw = raw.reshape(batch, self.num_regions, -1)
            raw_by_modality.append(raw)
            shared_by_modality.append(head(raw))
        raw = torch.stack(raw_by_modality, dim=2)
        shared = torch.stack(shared_by_modality, dim=2)
        tokens = torch.stack(tokens_by_modality, dim=2) if return_tokens else None
        return raw, shared, tokens

    def forward(
        self,
        images,
        labels=None,
        modality_mask=None,
        region_masks=None,
        return_extras=False,
        freeze_graph=False,
        anchor_prototypes=None,
    ):
        del labels, freeze_graph
        if anchor_prototypes is None:
            raise ValueError("Paper 4 requires anchor prototypes for diagnostic metric fusion")
        raw_features, modality_nodes, spatial_tokens = self.encode_modality_regions(
            images,
            region_masks,
            modality_mask,
            return_tokens=self.use_manifold_fusion,
        )
        if self.use_manifold_fusion:
            fusion = self.fusion(
                spatial_tokens,
                anchor_prototypes=anchor_prototypes,
                modality_mask=modality_mask,
            )
        else:
            fusion = self.fusion(
                modality_nodes,
                anchor_prototypes=anchor_prototypes,
                modality_mask=modality_mask,
                return_paths=return_extras,
            )
        shared = fusion["fused_nodes"]
        zero = shared.sum() * 0.0
        output = {
            "logits": None,
            "losses": {
                "geo_energy": fusion.get("geo_energy_loss", zero),
                "spd_condition": fusion.get("condition_loss", zero),
                "manifold_topology": fusion.get("topology_loss", zero),
                "cons": zero,
            },
        }
        if return_extras:
            extras = {
                "raw_features": raw_features,
                "shared_raw": shared,
                "shared": shared,
                "shared_mean": shared.mean(dim=1),
                "shared_norm": shared.norm(dim=-1).mean().detach(),
                "private_norm": zero.detach(),
                "diffusion_residual_norm": zero.detach(),
                "modality_nodes": fusion.get("modality_nodes", modality_nodes),
                "fusion_paths": fusion.get("paths"),
                "fusion_interior_paths": fusion.get("interior_paths", fusion.get("interior_path_embeddings")),
                "fusion_pair_indices": fusion["pair_indices"],
                "fusion_pair_valid": fusion["pair_valid"],
                "fusion_adjacency": fusion.get("adjacency"),
                "fusion_geodesic_energy": fusion.get("geodesic_energy"),
                "fusion_linear_energy": fusion.get("linear_energy"),
                "fusion_energy_ratio": fusion.get("energy_ratio"),
                "fusion_path_deviation": fusion.get("path_deviation"),
                "fusion_diagnostics": fusion["diagnostics"],
                "adjacency": None,
            }
            if self.use_manifold_fusion:
                extras.update(
                    {
                        "manifold_local_adjacency": fusion["local_adjacency"],
                        "manifold_upper_adjacency": fusion["upper_adjacency"],
                        "manifold_local_distances": fusion["local_distances"],
                        "manifold_upper_distances": fusion["upper_distances"],
                        "manifold_raw_scales": fusion["raw_scales"],
                        "manifold_raw_spd_traces": fusion["raw_spd_traces"],
                        "manifold_spd_eigenvalues": fusion["spd_eigenvalues"],
                        "manifold_condition_numbers": fusion["condition_numbers"],
                        "manifold_upper_node_names": fusion["upper_node_names"],
                    }
                )
            output["extras"] = extras
        return output


__all__ = ["GliomaGeodesicFusionNet"]
