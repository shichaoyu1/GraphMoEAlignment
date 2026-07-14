"""Paper 4 network for diagnostic metric-geodesic modality fusion."""

import torch
import torch.nn as nn

from glioma.models.encoders import ROI2DEncoder
from glioma.modules.geodesic_fusion import GeodesicModalityGraphFusion


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

        self.modality_encoders = nn.ModuleList(
            [ROI2DEncoder(in_ch=z_slices, feat_dim=feat_dim) for _ in range(num_modalities)]
        )
        self.modality_heads = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(feat_dim, shared_dim), nn.LayerNorm(shared_dim), nn.SiLU())
                for _ in range(num_modalities)
            ]
        )
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

    def encode_modality_regions(self, images, region_masks=None, modality_mask=None):
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
        for modality_idx, (encoder, head) in enumerate(zip(self.modality_encoders, self.modality_heads)):
            modality = images[:, modality_idx : modality_idx + 1].unsqueeze(1)
            masked = modality * region_masks[:, :, None]
            region_input = masked.reshape(batch * self.num_regions, depth, height, width)
            raw = encoder(region_input).reshape(batch, self.num_regions, -1)
            raw_by_modality.append(raw)
            shared_by_modality.append(head(raw))
        return torch.stack(raw_by_modality, dim=2), torch.stack(shared_by_modality, dim=2)

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
        raw_features, modality_nodes = self.encode_modality_regions(images, region_masks, modality_mask)
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
            "losses": {"geo_energy": fusion["geo_energy_loss"], "cons": zero},
        }
        if return_extras:
            output["extras"] = {
                "raw_features": raw_features,
                "shared_raw": shared,
                "shared": shared,
                "shared_mean": shared.mean(dim=1),
                "shared_norm": shared.norm(dim=-1).mean().detach(),
                "private_norm": zero.detach(),
                "diffusion_residual_norm": zero.detach(),
                "modality_nodes": fusion["modality_nodes"],
                "fusion_paths": fusion["paths"],
                "fusion_interior_paths": fusion["interior_paths"],
                "fusion_pair_indices": fusion["pair_indices"],
                "fusion_pair_valid": fusion["pair_valid"],
                "fusion_adjacency": fusion["adjacency"],
                "fusion_geodesic_energy": fusion["geodesic_energy"],
                "fusion_linear_energy": fusion["linear_energy"],
                "fusion_energy_ratio": fusion["energy_ratio"],
                "fusion_path_deviation": fusion["path_deviation"],
                "fusion_diagnostics": fusion["diagnostics"],
                "adjacency": None,
            }
        return output


__all__ = ["GliomaGeodesicFusionNet"]
