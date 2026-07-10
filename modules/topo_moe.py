"""Topology-guided mixture-of-anchor experts."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _row_normalize(matrix, eps=1e-8):
    return matrix / matrix.sum(dim=-1, keepdim=True).clamp(min=eps)


def _js_divergence(p, q, eps=1e-8):
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    midpoint = 0.5 * (p + q)
    return 0.5 * (p * (p.log() - midpoint.log())).sum(dim=-1) + 0.5 * (
        q * (q.log() - midpoint.log())
    ).sum(dim=-1)


class TopoMoE(nn.Module):
    """Anchor-family router with legacy v1 and probability-correct v2 paths."""

    def __init__(
        self,
        shared_dim: int,
        topo_prior: torch.Tensor,
        anchor_family_ids,
        num_families: int,
        topo_mode: str = "prior_plus_learned",
        temperature: float = 0.07,
        router_hidden: int = 128,
        version: str = "v2",
        topo_epsilon: float = 1e-4,
        topo_temperature: float = 1.0,
        topo_beta_init: float = 0.1,
        route_mixture: str = "log_prob",
        refine_prototypes: bool = True,
        specialize_margin: float = 0.05,
    ):
        super().__init__()
        if version not in {"v1", "v2"}:
            raise ValueError(f"Unsupported TopoMoE version: {version}")
        if topo_mode not in {"prior_only", "learned_only", "prior_plus_learned"}:
            raise ValueError(f"Unsupported topology mode: {topo_mode}")
        if route_mixture not in {"product", "log_prob"}:
            raise ValueError(f"Unsupported route mixture: {route_mixture}")

        num_anchors = int(topo_prior.shape[0])
        self.num_anchors = num_anchors
        self.num_families = int(num_families)
        self.residual_index = self.num_families - 1
        self.num_anchor_families = max(self.num_families - 1, 1)
        self.topo_mode = topo_mode
        self.temperature = float(temperature)
        self.version = version
        self.topo_epsilon = float(topo_epsilon)
        self.topo_temperature = float(topo_temperature)
        self.route_mixture = "product" if version == "v1" else route_mixture
        self.refine_prototypes = bool(refine_prototypes and version == "v2")
        self.specialize_margin = float(specialize_margin)

        self.register_buffer("A_prior", topo_prior.float())
        family_ids = torch.as_tensor(anchor_family_ids, dtype=torch.long)
        self.register_buffer("family_ids", family_ids)
        membership, has_anchor = self._build_membership(family_ids)
        self.register_buffer("family_membership", membership)
        self.register_buffer("family_has_anchor", has_anchor)

        init = torch.zeros(num_anchors, num_anchors)
        if version == "v1" and topo_mode == "learned_only":
            init = 0.01 * torch.randn(num_anchors, num_anchors)
        self.A_raw = nn.Parameter(init)

        if version == "v2":
            beta = min(max(float(topo_beta_init), 1e-5), 1.0 - 1e-5)
            self.topo_beta_logit = nn.Parameter(torch.tensor(math.log(beta / (1.0 - beta))))
        else:
            self.register_parameter("topo_beta_logit", None)

        self.router = nn.Sequential(
            nn.Linear(shared_dim + self.num_families, router_hidden),
            nn.SiLU(),
            nn.Linear(router_hidden, self.num_families),
        )
        self.family_projections = nn.ModuleList(
            [nn.Linear(shared_dim, shared_dim) for _ in range(self.num_anchor_families)]
        )
        self.residual_expert = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.SiLU(),
        )

    def _build_membership(self, family_ids):
        membership = torch.zeros(
            self.num_families,
            self.num_anchors,
            device=family_ids.device,
            dtype=torch.float32,
        )
        membership[family_ids, torch.arange(self.num_anchors, device=family_ids.device)] = 1.0
        has_anchor = (membership.sum(dim=1) > 0).float()
        membership = membership / membership.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return membership, has_anchor

    def _symmetric_delta(self):
        return 0.5 * (self.A_raw + self.A_raw.t())

    def initial_topology(self):
        if self.version == "v1":
            if self.topo_mode == "prior_only":
                return self.A_prior
            if self.topo_mode == "learned_only":
                weights = F.softplus(torch.zeros_like(self.A_raw))
                return _row_normalize(weights)
            return self.A_prior
        if self.topo_mode == "prior_only":
            return self.A_prior
        if self.topo_mode == "learned_only":
            return F.softmax(torch.zeros_like(self.A_raw), dim=-1)
        logits = torch.log(self.A_prior + self.topo_epsilon)
        return F.softmax(logits, dim=-1)

    def effective_topology(self):
        if self.version == "v1":
            if self.topo_mode == "prior_only":
                return self.A_prior
            if self.topo_mode == "learned_only":
                return _row_normalize(F.softplus(self.A_raw))
            weights = F.relu(self.A_prior + self.A_raw)
            row_sums = weights.sum(dim=1, keepdim=True)
            return torch.where(row_sums > 0, weights / row_sums.clamp(min=1e-8), self.A_prior)

        if self.topo_mode == "prior_only":
            return self.A_prior
        delta = self._symmetric_delta() / max(self.topo_temperature, 1e-8)
        if self.topo_mode == "learned_only":
            return F.softmax(delta, dim=-1)
        logits = torch.log(self.A_prior + self.topo_epsilon) + delta
        return F.softmax(logits, dim=-1)

    def _refined_prototypes(self, prototypes, adjacency):
        proto_norm = F.normalize(prototypes, dim=-1)
        if not self.refine_prototypes:
            return proto_norm
        beta = torch.sigmoid(self.topo_beta_logit)
        propagated = adjacency @ proto_norm
        return F.normalize((1.0 - beta) * proto_norm + beta * propagated, dim=-1)

    def _family_topology(self, adjacency, membership):
        block = membership @ adjacency @ membership.t()
        row_sums = block.sum(dim=1, keepdim=True)
        eye = torch.eye(self.num_families, device=block.device, dtype=block.dtype)
        return torch.where(row_sums > 0, block / row_sums.clamp(min=1e-8), eye)

    def _family_context(self, shared_nodes, prototypes, adjacency, membership, has_anchor):
        centroids = membership @ prototypes
        residual_centroid = prototypes.mean(dim=0, keepdim=True)
        has_anchor = has_anchor.unsqueeze(-1)
        centroids = has_anchor * centroids + (1.0 - has_anchor) * residual_centroid
        centroids = F.normalize(self._family_topology(adjacency, membership) @ centroids, dim=-1)
        return torch.einsum("bnd,fd->bnf", F.normalize(shared_nodes, dim=-1), centroids)

    def _topology_losses(self, adjacency):
        zero = adjacency.sum() * 0
        if self.topo_mode == "prior_only":
            return zero, zero
        if self.version == "v1":
            return (adjacency - self.A_prior).pow(2).mean(), zero
        prior_eps = _row_normalize(self.A_prior + self.topo_epsilon)
        prior_loss = _js_divergence(adjacency, prior_eps).mean()
        delta_loss = self._symmetric_delta().abs().mean()
        return prior_loss, delta_loss

    def _specialize_v2(self, routing_weights):
        anchor_routes = routing_weights[..., : self.num_anchor_families]
        anchor_routes = _row_normalize(anchor_routes)
        node_profiles = anchor_routes.mean(dim=0)
        penalties = []
        for left in range(node_profiles.shape[0]):
            for right in range(left + 1, node_profiles.shape[0]):
                divergence = _js_divergence(node_profiles[left], node_profiles[right])
                penalties.append(F.relu(self.specialize_margin - divergence))
        if not penalties:
            return routing_weights.sum() * 0
        return torch.stack(penalties).mean()

    @staticmethod
    def _specialize_v1(routing_weights):
        node_routing = routing_weights.mean(dim=0)
        if node_routing.shape[0] < 2:
            return routing_weights.sum() * 0
        normalized = F.normalize(node_routing, dim=-1)
        similarity = normalized @ normalized.t()
        off_diagonal = similarity - torch.diag(torch.diagonal(similarity))
        return off_diagonal.sum() / (node_routing.shape[0] * (node_routing.shape[0] - 1))

    def topology_diagnostics(self, adjacency=None):
        if adjacency is None:
            adjacency = self.effective_topology()
        initial = self.initial_topology().to(adjacency.device)
        prior_zero = self.A_prior <= 0
        increase = F.relu(adjacency - initial)
        support = ~prior_zero
        zero = adjacency.sum() * 0
        return {
            "topo_beta": torch.sigmoid(self.topo_beta_logit) if self.topo_beta_logit is not None else zero,
            "topo_delta_l1": self._symmetric_delta().abs().mean(),
            "topo_delta_fro": torch.linalg.vector_norm(self._symmetric_delta()),
            "effective_prior_fro": torch.linalg.vector_norm(adjacency - self.A_prior),
            "prior_support_reweight": (adjacency - self.A_prior).abs()[support].mean() if support.any() else zero,
            "new_edge_mass": increase[prior_zero].sum() if prior_zero.any() else zero,
            "opened_edge_count": ((adjacency > initial + 1e-6) & prior_zero).float().sum(),
        }

    def forward(
        self,
        shared_nodes,
        anchor_prototypes=None,
        topology_override=None,
        disabled_family_ids=None,
        routing_override=None,
        family_ids_override=None,
    ):
        batch, num_nodes, _ = shared_nodes.shape
        family_ids = self.family_ids if family_ids_override is None else family_ids_override.to(shared_nodes.device)
        if family_ids_override is None:
            membership = self.family_membership
            has_anchor = self.family_has_anchor
        else:
            membership, has_anchor = self._build_membership(family_ids)

        adjacency = self.effective_topology() if topology_override is None else topology_override.to(shared_nodes.device)
        refined_prototypes = None
        if anchor_prototypes is not None:
            refined_prototypes = self._refined_prototypes(anchor_prototypes, adjacency)
            context = self._family_context(shared_nodes, refined_prototypes, adjacency, membership, has_anchor)
        else:
            context = torch.zeros(batch, num_nodes, self.num_families, device=shared_nodes.device)

        router_logits = self.router(torch.cat([shared_nodes, context], dim=-1))
        for family_idx in disabled_family_ids or []:
            router_logits[..., int(family_idx)] = -1e9
        if routing_override is None:
            routing_weights = F.softmax(router_logits, dim=-1)
        else:
            routing_weights = _row_normalize(routing_override.to(shared_nodes.device).clamp(min=0.0))

        routed_scores = None
        routed_log_probs = None
        routed_logits = None
        family_log_probs = None
        if refined_prototypes is not None:
            routed_scores = torch.full(
                (batch, num_nodes, self.num_anchors),
                -1e9 if self.route_mixture == "log_prob" else 0.0,
                device=shared_nodes.device,
            )
            family_log_probs = torch.full_like(routed_scores, -1e9)
            for family in range(self.num_anchor_families):
                anchor_idx = torch.where(family_ids == family)[0]
                if anchor_idx.numel() == 0:
                    continue
                projected = F.normalize(self.family_projections[family](shared_nodes), dim=-1)
                scores = torch.matmul(projected, refined_prototypes[anchor_idx].t()) / self.temperature
                conditional = F.log_softmax(scores, dim=-1)
                family_log_probs[..., anchor_idx] = conditional
                if self.route_mixture == "log_prob":
                    mixed = torch.log(routing_weights[..., family].clamp(min=1e-8)).unsqueeze(-1) + conditional
                else:
                    mixed = routing_weights[..., family].unsqueeze(-1) * scores
                routed_scores[..., anchor_idx] = mixed
            if self.route_mixture == "log_prob":
                routed_log_probs = routed_scores
            elif self.version == "v1":
                routed_logits = routed_scores

        residual_gate = routing_weights[..., self.residual_index].unsqueeze(-1)
        residual_contribution = residual_gate * self.residual_expert(shared_nodes)
        sparse_loss = -(routing_weights * torch.log(routing_weights.clamp(min=1e-8))).sum(dim=-1).mean()

        if self.version == "v2":
            anchor_usage = _row_normalize(routing_weights[..., : self.num_anchor_families]).mean(dim=(0, 1))
            target = torch.full_like(anchor_usage, 1.0 / self.num_anchor_families)
            balance_loss = ((anchor_usage - target) ** 2).mean()
            specialize_loss = self._specialize_v2(routing_weights)
        else:
            mean_usage = routing_weights.mean(dim=(0, 1))
            target = torch.full_like(mean_usage, 1.0 / self.num_families)
            balance_loss = ((mean_usage - target) ** 2).mean()
            specialize_loss = self._specialize_v1(routing_weights)

        topo_prior_loss, topo_delta_loss = self._topology_losses(adjacency)
        diagnostics = self.topology_diagnostics(adjacency)
        diagnostics["residual_usage"] = routing_weights[..., self.residual_index].mean()

        output = {
            "routing_weights": routing_weights,
            "router_logits": router_logits,
            "routed_scores": routed_scores,
            "routed_log_probs": routed_log_probs,
            "family_log_probs": family_log_probs,
            "refined_prototypes": refined_prototypes,
            "effective_topology": adjacency,
            "residual_contribution": residual_contribution,
            "topo_prior_loss": topo_prior_loss,
            "topo_delta_loss": topo_delta_loss,
            "topo_loss": topo_prior_loss,
            "sparse_loss": sparse_loss,
            "balance_loss": balance_loss,
            "specialize_loss": specialize_loss,
            "diagnostics": diagnostics,
        }
        if self.version == "v1":
            output["routed_logits"] = routed_logits
        return output


__all__ = ["TopoMoE"]
