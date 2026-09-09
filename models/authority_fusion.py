"""Information-matched authority baselines and protected Log-SPD graph fusion."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from glioma.modules.authority_projection import GeneralAuthorityProjection, project_registered
from glioma.modules.authority_rules import compile_authority, relation_residuals
from glioma.modules.hierarchical_spd_fusion import spd_logm, spd_expm, symmetric_vectorize


TRAINED_VARIANTS = ("uniform", "learned", "reliability", "dominant", "transformer", "transformer_joint",
                    "learned_joint", "graph_only", "soft", "acf")
CORE_VARIANTS = ("learned", "learned_joint", "graph_only", "acf")
STATUS_NAMES = {0: "constrained", 1: "unconstrained", 2: "conflict", 3: "no_input", 4: "solver_failure"}


@dataclass
class AuthorityOutput:
    p0: torch.Tensor
    p_star: torch.Tensor
    probabilities: torch.Tensor
    protected_states: torch.Tensor
    protected_active: torch.Tensor
    relation_residuals: torch.Tensor
    status: torch.Tensor
    adjacency: torch.Tensor
    node_states: torch.Tensor
    compiled: object


def mean_augmented_spd(tokens, epsilon=1e-3):
    """SPD moment matrix with Schur complement covariance + diagonal regularizer."""
    mu = tokens.mean(-2)
    centered = tokens - mu.unsqueeze(-2)
    covariance = centered.transpose(-1, -2) @ centered / max(tokens.shape[-2] - 1, 1)
    top = torch.cat([covariance + mu.unsqueeze(-1) * mu.unsqueeze(-2), mu.unsqueeze(-1)], -1)
    bottom = torch.cat([mu, torch.ones_like(mu[..., :1])], -1).unsqueeze(-2)
    matrix = torch.cat([top, bottom], -2)
    jitter = torch.diag(torch.linspace(1, 2, matrix.shape[-1], device=matrix.device, dtype=matrix.dtype))
    matrix = matrix + epsilon * jitter
    return matrix / (matrix.diagonal(dim1=-2, dim2=-1).sum(-1) / matrix.shape[-1])[..., None, None]


class GraphUpdate(nn.Module):
    def __init__(self, dim, hidden, mass):
        super().__init__()
        self.mass = mass
        self.query = nn.Linear(dim, hidden, bias=False)
        self.key = nn.Linear(dim, hidden, bias=False)
        self.update = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, states, mask, available, num_modalities, mode, quality, bypass=False):
        count = states.shape[-2]
        diagonal = torch.eye(count, device=states.device, dtype=torch.bool)
        cross = mask & ~diagonal
        if mode == "uniform":
            scores = torch.zeros_like(mask, dtype=states.dtype)
        else:
            scores = self.query(states) @ self.key(states).transpose(-1, -2) / self.query.out_features ** .5
            if mode == "reliability":
                scores = scores + quality.clamp_min(1e-3).log()[..., None, :]
        weights = scores.masked_fill(~cross, -1e9).softmax(-1) * cross
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        budget = cross.any(-1).to(states.dtype) * self.mass
        adjacency = budget[..., None] * weights + (1 - budget)[..., None] * diagonal
        adjacency = adjacency * available[..., :, None]
        mixture = adjacency @ states
        updated = mixture + .1 * self.update(self.norm(mixture))
        if bypass:
            # Deliberately invalid shared gate used only as a positive leakage control.
            context = (states[..., :num_modalities, :] * available[..., :num_modalities, None]).sum(-2)
            context = context / available[..., :num_modalities].sum(-1).clamp_min(1)[..., None]
            updated = updated * (1 + .1 * context.sigmoid()[..., None, :])
        updated = updated * available[..., None]
        # Read-only support states cannot act as a sample-dependent relay.
        updated = torch.cat([updated[..., :num_modalities, :], states[..., num_modalities:, :]], -2)
        return updated, adjacency


class AuthorityFusion(nn.Module):
    def __init__(self, task, variant="acf", token_dim=8, num_modalities=4, spd_dim=8,
                 hidden_dim=64, layers=2, cross_mass=.35, geometry="spd", anchors=True,
                 mask_policy="authority", dominant_source=0, projection_backend="analytic"):
        super().__init__()
        if variant not in TRAINED_VARIANTS + ("rule_only",):
            raise ValueError("Unknown authority baseline")
        if geometry not in ("spd", "euclidean") or mask_policy not in ("authority", "role_shift"):
            raise ValueError("Unknown geometry or mask policy")
        if projection_backend not in ("analytic", "general"):
            raise ValueError("Unknown projection backend")
        self.projection_backend = projection_backend
        if spd_dim < 3 or layers < 1:
            raise ValueError("Need SPD dimension >=3 and >=1 update layer")
        self.task, self.variant, self.geometry = task, variant, geometry
        self.num_modalities, self.spd_dim = num_modalities, spd_dim
        self.relations = 1 if task == "s1" else 2
        self.classes = 4 if task == "s1" else 6
        self.supports = 4 if anchors else 0
        self.mask_policy, self.dominant_source = mask_policy, int(dominant_source)
        vector_dim = spd_dim * (spd_dim + 1) // 2
        self.adapters = nn.ModuleList([nn.Linear(token_dim, spd_dim-1) for _ in range(num_modalities)])
        self.context = nn.Linear(2*self.relations + 10, spd_dim-1)
        self.support_adapter = nn.Linear(8, spd_dim-1)
        self.register_buffer("semantic_templates", torch.eye(8)[:4])
        row, col = torch.triu_indices(spd_dim, spd_dim)
        basis = torch.zeros(vector_dim, spd_dim, spd_dim)
        basis[torch.arange(vector_dim), row, col] = torch.where(row == col, 1., 2.**-.5)
        basis[torch.arange(vector_dim), col, row] = torch.where(row == col, 1., 2.**-.5)
        self.register_buffer("symmetric_basis", basis)
        self.blocks = nn.ModuleList([GraphUpdate(vector_dim, hidden_dim, cross_mass) for _ in range(layers)])
        constraint_rows = 6 if task == "s1" else 4
        metadata_dim = 5*num_modalities + 2*num_modalities*self.relations + 2 + constraint_rows*(self.classes+2)
        self.policy_encoder = nn.Sequential(nn.Linear(metadata_dim, hidden_dim), nn.SiLU())
        self.is_transformer = variant.startswith("transformer")
        if self.is_transformer:
            self.transformer_input = nn.Linear(vector_dim, hidden_dim)
            block = nn.TransformerEncoderLayer(hidden_dim, 4, hidden_dim*2, dropout=0., batch_first=True)
            self.transformer = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
            readout_dim = hidden_dim
        else:
            readout_dim = vector_dim
        self.head = nn.Sequential(nn.Linear((num_modalities+self.supports)*readout_dim + hidden_dim, hidden_dim),
                                  nn.SiLU(), nn.Linear(hidden_dim, self.classes))

    def _policy_features(self, e, compiled):
        usable = (e.source_available & (e.verified_validity != 0)).to(e.tokens.dtype)
        values = [e.source_available.to(e.tokens.dtype), F.one_hot(e.verified_validity, 3).flatten(1).to(e.tokens.dtype),
                  e.source_quality * usable, (e.statements * usable[..., None]).flatten(1).to(e.tokens.dtype),
                  F.one_hot(e.authority_sources, self.num_modalities).flatten(1).to(e.tokens.dtype),
                  F.one_hot(e.target_id, 2).to(e.tokens.dtype),
                  (compiled.B * compiled.constraint_active[..., None]).flatten(1),
                  compiled.b * compiled.constraint_active, compiled.constraint_active.to(e.tokens.dtype)]
        return torch.cat(values, -1)

    def _encode(self, e, compiled):
        batch = len(e.tokens)
        roles = F.one_hot(e.authority_sources, self.num_modalities).transpose(1, 2).to(e.tokens.dtype)
        context = torch.cat([e.statements.to(e.tokens.dtype), e.source_quality[..., None],
                             F.one_hot(e.verified_validity, 3).to(e.tokens.dtype),
                             F.one_hot(e.target_id, 2)[:, None].expand(-1, self.num_modalities, -1).to(e.tokens.dtype),
                             torch.eye(self.num_modalities, device=e.tokens.device)[None].expand(batch, -1, -1), roles], -1)
        local_context = self.context(context)
        projected = torch.stack([adapter(e.tokens[:, 0, m]) + local_context[:, m, None]
                                 for m, adapter in enumerate(self.adapters)], 1)
        matrix = mean_augmented_spd(projected)
        if self.supports:
            support_mu = self.support_adapter(self.semantic_templates)
            support_matrix = mean_augmented_spd(support_mu[:, None].expand(-1, 2, -1))
            matrix = torch.cat([matrix, support_matrix[None].expand(batch, -1, -1, -1)], 1)
        representation = spd_logm(matrix) if self.geometry == "spd" else matrix
        vector = symmetric_vectorize(representation)
        return vector[:, None].expand(-1, self.relations, -1, -1)

    def _mask(self, compiled):
        mask = compiled.allowed_dependencies
        protect = self.variant in ("acf", "graph_only", "dominant")
        if not protect:
            available = compiled.node_available
            mask = available[..., :, None] & available[..., None, :]
            for s in range(self.num_modalities, mask.shape[-1]):
                mask[:, :, s] = False
                mask[:, :, s, s] = True
        elif self.variant == "dominant":
            available = compiled.node_available
            mask = available[..., :, None] & available[..., None, :]
            mask[:, :, self.dominant_source, :self.num_modalities] = False
            mask[:, :, self.dominant_source, self.dominant_source] = True
            for s in range(self.num_modalities, mask.shape[-1]):
                mask[:, :, s] = False
                mask[:, :, s, s] = True
        elif self.mask_policy == "role_shift":
            perm = torch.cat([torch.arange(self.num_modalities, device=mask.device).roll(1),
                              torch.arange(self.num_modalities, mask.shape[-1], device=mask.device)])
            mask = mask[..., perm, :][..., :, perm]
        return mask & compiled.node_available[..., :, None] & compiled.node_available[..., None, :]

    def forward(self, evidence, posthoc=False, bypass=False):
        compiled = compile_authority(evidence, self.supports)
        if self.variant == "rule_only":
            # No learned representation is evaluated and no protected-state claim is made.
            p0 = evidence.tokens.new_full((len(evidence.tokens), self.classes), 1/self.classes if self.task == "s1" else .5)
            projected = project_registered(p0, compiled)
            states = p0.new_zeros((*compiled.node_available.shape, 1))
            protected = p0.new_zeros((len(p0), self.relations, 1))
            return AuthorityOutput(p0, projected, projected, protected,
                                   torch.zeros_like(compiled.relation_active[:, :self.relations]),
                                   relation_residuals(projected, compiled), compiled.status,
                                   torch.zeros_like(compiled.allowed_dependencies, dtype=p0.dtype), states, compiled)
        states = self._encode(evidence, compiled)
        policy = self.policy_encoder(self._policy_features(evidence, compiled).to(states))
        mask = self._mask(compiled)
        quality = torch.cat([evidence.source_quality, torch.ones(len(states), self.supports, device=states.device)], -1)
        quality = quality[:, None].expand(-1, self.relations, -1)
        if self.is_transformer:
            source = self.transformer_input(states[:, 0])
            # Every baseline receives the same policy; the transformer can use it at every node.
            source = source + policy[:, None]
            padding = ~compiled.node_available[:, 0]
            safe_padding = padding.clone()
            safe_padding[padding.all(-1), 0] = False
            transformed = self.transformer(source, src_key_padding_mask=safe_padding)
            transformed = transformed * (~padding)[..., None]
            logits = self.head(torch.cat([transformed.flatten(1), policy], -1))
            adjacency = mask.to(states.dtype)
            # Expose final transformer states at the authorized source for honest diagnostics.
            states = transformed[:, None].expand(-1, self.relations, -1, -1)
        else:
            for block in self.blocks:
                states, adjacency = block(states, mask, compiled.node_available, self.num_modalities,
                                          self.variant, quality, bypass=bypass)
            logits = self.head(torch.cat([states.mean(1).flatten(1), policy], -1))
        p0 = logits.softmax(-1) if self.task == "s1" else logits.sigmoid()
        status = compiled.status
        if self.projection_backend == "general":
            active_rows = compiled.constraint_active
            projected, solver_status = GeneralAuthorityProjection()(
                p0, compiled.B * active_rows[..., None], compiled.b * active_rows, simplex=self.task == "s1")
            status = torch.where((status < 2) & (solver_status != 0), solver_status, status)
        else:
            projected = project_registered(p0, compiled)
        use_projection = posthoc or self.variant in ("acf", "learned_joint", "transformer_joint", "rule_only")
        probabilities = projected if use_projection else p0
        # One protected state per relation; inactive relations are explicitly marked, not successes.
        rows = torch.arange(len(states), device=states.device)[:, None]
        rel = torch.arange(self.relations, device=states.device)[None]
        protected = states[rows, rel, compiled.authorized_sources]
        return AuthorityOutput(p0, projected, probabilities, protected, compiled.relation_active[:, :self.relations],
                               relation_residuals(probabilities, compiled), status, adjacency, states, compiled)

    def manifold_states(self, output):
        if self.is_transformer:
            raise ValueError("Transformer states are not SPD matrices")
        symmetric = torch.einsum("...v,vij->...ij", output.node_states, self.symmetric_basis)
        return spd_expm(symmetric) if self.geometry == "spd" else symmetric


def authority_loss(output, labels, task, variant, projection_weight=.1):
    target = F.one_hot(labels.long(), 4).to(output.p0) if task == "s1" else labels.to(output.p0)
    valid = output.status < 2
    if not bool(valid.any()):
        raise ValueError("Training batch contains no feasible predictions")
    brier = (output.probabilities[valid] - target[valid]).square().mean()
    if variant in ("acf", "learned_joint", "transformer_joint"):
        brier = brier + projection_weight * (output.p_star[valid] - output.p0[valid]).square().mean()
    if variant == "soft":
        residual = relation_residuals(output.p0, output.compiled)
        active = output.compiled.relation_active & valid[:, None]
        penalty = residual[active].square().mean() if bool(active.any()) else residual.sum()*0
        brier = brier + .1 * penalty
    return brier
