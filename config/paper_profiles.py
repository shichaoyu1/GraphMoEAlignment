"""Paper-specific module-combination presets for semantic alignment runs."""


def apply_paper_profile(args):
    profile = getattr(args, "paper_config", "none")
    if profile == "none":
        return args

    if profile == "paper1":
        # paper1: graph + diffusion
        args.graph_type = "learnable"
        args.no_diffusion = False
        args.moe_module = "none"
        return args

    if profile == "paper2":
        if args.topomoe_version == "v1":
            # Frozen first-run baseline: MRI graph + private branch, without diffusion.
            args.graph_type = "learnable"
            args.no_private = False
            args.no_diffusion = True
        else:
            # Paper 2 v2: disease-anchor topology + MoE, without the legacy MRI graph.
            args.graph_type = "learnable" if args.variant == "graph_shared_only" else "no_graph"
            args.no_private = True
            args.no_diffusion = True
        args.moe_module = "topo_moe"
        # The parser defaults to v2; preserve an explicit v1 compatibility run.
        return args

    if profile == "paper3":
        # paper3: diffusion + MoE
        args.graph_type = "no_graph"
        args.no_diffusion = False
        args.moe_module = "diffusion_moe"
        return args

    if profile == "paper4":
        # Paper 4: modality-specific encoders plus a geodesic modality graph.
        args.node_mode = "regions"
        args.graph_type = "no_graph"
        args.no_private = True
        args.no_diffusion = True
        args.moe_module = "none"
        return args

    raise ValueError(f"Unsupported paper_config: {profile}")


def apply_paper2_experiment_profile(args):
    """Resolve the controlled Paper 2 model comparison into explicit flags."""
    profile = getattr(args, "experiment_profile", "legacy")
    if profile == "legacy":
        return args
    if profile not in {
        "direct_only",
        "unstructured_family_moe",
        "prior_guided_router",
        "prior_plus_learned",
    }:
        raise ValueError(f"Unsupported Paper 2 experiment profile: {profile}")

    args.paper_config = "paper2"
    args.topomoe_version = "v2"
    args.node_mode = "regions"
    args.graph_type = "no_graph"
    args.no_private = True
    args.no_diffusion = True
    args.moe_module = "topo_moe"
    args.routing_enabled = profile != "direct_only"
    args.use_residual_expert = False
    args.context_mode = "unstructured" if profile == "unstructured_family_moe" else "topology"
    args.topo_mode = "prior_plus_learned" if profile == "prior_plus_learned" else "prior_only"
    if profile == "unstructured_family_moe":
        args.disable_topology_refinement = True

    direct_epochs = getattr(args, "direct_stage_epochs", None)
    router_epochs = getattr(args, "router_stage_epochs", None)
    joint_epochs = getattr(args, "joint_stage_epochs", None)
    if direct_epochs is None:
        args.direct_stage_epochs = int(getattr(args, "epochs", 30))
    if router_epochs is None:
        args.router_stage_epochs = 0 if profile == "direct_only" else 15
    if joint_epochs is None:
        args.joint_stage_epochs = 0 if profile == "direct_only" else 10
    if getattr(args, "stage_patience", None) is None:
        args.stage_patience = 8
    return args


__all__ = ["apply_paper_profile", "apply_paper2_experiment_profile"]
