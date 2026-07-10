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

    raise ValueError(f"Unsupported paper_config: {profile}")


__all__ = ["apply_paper_profile"]
