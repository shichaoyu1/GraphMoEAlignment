"""Variant and semantic-alignment configuration helpers."""

from glioma.config.paper_profiles import apply_paper_profile


def apply_variant(args):
    if args.variant == "full":
        return args
    if args.variant == "clip":
        args.graph_type = "no_graph"
        args.no_private = True
        args.no_diffusion = True
        args.alignment_objective = "clip"
    elif args.variant == "medclip_style":
        args.graph_type = "no_graph"
        args.no_private = True
        args.no_diffusion = True
        args.alignment_objective = "medclip"
    elif args.variant == "dcca":
        args.graph_type = "no_graph"
        args.no_private = True
        args.no_diffusion = True
        args.alignment_objective = "dcca"
    elif args.variant in {"hgt", "graph_shared_only"}:
        args.graph_type = "learnable"
        args.no_private = True
        args.no_diffusion = True
        args.alignment_objective = "clip"
        args.variant = "graph_shared_only"
    elif args.variant == "no_anchor":
        args.exclude_pathology_anchors = True
    elif args.variant == "graph_only":
        args.no_diffusion = True
    elif args.variant == "modality_vector":
        args.node_mode = "modalities"
    elif args.variant == "no_private":
        args.no_private = True
        args.no_diffusion = True
    elif args.variant == "no_graph":
        args.graph_type = "no_graph"
    else:
        raise ValueError(f"Unsupported variant: {args.variant}")
    return args


__all__ = ["apply_variant", "apply_paper_profile"]
