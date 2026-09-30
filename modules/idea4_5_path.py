"""Training-only gradient routing for a directed graph message contribution."""

def route_message_gradient(messages, adjacency, representation, edge):
    """Scale one target<-source contribution's gradient without changing its value."""
    if edge is None:
        return messages
    target, source, strength = edge
    modalities = representation.shape[-3]
    if not (0 <= target < modalities and 0 <= source < modalities and target != source):
        raise ValueError("edge must name a directed cross-modal message")
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError("edge strength must be in [0, 1]")
    contribution = adjacency[..., target, source, None, None] * representation[..., source, :, :]
    routed = contribution.detach() + float(strength) * (contribution - contribution.detach())
    result = messages.clone()
    result[..., target, :, :] += routed - contribution
    return result


__all__ = ["route_message_gradient"]
