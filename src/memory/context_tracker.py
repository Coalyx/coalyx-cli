from src.core.schema import ContextBudget, ContextZone

ZONE_THRESHOLDS = {
    ContextZone.FREE: 0.0,
    ContextZone.MONITORED: 0.5,
    ContextZone.COMPACT_SUGGESTED: 0.7,
    ContextZone.CRITICAL: 0.9,
}


def create_budget(max_tokens: int) -> ContextBudget:
    """Create a fresh context budget for a session.

    Args:
        max_tokens: Maximum context window size of the model.

    Returns:
        A new ContextBudget with full capacity.
    """
    return ContextBudget(total_capacity=max_tokens)


def update_budget(budget: ContextBudget, tokens_added: int) -> ContextBudget:
    """Update the budget after tokens are consumed and recalculate the zone.

    Args:
        budget: Current context budget.
        tokens_added: Number of new tokens consumed.

    Returns:
        Updated ContextBudget with recalculated zone.
    """
    budget.used_tokens += tokens_added
    budget.zone = get_zone(budget)
    return budget


def get_zone(budget: ContextBudget) -> ContextZone:
    """Determine which memory zone the session is in based on usage ratio.

    Zones:
        - FREE (0–50%): No intervention needed.
        - MONITORED (50–70%): Show usage warnings in dashboard.
        - COMPACT_SUGGESTED (70–90%): Suggest /compact command.
        - CRITICAL (90%+): Force compaction or clear.

    Args:
        budget: Current context budget.

    Returns:
        The ContextZone corresponding to current usage.
    """
    if budget.total_capacity == 0:
        return ContextZone.CRITICAL

    ratio = budget.used_tokens / budget.total_capacity

    if ratio >= 0.9:
        return ContextZone.CRITICAL
    elif ratio >= 0.7:
        return ContextZone.COMPACT_SUGGESTED
    elif ratio >= 0.5:
        return ContextZone.MONITORED
    else:
        return ContextZone.FREE


def get_usage_ratio(budget: ContextBudget) -> float:
    """Get the usage ratio as a float between 0.0 and 1.0.

    Args:
        budget: Current context budget.

    Returns:
        Usage ratio clamped to [0.0, 1.0].
    """
    if budget.total_capacity == 0:
        return 1.0
    return min(budget.used_tokens / budget.total_capacity, 1.0)
