import psutil

from src.core.schema import MonitorStats, ContextBudget, ContextZone
from src.memory.context_tracker import create_budget, update_budget

VRAM_FALLBACK = "N/A"
SPEED_SMOOTHING = 0.7


def get_available_vram() -> str:
    """Return approximate available system memory as a formatted string."""
    try:
        mem = psutil.virtual_memory()
        return f"{mem.available / (1024 ** 3):.1f} GB (Sys RAM)"
    except Exception:
        return VRAM_FALLBACK


class SessionMonitor:
    """Unified session monitor tracking tokens, speed, and context budget."""

    def __init__(self, max_context_length: int = 4096):
        self.stats = MonitorStats()
        self.budget = create_budget(max_context_length)
        self.stats.remaining_context_length = max_context_length
        self.stats.available_vram = get_available_vram()

    @property
    def zone(self) -> ContextZone:
        return self.budget.zone

    def should_auto_compact(self) -> bool:
        return self.budget.zone == ContextZone.CRITICAL


    def update(self, tokens_used: int, duration_sec: float, model_name: str) -> None:
        """Update stats after a generation cycle."""
        self.stats.model_info = model_name
        self.stats.total_tokens_used += tokens_used

        self.budget = update_budget(self.budget, tokens_used)
        self.stats.remaining_context_length = max(
            self.budget.total_capacity - self.budget.used_tokens, 0
        )

        if duration_sec > 0:
            speed = tokens_used / duration_sec
            prev = self.stats.avg_speed_tokens_per_sec
            self.stats.avg_speed_tokens_per_sec = (
                speed if prev == 0.0
                else prev * SPEED_SMOOTHING + speed * (1 - SPEED_SMOOTHING)
            )

        self.stats.available_vram = get_available_vram()

    def reset_budget(self, max_tokens: int | None = None) -> None:
        """Reset context budget, optionally with a new capacity."""
        capacity = max_tokens or self.budget.total_capacity
        self.budget = create_budget(capacity)
        self.stats.remaining_context_length = capacity
