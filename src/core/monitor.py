import psutil
from src.core.schema import MonitorStats

def get_available_vram() -> str:
    """
    Attempt to get available VRAM.
    For local models, we might try NVML or similar.
    For now, we return N/A or system RAM if not a local GPU.
    """
    try:
        # Pynvml can be added later for accurate Nvidia VRAM.
        # Fallback to system RAM for now.
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
        return f"{available_gb:.1f} GB (Sys RAM)"
    except Exception:
        return "N/A"

class SessionMonitor:
    def __init__(self, max_context_length: int = 4096):
        self.stats = MonitorStats()
        self.stats.remaining_context_length = max_context_length
        self.stats.available_vram = get_available_vram()
        
    def update(self, tokens_used: int, duration_sec: float, model_name: str):
        """Update monitor stats after a generation."""
        self.stats.model_info = model_name
        self.stats.total_tokens_used += tokens_used
        
        # Approximate remaining context (very rough, assumes reset per message unless tracked otherwise)
        # Real context tracking requires counting tokens of all messages.
        self.stats.remaining_context_length -= tokens_used
        if self.stats.remaining_context_length < 0:
            self.stats.remaining_context_length = 0
            
        if duration_sec > 0:
            speed = tokens_used / duration_sec
            # Rolling average
            if self.stats.avg_speed_tokens_per_sec == 0.0:
                self.stats.avg_speed_tokens_per_sec = speed
            else:
                self.stats.avg_speed_tokens_per_sec = (self.stats.avg_speed_tokens_per_sec * 0.7) + (speed * 0.3)
                
        self.stats.available_vram = get_available_vram()
