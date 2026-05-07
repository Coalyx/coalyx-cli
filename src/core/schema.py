from enum import Enum
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field
from datetime import datetime


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class PipelineMode(str, Enum):
    INSTANT = "Instant"
    ADAPTIVE = "Adaptive Reasoning"


class HookEvent(str, Enum):
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    PRE_COMMIT = "pre_commit"
    NOTIFICATION = "notification"


class ContextZone(str, Enum):
    FREE = "free"
    MONITORED = "monitored"
    COMPACT_SUGGESTED = "compact_suggested"
    CRITICAL = "critical"


# --- Core Chat Schemas ---

class ToolCall(BaseModel):
    id: str
    name: str
    arguments: str

class Message(BaseModel):
    role: Role
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None


class ModelConfig(BaseModel):
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 2048


class GenerationResult(BaseModel):
    content: str
    tokens_used: int
    duration_sec: float
    tool_calls: Optional[List[ToolCall]] = None


class EmbeddingResult(BaseModel):
    vector: List[float]
    tokens_used: int


# --- Monitor Schemas ---

class MonitorStats(BaseModel):
    total_tokens_used: int = 0
    remaining_context_length: int = 0
    model_info: str = ""
    available_vram: str = "N/A"
    avg_speed_tokens_per_sec: float = 0.0


# --- Memory Schemas ---

class ContextBudget(BaseModel):
    """Tracks token usage against the model's context window."""
    total_capacity: int
    used_tokens: int = 0
    zone: ContextZone = ContextZone.FREE


class CompactionResult(BaseModel):
    """Output of a conversation compaction operation."""
    summary: str
    original_count: int
    compacted_count: int
    tokens_saved: int


class SessionSnapshot(BaseModel):
    """Serializable session state for persistence."""
    session_id: str
    model_name: str
    mode: PipelineMode
    messages: List[Message] = Field(default_factory=list)
    total_tokens_used: int = 0
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# --- Extension Schemas ---

class HookConfig(BaseModel):
    """Configuration for a lifecycle hook."""
    event: HookEvent
    command: str
    description: str = ""
    enabled: bool = True


class HookResult(BaseModel):
    """Result of executing a hook."""
    event: HookEvent
    success: bool
    output: str = ""
    error: str = ""


class SkillDefinition(BaseModel):
    """A skill loaded from .coalyx/skills/*.md."""
    name: str
    trigger_patterns: List[str] = Field(default_factory=list)
    instructions: str = ""
    source_file: str = ""


# --- Session State (used at runtime) ---

class SessionState(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    mode: PipelineMode = PipelineMode.INSTANT
    model_config_data: ModelConfig = Field(alias="model_config")

    class Config:
        populate_by_name = True
