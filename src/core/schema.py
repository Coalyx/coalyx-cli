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


class UncertaintyAction(str, Enum):
    ANSWER = "answer"
    ANSWER_WITH_CAVEATS = "answer_with_caveats"
    ASK_USER = "ask_user"
    RESEARCH = "research"
    VERIFY_WITH_TOOL = "verify_with_tool"
    ADVERSARIAL_REVIEW = "adversarial_review"
    REFUSE_OR_DEFER = "refuse_or_defer"


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
    max_tokens: int = 16384


class GenerationResult(BaseModel):
    content: str
    tokens_used: int
    duration_sec: float
    tool_calls: Optional[List[ToolCall]] = None


class EmbeddingResult(BaseModel):
    vector: List[float]
    tokens_used: int


class ConsistencyResult(BaseModel):
    """Semantic consistency measurement with reusable embeddings."""
    score: float
    embeddings: List[List[float]] = Field(default_factory=list)
    representative_idx: int = 0
    minority_idx: int = 0


class ToolCallLog(BaseModel):
    """Structured log entry for a single tool invocation."""
    timestamp: str
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    status: str = "success"
    output_preview: str = ""
    approved_by_user: bool = True


# --- Uncertainty Schemas ---

class ClaimConflict(BaseModel):
    claim_a: str
    claim_b: str
    conflict_type: str      # factual | structural | perspective
    severity: str           # low | medium | high


class Unknown(BaseModel):
    description: str
    kind: str               # user_intent | factual | contextual
    ask_user: bool
    researchable: bool


class UncertaintyReport(BaseModel):
    total_score: float = 0.0
    semantic_agreement: float = 0.0
    claim_conflicts: List[ClaimConflict] = Field(default_factory=list)
    unknowns: List[Unknown] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    research_questions: List[str] = Field(default_factory=list)
    clarification_questions: List[str] = Field(default_factory=list)
    computable_checks: List[str] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)
    recommended_action: UncertaintyAction = UncertaintyAction.ANSWER

    @property
    def confidence_score(self) -> float:
        """Confidence is the inverse of uncertainty. Range [0, 1]."""
        return max(0.0, 1.0 - self.total_score)


class ClarificationRequest(BaseModel):
    question: str
    default_assumptions: List[str] = Field(default_factory=list)
    fallback_plan: str = ""


# --- Adaptive Path Schemas ---

class PathMethod(str, Enum):
    DIRECT_SOLUTION = "direct_solution"
    FORMAL_PROOF = "formal_proof"
    COMPUTATIONAL_CHECK = "computational_check"
    ADVERSARIAL_CRITIC = "adversarial_critic"
    EDGE_CASE_SEARCH = "edge_case_search"
    SOURCE_GROUNDING = "source_grounding"
    RESEARCH = "research"
    IMPLEMENTATION_TEST = "implementation_test"


class PathSpec(BaseModel):
    id: str
    objective: str
    method: PathMethod
    system_prompt: str
    allowed_tools: List[str] = Field(default_factory=list)


class TaskProfile(BaseModel):
    complexity_score: float = 0.5
    requires_exact_answer: bool = False
    tool_resolvable: bool = False
    ambiguity_type: str = "none"
    risk_level: str = "low"


class PathBudgetRequest(BaseModel):
    initial_paths: int = 3
    max_desired_paths: int = 6
    reason: str = ""
    diversity_plan: List[str] = Field(default_factory=list)
    stop_condition: str = ""


class PathAuditResult(BaseModel):
    confidence: float = 0.0
    answer_agreement: bool = False
    high_conflict: bool = False
    conflict_is_tool_resolvable: bool = False
    requires_exact_answer: bool = False
    answer_disagreement: bool = False
    verified_answer_found: bool = False
    should_stop: bool = False
    should_expand: bool = False
    stop_reason: str = ""
    expansion_reason: str = ""


class AdaptivePathConfig(BaseModel):
    enabled: bool = True
    min_paths: int = 1
    default_paths: int = 3
    max_paths: int = 6
    max_waves: int = 3
    max_paths_per_wave: int = 3
    target_confidence: float = 0.78
    min_marginal_gain: float = 0.08
    max_total_path_tokens: int = 8192
    max_final_tokens: int = 1024


class PathBudget(BaseModel):
    config: AdaptivePathConfig = Field(default_factory=AdaptivePathConfig)
    paths_used: int = 0
    tokens_spent: int = 0
    waves_used: int = 0

    @property
    def paths_remaining(self) -> int:
        return self.config.max_paths - self.paths_used

    @property
    def tokens_remaining(self) -> int:
        return self.config.max_total_path_tokens - self.tokens_spent

    @property
    def min_tokens_per_path(self) -> int:
        return self.config.max_total_path_tokens // max(self.config.max_paths, 1)

    def consume(self, candidates: List["GenerationResult"]) -> None:
        self.paths_used += len(candidates)
        self.tokens_spent += sum(c.tokens_used for c in candidates)
        self.waves_used += 1


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
