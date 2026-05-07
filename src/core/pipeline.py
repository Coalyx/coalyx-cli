from typing import List, Tuple

from src.core.schema import Message, ModelConfig, GenerationResult, PipelineMode, Role
from src.core.model import generate_response
from src.core.embedding import calculate_group_consistency
from src.tools import get_all_tool_schemas, execute_tool
import json
import rich

STAGE1_CANDIDATES = 3
STAGE1_TEMPERATURE = 0.7
STAGE2_TEMPERATURE = 0.9
CERTAINTY_THRESHOLD = 0.75


def _is_ollama(model_name: str) -> bool:
    """Return True if the model is served via Ollama."""
    return model_name.lower().startswith("ollama/")


def run_instant_pipeline(
    messages: List[Message], config: ModelConfig, tools: list = None
) -> GenerationResult:
    """Run single-shot generation without uncertainty analysis.

    Args:
        messages: Conversation history.
        config: Model configuration.
        tools: Optional list of tool definitions.

    Returns:
        Single GenerationResult.
    """
    results = generate_response(
        messages=messages,
        model_name=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        n=1,
        tools=tools,
    )
    return results[0]


def run_adaptive_pipeline(
    messages: List[Message], config: ModelConfig, tools: list = None
) -> Tuple[GenerationResult, dict]:
    """Run the Adaptive Uncertainty-Aware Reasoning pipeline.

    Stage 1: Generate N candidates in a single batched call (saves API
    quota vs N separate calls). Falls back to sequential if the provider
    does not support the n parameter.

    Uncertainty Gate: Measure semantic consistency via embeddings.
    If consistency >= threshold, return the best candidate immediately.

    Stage 2 (Self-Doubt): If uncertain, inject conflicting candidates
    into a reflection prompt and synthesize a final answer.

    For Ollama thinking models (e.g. gemma4, deepseek-r1), the model-side
    thinking tokens are suppressed via ``extra_body={"think": False}``
    during both stages. Coalyx's own multi-candidate uncertainty pipeline
    already fulfils the deliberation role, so paying twice is wasteful.

    Args:
        messages: Conversation history.
        config: Model configuration.
        tools: Optional list of tool definitions.

    Returns:
        Tuple of (final GenerationResult, debug info dict).
    """
    debug_info = {
        "stage1_candidates": [],
        "consistency": 0.0,
        "stage2_triggered": False,
    }

    # Suppress built-in thinking for Ollama — Coalyx's pipeline handles reasoning.
    ollama_no_think = {"think": False} if _is_ollama(config.model_name) else None

    # STAGE 1: Fast Probe — single batched call with n=3
    # Note: Tools are intentionally NOT passed here. Stage 1 candidates are
    # only used for semantic uncertainty measurement, not for execution.
    # Passing tools to small models causes them to dump raw JSON in content.
    candidates = generate_response(
        messages=messages,
        model_name=config.model_name,
        temperature=STAGE1_TEMPERATURE,
        max_tokens=config.max_tokens,
        n=STAGE1_CANDIDATES,
        extra_body=ollama_no_think,
    )

    total_tokens = sum(c.tokens_used for c in candidates)
    total_duration = sum(c.duration_sec for c in candidates)

    texts = [c.content for c in candidates]
    debug_info["stage1_candidates"] = texts

    # MEASURE UNCERTAINTY
    try:
        consistency = calculate_group_consistency(texts)
    except Exception as e:
        consistency = 1.0
        debug_info["embedding_error"] = str(e)

    debug_info["consistency"] = consistency

    # UNCERTAINTY GATE
    if consistency >= CERTAINTY_THRESHOLD:
        best = candidates[0]
        best.tokens_used = total_tokens
        best.duration_sec = total_duration
        return best, debug_info

    # STAGE 2: Self-Doubt Activation
    debug_info["stage2_triggered"] = True

    unique_ans = "\n\n---\n\n".join(
        f"Candidate {i + 1}: {t}" for i, t in enumerate(texts)
    )
    self_doubt_msg = Message(
        role=Role.USER,
        content=(
            "We attempted to solve the previous query but received "
            "conflicting answers or perspectives. "
            "Here are the conflicting candidates we generated:\n\n"
            f"{unique_ans}\n\n"
            "Please carefully review these conflicting perspectives, "
            "analyze where they diverge or make mistakes, "
            "and provide a final, well-reasoned, and synthesis answer."
        ),
    )

    reflection_messages = messages + [self_doubt_msg]

    final_res = generate_response(
        messages=reflection_messages,
        model_name=config.model_name,
        temperature=STAGE2_TEMPERATURE,
        max_tokens=config.max_tokens,
        n=1,
        extra_body=ollama_no_think,
        tools=tools,
    )[0]

    final_res.tokens_used += total_tokens
    final_res.duration_sec += total_duration

    return final_res, debug_info


def run_pipeline(
    messages: List[Message], config: ModelConfig, mode: PipelineMode
) -> Tuple[GenerationResult, dict]:
    """Route to the appropriate pipeline based on mode.

    Args:
        messages: Conversation history.
        config: Model configuration.
        mode: Pipeline mode (Instant or Adaptive).

    Returns:
        Tuple of (GenerationResult, debug info dict).
    """
    tools = get_all_tool_schemas()
    current_messages = messages.copy()
    final_debug = {}
    while True:
        if mode == PipelineMode.INSTANT:
            res = run_instant_pipeline(current_messages, config, tools if tools else None)
            debug = {}
        else:
            res, debug = run_adaptive_pipeline(current_messages, config, tools if tools else None)
        
        final_debug.update(debug)
        
        if res.tool_calls:
            assistant_msg = Message(role=Role.ASSISTANT, content=res.content or "", tool_calls=res.tool_calls)
            current_messages.append(assistant_msg)
            messages.append(assistant_msg)
            
            for tc in res.tool_calls:
                rich.print(f"  [dim]⚡ Executing tool: [bold]{tc.name}[/bold][/dim]")
                try:
                    args = json.loads(tc.arguments) if isinstance(tc.arguments, str) else tc.arguments
                except:
                    args = {}
                output = execute_tool(tc.name, args)
                tool_msg = Message(role=Role.TOOL, content=str(output), tool_call_id=tc.id)
                current_messages.append(tool_msg)
                messages.append(tool_msg)
        else:
            return res, final_debug
