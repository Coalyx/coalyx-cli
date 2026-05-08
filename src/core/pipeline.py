from typing import List, Tuple, Union

from src.core.schema import Message, ModelConfig, GenerationResult, PipelineMode, Role, ToolCallLog, ClarificationRequest, UncertaintyAction
from src.core.model import generate_response
from datetime import datetime, timezone
from src.core.embedding import calculate_group_consistency
from src.core.uncertainty import analyze_uncertainty, decide_action
from src.tools import get_all_tool_schemas, execute_tool
import json
import rich

STAGE1_CANDIDATES = 5
STAGE1_TEMPERATURE = 0.5
STAGE2_TEMPERATURE = 0.9
CERTAINTY_THRESHOLD = 0.75
MAX_TOOL_ROUNDS = 10

CANDIDATE_ROLES = [
    "Answer the query directly and concisely.",
    (
        "Answer the query, but focus on potential issues, "
        "edge cases, and what could go wrong."
    ),
    (
        "Before answering, identify what context or information might be missing. "
        "Then answer with explicit assumptions stated."
    ),
]

SELF_DOUBT_PROMPT = (
    "We generated multiple candidate answers to the same query. "
    "They diverge in the following ways:\n\n"
    "{candidates}\n\n"
    "IMPORTANT:\n"
    "- The listed candidates may ALL be flawed. "
    "Do not assume the correct answer must be among them.\n"
    "- You may propose a completely new answer if warranted.\n\n"
    "Your task:\n"
    "1. Identify exactly where candidates AGREE and where they DISAGREE (especially numerical values).\n"
    "2. For each disagreement, determine the root cause (misinterpretation, calculation error, etc.).\n"
    "3. IMPORTANT: If there are 'EXTRA INSTRUCTIONS' below requesting research or verification, "
    "   you MUST use your available tools (search, code execution) to resolve the conflict "
    "   BEFORE providing your final synthesis. Do NOT rely on intuition if tools can provide facts.\n"
    "4. Synthesize the strongest final answer based on the evidence found.\n"
    "5. Explicitly state which candidate was wrong and why."
)

# ---------------------------------------------------------------------------
# Lexical fallback for when embedding API is unavailable
# ---------------------------------------------------------------------------

def _lexical_jaccard(text_a: str, text_b: str) -> float:
    """Jaccard similarity on lowercased word tokens."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a and not words_b:
        return 1.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union) if union else 1.0


def _lexical_group_consistency(texts: list) -> float:
    """Average pairwise Jaccard similarity — crude proxy for embedding consistency."""
    if len(texts) <= 1:
        return 1.0
    sims = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sims.append(_lexical_jaccard(texts[i], texts[j]))
    return sum(sims) / len(sims) if sims else 1.0


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
    messages: List[Message], config: ModelConfig, tools: list = None, is_loop: bool = False
) -> Tuple[Union[GenerationResult, ClarificationRequest], dict]:
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
        Tuple of (GenerationResult or ClarificationRequest, debug info dict).
    """
    debug_info = {
        "stage1_candidates": [],
        "consistency": 0.0,
        "stage2_triggered": False,
    }

    rich.print("  [dim]Sampling internal reasoning paths...[/dim]")
    # Note: Tools are intentionally NOT passed here. Stage 1 candidates are
    # only used for semantic uncertainty measurement, not for execution.
    candidates = []
    for role_prompt in CANDIDATE_ROLES:
        role_msg = Message(role=Role.SYSTEM, content=role_prompt)
        role_messages = [role_msg] + messages
        batch = generate_response(
            messages=role_messages,
            model_name=config.model_name,
            temperature=STAGE1_TEMPERATURE,
            max_tokens=config.max_tokens,
            n=1,
        )
        candidates.extend(batch)

    total_tokens = sum(c.tokens_used for c in candidates)
    total_duration = sum(c.duration_sec for c in candidates)

    texts = [c.content for c in candidates]
    debug_info["stage1_candidates"] = texts

    # MEASURE UNCERTAINTY
    rich.print("  [dim]Measuring semantic consistency...[/dim]")
    try:
        consistency_result = calculate_group_consistency(texts)
        consistency = consistency_result.score
        debug_info["representative_idx"] = consistency_result.representative_idx
        debug_info["minority_idx"] = consistency_result.minority_idx
    except Exception as e:
        # Embedding failed — use lexical overlap as a crude proxy instead
        # of defaulting to 0.0 (which would falsely trigger heavy
        # uncertainty for every transient API failure).
        consistency = _lexical_group_consistency(texts)
        debug_info["embedding_error"] = str(e)
        debug_info["consistency_method"] = "lexical_fallback"
        rich.print(f"  [dim yellow]Embedding failed, using lexical fallback (score={consistency:.2f})[/dim yellow]")
        # Create a dummy result for fallback
        from src.core.schema import ConsistencyResult
        consistency_result = ConsistencyResult(score=consistency)

    debug_info["consistency"] = consistency

    # STAGE 2: UNCERTAINTY ANALYSIS
    rich.print("  [dim]Analyzing uncertainty structured report...[/dim]")
    report = analyze_uncertainty(messages, candidates, consistency, config)
    debug_info["uncertainty_report"] = report.model_dump()
    last_user_message = ""
    for m in reversed(messages):
        if m.role == Role.USER:
            last_user_message = m.content
            break

    action = decide_action(report, last_user_message=last_user_message, is_loop=is_loop)
    
    rich.print(f"  [dim]Controller recommended action: [bold]{action.value}[/bold][/dim]")
    
    if action == UncertaintyAction.ASK_USER:
        rich.print("  [dim bold yellow]Uncertainty triggers Clarification Request...[/dim bold yellow]")
        question = (report.clarification_questions[0] 
                    if report.clarification_questions else "Bạn có thể cung cấp thêm ngữ cảnh được không?")
        req = ClarificationRequest(
            question=question,
            default_assumptions=report.assumptions,
            fallback_plan=f"Proceed with defaults due to {report.risk_flags[0] if report.risk_flags else 'ambiguity'}."
        )
        return req, debug_info

    # For RESEARCH or VERIFY_WITH_TOOL, we pass the questions/checks back into the prompt
    tool_prompt = ""
    if action == UncertaintyAction.RESEARCH and report.research_questions:
        tool_prompt = "Please research the following before answering:\n- " + "\n- ".join(report.research_questions)
    elif action == UncertaintyAction.VERIFY_WITH_TOOL and report.computable_checks:
        tool_prompt = "Please verify the following using tools before answering:\n- " + "\n- ".join(report.computable_checks)

    # STAGE 3: Self-Doubt / Synthesis Activation
    rich.print("  [dim bold yellow]Entering Resolution Loop for self-correction & synthesis...[/dim bold yellow]")
    debug_info["stage2_triggered"] = True

    unique_ans = "\n\n---\n\n".join(
        f"Candidate {i + 1}: {t}" for i, t in enumerate(texts)
    )
    resolution_prompt = SELF_DOUBT_PROMPT.format(candidates=unique_ans)
    
    if tool_prompt:
        resolution_prompt += f"\n\nEXTRA INSTRUCTIONS:\n{tool_prompt}"
        
    self_doubt_msg = Message(
        role=Role.USER,
        content=resolution_prompt,
    )

    reflection_messages = messages + [self_doubt_msg]

    final_res = generate_response(
        messages=reflection_messages,
        model_name=config.model_name,
        temperature=STAGE2_TEMPERATURE,
        max_tokens=config.max_tokens,
        n=1,
        tools=tools,
    )[0]

    # HARD RULE 3 & 4: Guardrail against Unverified Synthesis
    if action == UncertaintyAction.VERIFY_WITH_TOOL and not getattr(final_res, "tool_calls", None):
        high_conflict = report.claim_conflicts and any(c.severity == "high" for c in report.claim_conflicts)
        if high_conflict:
            warning = (
                "⚠️ **UNVERIFIED SYNTHESIS WARNING** ⚠️\n"
                "I detected severe logical conflicts in my reasoning and attempted to resolve them. "
                "However, I failed to programmatically verify the result using tools. "
                "The following explanation is an **unverified hypothesis** and may contain logical or arithmetic errors:\n\n"
                "---\n\n"
            )
            final_res.content = warning + (final_res.content or "")

    final_res.tokens_used += total_tokens

    final_res.duration_sec += total_duration

    return final_res, debug_info


def run_pipeline(
    messages: List[Message], config: ModelConfig, mode: PipelineMode, is_loop: bool = False
) -> Tuple[Union[GenerationResult, ClarificationRequest], dict]:
    """Route to the appropriate pipeline based on mode.

    Args:
        messages: Conversation history.
        config: Model configuration.
        mode: Pipeline mode (Instant or Adaptive).
        is_loop: Whether this is a retry from a previous clarification.

    Returns:
        Tuple of (GenerationResult, debug info dict).
    """
    tools = get_all_tool_schemas()
    current_messages = messages.copy()
    final_debug = {"tool_logs": []}
    res = None
    
    for _round in range(MAX_TOOL_ROUNDS):
        if mode == PipelineMode.INSTANT:
            res = run_instant_pipeline(current_messages, config, tools if tools else None)
            debug = {}
        else:
            res, debug = run_adaptive_pipeline(current_messages, config, tools if tools else None, is_loop=is_loop)
        
        final_debug.update(debug)

        if isinstance(res, ClarificationRequest):
            return res, final_debug
        
        if getattr(res, "tool_calls", None):
            assistant_msg = Message(role=Role.ASSISTANT, content=res.content or "", tool_calls=res.tool_calls)
            current_messages.append(assistant_msg)
            messages.append(assistant_msg)
            
            for tc in res.tool_calls:
                rich.print(f"  [dim]Executing tool: [bold]{tc.name}[/bold]...[/dim]")
                try:
                    args = json.loads(tc.arguments) if isinstance(tc.arguments, str) else tc.arguments
                except Exception as e:
                    args = {"__raw_arguments__": tc.arguments, "error": str(e)}
                output = execute_tool(tc.name, args)

                # Determine execution status from output
                output_str = str(output)
                if output_str.startswith("Error"):
                    status = "error"
                elif output_str.startswith("Tool '") and "denied" in output_str:
                    status = "denied"
                elif output_str.startswith("Security error"):
                    status = "denied"
                else:
                    status = "success"
                
                final_debug["tool_logs"].append(ToolCallLog(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    tool_name=tc.name,
                    arguments=args,
                    status=status,
                    output_preview=output_str[:500],
                    approved_by_user=(status != "denied"),
                ))
                
                tool_msg = Message(role=Role.TOOL, content=output_str, tool_call_id=tc.id)
                current_messages.append(tool_msg)
                messages.append(tool_msg)
        else:
            return res, final_debug
            
    return res, final_debug
