from typing import List, Tuple, Union

from src.core.schema import (
    Message, ModelConfig, GenerationResult, PipelineMode, Role,
    ToolCallLog, ClarificationRequest, UncertaintyAction,
)
from src.core.model import generate_response
from src.core.uncertainty import analyze_uncertainty, decide_action
from src.tools import get_all_tool_schemas, execute_tool
from datetime import datetime, timezone
import json
import rich

MAX_TOOL_ROUNDS = 10
DRAFT_TEMPERATURE = 0.5
SYNTHESIS_TEMPERATURE = 0.7

# Finalizer thresholds for runaway detection
_RUNAWAY_WAIT_THRESHOLD = 2
_RUNAWAY_FINAL_ANSWER_THRESHOLD = 2

METACOGNITIVE_SYSTEM_PROMPT = (
    "You are a rigorous reasoning assistant.\n"
    "Before answering, briefly identify:\n"
    "- What you know with high confidence.\n"
    "- What you are uncertain about or assuming.\n"
    "- Whether the answer requires external data, computation, or user clarification.\n\n"
    "For math or code verification, strictly use `bash` or `notebook_edit`. Do NOT use `write_file` for calculations.\n\n"
    "Then provide your best answer. Clearly mark uncertain claims. "
    "Do not pad the response — be direct and concise."
)

RESOLUTION_PROMPT = (
    "Review your previous draft answer below.\n\n"
    "Draft:\n{draft}\n\n"
    "IMPORTANT:\n"
    "- Your draft may contain errors. Do not assume it is correct.\n"
    "- You may propose a completely new answer if warranted.\n\n"
    "Your task:\n"
    "1. Identify any claims that are uncertain, conflicting, or unverified.\n"
    "2. If there are EXTRA INSTRUCTIONS below, you MUST follow them using your "
    "available tools BEFORE providing your final answer. Do NOT rely on intuition "
    "if tools can provide facts. Use `bash` or `notebook_edit` for computation, NEVER `write_file`.\n"
    "3. Produce the strongest final answer based on evidence.\n"
    "4. Explicitly state any remaining caveats."
)

COMPACT_FINALIZER_PROMPT = (
    "The following analysis contains the answer but is too verbose or repetitive.\n\n"
    "Analysis:\n{analysis}\n\n"
    "RULES (strict):\n"
    "- Do NOT include any 'Wait', 'Let me re-check', or self-correction loops.\n"
    "- Do NOT repeat the final answer more than once.\n"
    "- Give one concise proof sketch if needed, then state the final answer clearly.\n"
    "- Maximum length: 800 tokens. Be as concise as possible."
)


def _detect_runaway(text: str) -> Tuple[bool, List[str]]:
    """Detect final-channel self-doubt loop patterns in a response.

    Returns (is_runaway, list_of_flags).
    """
    if not text:
        return False, []

    flags = []
    lower = text.lower()

    wait_count = lower.count("wait")
    if wait_count >= _RUNAWAY_WAIT_THRESHOLD:
        flags.append(f"self_check_loop:wait_x{wait_count}")

    final_answer_count = lower.count("final answer")
    if final_answer_count >= _RUNAWAY_FINAL_ANSWER_THRESHOLD:
        flags.append(f"repeated_final_answer:x{final_answer_count}")

    recheck_count = lower.count("let me re-check") + lower.count("let me recheck")
    if recheck_count >= 2:
        flags.append(f"recheck_loop:x{recheck_count}")

    return bool(flags), flags


def _compact_final(
    analysis: str,
    messages: List[Message],
    config: ModelConfig,
) -> GenerationResult:
    """Regenerate a clean, compact final answer from a runaway response."""
    prompt = COMPACT_FINALIZER_PROMPT.format(analysis=analysis[:6000])
    compact_msg = Message(role=Role.USER, content=prompt)
    return generate_response(
        messages=messages + [compact_msg],
        model_name=config.model_name,
        temperature=0.3,
        max_tokens=1024,
        n=1,
    )[0]


def run_instant_pipeline(
    messages: List[Message], config: ModelConfig, tools: list = None
) -> GenerationResult:
    """Run single-shot generation without uncertainty analysis."""
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

    Four steps:
    1. Draft a calibrated answer with a metacognitive system prompt.
    2. Analyze uncertainty of the draft via structured JSON controller.
    3. Rule engine deterministically selects an action.
    4. Execute the action: answer | ask user | verify with tools | research | refine.
       A finalizer pass sanitizes runaway self-doubt loops before returning.

    Args:
        messages: Conversation history.
        config: Model configuration.
        tools: Optional list of tool definitions.
        is_loop: Whether this is a retry from a previous clarification.

    Returns:
        Tuple of (GenerationResult or ClarificationRequest, debug info dict).
    """
    debug_info: dict = {}

    # --- Step 1: Draft ---
    rich.print("  [dim]Drafting calibrated answer...[/dim]")
    draft_messages = [Message(role=Role.SYSTEM, content=METACOGNITIVE_SYSTEM_PROMPT)] + messages
    draft = generate_response(
        messages=draft_messages,
        model_name=config.model_name,
        temperature=DRAFT_TEMPERATURE,
        max_tokens=config.max_tokens,
        n=1,
    )[0]

    # --- Step 2: Analyze uncertainty ---
    rich.print("  [dim]Analyzing uncertainty...[/dim]")
    report = analyze_uncertainty(messages, [draft], consistency=1.0, config=config)
    debug_info["uncertainty_report"] = report.model_dump()

    last_user_message = ""
    for m in reversed(messages):
        if m.role == Role.USER:
            last_user_message = m.content
            break

    # --- Step 3: Rule engine ---
    action = decide_action(report, last_user_message=last_user_message, is_loop=is_loop)
    debug_info["controller_action"] = action.value
    rich.print(f"  [dim]Controller action: [bold]{action.value}[/bold][/dim]")

    # --- Step 4: Execute ---
    if action in (UncertaintyAction.ANSWER, UncertaintyAction.ANSWER_WITH_CAVEATS):
        if action == UncertaintyAction.ANSWER_WITH_CAVEATS and report.risk_flags:
            caveat = f"\n\n> Note: {report.risk_flags[0]}"
            draft.content = (draft.content or "") + caveat
        return _apply_finalizer(draft, messages, config, debug_info), debug_info
    if action == UncertaintyAction.ASK_USER:
        rich.print("  [dim yellow]Action: ask user for clarification[/dim yellow]")
        question = (
            report.clarification_questions[0]
            if report.clarification_questions
            else "Could you provide more context?"
        )
        req = ClarificationRequest(
            question=question,
            default_assumptions=report.assumptions,
            fallback_plan=f"Proceed with defaults due to {report.risk_flags[0] if report.risk_flags else 'ambiguity'}.",
        )
        return req, debug_info

    rich.print("  [dim yellow]Action: refinement pass[/dim yellow]")
    debug_info["refinement_triggered"] = True

    extra_instructions = ""
    if action == UncertaintyAction.RESEARCH and report.research_questions:
        extra_instructions = "Please research the following before answering:\n- " + "\n- ".join(report.research_questions)
    elif action == UncertaintyAction.VERIFY_WITH_TOOL and report.computable_checks:
        extra_instructions = "Please verify the following using tools before answering:\n- " + "\n- ".join(report.computable_checks)

    resolution_content = RESOLUTION_PROMPT.format(draft=draft.content or "")
    if extra_instructions:
        resolution_content += f"\n\nEXTRA INSTRUCTIONS:\n{extra_instructions}"

    resolution_msg = Message(role=Role.USER, content=resolution_content)
    refined = generate_response(
        messages=messages + [resolution_msg],
        model_name=config.model_name,
        temperature=SYNTHESIS_TEMPERATURE,
        max_tokens=config.max_tokens,
        n=1,
        tools=tools,
    )[0]

    if action == UncertaintyAction.VERIFY_WITH_TOOL and not getattr(refined, "tool_calls", None):
        high_conflict = report.claim_conflicts and any(c.severity == "high" for c in report.claim_conflicts)
        if high_conflict:
            warning = (
                "⚠️ **UNVERIFIED SYNTHESIS WARNING** ⚠️\n"
                "Severe logical conflicts were detected but could not be resolved with tools. "
                "The following answer is an **unverified hypothesis**:\n\n---\n\n"
            )
            refined.content = warning + (refined.content or "")

    refined.tokens_used += draft.tokens_used
    refined.duration_sec += draft.duration_sec
    if getattr(refined, "tool_calls", None):
        return refined, debug_info

    return _apply_finalizer(refined, messages, config, debug_info), debug_info


def _apply_finalizer(
    result: GenerationResult,
    messages: List[Message],
    config: ModelConfig,
    debug_info: dict,
) -> GenerationResult:
    """Detect and fix runaway self-doubt loops in the final response text."""
    is_runaway, flags = _detect_runaway(result.content or "")
    debug_info["presentation_risk_flags"] = flags

    if not is_runaway:
        return result

    rich.print(f"  [dim red]Runaway detected: {', '.join(flags)} — compacting final response[/dim red]")
    debug_info["compact_finalizer_triggered"] = True

    compact = _compact_final(result.content or "", messages, config)
    compact.tokens_used += result.tokens_used
    compact.duration_sec += result.duration_sec
    return compact


def run_pipeline(
    messages: List[Message], config: ModelConfig, mode: PipelineMode, is_loop: bool = False
) -> Tuple[Union[GenerationResult, ClarificationRequest], dict]:
    """Route to the appropriate pipeline based on mode."""
    tools = get_all_tool_schemas()
    current_messages = messages.copy()
    final_debug = {"tool_logs": []}
    res = None

    for _round in range(MAX_TOOL_ROUNDS):
        if mode == PipelineMode.INSTANT:
            res = run_instant_pipeline(current_messages, config, tools if tools else None)
            debug = {}
        else:
            if current_messages and current_messages[-1].role == Role.TOOL:
                rich.print("  [dim yellow]Action: final synthesis after tools[/dim yellow]")
                synth_prompt = (
                    "You have executed tools to verify the uncertainties. The results are above.\n"
                    "Synthesize the final answer based on the evidence.\n"
                    "RULES:\n"
                    "- Do NOT use any more tools.\n"
                    "- Do NOT include any 'Wait', 'Let me re-check', or self-correction loops.\n"
                    "- Provide one concise proof or explanation, followed by the clear final answer."
                )
                synth_msg = Message(role=Role.USER, content=synth_prompt)
                synth = generate_response(
                    messages=current_messages + [synth_msg],
                    model_name=config.model_name,
                    temperature=0.3,
                    max_tokens=config.max_tokens,
                    n=1,
                    tools=None,
                )[0]
                debug = {"synthesis_triggered_after_tools": True}
                res = _apply_finalizer(synth, current_messages, config, debug)
                final_debug.update(debug)
                return res, final_debug
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
