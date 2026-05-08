from typing import List, Tuple, Union

from src.core.schema import (
    Message, ModelConfig, GenerationResult, PipelineMode, Role,
    ToolCallLog, ClarificationRequest, UncertaintyAction,
    AdaptivePathConfig, PathBudget,
)
from src.core.model import generate_response
from src.core.config import load_config
from src.core.path_planner import profile_task, plan_initial_paths, request_additional_paths
from src.core.path_auditor import audit_paths, should_expand_paths
from src.core.uncertainty import analyze_uncertainty, decide_action
from src.tools import get_all_tool_schemas, execute_tool
from datetime import datetime, timezone
import json
import rich

PATH_TEMPERATURE = 0.5
SYNTHESIS_TEMPERATURE = 0.9
MAX_TOOL_ROUNDS = 10

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


def _load_path_config() -> AdaptivePathConfig:
    cfg = load_config()
    ap = cfg.get("adaptive_paths", {})
    return AdaptivePathConfig(**ap)


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


def _run_path_wave(
    path_specs: list,
    messages: List[Message],
    config: ModelConfig,
) -> List[GenerationResult]:
    candidates = []
    for spec in path_specs:
        role_msg = Message(role=Role.SYSTEM, content=spec.system_prompt)
        role_messages = [role_msg] + messages
        batch = generate_response(
            messages=role_messages,
            model_name=config.model_name,
            temperature=PATH_TEMPERATURE,
            max_tokens=config.max_tokens,
            n=1,
        )
        candidates.extend(batch)
    return candidates


def _synthesize_with_audit(
    messages: List[Message],
    candidates: List[GenerationResult],
    audit_debug: dict,
    config: ModelConfig,
    tools: list = None,
    is_loop: bool = False,
) -> Tuple[Union[GenerationResult, ClarificationRequest], dict]:
    texts = [c.content for c in candidates]
    total_tokens = sum(c.tokens_used for c in candidates)
    total_duration = sum(c.duration_sec for c in candidates)

    uncertainty_report = audit_debug.get("uncertainty_report")
    controller_action = audit_debug.get("controller_action", "answer")

    from src.core.schema import UncertaintyReport
    report = UncertaintyReport(**uncertainty_report) if uncertainty_report else None

    last_user_message = ""
    for m in reversed(messages):
        if m.role == Role.USER:
            last_user_message = m.content
            break

    if report:
        action = decide_action(report, last_user_message=last_user_message, is_loop=is_loop)
    else:
        action = UncertaintyAction.ANSWER

    rich.print(f"  [dim]Controller recommended action: [bold]{action.value}[/bold][/dim]")

    if action == UncertaintyAction.ASK_USER:
        rich.print("  [dim bold yellow]Uncertainty triggers Clarification Request...[/dim bold yellow]")
        question = (
            report.clarification_questions[0]
            if report and report.clarification_questions
            else "Bạn có thể cung cấp thêm ngữ cảnh được không?"
        )
        req = ClarificationRequest(
            question=question,
            default_assumptions=report.assumptions if report else [],
            fallback_plan=f"Proceed with defaults due to {report.risk_flags[0] if report and report.risk_flags else 'ambiguity'}.",
        )
        return req, audit_debug

    tool_prompt = ""
    if report:
        if action == UncertaintyAction.RESEARCH and report.research_questions:
            tool_prompt = "Please research the following before answering:\n- " + "\n- ".join(report.research_questions)
        elif action == UncertaintyAction.VERIFY_WITH_TOOL and report.computable_checks:
            tool_prompt = "Please verify the following using tools before answering:\n- " + "\n- ".join(report.computable_checks)

    rich.print("  [dim bold yellow]Entering Resolution Loop for self-correction & synthesis...[/dim bold yellow]")
    audit_debug["stage2_triggered"] = True

    unique_ans = "\n\n---\n\n".join(
        f"Candidate {i + 1}: {t}" for i, t in enumerate(texts)
    )
    resolution_prompt = SELF_DOUBT_PROMPT.format(candidates=unique_ans)

    if tool_prompt:
        resolution_prompt += f"\n\nEXTRA INSTRUCTIONS:\n{tool_prompt}"

    self_doubt_msg = Message(role=Role.USER, content=resolution_prompt)
    reflection_messages = messages + [self_doubt_msg]

    final_res = generate_response(
        messages=reflection_messages,
        model_name=config.model_name,
        temperature=SYNTHESIS_TEMPERATURE,
        max_tokens=config.max_tokens,
        n=1,
        tools=tools,
    )[0]

    if action == UncertaintyAction.VERIFY_WITH_TOOL and not getattr(final_res, "tool_calls", None):
        high_conflict = report and report.claim_conflicts and any(c.severity == "high" for c in report.claim_conflicts)
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

    return final_res, audit_debug


def _format_competing_hypotheses(candidates: List[GenerationResult]) -> str:
    parts = []
    for i, c in enumerate(candidates):
        preview = c.content[:300] if c.content else "(empty)"
        parts.append(f"**Hypothesis {i+1}:**\n{preview}")
    return "\n\n---\n\n".join(parts)


def run_adaptive_pipeline(
    messages: List[Message], config: ModelConfig, tools: list = None, is_loop: bool = False
) -> Tuple[Union[GenerationResult, ClarificationRequest], dict]:
    """Run the Progressive Parallel Paths reasoning pipeline.

    1. Profile the task to determine complexity and requirements.
    2. Plan initial paths with diverse objectives.
    3. Run paths in waves, auditing after each wave.
    4. Expand with additional paths if audit detects unresolved conflicts.
    5. Synthesize the final answer with full audit context.
    """
    path_config = _load_path_config()
    debug_info = {
        "stage1_candidates": [],
        "consistency": 0.0,
        "stage2_triggered": False,
        "path_budget": {},
        "path_expansion_events": [],
    }

    # 1. Profile task
    rich.print("  [dim]Profiling task complexity...[/dim]")
    profile = profile_task(messages, config)
    debug_info["task_profile"] = profile.model_dump()
    rich.print(f"  [dim]Task profile: complexity={profile.complexity_score:.2f}, exact={profile.requires_exact_answer}, risk={profile.risk_level}[/dim]")

    # 2. Plan initial paths
    rich.print("  [dim]Planning reasoning paths...[/dim]")
    path_specs, budget_request = plan_initial_paths(profile, messages, config, path_config)
    budget = PathBudget(config=path_config)
    debug_info["initial_path_count"] = len(path_specs)
    debug_info["path_specs_used"] = [s.model_dump() for s in path_specs]
    rich.print(f"  [dim]Allocated {len(path_specs)} initial paths (max {path_config.max_paths})[/dim]")

    # 3. Progressive wave loop
    all_candidates: List[GenerationResult] = []
    audit_result = None
    audit_debug = {}

    for wave_idx in range(path_config.max_waves):
        wave_label = f"Wave {wave_idx + 1}"
        rich.print(f"  [dim]Running {wave_label}: {len(path_specs)} paths...[/dim]")

        wave_candidates = _run_path_wave(path_specs, messages, config)
        all_candidates.extend(wave_candidates)
        budget.consume(wave_candidates)

        texts = [c.content for c in all_candidates]
        debug_info["stage1_candidates"] = texts

        rich.print(f"  [dim]Auditing {len(all_candidates)} candidates...[/dim]")
        audit_result, audit_debug = audit_paths(
            all_candidates, messages, profile, config, path_config
        )
        debug_info["consistency"] = audit_debug.get("consistency", 0.0)
        debug_info.update({k: v for k, v in audit_debug.items() if k != "stage1_candidates"})

        rich.print(
            f"  [dim]Audit: confidence={audit_result.confidence:.2f}, "
            f"conflict={audit_result.high_conflict}, "
            f"stop={audit_result.should_stop}[/dim]"
        )

        if audit_result.should_stop:
            rich.print(f"  [dim]Stopping: {audit_result.stop_reason}[/dim]")
            break

        expand, reason = should_expand_paths(audit_result, budget, path_config)
        if not expand:
            rich.print(f"  [dim]No expansion: {reason}[/dim]")
            break

        rich.print(f"  [dim bold yellow]Expanding paths: {reason}[/dim bold yellow]")
        path_specs = request_additional_paths(
            messages, all_candidates, audit_result, budget, config
        )

        if not path_specs:
            rich.print("  [dim]No additional paths generated, stopping.[/dim]")
            break

        debug_info["path_expansion_events"].append({
            "wave": wave_idx + 2,
            "from": budget.paths_used - len(path_specs),
            "added": len(path_specs),
            "reason": reason,
        })

    # Record budget summary
    stop_reason = audit_result.stop_reason if audit_result else "single_wave"
    debug_info["path_budget"] = {
        "initial_paths": debug_info.get("initial_path_count", 0),
        "max_paths": path_config.max_paths,
        "paths_used": budget.paths_used,
        "waves_used": budget.waves_used,
        "stop_reason": stop_reason,
        "tokens_spent": budget.tokens_spent,
    }

    # 4. Safety check: budget exhausted with high conflict
    if (
        budget.paths_remaining <= 0
        and audit_result
        and audit_result.high_conflict
    ):
        rich.print("  [dim bold red]Budget exhausted with unresolved conflict.[/dim bold red]")
        unresolved = GenerationResult(
            content=(
                "I could not resolve the conflict within the path budget.\n"
                "Here are the competing hypotheses and the strongest evidence:\n\n"
                + _format_competing_hypotheses(all_candidates)
            ),
            tokens_used=budget.tokens_spent,
            duration_sec=sum(c.duration_sec for c in all_candidates),
        )
        return unresolved, debug_info

    # 5. Synthesize final answer
    rich.print("  [dim]Synthesizing final answer...[/dim]")
    return _synthesize_with_audit(
        messages, all_candidates, audit_debug, config,
        tools=tools, is_loop=is_loop,
    )


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
