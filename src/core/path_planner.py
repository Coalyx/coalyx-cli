import json
import logging
from typing import List, Tuple

from src.core.schema import (
    Message, Role, ModelConfig, TaskProfile, PathSpec, PathMethod,
    PathBudgetRequest, AdaptivePathConfig, PathBudget,
)
from src.core.model import generate_response

logger = logging.getLogger(__name__)

_JSON_RESPONSE_FORMAT = {"type": "json_object"}

PROFILER_PROMPT = """You are a task profiler for a reasoning pipeline.
Analyze the user query and return ONLY valid JSON with these fields:

{
  "complexity_score": 0.0-1.0,
  "requires_exact_answer": true/false,
  "tool_resolvable": true/false,
  "ambiguity_type": "none|intent|factual|structural",
  "risk_level": "low|medium|high"
}

Guidelines:
- complexity_score: 0.0 = trivial arithmetic, 0.3 = factual lookup, 0.5 = explanation, 0.7 = multi-step reasoning, 1.0 = research-grade
- requires_exact_answer: true for math, counting, specific data queries
- tool_resolvable: true if code execution, search, or computation can verify
- ambiguity_type: "none" if the query is clear and self-contained
- risk_level: "high" for security, financial, medical, or destructive operations
"""

PATH_PLANNER_PROMPT = """You are a path planner for a multi-candidate reasoning pipeline.

Given the task profile and user query, design reasoning paths.
Each path must have a distinct purpose. Do NOT create redundant paths.

Return ONLY valid JSON:
{{
  "paths": [
    {{
      "id": "path_<name>",
      "objective": "What this path should accomplish",
      "method": "<method>",
      "system_prompt": "System instruction for this reasoning path"
    }}
  ],
  "reason": "Why these paths are needed",
  "stop_condition": "When to stop adding paths"
}}

Valid methods: direct_solution, formal_proof, computational_check, adversarial_critic, edge_case_search, source_grounding, research, implementation_test

Task profile:
{profile_json}

Requested path count: {path_count}

RULES:
- Generate exactly {path_count} paths
- Each path must have a unique method or distinct objective
- system_prompt must be a concise instruction (1-3 sentences)
- Do NOT include paths with vague objectives like "explore more"
"""

EXPANSION_PROMPT = """You are expanding the reasoning path set because the auditor detected unresolved issues.

Current candidates produced these results:
{candidates_summary}

Audit findings:
- Confidence: {confidence}
- High conflict: {high_conflict}
- Conflict is tool-resolvable: {tool_resolvable}
- Answer disagreement: {disagreement}

Design {count} additional paths to resolve the identified issues.
Each path must target a specific gap. Do NOT repeat existing approaches.

Return ONLY valid JSON:
{{
  "paths": [
    {{
      "id": "path_expand_<name>",
      "objective": "What this path should resolve",
      "method": "<method>",
      "system_prompt": "System instruction for this path"
    }}
  ],
  "reason": "Why these specific paths will reduce uncertainty"
}}

Valid methods: direct_solution, formal_proof, computational_check, adversarial_critic, edge_case_search, source_grounding, research, implementation_test
"""

_DEFAULT_PATHS = [
    PathSpec(
        id="path_direct",
        objective="Answer the query directly and concisely.",
        method=PathMethod.DIRECT_SOLUTION,
        system_prompt="Answer the query directly and concisely.",
    ),
    PathSpec(
        id="path_critic",
        objective="Focus on potential issues, edge cases, and what could go wrong.",
        method=PathMethod.ADVERSARIAL_CRITIC,
        system_prompt=(
            "Answer the query, but focus on potential issues, "
            "edge cases, and what could go wrong."
        ),
    ),
    PathSpec(
        id="path_context",
        objective="Identify missing context or information, then answer with explicit assumptions.",
        method=PathMethod.SOURCE_GROUNDING,
        system_prompt=(
            "Before answering, identify what context or information might be missing. "
            "Then answer with explicit assumptions stated."
        ),
    ),
]


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def profile_task(messages: List[Message], config: ModelConfig) -> TaskProfile:
    last_user_message = ""
    for m in reversed(messages):
        if m.role == Role.USER:
            last_user_message = m.content
            break

    if not last_user_message:
        return TaskProfile()

    try:
        res = generate_response(
            messages=[
                Message(role=Role.SYSTEM, content=PROFILER_PROMPT),
                Message(role=Role.USER, content=last_user_message),
            ],
            model_name=config.model_name,
            temperature=0.1,
            max_tokens=512,
            n=1,
            response_format=_JSON_RESPONSE_FORMAT,
        )[0]

        content = res.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
        if content.endswith("```"):
            content = content[:-3]

        data = json.loads(content.strip())
        return TaskProfile(**data)
    except Exception as e:
        logger.warning("Task profiling failed, using defaults: %s", e)
        return TaskProfile()


def _compute_heuristic_count(profile: TaskProfile, config: AdaptivePathConfig) -> int:
    n = 1
    if profile.complexity_score >= 0.35:
        n += 1
    if profile.requires_exact_answer:
        n += 1
    if profile.tool_resolvable:
        n += 1
    if profile.risk_level == "high":
        n += 1
    return _clamp(n, config.min_paths, config.max_paths)


def _parse_path_specs(content: str, max_count: int) -> List[PathSpec]:
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
    if content.endswith("```"):
        content = content[:-3]

    data = json.loads(content.strip())
    paths_data = data.get("paths", [])

    specs = []
    for p in paths_data[:max_count]:
        try:
            specs.append(PathSpec(
                id=p.get("id", f"path_{len(specs)}"),
                objective=p.get("objective", ""),
                method=PathMethod(p.get("method", "direct_solution")),
                system_prompt=p.get("system_prompt", p.get("objective", "")),
            ))
        except (ValueError, KeyError):
            continue

    return specs


def plan_initial_paths(
    profile: TaskProfile,
    messages: List[Message],
    config: ModelConfig,
    path_config: AdaptivePathConfig,
) -> Tuple[List[PathSpec], PathBudgetRequest]:
    path_count = _compute_heuristic_count(profile, path_config)

    if path_count == 1:
        return (
            [_DEFAULT_PATHS[0]],
            PathBudgetRequest(
                initial_paths=1,
                max_desired_paths=2,
                reason="simple task, single path sufficient",
                stop_condition="Stop after first path if no conflict.",
            ),
        )

    last_user_message = ""
    for m in reversed(messages):
        if m.role == Role.USER:
            last_user_message = m.content
            break

    try:
        prompt = PATH_PLANNER_PROMPT.format(
            profile_json=profile.model_dump_json(indent=2),
            path_count=path_count,
        )
        res = generate_response(
            messages=[
                Message(role=Role.SYSTEM, content=prompt),
                Message(role=Role.USER, content=last_user_message),
            ],
            model_name=config.model_name,
            temperature=0.2,
            max_tokens=1024,
            n=1,
            response_format=_JSON_RESPONSE_FORMAT,
        )[0]

        specs = _parse_path_specs(res.content, path_count)
        if len(specs) >= 2:
            budget = PathBudgetRequest(
                initial_paths=len(specs),
                max_desired_paths=path_config.max_paths,
                reason=json.loads(res.content.strip()).get("reason", ""),
                stop_condition=json.loads(res.content.strip()).get("stop_condition", ""),
            )
            return specs, budget
    except Exception as e:
        logger.warning("Path planning LLM call failed, using defaults: %s", e)

    specs = _DEFAULT_PATHS[:path_count]
    return (
        specs,
        PathBudgetRequest(
            initial_paths=len(specs),
            max_desired_paths=path_config.max_paths,
            reason="heuristic fallback",
            stop_condition="Stop when confidence target met.",
        ),
    )


def request_additional_paths(
    messages: List[Message],
    candidates: list,
    audit,
    budget: PathBudget,
    config: ModelConfig,
) -> List[PathSpec]:
    allowed = min(
        budget.config.max_paths_per_wave,
        budget.paths_remaining,
        budget.tokens_remaining // max(budget.min_tokens_per_path, 1),
    )

    if allowed <= 0:
        return []

    candidates_summary = "\n".join(
        f"Candidate {i+1}: {c.content[:200]}..."
        for i, c in enumerate(candidates)
    )

    try:
        prompt = EXPANSION_PROMPT.format(
            candidates_summary=candidates_summary,
            confidence=audit.confidence,
            high_conflict=audit.high_conflict,
            tool_resolvable=audit.conflict_is_tool_resolvable,
            disagreement=audit.answer_disagreement,
            count=allowed,
        )

        last_user_message = ""
        for m in reversed(messages):
            if m.role == Role.USER:
                last_user_message = m.content
                break

        res = generate_response(
            messages=[
                Message(role=Role.SYSTEM, content=prompt),
                Message(role=Role.USER, content=last_user_message),
            ],
            model_name=config.model_name,
            temperature=0.2,
            max_tokens=1024,
            n=1,
            response_format=_JSON_RESPONSE_FORMAT,
        )[0]

        content = res.content.strip()
        data = json.loads(content if not content.startswith("```") else content.split("\n", 1)[-1].rstrip("`"))
        reason = data.get("reason", "")

        if not reason or len(reason) < 10:
            logger.warning("Expansion request rejected: reason too vague")
            return []

        return _parse_path_specs(res.content, allowed)
    except Exception as e:
        logger.warning("Path expansion failed: %s", e)
        return []
