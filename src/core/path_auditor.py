import logging
from typing import List, Tuple

from src.core.schema import (
    Message, Role, ModelConfig, GenerationResult,
    TaskProfile, PathAuditResult, AdaptivePathConfig, PathBudget,
    UncertaintyAction,
)
from src.core.embedding import calculate_group_consistency
from src.core.uncertainty import analyze_uncertainty, decide_action

logger = logging.getLogger(__name__)


def _lexical_jaccard(text_a: str, text_b: str) -> float:
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a and not words_b:
        return 1.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union) if union else 1.0


def _lexical_group_consistency(texts: list) -> float:
    if len(texts) <= 1:
        return 1.0
    sims = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sims.append(_lexical_jaccard(texts[i], texts[j]))
    return sum(sims) / len(sims) if sims else 1.0


def audit_paths(
    candidates: List[GenerationResult],
    messages: List[Message],
    profile: TaskProfile,
    config: ModelConfig,
    path_config: AdaptivePathConfig,
) -> Tuple[PathAuditResult, dict]:
    texts = [c.content for c in candidates]

    try:
        consistency_result = calculate_group_consistency(texts)
        consistency = consistency_result.score
    except Exception as e:
        consistency = _lexical_group_consistency(texts)
        logger.warning("Embedding failed in auditor, lexical fallback: %s", e)

    report = analyze_uncertainty(messages, candidates, consistency, config)

    last_user_message = ""
    for m in reversed(messages):
        if m.role == Role.USER:
            last_user_message = m.content
            break

    action = decide_action(report, last_user_message=last_user_message)

    high_conflict = bool(
        report.claim_conflicts
        and any(c.severity == "high" for c in report.claim_conflicts)
    )
    has_computable = bool(report.computable_checks)
    answer_disagreement = high_conflict or consistency < 0.6

    confidence = (
        0.4 * consistency
        + 0.3 * report.semantic_agreement
        + 0.3 * (1.0 - report.total_score)
    )

    verified = (
        action == UncertaintyAction.ANSWER
        and confidence >= path_config.target_confidence
    )

    should_stop = verified or (
        confidence >= path_config.target_confidence and not high_conflict
    )

    should_expand, expansion_reason = _should_expand(
        PathAuditResult(
            confidence=confidence,
            high_conflict=high_conflict,
            conflict_is_tool_resolvable=has_computable,
            requires_exact_answer=profile.requires_exact_answer,
            answer_disagreement=answer_disagreement,
            verified_answer_found=verified,
        ),
        path_config,
    )

    audit = PathAuditResult(
        confidence=confidence,
        answer_agreement=consistency >= 0.8,
        high_conflict=high_conflict,
        conflict_is_tool_resolvable=has_computable,
        requires_exact_answer=profile.requires_exact_answer,
        answer_disagreement=answer_disagreement,
        verified_answer_found=verified,
        should_stop=should_stop,
        should_expand=should_expand,
        stop_reason="" if not should_stop else "confidence_target_met",
        expansion_reason=expansion_reason,
    )

    debug = {
        "consistency": consistency,
        "uncertainty_report": report.model_dump(),
        "controller_action": action.value,
        "audit_result": audit.model_dump(),
    }

    return audit, debug


def _should_expand(audit: PathAuditResult, config: AdaptivePathConfig) -> Tuple[bool, str]:
    if audit.verified_answer_found:
        return False, "verified_answer_found"

    if audit.confidence >= config.target_confidence and not audit.high_conflict:
        return False, "confidence_target_met"

    if audit.high_conflict and audit.conflict_is_tool_resolvable:
        return True, "high_conflict_tool_resolvable"

    if audit.answer_disagreement and audit.requires_exact_answer:
        return True, "exact_answer_disagreement"

    return False, "no_useful_expansion"


def should_expand_paths(
    audit: PathAuditResult,
    budget: PathBudget,
    config: AdaptivePathConfig,
) -> Tuple[bool, str]:
    if budget.paths_remaining <= 0:
        return False, "max_paths_reached"

    if budget.tokens_remaining < budget.min_tokens_per_path:
        return False, "token_budget_exhausted"

    if audit.verified_answer_found:
        return False, "verified_answer_found"

    if audit.confidence >= config.target_confidence and not audit.high_conflict:
        return False, "confidence_target_met"

    if audit.high_conflict and audit.conflict_is_tool_resolvable:
        return True, "high_conflict_tool_resolvable"

    if audit.answer_disagreement and audit.requires_exact_answer:
        return True, "exact_answer_disagreement"

    return False, "no_useful_expansion"
