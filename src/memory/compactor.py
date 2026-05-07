from typing import List

from src.core.schema import Message, Role, CompactionResult
from src.core.model import generate_response

COMPACT_SYSTEM_PROMPT = (
    "You are a conversation summarizer. Condense the following conversation "
    "into a concise summary that preserves all key decisions, facts, code snippets, "
    "and action items. Output ONLY the summary, no preamble."
)
MIN_MESSAGES_TO_COMPACT = 2
KEEP_RECENT_DEFAULT = 4


def compact_messages(
    messages: List[Message],
    model_name: str,
    keep_recent: int = KEEP_RECENT_DEFAULT,
) -> CompactionResult:
    """Compress older messages into a summary, preserving recent ones.

    Args:
        messages: Full conversation history.
        model_name: Model to use for summarization.
        keep_recent: Number of recent messages to exclude from compaction.

    Returns:
        CompactionResult with summary of the older portion.
    """
    if len(messages) <= MIN_MESSAGES_TO_COMPACT:
        return CompactionResult(
            summary="",
            original_count=len(messages),
            compacted_count=len(messages),
            tokens_saved=0,
        )

    to_compact = messages[:-keep_recent] if len(messages) > keep_recent else messages
    if len(to_compact) <= MIN_MESSAGES_TO_COMPACT:
        return CompactionResult(
            summary="",
            original_count=len(messages),
            compacted_count=len(messages),
            tokens_saved=0,
        )

    conversation_text = "\n".join(
        f"[{m.role.value}]: {m.content}" for m in to_compact
    )

    summary_prompt = [
        Message(role=Role.SYSTEM, content=COMPACT_SYSTEM_PROMPT),
        Message(role=Role.USER, content=conversation_text),
    ]

    result = generate_response(
        messages=summary_prompt,
        model_name=model_name,
        temperature=0.3,
        max_tokens=1024,
        n=1,
    )[0]

    original_tokens = sum(len(m.content) // 4 for m in to_compact)
    summary_tokens = len(result.content) // 4

    return CompactionResult(
        summary=result.content,
        original_count=len(to_compact),
        compacted_count=1,
        tokens_saved=max(original_tokens - summary_tokens, 0),
    )


def apply_compaction(
    messages: List[Message],
    result: CompactionResult,
    keep_recent: int = KEEP_RECENT_DEFAULT,
) -> List[Message]:
    """Replace older messages with summary, keeping recent messages intact.

    Args:
        messages: Full conversation history.
        result: Compaction result containing the summary text.
        keep_recent: Number of recent messages to preserve after summary.

    Returns:
        New message list: [summary] + [recent messages].
    """
    if not result.summary:
        return messages

    summary_msg = Message(
        role=Role.SYSTEM,
        content=f"[Compacted conversation summary]\n{result.summary}",
    )

    recent = messages[-keep_recent:] if len(messages) > keep_recent else []
    return [summary_msg] + recent
