from typing import List

from src.core.schema import Message, Role, CompactionResult, SessionState
from src.core.model import generate_response

COMPACT_SYSTEM_PROMPT = (
    "You are a conversation summarizer. Condense the following conversation "
    "into a concise summary that preserves all key decisions, facts, code snippets, "
    "and action items. Output ONLY the summary, no preamble."
)


def compact_messages(messages: List[Message], model_name: str) -> CompactionResult:
    """Compress a list of messages into a single summary message via LLM.

    Uses Instant mode (single call) to avoid recursive multi-path sampling.

    Args:
        messages: The conversation messages to compact.
        model_name: Model identifier for the summarization call.

    Returns:
        CompactionResult with the summary and token savings metadata.
    """
    if len(messages) <= 2:
        return CompactionResult(
            summary="",
            original_count=len(messages),
            compacted_count=len(messages),
            tokens_saved=0,
        )

    conversation_text = "\n".join(
        f"[{m.role.value}]: {m.content}" for m in messages
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

    original_tokens = sum(len(m.content) // 4 for m in messages)
    summary_tokens = len(result.content) // 4

    return CompactionResult(
        summary=result.content,
        original_count=len(messages),
        compacted_count=1,
        tokens_saved=max(original_tokens - summary_tokens, 0),
    )


def apply_compaction(messages: List[Message], result: CompactionResult) -> List[Message]:
    """Replace old messages with a single summary message.

    Args:
        messages: Current message list.
        result: The compaction result containing the summary.

    Returns:
        New message list starting with the compacted summary.
    """
    if not result.summary:
        return messages

    summary_msg = Message(
        role=Role.SYSTEM,
        content=f"[Compacted conversation summary]\n{result.summary}",
    )
    return [summary_msg]
