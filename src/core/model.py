import time
import logging
from typing import List, Optional

import litellm
from litellm import completion

from src.core.schema import Message, GenerationResult

litellm.suppress_debug_info = False
litellm.set_verbose = False
logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 5
DEFAULT_BASE_DELAY = 2.0
DEFAULT_MAX_DELAY = 60.0

PROVIDERS_NO_BATCH = {"gemini", "google", "claude", "anthropic", "ollama"}
SEQUENTIAL_CALL_DELAY = 4.0


def generate_response(
    messages: List[Message],
    model_name: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    n: int = 1,
    max_retries: int = DEFAULT_MAX_RETRIES,
    extra_body: Optional[dict] = None,
) -> List[GenerationResult]:
    """Generate response(s) using litellm with automatic retry on rate limits.

    Implements exponential backoff with jitter for RateLimitError.
    When n > 1, attempts a single batched call first. If the provider
    does not support the n parameter, falls back to sequential calls
    with rate-limit-safe spacing.

    Args:
        messages: Conversation messages.
        model_name: Model identifier (e.g. 'gemini/gemini-2.0-flash').
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        n: Number of candidates to generate.
        max_retries: Maximum retry attempts on rate limit errors.
        extra_body: Optional extra payload passed directly to the provider
            (e.g. ``{"think": False}`` to disable Ollama thinking tokens).

    Returns:
        List of GenerationResult.
    """
    formatted = [{"role": m.role.value, "content": m.content} for m in messages]

    if n > 1 and not _is_no_batch_provider(model_name):
        try:
            return _call_with_retry(
                formatted, model_name, temperature, max_tokens, n, max_retries,
                extra_body=extra_body,
            )
        except RuntimeError:
            pass  # fall through to sequential

    if n > 1:
        return _sequential_generate(
            formatted, model_name, temperature, max_tokens, n, max_retries,
            extra_body=extra_body,
        )

    return _call_with_retry(
        formatted, model_name, temperature, max_tokens, 1, max_retries,
        extra_body=extra_body,
    )


def _call_with_retry(
    formatted_messages: list,
    model_name: str,
    temperature: float,
    max_tokens: int,
    n: int,
    max_retries: int,
    extra_body: Optional[dict] = None,
) -> List[GenerationResult]:
    """Execute a single litellm call with exponential backoff retry.

    Args:
        formatted_messages: Pre-formatted message dicts.
        model_name: Model identifier.
        temperature: Sampling temperature.
        max_tokens: Max tokens.
        n: Number of choices.
        max_retries: Max retry count.
        extra_body: Optional extra payload forwarded to the provider API.

    Returns:
        List of GenerationResult.

    Raises:
        RuntimeError: If all retries are exhausted or a non-retryable error occurs.
    """
    delay = DEFAULT_BASE_DELAY
    call_kwargs = dict(
        model=model_name,
        messages=formatted_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
    )
    if extra_body:
        call_kwargs["extra_body"] = extra_body

    for attempt in range(max_retries + 1):
        start = time.time()
        try:
            response = completion(**call_kwargs)
            duration = time.time() - start
            return _parse_response(response, duration)

        except litellm.AuthenticationError as e:
            raise RuntimeError(
                f"Authentication failed — check your API key.\n"
                f"Run: coalyx config set gemini-api-key <your_key>\n"
                f"Detail: {e}"
            )

        except litellm.RateLimitError as e:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Rate limit exceeded after {max_retries} retries.\n"
                    f"Last error: {e}"
                )

            retry_after = _extract_retry_after(e)
            wait = retry_after if retry_after else delay

            logger.warning(
                "Rate limited (attempt %d/%d). Waiting %.1fs... [%s]",
                attempt + 1, max_retries, wait, str(e)[:120],
            )
            time.sleep(wait)
            delay = min(delay * 2, DEFAULT_MAX_DELAY)

        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")


def _sequential_generate(
    formatted_messages: list,
    model_name: str,
    temperature: float,
    max_tokens: int,
    n: int,
    max_retries: int,
    extra_body: Optional[dict] = None,
) -> List[GenerationResult]:
    """Generate n candidates via sequential single calls with spacing.

    Fallback when the provider does not support the n parameter.
    Inserts a small delay between calls to avoid hitting rate limits.

    Args:
        formatted_messages: Pre-formatted message dicts.
        model_name: Model identifier.
        temperature: Sampling temperature.
        max_tokens: Max tokens.
        n: Number of candidates.
        max_retries: Max retry count per call.
        extra_body: Optional extra payload forwarded to the provider API.

    Returns:
        List of GenerationResult.
    """
    results: List[GenerationResult] = []
    inter_call_delay = SEQUENTIAL_CALL_DELAY

    for i in range(n):
        if i > 0:
            time.sleep(inter_call_delay)

        batch = _call_with_retry(
            formatted_messages, model_name, temperature, max_tokens, 1, max_retries,
            extra_body=extra_body,
        )
        results.extend(batch)

    return results


def _parse_response(response, duration: float) -> List[GenerationResult]:
    """Parse a litellm response into GenerationResult objects.

    Args:
        response: Raw litellm completion response.
        duration: Wall-clock seconds for the call.

    Returns:
        List of GenerationResult.
    """
    results = []
    total_tokens = getattr(response.usage, "total_tokens", 0)
    choice_count = len(response.choices)

    for choice in response.choices:
        content = choice.message.content or ""
        tokens = total_tokens // choice_count

        results.append(
            GenerationResult(
                content=content,
                tokens_used=tokens,
                duration_sec=duration,
            )
        )

    return results


def _is_no_batch_provider(model_name: str) -> bool:
    """Check if a model's provider does not support n>1 batch completions.

    Args:
        model_name: Model identifier string (e.g. 'gemini/gemini-2.0-flash').

    Returns:
        True if the provider only supports n=1 per call.
    """
    lower = model_name.lower()
    return any(p in lower for p in PROVIDERS_NO_BATCH)


def _extract_retry_after(error: Exception) -> float:
    """Try to extract a Retry-After hint from a rate limit error.

    Args:
        error: The rate limit exception.

    Returns:
        Seconds to wait, or 0.0 if not available.
    """
    msg = str(error)
    if "retry after" in msg.lower():
        import re
        match = re.search(r"(\d+(?:\.\d+)?)\s*s", msg.lower())
        if match:
            return float(match.group(1))
    return 0.0
