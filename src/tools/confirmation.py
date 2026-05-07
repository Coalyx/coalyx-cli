"""Tool confirmation gate for dangerous operations.

Classifies tools by danger level and requires user confirmation
before executing high-risk operations (shell commands, file writes,
code execution). Supports an auto-approve mode for power users.
"""

import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Danger classification
# ---------------------------------------------------------------------------

DANGER_HIGH = "high"
DANGER_MEDIUM = "medium"
DANGER_LOW = "low"
DANGER_NONE = "none"

TOOL_DANGER_LEVELS: Dict[str, str] = {
    # Arbitrary command / code execution
    "bash": DANGER_HIGH,
    "powershell": DANGER_HIGH,
    "repl": DANGER_HIGH,
    # File mutation
    "write_file": DANGER_MEDIUM,
    "edit_file": DANGER_MEDIUM,
    "notebook_edit": DANGER_MEDIUM,
    # Network with SSRF risk
    "web_fetch": DANGER_LOW,
    # Task / agent stubs — currently no-ops but will need review when live
    "task_create": DANGER_LOW,
    "cron_create": DANGER_LOW,
}


def get_danger_level(tool_name: str) -> str:
    """Return the danger classification for a tool.

    Defaults to ``DANGER_NONE`` for tools not explicitly listed
    (read-only tools, search, etc.).
    """
    return TOOL_DANGER_LEVELS.get(tool_name, DANGER_NONE)


# ---------------------------------------------------------------------------
# Confirmation callback protocol
# ---------------------------------------------------------------------------

# The confirmation callback receives (tool_name, kwargs, danger_level)
# and returns True if the user approves execution, False otherwise.
ConfirmationCallback = Callable[[str, Dict[str, Any], str], bool]

# Module-level state — configured once at startup from the CLI layer.
_confirmation_callback: Optional[ConfirmationCallback] = None
_auto_approve: bool = False


def configure_confirmation(
    callback: ConfirmationCallback,
    auto_approve: bool = False,
) -> None:
    """Set up the confirmation gate.

    Args:
        callback: Function that prompts the user and returns True/False.
        auto_approve: If True, skip all confirmations (--auto-approve mode).
    """
    global _confirmation_callback, _auto_approve
    _confirmation_callback = callback
    _auto_approve = auto_approve


def request_confirmation(
    tool_name: str,
    kwargs: Dict[str, Any],
) -> bool:
    """Check whether execution of *tool_name* should proceed.

    Returns True if:
    - the tool is classified as ``DANGER_NONE`` (read-only, safe)
    - auto-approve mode is enabled
    - no callback is configured (legacy / testing path)
    - the user explicitly approves via the callback

    Returns False only when the user actively denies execution.
    """
    level = get_danger_level(tool_name)

    if level == DANGER_NONE:
        return True

    if _auto_approve:
        logger.info(
            "Auto-approved %s tool '%s'", level, tool_name
        )
        return True

    if _confirmation_callback is None:
        logger.warning(
            "No confirmation callback configured; BLOCKING '%s' (fail-closed)",
            tool_name,
        )
        return False

    return _confirmation_callback(tool_name, kwargs, level)
