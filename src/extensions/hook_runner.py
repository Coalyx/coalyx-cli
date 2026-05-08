import logging
import subprocess
from typing import List, Dict, Any

from src.core.schema import HookEvent, HookResult
from src.extensions.registry import ExtensionRegistry, get_hooks_for_event

logger = logging.getLogger(__name__)

# Characters / patterns that indicate shell injection risk in hook commands
_DANGEROUS_PATTERNS = [
    "`",       # backtick command substitution
    "$(",      # subshell
    "&&",      # command chaining
    "||",      # command chaining
    ";",       # command separator
    "|",       # pipe
    ">",       # redirect
    "<",       # redirect
]


def _validate_hook_command(command: str) -> None:
    """Raise ValueError if *command* contains dangerous shell patterns.

    Hook commands loaded from extension config files should be simple
    executables with arguments. Complex shell constructs (pipes,
    redirects, backticks, chaining) are blocked to reduce the risk
    of config-file tampering leading to arbitrary code execution.
    """
    for pattern in _DANGEROUS_PATTERNS:
        if pattern in command:
            raise ValueError(
                f"Hook command contains dangerous pattern '{pattern}': {command!r}"
            )


def run_hooks(
    registry: ExtensionRegistry, event: HookEvent, context: Dict[str, Any] = None
) -> List[HookResult]:
    """Execute all registered hooks for a given lifecycle event.

    Args:
        registry: The extension registry containing hook configurations.
        event: The lifecycle event being triggered.
        context: Optional dictionary of contextual data passed to hooks
                 as environment variables prefixed with COALYX_.

    Returns:
        List of HookResult for each executed hook.
    """
    if context is None:
        context = {}

    hooks = get_hooks_for_event(registry, event)
    results = []

    for hook in hooks:
        try:
            _validate_hook_command(hook.command)
        except ValueError as e:
            logger.warning("Skipping unsafe hook: %s", e)
            results.append(HookResult(
                event=event,
                success=False,
                error=str(e),
            ))
            continue

        result = execute_shell_hook(hook.command, context)
        result.event = event
        logger.info(
            "Hook [%s] %s: %s",
            event.value,
            "OK" if result.success else "FAIL",
            hook.command,
        )
        results.append(result)

    return results


def execute_shell_hook(command: str, context: Dict[str, Any] = None) -> HookResult:
    """Execute a single shell hook command.

    Context values are injected as environment variables with the
    COALYX_ prefix (e.g., context key "session_id" becomes
    COALYX_SESSION_ID).

    Args:
        command: Shell command string to execute.
        context: Optional context data exposed as env vars.

    Returns:
        HookResult with success status and captured output.
    """
    import shlex
    from src.tools.sandbox import get_safe_env

    env = get_safe_env()
    if context:
        for key, value in context.items():
            env_key = f"COALYX_{key.upper()}"
            env[env_key] = str(value)

    try:
        args = shlex.split(command)
        proc = subprocess.run(
            args,
            shell=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        return HookResult(
            event=HookEvent.NOTIFICATION,
            success=proc.returncode == 0,
            output=proc.stdout.strip(),
            error=proc.stderr.strip(),
        )
    except subprocess.TimeoutExpired:
        return HookResult(
            event=HookEvent.NOTIFICATION,
            success=False,
            error=f"Hook timed out after 30s: {command}",
        )
    except Exception as e:
        return HookResult(
            event=HookEvent.NOTIFICATION,
            success=False,
            error=str(e),
        )
