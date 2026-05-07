import subprocess
from typing import List, Dict, Any

from src.core.schema import HookEvent, HookResult
from src.extensions.registry import ExtensionRegistry, get_hooks_for_event


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
        result = execute_shell_hook(hook.command, context)
        result.event = event
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
    import os

    env = os.environ.copy()
    if context:
        for key, value in context.items():
            env_key = f"COALYX_{key.upper()}"
            env[env_key] = str(value)

    try:
        proc = subprocess.run(
            command,
            shell=True,
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
