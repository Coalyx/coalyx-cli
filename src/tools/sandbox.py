"""Path sandboxing for file-system tools.

All file-operation tools must call :func:`resolve_safe_path` before
touching the file system.  The function resolves symlinks, normalises
the path, and verifies that the result stays within the project root.
"""

import os
from pathlib import Path
from typing import Dict, Optional

# Module-level project root — set once via :func:`set_project_root`.
_project_root: Optional[Path] = None


class PathSecurityError(Exception):
    """Raised when a path resolves outside the allowed project root."""


def set_project_root(root: str) -> None:
    """Configure the project root used for all path validation.

    Must be called before any file-system tool is executed (typically
    during ``setup_tools()`` in the CLI entry-point).
    """
    global _project_root
    _project_root = Path(root).resolve()


def get_project_root() -> Optional[Path]:
    """Return the configured project root, or None if not yet set."""
    return _project_root


def resolve_safe_path(filepath: str) -> str:
    """Resolve *filepath* and ensure it lives inside the project root.

    The function:
    1. Expands ``~`` (home-dir shorthand).
    2. Resolves relative paths against the project root.
    3. Resolves symlinks via :meth:`Path.resolve`.
    4. Checks that the final path is equal to or a child of the root.

    Returns:
        The absolute, resolved path as a string.

    Raises:
        PathSecurityError: if the path escapes the project root.
        RuntimeError: if the project root has not been configured.
    """
    if _project_root is None:
        raise RuntimeError(
            "Project root has not been configured. "
            "Call sandbox.set_project_root() before using file tools."
        )

    raw = Path(filepath).expanduser()

    # Make relative paths relative to project root, not cwd
    if not raw.is_absolute():
        raw = _project_root / raw

    resolved = raw.resolve()

    try:
        resolved.relative_to(_project_root)
    except ValueError:
        raise PathSecurityError(
            f"Access denied: '{filepath}' resolves to '{resolved}' "
            f"which is outside the project root '{_project_root}'."
        )

    return str(resolved)


def validate_no_package_installation(command: str) -> None:
    """Raise PermissionError if the command attempts to install packages."""
    cmd_lower = command.lower()
    blocked_patterns = [
        "pip install", "pip3 install", "pip2 install",
        "conda install", "poetry add",
        "uv pip install", "uv add"
    ]
    for pattern in blocked_patterns:
        if pattern in cmd_lower:
            raise PermissionError(
                f"Agent is not allowed to install packages. Attempted: '{pattern}'"
            )


def get_safe_env() -> Dict[str, str]:
    """Return a sanitized copy of the current environment variables.

    Removes variables that commonly contain secrets or sensitive tokens
    to prevent leakage to subprocesses (bash, repl, hooks).
    """
    env = os.environ.copy()
    sensitive_patterns = [
        "SECRET", "KEY", "TOKEN", "PASSWORD", "AUTH", 
        "CREDENTIAL", "API", "AWS_", "GCP_", "AZURE_"
    ]
    
    safe_env = {}
    for k, v in env.items():
        k_upper = k.upper()
        if not any(p in k_upper for p in sensitive_patterns):
            safe_env[k] = v
            
    from src.core.env import get_venv_bin_dir
    from pathlib import Path
    venv_bin = str(get_venv_bin_dir(Path.home() / ".coalyx"))
    safe_env["PATH"] = f"{venv_bin}{os.pathsep}{safe_env.get('PATH', '')}"
            
    return safe_env
