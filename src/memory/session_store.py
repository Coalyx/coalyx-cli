import json
import uuid
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from src.core.schema import SessionSnapshot


def _ensure_dir(directory: Path) -> None:
    """Create directory if it doesn't exist."""
    directory.mkdir(parents=True, exist_ok=True)


def generate_session_id() -> str:
    """Generate a unique session identifier.

    Returns:
        A short UUID string.
    """
    return uuid.uuid4().hex[:12]


def save_session(snapshot: SessionSnapshot, session_dir: Path) -> Path:
    """Persist a session snapshot to disk as JSON.

    Args:
        snapshot: The session state to save.
        session_dir: Directory to store session files.

    Returns:
        Path to the saved session file.
    """
    _ensure_dir(session_dir)
    snapshot.updated_at = datetime.utcnow().isoformat()

    filepath = session_dir / f"{snapshot.session_id}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(snapshot.model_dump(), f, indent=2, ensure_ascii=False)

    return filepath


def load_session(session_id: str, session_dir: Path) -> Optional[SessionSnapshot]:
    """Load a session snapshot from disk.

    Args:
        session_id: The session identifier.
        session_dir: Directory containing session files.

    Returns:
        SessionSnapshot if found, None otherwise.
    """
    filepath = session_dir / f"{session_id}.json"
    if not filepath.exists():
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return SessionSnapshot(**data)


def list_sessions(session_dir: Path) -> List[SessionSnapshot]:
    """List all saved sessions ordered by most recently updated.

    Args:
        session_dir: Directory containing session files.

    Returns:
        List of SessionSnapshot objects, newest first.
    """
    if not session_dir.exists():
        return []

    sessions = []
    for filepath in session_dir.glob("*.json"):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            sessions.append(SessionSnapshot(**data))
        except (json.JSONDecodeError, ValueError):
            continue

    sessions.sort(key=lambda s: s.updated_at, reverse=True)
    return sessions
