import json
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List

from src.core.schema import Message, ToolCallLog

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
    re.compile(r"ghp_[0-9A-Za-z_]{30,}"),
    re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----"),
]


class SessionWorkspace:
    """Structured reasoning workspace for a single session."""

    def __init__(self, session_id: str, coalyx_root: Path):
        self.session_id = session_id
        self.root = coalyx_root / "sessions" / session_id
        self._ensure_structure()

    def _ensure_structure(self) -> None:
        for subdir in ["artifacts", "tools", "uncertainty"]:
            (self.root / subdir).mkdir(parents=True, exist_ok=True)

    # --- Manifest ---

    def save_manifest(self, data: dict) -> None:
        path = self.root / "manifest.json"
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def load_manifest(self) -> Optional[dict]:
        path = self.root / "manifest.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    # --- Messages (append-only JSONL) ---

    def append_message(self, message: Message) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **message.model_dump(),
        }
        entry = self._redact_secrets(entry)
        with open(self.root / "messages.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def load_messages(self) -> List[dict]:
        path = self.root / "messages.jsonl"
        if not path.exists():
            return []
        entries = []
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries

    # --- Tool call logging ---

    def log_tool_call(self, log: ToolCallLog) -> None:
        entry = self._redact_secrets(log.model_dump())
        with open(self.root / "tools" / "calls.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # --- Artifacts ---


    def list_artifacts(self) -> List[str]:
        artifacts_dir = self.root / "artifacts"
        if not artifacts_dir.exists():
            return []
        return [f.name for f in sorted(artifacts_dir.iterdir()) if f.is_file()]

    # --- Uncertainty reports ---

    def log_uncertainty(self, report: dict) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **report,
        }
        with open(
            self.root / "uncertainty" / "reports.jsonl", "a", encoding="utf-8"
        ) as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # --- Security: redact secrets before persisting ---

    def _redact_secrets(self, data: dict) -> dict:
        text = json.dumps(data, ensure_ascii=False)
        for pattern in SECRET_PATTERNS:
            text = pattern.sub("[REDACTED]", text)
        return json.loads(text)
