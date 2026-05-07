from pathlib import Path

MAX_MEMORY_LINES = 500
MEMORY_FILENAME = "COALYX.md"


def load_project_memory(project_root: Path) -> str:
    """Load the project memory anchor file (COALYX.md) as a system prompt.

    This file stores persistent project context: tech stack, conventions,
    security rules, and other invariant project knowledge. Kept under
    500 lines to avoid context bloat.

    Args:
        project_root: Root directory of the project.

    Returns:
        Contents of COALYX.md, or empty string if not found.
    """
    memory_path = project_root / MEMORY_FILENAME
    if not memory_path.exists():
        return ""

    content = memory_path.read_text(encoding="utf-8")

    if not validate_memory_file(content):
        return content

    return content


def validate_memory_file(content: str) -> bool:
    """Check that the memory file stays under the recommended line limit.

    Args:
        content: The raw text content of COALYX.md.

    Returns:
        True if within limits, False if exceeding MAX_MEMORY_LINES.
    """
    line_count = content.count("\n") + 1
    return line_count <= MAX_MEMORY_LINES


def scaffold_memory_file(project_root: Path) -> Path:
    """Create a starter COALYX.md file in the project root.

    Args:
        project_root: Root directory of the project.

    Returns:
        Path to the created file.
    """
    memory_path = project_root / MEMORY_FILENAME
    if memory_path.exists():
        return memory_path

    template = (
        "# Project Memory\n"
        "\n"
        "This file is the project memory anchor for Coalyx CLI.\n"
        "Store persistent project context here: tech stack, conventions,\n"
        "security rules, and other invariant project knowledge.\n"
        "\n"
        "Keep this file under 500 lines to avoid context bloat.\n"
        "\n"
        "## Tech Stack\n"
        "\n"
        "- Language: \n"
        "- Framework: \n"
        "\n"
        "## Conventions\n"
        "\n"
        "- \n"
        "\n"
        "## Security Rules\n"
        "\n"
        "- Never commit secrets or API keys\n"
    )

    memory_path.write_text(template, encoding="utf-8")
    return memory_path
