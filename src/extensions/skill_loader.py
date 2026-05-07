import re
from pathlib import Path
from typing import List

from src.core.schema import SkillDefinition


def discover_skills(skills_dir: Path) -> List[SkillDefinition]:
    """Discover and parse all skill files from a directory.

    Scans for *.md files in the given directory and parses each one.

    Args:
        skills_dir: Path to the .coalyx/skills/ directory.

    Returns:
        List of parsed SkillDefinition objects.
    """
    if not skills_dir.exists():
        return []

    skills = []
    for filepath in sorted(skills_dir.glob("*.md")):
        try:
            skill = parse_skill_file(filepath)
            skills.append(skill)
        except ValueError:
            continue

    return skills


def parse_skill_file(filepath: Path) -> SkillDefinition:
    """Parse a single skill markdown file.

    Expected format:
        ---
        name: My Skill
        triggers: pattern1, pattern2, regex.*pattern
        ---
        The body below is the instruction text that gets injected
        into the system prompt when the skill is activated.

    Args:
        filepath: Path to the .md skill file.

    Returns:
        Parsed SkillDefinition.

    Raises:
        ValueError: If the file cannot be parsed.
    """
    content = filepath.read_text(encoding="utf-8")

    frontmatter_match = re.match(
        r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL
    )

    if not frontmatter_match:
        raise ValueError(f"No valid frontmatter found in {filepath}")

    header_block = frontmatter_match.group(1)
    body = frontmatter_match.group(2).strip()

    name = filepath.stem
    triggers: List[str] = []

    for line in header_block.splitlines():
        line = line.strip()
        if line.lower().startswith("name:"):
            name = line.split(":", 1)[1].strip()
        elif line.lower().startswith("triggers:"):
            raw = line.split(":", 1)[1].strip()
            triggers = [t.strip() for t in raw.split(",") if t.strip()]

    return SkillDefinition(
        name=name,
        trigger_patterns=triggers,
        instructions=body,
        source_file=str(filepath),
    )
