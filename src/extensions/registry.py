from typing import List, Optional, Dict
import re

from src.core.schema import HookEvent, HookConfig, SkillDefinition


class ExtensionRegistry:
    """Central registry holding all registered hooks and loaded skills."""

    def __init__(self):
        self._hooks: Dict[HookEvent, List[HookConfig]] = {e: [] for e in HookEvent}
        self._skills: List[SkillDefinition] = []


def create_registry() -> ExtensionRegistry:
    """Create a fresh extension registry.

    Returns:
        An empty ExtensionRegistry.
    """
    return ExtensionRegistry()


def register_hook(registry: ExtensionRegistry, config: HookConfig) -> None:
    """Register a hook into the registry for a specific event.

    Args:
        registry: The extension registry.
        config: Hook configuration specifying event and command.
    """
    registry._hooks[config.event].append(config)


def register_skill(registry: ExtensionRegistry, skill: SkillDefinition) -> None:
    """Register a skill into the registry.

    Args:
        registry: The extension registry.
        skill: The parsed skill definition.
    """
    registry._skills.append(skill)


def get_hooks_for_event(
    registry: ExtensionRegistry, event: HookEvent
) -> List[HookConfig]:
    """Retrieve all enabled hooks for a given event type.

    Args:
        registry: The extension registry.
        event: The lifecycle event to query.

    Returns:
        List of enabled HookConfig for the event.
    """
    return [h for h in registry._hooks.get(event, []) if h.enabled]


def match_skill(
    registry: ExtensionRegistry, user_input: str
) -> Optional[SkillDefinition]:
    """Find the first skill whose trigger patterns match the user input.

    Patterns are matched as case-insensitive regex against the user input.

    Args:
        registry: The extension registry.
        user_input: The raw user input string.

    Returns:
        The first matching SkillDefinition, or None.
    """
    for skill in registry._skills:
        for pattern in skill.trigger_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                return skill
    return None
