import logging
from typing import Dict, List, Any
from src.tools.base import ToolWrapper

logger = logging.getLogger(__name__)

class ToolRegistry:
    """Registry to hold and dispatch AI tools."""
    def __init__(self):
        self._tools: Dict[str, ToolWrapper] = {}

    def register(self, tool: ToolWrapper):
        self._tools[tool.name] = tool
        
    def get_all_schemas(self) -> List[Dict[str, Any]]:
        return [t.schema for t in self._tools.values()]
        
    def execute(self, name: str, kwargs: Dict[str, Any]) -> str:
        if name not in self._tools:
            return f"Error: Tool '{name}' not found."

        # --- Confirmation gate ---
        from src.tools.confirmation import request_confirmation
        if not request_confirmation(name, kwargs):
            logger.info("User denied execution of tool '%s'", name)
            return f"Tool '{name}' execution was denied by the user."

        try:
            result = self._tools[name](**kwargs)
            return str(result) if result is not None else "Success (no output)"
        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            return f"Error executing tool '{name}': {str(e)}"

# Global registry instance
_global_registry = ToolRegistry()

def register_tool(tool: ToolWrapper):
    """Register a tool globally."""
    _global_registry.register(tool)
    
def get_all_tool_schemas() -> List[Dict[str, Any]]:
    """Get schemas for all registered tools."""
    return _global_registry.get_all_schemas()

def execute_tool(name: str, kwargs: Dict[str, Any]) -> str:
    """Execute a registered tool by name with arguments."""
    return _global_registry.execute(name, kwargs)

def setup_tools():
    """Import all tool modules to trigger their @tool decorators and register them."""
    import src.tools.system
    import src.tools.web
    import src.tools.task
    import src.tools.protocol
    import src.tools.utility
