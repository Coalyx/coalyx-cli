import inspect
from typing import Callable, Any, Dict, List

def _python_type_to_json_schema(py_type: Any) -> Dict[str, Any]:
    """Convert a Python type hint to a JSON Schema type."""
    if py_type == int:
        return {"type": "integer"}
    elif py_type == float:
        return {"type": "number"}
    elif py_type == bool:
        return {"type": "boolean"}
    elif hasattr(py_type, "__origin__") and py_type.__origin__ == list:
        return {"type": "array", "items": {"type": "string"}}
    return {"type": "string"}

def generate_tool_schema(func: Callable) -> Dict[str, Any]:
    """Generate OpenAI-compatible tool schema from a Python function."""
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    
    properties = {}
    required = []
    
    for name, param in sig.parameters.items():
        if name == "self":
            continue
            
        param_schema = _python_type_to_json_schema(param.annotation)
        param_schema["description"] = f"Parameter: {name}"
        
        properties[name] = param_schema
        if param.default == inspect.Parameter.empty:
            required.append(name)
            
    desc = doc.split("\n")[0].strip() if doc else f"Function {func.__name__}"
            
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }

class ToolWrapper:
    """Wraps a python function to expose it as an AI tool."""
    def __init__(self, func: Callable, name: str = None, description: str = None):
        self.func = func
        self.name = name or func.__name__
        self.schema = generate_tool_schema(func)
        if name:
            self.schema["function"]["name"] = name
        if description:
            self.schema["function"]["description"] = description
            
    def __call__(self, **kwargs):
        return self.func(**kwargs)

def tool(name: str = None, description: str = None):
    """Decorator to mark a function as an AI tool."""
    def decorator(func: Callable):
        return ToolWrapper(func, name, description)
    return decorator
