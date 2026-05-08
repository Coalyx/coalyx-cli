import inspect
from typing import Callable, Any, Dict, List, Optional, Union, get_args, get_origin

def _python_type_to_json_schema(py_type: Any) -> Dict[str, Any]:
    """Convert a Python type hint to a JSON Schema type.

    Handles primitives, List[T], Dict[K,V], Optional[T], and
    Union types. Falls back to ``{"type": "string"}`` for
    unrecognised annotations.
    """
    if py_type is inspect.Parameter.empty or py_type is None:
        return {"type": "string"}

    # Primitives
    if py_type == int:
        return {"type": "integer"}
    elif py_type == float:
        return {"type": "number"}
    elif py_type == bool:
        return {"type": "boolean"}
    elif py_type == str:
        return {"type": "string"}

    origin = get_origin(py_type)
    args = get_args(py_type)

    # list / List[T]
    if origin is list:
        if args:
            item_schema = _python_type_to_json_schema(args[0])
        else:
            item_schema = {"type": "string"}
        return {"type": "array", "items": item_schema}

    # dict / Dict[K, V]
    if origin is dict:
        return {"type": "object"}

    # Optional[T] is Union[T, None]
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _python_type_to_json_schema(non_none[0])
        # True union — just use string as a safe fallback
        return {"type": "string"}

    return {"type": "string"}

def generate_tool_schema(func: Callable) -> Dict[str, Any]:
    """Generate OpenAI-compatible tool schema from a Python function."""
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    
    param_docs = {}
    for line in doc.split("\n"):
        line = line.strip()
        if ":" in line and ("Args:" not in line):
            parts = line.split(":", 1)
            p_name = parts[0].strip().split()[0]
            param_docs[p_name] = parts[1].strip()

    properties = {}
    required = []
    
    for name, param in sig.parameters.items():
        if name == "self":
            continue
            
        param_schema = _python_type_to_json_schema(param.annotation)
        param_schema["description"] = param_docs.get(name, f"The {name} for the tool.")
        
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
    def __init__(self, func: Callable, name: Optional[str] = None, description: Optional[str] = None):
        self.func = func
        self.name = name or func.__name__
        self.schema = generate_tool_schema(func)
        if name:
            self.schema["function"]["name"] = name
        if description:
            self.schema["function"]["description"] = description
            
    def __call__(self, **kwargs):
        return self.func(**kwargs)

def tool(name: Optional[str] = None, description: Optional[str] = None):
    """Decorator to mark a function as an AI tool."""
    def decorator(func: Callable):
        return ToolWrapper(func, name, description)
    return decorator
