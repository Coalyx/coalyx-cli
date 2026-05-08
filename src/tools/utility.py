import subprocess
import sys
from src.tools.base import tool
from src.tools.registry import register_tool
from src.tools.sandbox import get_project_root, get_safe_env


@tool(name="todo_write", description="Add an item to the current session's todo list.")
def todo_write(task: str) -> str:
    try:
        with open(".coalyx/TODO.md", "a") as f:
            f.write(f"- [ ] {task}\n")
        return f"Added '{task}' to TODO list."
    except Exception as e:
        return f"Failed to write todo: {e}"

@tool(name="config", description="View or modify Coalyx settings.")
def config_tool(action: str, key: str = "", value: str = "") -> str:
    return f"Stub: Config {action} executed."

@tool(name="enter_plan_mode", description="Switch into planning mode.")
def enter_plan_mode() -> str:
    return "Stub: Entered planning mode."

@tool(name="exit_plan_mode", description="Exit planning mode.")
def exit_plan_mode() -> str:
    return "Stub: Exited planning mode."

@tool(name="repl", description="Execute code in a Python REPL.")
def repl(code: str) -> str:
    """Execute Python code in an isolated subprocess.

    The code runs in a separate Python process with a 30-second timeout,
    so it cannot access or mutate Coalyx's own memory space.
    """
    try:
        cwd = str(get_project_root()) if get_project_root() else None
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=cwd,
            env=get_safe_env(),
        )
        out = result.stdout
        if result.stderr:
            out += "\nSTDERR:\n" + result.stderr
        return out.strip() if out.strip() else "Code executed successfully."
    except subprocess.TimeoutExpired:
        return "REPL Error: Execution timed out after 30 seconds."
    except Exception as e:
        return f"REPL Error: {e}"

for t in [todo_write, config_tool, enter_plan_mode, exit_plan_mode, repl]:
    register_tool(t)
