from src.tools.base import tool
from src.tools.registry import register_tool

@tool(name="ask_user_question", description="Ask the user a question and wait for their input.")
def ask_user_question(question: str) -> str:
    print(f"\nAgent asks: {question}")
    try:
        answer = input("Your answer: ")
        return answer
    except Exception as e:
        return f"User did not answer: {e}"

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
    try:
        import io, sys
        old_stdout = sys.stdout
        redirected_output = sys.stdout = io.StringIO()
        exec(code, {})
        sys.stdout = old_stdout
        out = redirected_output.getvalue()
        return out if out else "Code executed successfully."
    except Exception as e:
        import sys
        sys.stdout = sys.__stdout__
        return f"REPL Error: {e}"

for t in [ask_user_question, todo_write, config_tool, enter_plan_mode, exit_plan_mode, repl]:
    register_tool(t)
