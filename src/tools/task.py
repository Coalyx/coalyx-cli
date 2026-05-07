from src.tools.base import tool
from src.tools.registry import register_tool

@tool(name="agent", description="Launch a specialized sub-agent (e.g. Explorer, Planner).")
def agent(task: str, role: str = "assistant") -> str:
    return f"Stub: Agent {role} started for task '{task}'. (Feature in development)"

@tool(name="team_create", description="Create a team of sub-agents to process parallel tasks.")
def team_create(name: str, members: list) -> str:
    return f"Stub: Team '{name}' created with members {members}. (Feature in development)"

@tool(name="team_delete", description="Delete an existing team.")
def team_delete(name: str) -> str:
    return f"Stub: Team '{name}' deleted."

@tool(name="task_create", description="Create a background task.")
def task_create(command: str) -> str:
    return f"Stub: Background task '{command}' created."

@tool(name="task_list", description="List running background tasks.")
def task_list() -> str:
    return "Stub: No active tasks."

@tool(name="task_stop", description="Stop a background task.")
def task_stop(task_id: str) -> str:
    return f"Stub: Task {task_id} stopped."

@tool(name="cron_create", description="Create a cron job.")
def cron_create(schedule: str, command: str) -> str:
    return f"Stub: Cron job '{command}' scheduled for '{schedule}'."

@tool(name="cron_list", description="List active cron jobs.")
def cron_list() -> str:
    return "Stub: No active cron jobs."

for t in [agent, team_create, team_delete, task_create, task_list, task_stop, cron_create, cron_list]:
    register_tool(t)
