import json
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

import typer
from rich.table import Table
from rich.panel import Panel
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.formatted_text import HTML

from src.core.config import set_config_value, setup_environment, load_config
from src.core.schema import (
    Message, Role, ModelConfig, PipelineMode,
    SessionSnapshot, HookEvent, HookConfig,
)
from src.core.pipeline import run_pipeline
from src.core.monitor import SessionMonitor
from src.memory.compactor import compact_messages, apply_compaction, KEEP_RECENT_DEFAULT
from src.memory.session_store import (
    generate_session_id, save_session, load_session, list_sessions,
)
from src.memory.project_memory import (
    load_project_memory, scaffold_memory_file, validate_memory_file,
)
from src.extensions.registry import (
    create_registry, register_hook, register_skill, match_skill,
)
from src.extensions.skill_loader import discover_skills
from src.extensions.hook_runner import run_hooks
from src.tools import setup_tools
from src.cli.ui import (
    console, print_welcome, print_error, print_warning, print_info,
    render_dashboard, print_message, print_debug_info,
    print_zone_warning, print_slash_commands,
)

app = typer.Typer(help="Coalyx CLI — Multi-Task AI Chat with Adaptive Reasoning")

COALYX_DIR = ".coalyx"
SESSIONS_DIR = "sessions"
SKILLS_DIR = "skills"
SETTINGS_FILE = "settings.local.json"

MODEL_CONTEXT_LIMITS = {
    "gemini": 1_000_000,
    "gpt-4o": 128_000,
    "gpt-4": 128_000,
    "gpt-3.5": 16_385,
    "claude": 200_000,
    "ollama": 16_385,
}


def _get_context_limit(model_name: str) -> int:
    """Resolve approximate context window size for a model name."""
    lower = model_name.lower()
    for key, limit in MODEL_CONTEXT_LIMITS.items():
        if key in lower:
            return limit
    return 8_192


def _get_project_root() -> Path:
    """Resolve the project root directory."""
    return Path.cwd()


def _get_coalyx_dir() -> Path:
    """Resolve the .coalyx runtime directory."""
    return _get_project_root() / COALYX_DIR


def _load_hooks_from_settings(registry, coalyx_dir: Path) -> None:
    """Load hook configurations from settings.local.json."""
    settings_path = coalyx_dir / SETTINGS_FILE
    if not settings_path.exists():
        return

    try:
        with open(settings_path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return

    for hook_data in data.get("hooks", []):
        try:
            config = HookConfig(**hook_data)
            register_hook(registry, config)
        except (ValueError, TypeError):
            continue


def ensure_initialized():
    """Silently scaffold .coalyx/ directory if it doesn't exist."""
    root = _get_project_root()
    coalyx_dir = root / COALYX_DIR
    if not coalyx_dir.exists():
        (coalyx_dir / SESSIONS_DIR).mkdir(parents=True, exist_ok=True)
        (coalyx_dir / SKILLS_DIR).mkdir(parents=True, exist_ok=True)
        settings_path = coalyx_dir / SETTINGS_FILE
        settings_path.write_text(json.dumps({"hooks": []}, indent=2))
        scaffold_memory_file(root)

def interactive_setup():
    """Prompt user for essential API keys."""
    console.print("\n[bold yellow]Welcome to Coalyx CLI![/bold yellow]")
    console.print("Let's do a quick setup. You can always run [bold cyan]/config[/bold cyan] inside the chat to change these later.\n")
    
    console.print("[dim]Coalyx requires a Gemini API Key for its Adaptive Reasoning (Embeddings).[/dim]")
    console.print("[dim]Get your API key here: https://aistudio.google.com/api-keys[/dim]")
    gemini_key = console.input("\nEnter Gemini API Key (leave blank to skip): ").strip()
    if gemini_key:
        set_config_value("gemini-api-key", gemini_key)
    
    openai_key = console.input("Enter OpenAI API Key (leave blank to skip): ").strip()
    if openai_key:
        set_config_value("openai-api-key", openai_key)
        
    print_info("Setup complete! Starting chat...\n")

@app.callback(invoke_without_command=True)
def main(
    model: str = typer.Option(
        "gemini/gemini-2.0-flash",
        help="Model identifier (e.g. gpt-4o, gemini/gemini-2.0-flash, ollama/llama3)",
    ),
    mode: str = typer.Option("instant", help="Mode: 'instant' or 'adaptive'"),
    temperature: float = typer.Option(0.7, help="Generation temperature"),
    resume: str = typer.Option("", help="Resume a saved session by ID"),
):
    """Start an interactive chat session."""
    ensure_initialized()
    
    config = load_config()
    if not config.get("gemini-api-key") and not config.get("openai-api-key") and not config.get("ollama-api-base"):
        interactive_setup()
        setup_environment() # reload config
        
    setup_environment()

    pipeline_mode = PipelineMode.ADAPTIVE if mode.lower() == "adaptive" else PipelineMode.INSTANT
    model_config = ModelConfig(model_name=model, temperature=temperature)
    context_limit = _get_context_limit(model)
    monitor = SessionMonitor(max_context_length=context_limit)
    session_id = generate_session_id()
    messages: list[Message] = []
    injected_skills: set[str] = set()

    # --- Setup extensions and tools ---
    setup_tools()
    registry = create_registry()
    coalyx_dir = _get_coalyx_dir()
    _load_hooks_from_settings(registry, coalyx_dir)

    skills_dir = coalyx_dir / SKILLS_DIR
    for skill in discover_skills(skills_dir):
        register_skill(registry, skill)

    # --- Load project memory ---
    project_memory = load_project_memory(_get_project_root())
    if project_memory:
        if not validate_memory_file(project_memory):
            print_warning("COALYX.md exceeds 500 lines. Consider trimming it.")
        messages.append(Message(role=Role.SYSTEM, content=project_memory))
        print_info("Loaded project memory from COALYX.md")

    # --- Resume session ---
    snapshot = None
    if resume:
        session_dir = coalyx_dir / SESSIONS_DIR
        snapshot = load_session(resume, session_dir)
        if snapshot:
            messages = snapshot.messages
            session_id = snapshot.session_id
            model = snapshot.model_name
            pipeline_mode = snapshot.mode
            model_config.model_name = model
            context_limit = _get_context_limit(model)
            monitor = SessionMonitor(max_context_length=context_limit)
            monitor.update(tokens_used=snapshot.total_tokens_used, duration_sec=0.0, model_name=model)
            print_info(f"Resumed session {session_id} ({len(messages)} messages)")
        else:
            print_error(f"Session '{resume}' not found.")
            return

    # --- Check API key for Adaptive mode ---
    config = load_config()
    if pipeline_mode == PipelineMode.ADAPTIVE and not config.get("gemini-api-key"):
        print_warning("Adaptive mode requires Gemini API key. Falling back to Instant mode.")
        pipeline_mode = PipelineMode.INSTANT

    # --- Fire SESSION_START hooks ---
    run_hooks(registry, HookEvent.SESSION_START, {"session_id": session_id, "model": model})

    print_welcome()
    console.print(
        f"  Mode: [bold]{pipeline_mode.value}[/bold] │ "
        f"Model: [bold]{model}[/bold] │ "
        f"Session: [dim]{session_id}[/dim]\n"
    )

    if resume and snapshot:
        for msg in messages:
            if msg.role == Role.USER:
                print_message("user", msg.content)
            elif msg.role == Role.ASSISTANT and msg.content:
                print_message("assistant", msg.content)

    print_info("Type /help for available commands.\n")

    try:
        while True:
            try:
                user_input = _read_user_input()
            except EOFError:
                break

            if user_input is None or not user_input.strip():
                continue

            # --- Slash commands ---
            if user_input.startswith("/"):
                parts = user_input.strip().split(maxsplit=1)
                cmd = parts[0].lower()

                if cmd in ("/quit", "/exit", "/q"):
                    break

                elif cmd == "/help":
                    print_slash_commands()
                    continue

                elif cmd == "/config":
                    console.print("\n[bold yellow]Configuration Setup[/bold yellow]")
                    console.print("[dim]Get your Gemini API key here: https://aistudio.google.com/api-keys[/dim]")
                    gemini_key = console.input("\nEnter Gemini API Key (leave blank to skip): ").strip()
                    if gemini_key:
                        set_config_value("gemini-api-key", gemini_key)
                        print_info("Gemini API Key updated.")
                    openai_key = console.input("Enter OpenAI API Key (leave blank to skip): ").strip()
                    if openai_key:
                        set_config_value("openai-api-key", openai_key)
                        print_info("OpenAI API Key updated.")
                    ollama_base = console.input("Enter Ollama API Base (leave blank to skip): ").strip()
                    if ollama_base:
                        set_config_value("ollama-api-base", ollama_base)
                        print_info("Ollama API Base updated.")
                    setup_environment() # apply new keys
                    continue

                elif cmd == "/sessions":
                    session_dir = coalyx_dir / SESSIONS_DIR
                    sessions = list_sessions(session_dir)
                    if not sessions:
                        print_info("No saved sessions found.")
                    else:
                        table = Table(show_header=True, box=None, padding=(0, 1))
                        table.add_column("Session ID", style="cyan")
                        table.add_column("Model", style="magenta")
                        table.add_column("Messages", style="green")
                        table.add_column("Last Updated", style="dim")
                        for s in sessions[:10]:
                            try:
                                dt = datetime.fromisoformat(s.updated_at)
                                dt_str = dt.strftime("%Y-%m-%d %H:%M")
                            except Exception:
                                dt_str = s.updated_at
                            table.add_row(s.session_id, s.model_name, str(len(s.messages)), dt_str)
                        console.print(Panel(table, title="[bold]Recent Sessions[/bold]", border_style="dim", expand=False))
                        print_info("To resume, start the app with: coalyx --resume <Session ID>")
                    continue

                elif cmd == "/status":
                    console.print(render_dashboard(monitor.stats, monitor.budget))
                    continue

                elif cmd == "/mode":
                    if pipeline_mode == PipelineMode.INSTANT:
                        pipeline_mode = PipelineMode.ADAPTIVE
                    else:
                        pipeline_mode = PipelineMode.INSTANT
                    print_info(f"Switched to {pipeline_mode.value} mode.")
                    console.print(f"  Mode: [bold]{pipeline_mode.value}[/bold] │ Model: [bold]{model}[/bold] │ Session: [dim]{session_id}[/dim]\n")
                    continue

                elif cmd == "/model":
                    if len(parts) < 2:
                        print_warning("Usage: /model <model_name>")
                        continue
                    model = parts[1]
                    model_config.model_name = model
                    context_limit = _get_context_limit(model)
                    monitor = SessionMonitor(max_context_length=context_limit)
                    print_info(f"Model set to {model}. Context limit: {context_limit}")
                    console.print(f"  Mode: [bold]{pipeline_mode.value}[/bold] │ Model: [bold]{model}[/bold] │ Session: [dim]{session_id}[/dim]\n")
                    continue

                elif cmd == "/file":
                    if len(parts) < 2:
                        print_warning("Usage: /file <path>")
                        continue
                    file_content = _load_file_as_input(parts[1])
                    if file_content is None:
                        continue
                    user_input = file_content
                    print_info(f"Loaded {len(user_input)} chars from {parts[1]}")

                elif cmd == "/compact":
                    if len(messages) <= 2:
                        print_info("Not enough messages to compact.")
                        continue
                    with console.status("[bold yellow]Compacting...[/bold yellow]", spinner="dots"):
                        result = compact_messages(messages, model)
                    if result.summary:
                        messages = apply_compaction(messages, result)
                        monitor.reset_budget()
                        monitor.update(
                            tokens_used=len(result.summary) // 4,
                            duration_sec=0.0,
                            model_name=model,
                        )
                        print_info(
                            f"Compacted {result.original_count} messages → 1 summary. "
                            f"~{result.tokens_saved} tokens freed."
                        )
                    else:
                        print_info("Nothing to compact.")
                    continue

                elif cmd == "/clear":
                    messages = []
                    monitor.reset_budget()
                    project_memory = load_project_memory(_get_project_root())
                    if project_memory:
                        messages.append(Message(role=Role.SYSTEM, content=project_memory))
                    print_info("Session cleared. Project memory reloaded.")
                    continue

                else:
                    print_warning(f"Unknown command: {cmd}. Type /help for available commands.")
                    continue

            # --- Skill matching ---
            matched_skill = match_skill(registry, user_input)
            if matched_skill and matched_skill.name not in injected_skills:
                injected_skills.add(matched_skill.name)
                print_info(f"Skill activated: {matched_skill.name}")
                messages.append(
                    Message(role=Role.SYSTEM, content=matched_skill.instructions)
                )

            # --- Generate response ---
            user_msg = Message(role=Role.USER, content=user_input)
            messages.append(user_msg)

            with console.status("[bold green]Thinking...[/bold green]", spinner="dots"):
                result, debug_info = run_pipeline(messages, model_config, pipeline_mode)

            monitor.update(
                tokens_used=result.tokens_used,
                duration_sec=result.duration_sec,
                model_name=model,
            )
            assistant_msg = Message(role=Role.ASSISTANT, content=result.content)
            messages.append(assistant_msg)

            print_debug_info(debug_info)
            print_message("assistant", result.content)
            console.print(render_dashboard(monitor.stats, monitor.budget))
            print_zone_warning(monitor.zone)

            if monitor.should_auto_compact() and len(messages) > KEEP_RECENT_DEFAULT:
                print_warning("Context critical (>90%). Auto-compacting...")
                with console.status("[bold yellow]Compacting...[/bold yellow]", spinner="dots"):
                    compact_result = compact_messages(messages, model)
                if compact_result.summary:
                    messages[:] = apply_compaction(messages, compact_result)
                    monitor.reset_budget()
                    monitor.update(
                        tokens_used=len(compact_result.summary) // 4,
                        duration_sec=0.0,
                        model_name=model,
                    )
                    print_info(
                        f"Auto-compacted {compact_result.original_count} messages. "
                        f"~{compact_result.tokens_saved} tokens freed."
                    )

    except KeyboardInterrupt:
        console.print("\n")

    # --- Fire SESSION_END hooks & save ---
    snapshot = SessionSnapshot(
        session_id=session_id,
        model_name=model,
        mode=pipeline_mode,
        messages=messages,
        total_tokens_used=monitor.stats.total_tokens_used,
    )

    session_dir = _get_coalyx_dir() / SESSIONS_DIR
    try:
        session_dir.mkdir(parents=True, exist_ok=True)
        save_session(snapshot, session_dir)
        print_info(f"Session saved: {session_id}")
    except OSError as e:
        print_warning(f"Could not save session: {e}")

    run_hooks(
        registry, HookEvent.SESSION_END,
        {"session_id": session_id, "total_tokens": monitor.stats.total_tokens_used},
    )

    console.print("[dim]Goodbye.[/dim]")


def _create_prompt_session() -> PromptSession:
    """Create a prompt_toolkit session with multiline paste support.

    Keybindings:
        - Enter: Submit input
        - Alt+Enter / Escape then Enter: Insert a newline (manual multiline)
        - Pasted text: Automatically captured as-is (bracket paste mode)

    Returns:
        Configured PromptSession.
    """
    bindings = KeyBindings()

    @bindings.add("escape", "enter")
    def _insert_newline(event):
        event.current_buffer.insert_text("\n")

    return PromptSession(
        key_bindings=bindings,
        multiline=False,
        enable_open_in_editor=False,
    )


_prompt_session: Optional[PromptSession] = None


def _read_user_input() -> Optional[str]:
    """Read user input with native multiline paste support.

    Uses prompt_toolkit bracket paste mode to auto-detect pasted
    multi-line text. For manual newlines, press Alt+Enter or
    Escape then Enter.

    Returns:
        The user input string, or None if empty.
    """
    global _prompt_session
    if _prompt_session is None:
        _prompt_session = _create_prompt_session()

    try:
        text = _prompt_session.prompt(
            HTML("<b><cyan>You: </cyan></b>"),
        )
    except EOFError:
        raise
    except KeyboardInterrupt:
        return None

    return text


def _load_file_as_input(filepath: str) -> Optional[str]:
    """Load a file's content to use as user input.

    Args:
        filepath: Path to the file to load.

    Returns:
        File content as string, or None on error.
    """
    path = Path(filepath).expanduser()
    if not path.exists():
        print_error(f"File not found: {path}")
        return None
    try:
        content = path.read_text(encoding="utf-8")
        return content
    except OSError as e:
        print_error(f"Cannot read file: {e}")
        return None


if __name__ == "__main__":
    app()
