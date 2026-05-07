from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.progress_bar import ProgressBar
from rich.text import Text
from rich.rule import Rule

from src.core.schema import MonitorStats, ContextBudget, ContextZone

console = Console()

ZONE_COLORS = {
    ContextZone.FREE: "green",
    ContextZone.MONITORED: "yellow",
    ContextZone.COMPACT_SUGGESTED: "bright_red",
    ContextZone.CRITICAL: "bold red",
}

ZONE_LABELS = {
    ContextZone.FREE: "Free",
    ContextZone.MONITORED: "Monitored",
    ContextZone.COMPACT_SUGGESTED: "Compact Suggested",
    ContextZone.CRITICAL: "CRITICAL",
}


def print_welcome():
    """Display the welcome banner."""
    banner = Text()
    banner.append("╔══════════════════════════════════════╗\n", style="cyan")
    banner.append("║       ", style="cyan")
    banner.append("C O A L Y X   C L I", style="bold bright_cyan")
    banner.append("            ║\n", style="cyan")
    banner.append("║  ", style="cyan")
    banner.append("Adaptive Reasoning", style="italic dim")
    banner.append("                  ║\n", style="cyan")
    banner.append("╚══════════════════════════════════════╝", style="cyan")
    console.print(banner)
    console.print()


def print_error(msg: str):
    """Display an error message."""
    console.print(f"[bold red]✗ Error:[/bold red] {msg}")


def print_warning(msg: str):
    """Display a warning message."""
    console.print(f"[bold yellow]⚠ Warning:[/bold yellow] {msg}")


def print_info(msg: str):
    """Display an informational message."""
    console.print(f"[dim cyan]ℹ {msg}[/dim cyan]")


def render_memory_bar(budget: ContextBudget) -> Text:
    """Render a visual memory usage bar with zone coloring.

    Args:
        budget: Current context budget.

    Returns:
        Rich Text object representing the memory bar.
    """
    if budget.total_capacity == 0:
        return Text("N/A")

    ratio = min(budget.used_tokens / budget.total_capacity, 1.0)
    bar_width = 30
    filled = int(ratio * bar_width)
    empty = bar_width - filled

    color = ZONE_COLORS[budget.zone]
    label = ZONE_LABELS[budget.zone]

    bar = Text()
    bar.append("│", style="dim")
    bar.append("█" * filled, style=color)
    bar.append("░" * empty, style="dim")
    bar.append("│", style="dim")
    bar.append(f" {ratio:.0%} ", style=f"bold {color}")
    bar.append(f"[{label}]", style=color)

    return bar


def render_dashboard(stats: MonitorStats, budget: ContextBudget = None) -> Panel:
    """Render the session monitor dashboard.

    Args:
        stats: Current session monitor statistics.
        budget: Optional context budget for memory bar display.

    Returns:
        Rich Panel containing the dashboard.
    """
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Metric", style="cyan", min_width=14)
    table.add_column("Value", style="magenta")

    table.add_row("  Model", stats.model_info)
    table.add_row("  Tokens Used", f"{stats.total_tokens_used:,}")
    table.add_row("  Context Left", f"{stats.remaining_context_length:,}")
    table.add_row("  Speed", f"{stats.avg_speed_tokens_per_sec:.1f} tok/s")
    table.add_row("  Memory", stats.available_vram)

    if budget:
        memory_bar = render_memory_bar(budget)
        table.add_row("  Context", memory_bar)

    return Panel(
        table,
        title="[bold bright_cyan]Session Monitor[/bold bright_cyan]",
        border_style="bright_blue",
        expand=False,
        padding=(0, 1),
    )


def print_message(role: str, content: str):
    """Display a chat message in a styled format without full box borders 
    to prevent alignment issues with wide characters (emojis, math).

    Args:
        role: Message role ('user' or 'assistant').
        content: The message content (rendered as markdown for assistant).
    """
    if role == "user":
        console.print(f"\n[bold green]You:[/bold green] {content}\n")
    else:
        console.print(Rule(title="[bold bright_blue]Coalyx[/bold bright_blue]", style="bright_blue", align="left"))
        if not content or not content.strip():
            console.print(Text("(no response)", style="dim italic"))
        else:
            import time
            from rich.live import Live
            # Simulate token streaming effect
            chunk_size = 8
            delay = 0.015
            with Live(auto_refresh=False, console=console) as live:
                for i in range(0, len(content) + chunk_size, chunk_size):
                    chunk = content[:i]
                    if chunk:
                        try:
                            live.update(Markdown(chunk), refresh=True)
                        except Exception:
                            live.update(Text(chunk), refresh=True)
                    time.sleep(delay)
                # Ensure the final full content is rendered properly
                try:
                    live.update(Markdown(content), refresh=True)
                except Exception:
                    live.update(Text(content), refresh=True)

        console.print(Rule(style="bright_blue"))
        console.print()


def print_debug_info(debug_info: dict):
    """Display pipeline debug information (Adaptive mode only).

    Args:
        debug_info: Dictionary with consistency score and stage2 status.
    """
    if not debug_info:
        return

    consistency = debug_info.get("consistency", 0.0)
    stage2 = debug_info.get("stage2_triggered", False)

    msg = Text()
    msg.append("  ◈ ", style="dim")
    msg.append(f"Consistency: {consistency:.2f}", style="dim")
    msg.append(" │ ", style="dim")

    if stage2:
        msg.append("Stage 2 Activated (Self-Doubt)", style="bold yellow")
    else:
        msg.append("High Certainty (Stage 1 Only)", style="bold green")

    console.print(msg)


def print_zone_warning(zone: ContextZone):
    """Display a warning based on the current context zone.

    Args:
        zone: The current context zone.
    """
    if zone == ContextZone.MONITORED:
        print_warning("Context usage above 50%. Monitoring token budget.")
    elif zone == ContextZone.COMPACT_SUGGESTED:
        print_warning("Context usage above 70%. Consider running /compact to free space.")
    elif zone == ContextZone.CRITICAL:
        print_error("Context usage above 90%! Run /compact or /clear immediately.")


def print_slash_commands():
    """Display available slash commands."""
    table = Table(show_header=True, box=None, padding=(0, 1))
    table.add_column("Command", style="cyan")
    table.add_column("Description", style="dim")

    table.add_row("/file <path>", "Load a file as input prompt")
    table.add_row("/model <name>", "Switch the active AI model")
    table.add_row("/compact", "Compress conversation history to free context")
    table.add_row("/clear", "Clear all messages and reset session")
    table.add_row("/mode", "Toggle between Instant and Adaptive Reasoning")
    table.add_row("/status", "Show current session status and dashboard")
    table.add_row("/help", "Show this help message")
    table.add_row("/quit", "Exit the chat session")
    table.add_row("", "")
    table.add_row('\"\"\"', "Start/end multiline input block")
    table.add_row("line\\\\", "Continue input on next line")

    console.print(Panel(table, title="[bold]Commands[/bold]", border_style="dim", expand=False))
