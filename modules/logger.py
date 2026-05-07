import json
import shutil
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Iterable, Optional, List, Dict

from rich import box
from rich.console import Console  # type: ignore[import-not-found]
from rich.panel import Panel  # type: ignore[import-not-found]
from rich.text import Text
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rich.live import Live
from rich.table import Table
from rich.columns import Columns


class BaseAgentLogger:
    """
    Minimal logger interface used by agents.

    Implementations should be resilient: never raise from logging.
    """

    def info(self, message: str) -> None:
        return None

    def success(self, message: str) -> None:
        return None

    def warning(self, message: str) -> None:
        return None

    def error(self, message: str) -> None:
        return None

    def debug(self, message: str) -> None:
        return None

    def log_agent_action(
        self, agent_name: str, entity: str, action: str, details: Optional[str] = None
    ) -> None:
        return None

    def log_agent_tool_execution(
        self, agent_name: str, tool_name: str, args: dict
    ) -> None:
        return None

    def log_agent_result(
        self, agent_name: str, result: str, success: bool = True
    ) -> None:
        return None


class Logger:
    """
    Simple JSONL logger that records user/assistant/tool messages.
    """

    def __init__(self, path: str) -> None:
        self.path = path

        # create dir if not exists
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)

        # empty the file
        with open(self.path, "w") as f:
            f.write("")

    def _write(self, record: dict) -> None:
        try:
            p = Path(self.path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            # Never raise from logging
            pass

    def _reset(self) -> None:
        with open(self.path, "w") as f:
            f.write("")

    def log_user(self, content: str) -> None:
        self._write({"role": "user", "content": content})

    def log_assistant(self, content: str) -> None:
        self._write({"role": "assistant", "content": content})

    def log_tool(self, tool: str, content: Any) -> None:
        self._write({"role": "tool", "tool": tool, "content": content})

    def log_messages(self, messages: list[dict[str, str]]) -> None:
        for message in messages:
            self._write(message)

    def reset_and_log_messages(self, messages: list[dict[str, str]]) -> None:
        self._reset()
        self.log_messages(messages)


class QAGeneratorLogger(BaseAgentLogger):
    """
    Comprehensive terminal logger for the knowledge graph generation process.
    Provides beautiful, structured output with progress tracking and color coding.
    """

    def __init__(
        self, enable_color: bool | None = None, width: int | None = None
    ) -> None:
        self.enable_color = (
            bool(enable_color) if enable_color is not None else sys.stdout.isatty()
        )
        try:
            terminal_width = shutil.get_terminal_size((100, 20)).columns
        except OSError:
            terminal_width = 100
        self.width = width or terminal_width
        self.console = Console(width=self.width, no_color=not self.enable_color)
        self.start_time = time.time()

        # Progress tracking
        self.current_run = None
        self.current_graph = None
        self.stats = {
            "graphs_completed": 0,
            "graphs_failed": 0,
            "seeds_generated": 0,
            "total_time": 0,
        }

    # ===== RUN-LEVEL LOGGING =====

    def log_run_start(self, run_id: str, total_graphs: int = None) -> None:
        """Log the start of a new run."""
        self.current_run = run_id
        self.stats = {
            "graphs_completed": 0,
            "graphs_failed": 0,
            "seeds_generated": 0,
            "total_time": 0,
        }

        title = f"🚀 Starting Knowledge Graph Generation Run"
        lines = [
            f"Run ID: [bold cyan]{run_id}[/bold cyan]",
            f"Target Graphs: [bold yellow]{total_graphs or 'Unlimited'}[/bold yellow]",
            f"Start Time: [dim]{time.strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
        ]
        self._emit_panel(title, lines, "green", border_style="green")

    def log_run_end(self, run_id: str, completed: int, total_time: float) -> None:
        """Log the end of a run."""
        avg_time = total_time / completed if completed > 0 else 0

        title = f"✅ Run Completed"
        lines = [
            f"Run ID: [bold cyan]{run_id}[/bold cyan]",
            f"Graphs Completed: [bold green]{completed}[/bold green]",
            f"Total Time: [bold blue]{total_time:.2f}s[/bold blue] ({total_time/60:.1f}min)",
            f"Average per Graph: [bold blue]{avg_time:.2f}s[/bold blue] ({avg_time/60:.1f}min)",
        ]
        self._emit_panel(title, lines, "green", border_style="green")

    # ===== GRAPH-LEVEL LOGGING =====

    def log_graph_start(self, graph_id: str, run_id: str, seed_name: str) -> None:
        """Log the start of graph generation."""
        self.current_graph = graph_id
        title = f"📊 Starting Graph Generation"
        lines = [
            f"Graph ID: [bold cyan]{graph_id}[/bold cyan]",
            f"Run ID: [bold magenta]{run_id}[/bold magenta]",
            f"Seed Entity: [bold yellow]{seed_name}[/bold yellow]",
        ]
        self._emit_panel(title, lines, "blue", border_style="blue")

    def log_graph_checkpoint_resume(self, graph_id: str, run_id: str) -> None:
        """Log resuming from checkpoint."""
        title = f"📂 Resuming from Checkpoint"
        lines = [
            f"Graph ID: [bold cyan]{graph_id}[/bold cyan]",
            f"Run ID: [bold magenta]{run_id}[/bold magenta]",
        ]
        self._emit_panel(title, lines, "yellow", border_style="yellow")

    def log_graph_expansion_start(
        self,
        graph_id: str,
        run_id: str,
        seed_name: str,
        expansion_type: str,
        complexity_budget: Optional[int] = None,
    ) -> None:
        """Log the start of graph expansion with unified status."""
        self.current_graph = graph_id

        type_icons = {"checkpoint": "📂", "existing_seed": "🌱", "new_seed": "🆕"}

        type_descriptions = {
            "checkpoint": "Resuming from checkpoint",
            "existing_seed": "Expanding from existing seed",
            "new_seed": "Expanding from newly generated seed",
        }

        icon = type_icons.get(expansion_type, "📊")
        description = type_descriptions.get(
            expansion_type, f"Starting {expansion_type}"
        )

        title = f"{icon} {description}"
        lines = [
            f"Graph ID: [bold cyan]{graph_id}[/bold cyan]",
            f"Run ID: [bold magenta]{run_id}[/bold magenta]",
            f"Seed Entity: [bold yellow]{seed_name}[/bold yellow]",
        ]

        if complexity_budget:
            lines.append(
                f"Target Complexity: [bold green]{complexity_budget}[/bold green] entities"
            )

        color = "yellow" if expansion_type == "checkpoint" else "blue"
        self._emit_panel(title, lines, color, border_style=color)

    def log_graph_saved(
        self, graph_id: str, graph_type: str, entity_count: int, predicate_count: int
    ) -> None:
        """Log successful graph save."""
        type_color = "green" if graph_type == "fuzzy" else "blue"
        title = f"💾 Graph Saved ({graph_type.title()})"
        lines = [
            f"Graph ID: [bold cyan]{graph_id}[/bold cyan]",
            f"Type: [bold {type_color}]{graph_type}[/bold {type_color}]",
            f"Entities: [bold green]{entity_count}[/bold green]",
            f"Predicates: [bold green]{predicate_count}[/bold green]",
        ]
        self._emit_panel(title, lines, type_color, border_style=type_color)

    def log_graph_error(self, graph_id: str, error: str) -> None:
        """Log graph generation error."""
        title = f"❌ Graph Generation Failed"
        lines = [
            f"Graph ID: [bold cyan]{graph_id}[/bold cyan]",
            f"Error: [bold red]{error}[/bold red]",
        ]
        self._emit_panel(title, lines, "red", border_style="red")

    # ===== SEED GENERATION LOGGING =====

    def log_seed_generation_start(
        self,
        run_id: str,
        current_count: int,
        target_count: int,
        topic: Optional[str] = None,
        category: Optional[str] = None,
    ) -> None:
        """Log start of seed generation."""
        title = f"🌱 Generating New Seed"
        lines = [
            f"Run ID: [bold magenta]{run_id}[/bold magenta]",
            f"Progress: [bold cyan]{current_count + 1}/{target_count}[/bold cyan]",
        ]
        if topic:
            lines.append(f"Topic: [bold magenta]{topic}[/bold magenta]")
        if category:
            lines.append(f"Category: [bold green]{category}[/bold green]")
        self._emit_panel(title, lines, "cyan", border_style="cyan")

    def log_seed_generated(
        self,
        seed_name: str,
        category: str,
        generation_time: float,
        description: Optional[str] = None,
        mongo_id: Optional[str] = None,
    ) -> None:
        """Log successful seed generation."""
        title = f"✅ Seed Generated"
        lines = [
            f"Entity: [bold yellow]{seed_name}[/bold yellow]",
            f"Category: [bold green]{category}[/bold green]",
            f"Time: [bold blue]{generation_time:.2f}s[/bold blue]",
        ]
        if mongo_id:
            lines.append(f"Mongo ID: [bold cyan]{mongo_id}[/bold cyan]")
        if description:
            # Wrap description if it's long
            wrapped_desc = textwrap.fill(description, width=min(self.width, 70))
            lines.append(f"Description: [dim]{wrapped_desc}[/dim]")
        self._emit_panel(title, lines, "green", border_style="green")

    def log_seed_generation_failed(self, error: str, generation_time: float) -> None:
        """Log seed generation failure."""
        title = f"❌ Seed Generation Failed"
        lines = [
            f"Error: [bold red]{error}[/bold red]",
            f"Time: [bold blue]{generation_time:.2f}s[/bold blue]",
        ]
        self._emit_panel(title, lines, "red", border_style="red")

    # ===== AGENT ACTIVITY LOGGING =====

    def log_agent_action(
        self, agent_name: str, entity: str, action: str, details: Optional[str] = None
    ) -> None:
        """Log agent performing an action."""
        agent_colors = {
            "Fuzzifier": "magenta",
            "KnowledgeSearch": "cyan",
            "PredicateSampler": "yellow",
            "SeedGen": "green",
        }
        color = agent_colors.get(agent_name, "blue")

        if details:
            message = f"[bold {color}]{agent_name}[/] [dim]for[/dim] [bold white]{entity}[/] [yellow]{action}[/]: [dim]{details}[/dim]"
        else:
            message = f"[bold {color}]{agent_name}[/] [dim]for[/dim] [bold white]{entity}[/] [yellow]{action}[/]"

        self.console.print(message)

    def log_agent_tool_execution(
        self, agent_name: str, tool_name: str, args: dict
    ) -> None:
        """Log agent tool execution."""
        agent_colors = {
            "Fuzzifier": "magenta",
            "KnowledgeSearch": "cyan",
            "PredicateSampler": "yellow",
            "SeedGen": "green",
        }
        color = agent_colors.get(agent_name, "blue")

        # Format args nicely
        if len(str(args)) > 100:
            args_str = str(args)[:97] + "..."
        else:
            args_str = str(args)

        message = f"[dim]├─[/dim] [bold {color}]{agent_name}[/] [blue]executes[/blue] [bold cyan]{tool_name}[/] [dim]{args_str}[/dim]"
        self.console.print(message)

    def log_agent_result(
        self, agent_name: str, result: str, success: bool = True
    ) -> None:
        """Log agent completion result."""
        agent_colors = {
            "Fuzzifier": "magenta",
            "KnowledgeSearch": "cyan",
            "PredicateSampler": "yellow",
            "SeedGen": "green",
        }
        color = agent_colors.get(agent_name, "blue")
        status_icon = "✅" if success else "❌"
        status_color = "green" if success else "red"

        if len(result) > 100:
            result = result[:97] + "..."

        if success:
            message = f"[dim]└─[/dim] [bold {color}]{agent_name}[/] {status_icon} [dim]{result}[/dim]"
        else:
            message = f"[bold red]└─ {agent_name} {status_icon} {result}[/bold red]"
        self.console.print(message)

    def log_sampler_entity_added(
        self, entity_id: str, name: str, category: str
    ) -> None:
        """Log an entity added by the PredicateSampler."""
        message = (
            f"[dim]├─[/dim] [bold cyan]Entity[/] "
            f"[bold yellow]{entity_id}[/] [bold white]{name}[/] [dim]({category})[/dim]"
        )
        self.console.print(message)

    def log_sampler_predicate_added(self, content: str) -> None:
        """Log a predicate added by the PredicateSampler with stylized entity placeholders."""
        import re

        # Find placeholders like <|-E0-|> and stylize them
        # We use a background color for the entity IDs to make them pop
        stylized_content = re.sub(
            r"<\|-(E\d+)-\|>", r"[bold black on bright_cyan] \1 [/]", content
        )

        message = f"[dim]├─[/dim] [bold green]Predicate[/] [white]{stylized_content}[/]"
        self.console.print(message)

    def log_sampling_complete(
        self, entity_name: str, entities_added: int, predicates_added: int, message: str
    ) -> None:
        """Log completion of predicate sampling with summary stats."""
        # Use a closing branch for the summary to tie it all together
        status_icon = "✅" if entities_added > 0 or predicates_added > 0 else "ℹ️"
        summary_msg = (
            f"[dim]└─[/dim] [bold green]{status_icon} Sampling Complete:[/] "
            f"Added [bold cyan]{entities_added}[/] entities and [bold green]{predicates_added}[/] predicates. "
            f"[dim]{message}[/dim]"
        )
        self.console.print(summary_msg)
        self.console.print()  # Spacing

    def log_knowledge_search_complete(
        self,
        entity_name: str,
        predicates_found: int,
        urls_processed: int,
        search_time: float,
    ) -> None:
        """Log completion of knowledge search with summary."""
        title = f"🔍 Knowledge Search Complete"
        lines = [
            f"Entity: [bold cyan]{entity_name}[/bold cyan]",
            f"Predicates Found: [bold green]{predicates_found}[/bold green]",
            f"URLs Processed: [bold blue]{urls_processed}[/bold blue]",
            f"Search Time: [bold yellow]{search_time:.2f}s[/bold yellow]",
        ]
        self._emit_panel(title, lines, "cyan", border_style="cyan")

    # ===== PROGRESS AND STATUS LOGGING =====

    def log_entity_expansion_start(
        self,
        entity_name: str,
        entity_id: str,
        expansion_probability: float,
        current_complexity: int,
        max_complexity: int,
    ) -> None:
        """Log the start of entity expansion with probability and progress info."""
        progress_percent = min(100, (current_complexity / max_complexity) * 100)
        prob_percent = expansion_probability * 100

        title = f"🔍 Expanding Entity"
        lines = [
            f"Entity: [bold yellow]{entity_name}[/bold yellow] ([dim]{entity_id}[/dim])",
            f"Expansion Probability: [bold magenta]{prob_percent:.1f}%[/bold magenta]",
            f"Graph Progress: [bold green]{current_complexity}/{max_complexity}[/bold green] ({progress_percent:.1f}%)",
        ]
        self._emit_panel(title, lines, "cyan", border_style="cyan")

    def log_knowledge_search_turn(
        self, entity_name: str, turn_number: int, max_turns: int, summary: str
    ) -> None:
        """Log a knowledge search turn result in branching format with improved styling."""
        # Format the result as a branching message
        step_indicator = f"{turn_number}/{max_turns}"
        message = f"[dim]└─[/dim] [bold cyan]KnowledgeSearch[/] [cyan]result[/cyan] [yellow]for turn[/yellow] [bold yellow]{step_indicator}[/bold yellow]: [dim white]{summary}[/dim white]"
        self.console.print(message)

    def log_predicates_discovered(self, entity_name: str, predicate_count: int) -> None:
        """Log predicates discovered during knowledge search."""
        title = f"📋 Predicates Discovered"
        lines = [
            f"Entity: [bold cyan]{entity_name}[/bold cyan]",
            f"Predicates Found: [bold green]{predicate_count}[/bold green]",
        ]
        self._emit_panel(title, lines, "green", border_style="green")

    def log_predicate_sampling_start(
        self, entity_name: str, predicate_count: int
    ) -> None:
        """Log start of predicate sampling."""
        message = f"[bold yellow]PredicateSampler[/] [dim]for[/dim] [bold white]{entity_name}[/] [yellow]sampling from[/] [bold yellow]{predicate_count}[/] [yellow]candidates...[/]"
        self.console.print(message)

    def display_predicates_table(
        self, predicates: List[Dict[str, Any]], title: str = "Available Predicates"
    ) -> None:
        """Display predicates in a formatted table with improved styling."""
        if not predicates:
            self.console.print(f"[dim]No predicates available[/dim]")
            return

        table = Table(
            title=f"[bold cyan]{title}[/bold cyan]",
            show_header=True,
            header_style="bold white on blue",
            border_style="bright_blue",
            box=box.ROUNDED,
            expand=False,
        )
        table.add_column("#", style="dim", width=4, justify="center")
        table.add_column("Predicate Content", style="white")
        table.add_column("Src", style="dim green", width=5, justify="center")

        for i, pred in enumerate(predicates, 1):
            content = pred.get("content", "")
            topic = pred.get("topic", "General")
            links = pred.get("links", [])

            # Combine topic and content
            full_content = f"[bold cyan]{topic}[/bold cyan]\n{content}"

            # Truncate content if too long for the simple table
            if len(full_content) > 120:
                full_content = full_content[:117] + "..."

            # Format sources count
            sources_count = str(len(links)) if links else "0"

            table.add_row(str(i), full_content, sources_count)

        self.console.print(table)
        self.console.print()

    def display_predicates_table_detailed(
        self,
        predicates: List[Dict[str, Any]],
        title: str = "Available Predicates",
        entity_name: Optional[str] = None,
    ) -> None:
        """Display predicates in a detailed table format with improved styling (no truncation, full sources)."""
        if not predicates:
            self.console.print(f"[dim]No predicates available[/dim]")
            return

        # Create enhanced title with discovery information
        enhanced_title = title
        if entity_name:
            enhanced_title = (
                f"{title} • Entity: [bold yellow]{entity_name}[/bold yellow]"
            )

        table = Table(
            title=f"[bold cyan]{enhanced_title}[/bold cyan]",
            show_header=True,
            header_style="bold white on blue",
            border_style="bright_blue",
            box=box.ROUNDED,
            expand=False,
        )
        table.add_column("#", style="dim", width=4, justify="center")
        table.add_column("Predicate Content & Evidence", style="white")

        for i, pred in enumerate(predicates, 1):
            content = pred.get("content", "")
            topic = pred.get("topic", "General")
            links = pred.get("links", [])

            # Format links with icons
            links_text = (
                "\n".join([f"[dim]🔗 {link}[/dim]" for link in links])
                if links
                else "[dim italic]No sources available[/dim italic]"
            )

            # Combine topic, content and links
            cell_content = f"[bold cyan]{topic}[/bold cyan]\n{content}\n\n[bold white]Sources:[/bold white]\n{links_text}"

            table.add_row(str(i), cell_content)

        self.console.print(table)
        self.console.print()

    def log_fuzzification_start(self) -> None:
        """Log start of graph fuzzification."""
        title = f"🎭 Starting Graph Fuzzification"
        lines = ["Converting original graph to fuzzy representation..."]
        self._emit_panel(title, lines, "magenta", border_style="magenta")

    def log_fuzzification_complete(self, changes_made: int) -> None:
        """Log completion of fuzzification."""
        title = f"🎭 Fuzzification Complete"
        lines = [
            f"Applied [bold magenta]{changes_made}[/bold magenta] fuzzy transformations"
        ]
        self._emit_panel(title, lines, "magenta", border_style="magenta")

    # ===== CHECKPOINT AND RESUME LOGGING =====

    def log_checkpoint_resume_attempt(
        self, graph_id: str, run_id: str, checkpoint_count: int
    ) -> None:
        """Log checkpoint resume attempt."""
        title = f"📂 Attempting Checkpoint Resume"
        lines = [
            f"Graph ID: [bold cyan]{graph_id}[/bold cyan]",
            f"Run ID: [bold magenta]{run_id}[/bold magenta]",
            f"Available Checkpoints: [bold yellow]{checkpoint_count}[/bold yellow]",
        ]
        self._emit_panel(title, lines, "yellow", border_style="yellow")

    def log_checkpoint_skip(self, reason: str, count: int) -> None:
        """Log checkpoint skip."""
        # Show concise skip message during processing
        message = f"[dim]⏭️  Skipped {count} checkpoint{'s' if count != 1 else ''} (fuzzy graph exists)[/dim]"
        self.console.print(message)

    def log_unused_seeds_processing(self, seed_count: int) -> None:
        """Log processing of unused seeds."""
        title = f"🔄 Processing Unused Seeds"
        lines = [
            f"Found [bold yellow]{seed_count}[/bold yellow] unused seeds to expand"
        ]
        self._emit_panel(title, lines, "blue", border_style="blue")

    # ===== GENERAL UTILITY LOGGING =====

    def info(self, message: str) -> None:
        """General info message."""
        self.console.print(f"[dim]ℹ️  {message}[/dim]")

    def success(self, message: str) -> None:
        """Success message."""
        self.console.print(f"[bold green]✅ {message}[/bold green]")

    def warning(self, message: str) -> None:
        """Warning message."""
        self.console.print(f"[bold yellow]⚠️  {message}[/bold yellow]")

    def error(self, message: str) -> None:
        """Error message."""
        self.console.print(f"[bold red]❌ {message}[/bold red]")

    def debug(self, message: str) -> None:
        """Debug message."""
        self.console.print(f"[dim]🔍 {message}[/dim]")

    def separator(self, title: Optional[str] = None) -> None:
        """Print a separator line."""
        if title:
            self.console.print(f"\n[bold blue]{'='*10} {title} {'='*10}[/bold blue]\n")
        else:
            self.console.print(f"[dim]{'─'*50}[/dim]")

    # ===== PRIVATE HELPER METHODS =====

    def _emit_panel(
        self, title: str, lines: list[str], color: str, border_style: str = "blue"
    ) -> None:
        """Emit a formatted panel."""
        body = "\n".join(lines)
        panel = Panel(
            body,
            title=f"[bold {color}]{title}[/bold {color}]",
            expand=False,
            width=min(self.width, 80),
            border_style=border_style if self.enable_color else "",
            padding=(1, 2),
        )
        self.console.print(panel)
        self.console.print()  # Add spacing

    def _short_agent(self, agent_id: str) -> str:
        """Return the last UUID segment if the id looks like a UUID; otherwise, return the tail segment after '-'."""
        try:
            parts = (agent_id or "").split("-")
            if parts:
                return parts[-1] or agent_id
            return agent_id
        except Exception:
            return agent_id

    # ===== LEGACY METHODS (for backward compatibility) =====

    def print_current_entity(
        self, entity: str, depth: int, predicate_budget: int, entity_budget: int
    ) -> None:
        """Legacy method for backward compatibility."""
        self.log_expansion_progress(entity, depth, entity_budget, predicate_budget)

    def log_fact_action(
        self, agent_id: str, entity: str, tool_name: str, reference: str | None
    ) -> None:
        """Legacy method for backward compatibility."""
        self.log_agent_tool_execution(agent_id, tool_name, {"reference": reference})

    def log_fact_yield(
        self, agent_id: str, entity: str, extracted_count: int, total_count: int
    ) -> None:
        """Legacy method for backward compatibility."""
        self.log_agent_result(
            agent_id, f"Extracted {extracted_count} facts (total: {total_count})", True
        )


# Backward-compatible alias (older scripts/tests may still import TerminalLogger).
TerminalLogger = QAGeneratorLogger
